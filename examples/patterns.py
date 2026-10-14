"""Every composition block, run against a real local model, each run
captured as a flow-xray trace.

    ollama pull llama3.2:3b
    pip install "agent-runloop[trace]"
    python examples/patterns.py            # traces in ./traces, report at the end
"""
from __future__ import annotations

import asyncio
import re
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from agent_loop import (
    Agent, Budget, Context, Fallback, Handoff, LLMClassifier, LLMCritic, Parallel, Refine, Router, Sequence,
    SqliteAgentRunStore, Timeout, ToolSpec, Verdict, majority_vote, step,
)
from agent_loop.ollama import OllamaBackend

KB = {
    "refunds": "Refunds are paid back to the original card within 5 to 7 business days.",
    "pro-plan": "The Pro plan costs 29 USD per month, billed monthly.",
    "export": "If the export button does nothing, clear the browser cache and retry; it is a known caching bug.",
    "warranty": "Monitors have a 3 year warranty.",
}


class SearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=200)


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"\w+", text.lower()) if len(w) > 3}


async def search(args: SearchArgs, trace_id: str):
    """Toy search: documents sharing a word stem with the query, best first."""
    query = _stems(args.query)
    scored = sorted(((len(query & _stems(v)), k, v) for k, v in KB.items()), reverse=True)
    hits = [{"doc_id": k, "text": v} for score, k, v in scored if score]
    return hits[:2] or [{"doc_id": None, "text": "no results"}]


class TicketArgs(BaseModel):
    summary: str = Field(min_length=5, max_length=300)


async def create_ticket(args: TicketArgs, trace_id: str):
    return {"ticket_id": "T-1", "summary": args.summary}


SEARCH = ToolSpec("search_knowledge_base", "Search the support knowledge base.", SearchArgs, search)
TICKET = ToolSpec("create_ticket", "Open a support ticket.", TicketArgs, create_ticket, writes=True)
ANSWER_FROM_TOOLS = "Answer only from search results. Be brief."

support = Agent("support", f"You are a support agent. {ANSWER_FROM_TOOLS}", [SEARCH])
billing = Agent("billing", f"You are the billing agent. {ANSWER_FROM_TOOLS}", [SEARCH])
technical = Agent("technical", f"You are the technical support agent. {ANSWER_FROM_TOOLS}", [SEARCH])


async def keyword_route(task: str, ctx: Context) -> str:
    return "billing" if re.search(r"refund|price|plan|invoice|cost", task, re.I) else "technical"


async def word_count(task: str, ctx: Context) -> str:
    return str(len(task.split()))


def mentions_a_number(task: str, answer: str, ctx: Context) -> Verdict:
    return Verdict(ok=bool(re.search(r"\d", answer)), feedback="State the exact number of days.")


def mentions_the_card(task: str, answer: str, ctx: Context) -> Verdict:
    return Verdict(ok="card" in answer.lower(), feedback="Say where the money goes: to the original card.")


def scenarios():
    # Voters must differ (prompt, model or temperature): identical agents at
    # temperature 0 give one answer three times and the vote checks nothing.
    styles = ["Classify the sentiment of the user's message.",
              "You are a strict reviewer of customer feedback. Decide its overall tone.",
              "Read the message as a support lead would and judge how the customer feels."]
    labeler = lambda n: Agent(f"labeler{n}", f"{styles[n - 1]} Answer with exactly one word: "  # noqa: E731
                                              "positive, negative or neutral.")
    return [
        ("agent", support, "How long does a refund take?",
         lambda o: o.ok and "5" in (o.answer or "")),
        ("router(function)", Router({"billing": billing, "technical": technical}, keyword_route),
         "How much is the Pro plan per month?", lambda o: o.ok and o.data["route"] == "billing"),
        ("router(llm)", Router({"billing": billing, "technical": technical},
                               LLMClassifier({"billing": "payments, refunds, prices, plans",
                                              "technical": "bugs, errors, buttons not working"}), default="technical"),
         "The export button does nothing when I click it.", lambda o: o.ok and o.data["route"] == "technical"),
        ("sequence(waterfall)", Sequence(support, Agent("writer", "Rewrite the result of the previous step as one "
                                                                  "polite sentence to a customer.")),
         "How long does a refund take?", lambda o: o.ok and len(o.runs) == 2 and "5" in (o.answer or "")),
        ("parallel(vote)", Parallel(labeler(1), labeler(2), labeler(3), combine=majority_vote),
         "I love this product, it works great!", lambda o: o.ok and o.answer == "positive"),
        ("parallel(aggregator agent)", Parallel(billing, technical, combine=Agent(
            "editor", "Merge the candidate answers into one short answer.")),
         "How much is the Pro plan per month?", lambda o: o.ok and len(o.runs) == 3),
        ("handoff", Handoff(Agent("triage", "You are triage. You cannot answer billing questions yourself: hand "
                                            "billing questions over to the billing agent."),
                            {"billing": billing}, descriptions={"billing": "For refunds, prices and plans."}),
         "How long does a refund take?", lambda o: o.data.get("handoff") == "billing" and o.ok),
        ("refine(rule critic)", Refine(support, mentions_a_number, max_rounds=2),
         "How long does a refund take?", lambda o: o.ok and o.data["reviews"][-1]["ok"]),
        ("refine(rejected, then revised)", Refine(support, mentions_the_card, max_rounds=3),
         "How long does a refund take?",
         lambda o: o.ok and o.data["rounds"] >= 2 and not o.data["reviews"][0]["ok"] and "card" in o.answer.lower()),
        ("refine(llm critic)", Refine(support, LLMCritic("The answer must state how many business days a refund "
                                                         "takes."), max_rounds=2),
         "How long does a refund take?", lambda o: o.status in ("answered", "rejected_final") and o.data["reviews"]),
        ("fallback", Fallback(Agent("tiny", "Answer.", [SEARCH], budget=Budget(max_tokens=50)), support),
         "What is the warranty on monitors?", lambda o: o.ok and o.data["tried"][0] == "budget_exceeded"),
        ("timeout", Timeout(support, seconds=0.2), "How long does a refund take?",
         lambda o: o.status == "timeout"),
        ("custom function step", Sequence(step(word_count)), "one two three", lambda o: o.answer == "3"),
        ("nested: router -> sequence", Router({"billing": Sequence(billing, Agent("writer", "Rewrite the previous "
                                                                                           "result as one sentence.")),
                                               "technical": technical}, keyword_route),
         "When will my refund arrive?",
         lambda o: o.ok and o.data["route"] == "billing" and len(o.runs) == 2 and "5" in (o.answer or "")),
        ("approval pause stops the composite", Sequence(Agent("ticketer", "Open a support ticket for the user's "
                                                                          "problem.", [TICKET]), support),
         "Please open a ticket: export to PDF crashes.",
         lambda o: o.status == "awaiting_approval" and o.runs[-1].pending_call["tool"] == "create_ticket"),
    ]


async def main() -> int:
    try:
        from flow_xray import TraceStore
    except ImportError:
        TraceStore = None
    store = TraceStore(Path("traces"), max_llm_calls=12) if TraceStore else None
    backend = OllamaBackend("llama3.2:3b")
    runs_db = str(Path(tempfile.mkdtemp()) / "runs.sqlite3")
    failed = 0
    for name, block, task, expect in scenarios():
        ctx = Context(backend=backend, run_store=SqliteAgentRunStore(runs_db), context={"subject": "demo"})
        capture = store.capture("example", name, ctx.trace_id) if store else _nothing()
        with capture:
            outcome = await block(task, ctx)
        ok = bool(expect(outcome))
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name:<36} status={outcome.status:<17} runs={len(outcome.runs)} "
              f"data={ {k: v for k, v in outcome.data.items() if k != 'reviews'} } answer={(outcome.answer or '')[:70]!r}")
    if store:
        print("\n" + store.report(limit=len(scenarios())))
    print(f"\n{len(scenarios()) - failed}/{len(scenarios())} scenarios as expected")
    return 1 if failed else 0


class _nothing:
    def __enter__(self):
        return {}

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
