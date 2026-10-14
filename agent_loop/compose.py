"""Composing agents: every building block is a step, and every step is

    async step(task: str, ctx: Context) -> Outcome

so blocks nest freely: a Router whose routes are Sequences, a Parallel
inside a Fallback, an ordinary async function in the middle of a chain.

    Agent       one run of the agent loop (instructions, tools, model, budget)
    Sequence    waterfall: each step's answer becomes part of the next task
    Router      a classifier (function or model) picks one route
    Parallel    steps run concurrently; a combiner merges their answers
    Handoff     an agent hands the case to another with a structured message
    Refine      worker answers, critic reviews, worker revises (max_rounds)
    Fallback    the first step that answers wins
    Timeout     a deadline around any step
    step(fn)    any `async (task, ctx) -> str | Outcome` function as a step

Limits, stated: when an agent inside a composite pauses for human
approval, the composite stops and returns that `awaiting_approval`
outcome; resuming the paused agent run (`resume_agent`) does not continue
the rest of the composite. Token and tool budgets are per agent run;
`Timeout` bounds wall time across a whole composite.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError, create_model

from agent_loop._trace import trace
from agent_loop.backend import LLMBackend
from agent_loop.hooks import bind_trace, log_step, new_trace_id
from agent_loop.loop import AgentResult, Budget, FinalCheck, ToolSpec, run_agent
from agent_loop.runs import SqliteAgentRunStore


@dataclass
class Context:
    """What every step of one composite run shares. `context` is the
    caller's trusted data (user, tenant, ...), passed to every agent run
    and to tool authorization; it never comes from model output."""
    backend: LLMBackend
    run_store: SqliteAgentRunStore | None = None
    context: dict[str, Any] = field(default_factory=dict)
    trace_id: str = field(default_factory=new_trace_id)
    events: asyncio.Queue | None = None


@dataclass
class Outcome:
    status: str  # "answered", an agent run status, or a composite one ("unroutable", "failed")
    answer: str | None
    runs: list[AgentResult] = field(default_factory=list)  # every agent run inside, in order
    data: dict[str, Any] = field(default_factory=dict)  # what the block decided (route, votes, rounds...)

    @property
    def ok(self) -> bool:
        return self.status == "answered"


class Step(Protocol):
    async def __call__(self, task: str, ctx: Context) -> Outcome: ...


StepLike = Step | Callable[[str, Context], Awaitable[Any]]


def step(fn: Callable[[str, Context], Awaitable[Any]]) -> Step:
    """An async function as a step. It may return an Outcome, or a string
    (answered) / None (failed)."""
    if isinstance(fn, _Block):
        return fn
    return _FunctionStep(fn)


class _Block:
    name: str = ""

    async def __call__(self, task: str, ctx: Context) -> Outcome:
        with bind_trace(ctx.trace_id):
            return await self.run(task, ctx)

    async def run(self, task: str, ctx: Context) -> Outcome:  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:  # traces show the block, not its whole config
        return f"{type(self).__name__}({self.name!r})" if self.name else f"{type(self).__name__}()"


class _FunctionStep(_Block):
    def __init__(self, fn):
        self.fn, self.name = fn, getattr(fn, "__name__", "step")

    @trace
    async def run(self, task: str, ctx: Context) -> Outcome:
        out = await self.fn(task, ctx)
        if isinstance(out, Outcome):
            return out
        return Outcome("answered" if out is not None else "failed", None if out is None else str(out))


def _as_step(s: StepLike) -> Step:
    return s if isinstance(s, _Block) else step(s)


# -- one agent ----------------------------------------------------------------


class Agent(_Block):
    """An agent as configuration; calling it runs the loop once.
    `backend` and `budget` default to the context's backend and Budget()."""

    def __init__(self, name: str, instructions: str, tools: list[ToolSpec] | None = None, *,
                 backend: LLMBackend | None = None, budget: Budget | None = None,
                 final_check: FinalCheck | None = None,
                 prefetch: Callable[[str], list[tuple[str, dict]]] | None = None):
        self.name, self.instructions, self.tools = name, instructions, list(tools or [])
        self.backend, self.budget, self.final_check, self.prefetch = backend, budget, final_check, prefetch

    @trace
    async def run(self, task: str, ctx: Context) -> Outcome:
        result = await run_agent(
            task, self.instructions, self.tools, backend=self.backend or ctx.backend,
            trace_id=ctx.trace_id, final_check=self.final_check, run_store=ctx.run_store,
            context={**ctx.context, "agent_role": self.name}, budget=self.budget or Budget(),
            events=ctx.events, prefetch=self.prefetch(task) if self.prefetch else None)
        return Outcome(result.status, result.answer, [result], {"agent": self.name})


# -- waterfall ----------------------------------------------------------------


def chain_task(task: str, previous: Outcome, index: int) -> str:
    """Default hand-over between Sequence steps: the original task plus
    the previous step's answer, labelled."""
    return f"{task}\n\nResult of the previous step:\n{previous.answer}"


class Sequence(_Block):
    """Waterfall. Stops at the first step that does not answer and
    returns its status. `carry(task, previous_outcome, index) -> next task`."""

    def __init__(self, *steps: StepLike, name: str = "", carry: Callable[[str, Outcome, int], str] = chain_task):
        if not steps:
            raise ValueError("Sequence needs at least one step")
        self.steps, self.name, self.carry = [_as_step(s) for s in steps], name, carry

    @trace
    async def run(self, task: str, ctx: Context) -> Outcome:
        runs: list[AgentResult] = []
        current, outcome = task, None
        for i, s in enumerate(self.steps):
            outcome = await s(current, ctx)
            runs += outcome.runs
            if not outcome.ok:
                return replace(outcome, runs=runs, data={**outcome.data, "stopped_at": i})
            if i + 1 < len(self.steps):
                current = self.carry(task, outcome, i)
        return replace(outcome, runs=runs, data={**outcome.data, "steps": len(self.steps)})


# -- router -------------------------------------------------------------------


class LLMClassifier:
    """A model picks one route by name, constrained to the route names by
    the JSON schema (an enum), with a reason. `descriptions` tell the
    model what each route is for."""

    def __init__(self, descriptions: Mapping[str, str], backend: LLMBackend | None = None,
                 instructions: str = "Pick the one route that should handle the task."):
        self.descriptions, self.backend, self.instructions = dict(descriptions), backend, instructions
        names = tuple(self.descriptions)
        self.schema = create_model("RouteChoice", route=(Literal[names], ...), reason=(str, ""))  # type: ignore[valid-type]

    @trace
    async def __call__(self, task: str, ctx: Context) -> str | None:
        from agent_loop.prompts import build_system_message
        catalog = "\n".join(f"- {k}: {v}" for k, v in self.descriptions.items())
        messages = [build_system_message(f"{self.instructions}\nRoutes:\n{catalog}\n"
                                         'Reply with one JSON object: {"route": "<route name>", "reason": "..."}'),
                    {"role": "user", "content": task}]
        raw = await (self.backend or ctx.backend).generate(
            messages, format=self.schema.model_json_schema(), options={"temperature": 0, "seed": 42},
            trace_id=ctx.trace_id)
        try:
            choice = self.schema.model_validate_json(raw)
        except ValidationError as e:
            log_step(ctx.trace_id, "route_invalid", raw=raw[:300], detail=str(e)[:300])
            return None
        log_step(ctx.trace_id, "route_chosen", route=choice.route, reason=choice.reason)
        return choice.route


class Router(_Block):
    """`classify(task, ctx) -> route name` (sync or async; e.g. keyword
    rules, an embedding classifier, or LLMClassifier). An unknown or
    missing route goes to `default`, or ends as status "unroutable"."""

    def __init__(self, routes: Mapping[str, StepLike], classify: Callable[[str, Context], Any], *,
                 default: str | None = None, name: str = ""):
        if default is not None and default not in routes:
            raise ValueError(f"default route {default!r} is not one of {sorted(routes)}")
        self.routes = {k: _as_step(v) for k, v in routes.items()}
        self.classify, self.default, self.name = classify, default, name

    @trace
    async def run(self, task: str, ctx: Context) -> Outcome:
        chosen = self.classify(task, ctx)
        if inspect.isawaitable(chosen):
            chosen = await chosen
        route = chosen if chosen in self.routes else self.default
        log_step(ctx.trace_id, "routed", chosen=chosen, route=route)
        if route is None:
            return Outcome("unroutable", None, data={"route": None, "chosen": chosen})
        outcome = await self.routes[route](task, ctx)
        return replace(outcome, data={**outcome.data, "route": route, "chosen": chosen})


# -- parallel -----------------------------------------------------------------


def join_answers(task: str, outcomes: list[Outcome]) -> Outcome:
    """Default combiner: the answered results, labelled by position.
    Fails only when none answered."""
    answered = [(i, o) for i, o in enumerate(outcomes) if o.ok]
    if not answered:
        return Outcome("failed", None, data={"statuses": [o.status for o in outcomes]})
    return Outcome("answered", "\n\n".join(f"[{i + 1}] {o.answer}" for i, o in answered))


def majority_vote(task: str, outcomes: list[Outcome]) -> Outcome:
    """The answer most branches gave (exact match after strip/lower); for
    short labels or numbers, not free text."""
    votes = Counter((o.answer or "").strip().lower() for o in outcomes if o.ok)
    if not votes:
        return Outcome("failed", None, data={"votes": {}})
    winner, count = votes.most_common(1)[0]
    return Outcome("answered", winner, data={"votes": dict(votes), "agreement": count / len(outcomes)})


class Parallel(_Block):
    """Steps run concurrently on the same task. `combine` is a function
    `(task, outcomes) -> Outcome` (join_answers, majority_vote, your own)
    or a step (e.g. an aggregator Agent) that gets the task plus every
    branch's answer. A branch that raises counts as a "failed" outcome."""

    def __init__(self, *steps: StepLike, combine: Callable[[str, list[Outcome]], Outcome] | StepLike = join_answers,
                 name: str = ""):
        if not steps:
            raise ValueError("Parallel needs at least one step")
        self.steps, self.combine, self.name = [_as_step(s) for s in steps], combine, name

    @trace
    async def run(self, task: str, ctx: Context) -> Outcome:
        results = await asyncio.gather(*(s(task, ctx) for s in self.steps), return_exceptions=True)
        outcomes = []
        for r in results:
            if isinstance(r, BaseException):
                if isinstance(r, asyncio.CancelledError):
                    raise r
                log_step(ctx.trace_id, "branch_failed", detail=f"{type(r).__name__}: {r}")
                outcomes.append(Outcome("failed", None, data={"error": f"{type(r).__name__}: {r}"}))
            else:
                outcomes.append(r)
        runs = [run for o in outcomes for run in o.runs]
        if isinstance(self.combine, _Block):
            listing = "\n\n".join(f"Answer {i + 1} ({o.status}):\n{o.answer}" for i, o in enumerate(outcomes))
            merged = await self.combine(f"{task}\n\nCandidate answers:\n{listing}", ctx)
        else:
            merged = self.combine(task, outcomes)
        return replace(merged, runs=runs + merged.runs,
                       data={**merged.data, "branches": [o.status for o in outcomes]})


# -- handoff ------------------------------------------------------------------


class HandoffNote(BaseModel):
    """Default structured message from the source agent to the target.
    Flat strings on purpose: with a `dict` field, llama3.2:3b sent the
    string "{}" on every attempt and never got a valid handoff through."""
    summary: str = Field(min_length=10, max_length=1000)
    details: str = Field(default="", max_length=2000)


def render_note(note: BaseModel) -> str:
    return "Case handed over by another agent:\n" + json.dumps(note.model_dump(), ensure_ascii=False, indent=2)


class Handoff(_Block):
    """The source agent gets one `handoff_to_<target>` tool per target. If
    it calls one, the target step runs on the structured message only
    (not the source's conversation). At most one handoff per run: targets
    get no handoff tools from here, so agents cannot bounce a case back
    and forth. If the source never hands off, its own outcome is returned.

    `message` is the pydantic model of the note (validated like any tool
    argument); `render(note) -> task text` for the target."""

    def __init__(self, source: Agent, targets: Mapping[str, StepLike], *, message: type[BaseModel] = HandoffNote,
                 render: Callable[[BaseModel], str] = render_note, descriptions: Mapping[str, str] | None = None,
                 name: str = ""):
        self.source, self.targets = source, {k: _as_step(v) for k, v in targets.items()}
        self.message, self.render, self.name = message, render, name or source.name
        self.descriptions = dict(descriptions or {})

    def _tools(self) -> list[ToolSpec]:
        async def accepted(args: BaseModel, trace_id: str) -> Any:
            return {"accepted": True, "note": "Another agent will take over. Tell the user it is being handled."}
        return [ToolSpec(f"handoff_to_{k}", self.descriptions.get(k, f"Hand the case over to the {k} agent."),
                         self.message, accepted) for k in self.targets]

    @trace
    async def run(self, task: str, ctx: Context) -> Outcome:
        source = Agent(self.source.name, self.source.instructions, self.source.tools + self._tools(),
                       backend=self.source.backend, budget=self.source.budget,
                       final_check=self.source.final_check, prefetch=self.source.prefetch)
        first = await source(task, ctx)
        steps = first.runs[-1].steps if first.runs else []
        call = next((s for s in steps if s.kind == "tool_call" and (s.tool or "").startswith("handoff_to_")), None)
        if call is None:
            return replace(first, data={**first.data, "handoff": None})
        target = call.tool.removeprefix("handoff_to_")
        note = self.message.model_validate(call.arguments)
        log_step(ctx.trace_id, "handoff", source=self.source.name, target=target, message=note.model_dump())
        second = await self.targets[target](self.render(note), ctx)
        return replace(second, runs=first.runs + second.runs, data={**second.data, "handoff": target})


# -- evaluator / optimizer ----------------------------------------------------


class Verdict(BaseModel):
    ok: bool
    feedback: str = ""


class LLMCritic:
    """A model reviews an answer against `criteria` and returns a Verdict."""

    def __init__(self, criteria: str, backend: LLMBackend | None = None):
        self.criteria, self.backend = criteria, backend

    @trace
    async def __call__(self, task: str, answer: str, ctx: Context) -> Verdict:
        from agent_loop.prompts import build_system_message
        messages = [build_system_message(
            "You review an answer to a task against the criteria below. Reply with one JSON object: "
            '{"ok": true|false, "feedback": "<what must change, empty if ok>"}.\n'
            f"Criteria:\n{self.criteria}"),
            {"role": "user", "content": f"Task:\n{task}\n\nAnswer to review:\n{answer}"}]
        raw = await (self.backend or ctx.backend).generate(
            messages, format=Verdict.model_json_schema(), options={"temperature": 0, "seed": 42},
            trace_id=ctx.trace_id)
        try:
            return Verdict.model_validate_json(raw)
        except ValidationError:
            return Verdict(ok=False, feedback="the review could not be parsed; answer again, precisely")


class Refine(_Block):
    """Worker answers; `critic(task, answer, ctx) -> Verdict` (sync or
    async; a rule function or LLMCritic) reviews it; on a rejection the
    worker answers again with the feedback, up to `max_rounds` answers.
    Ends "answered" when the critic accepts, else "rejected_final" with
    the last answer kept in data["last_answer"]."""

    def __init__(self, worker: StepLike, critic: Callable[..., Any], *, max_rounds: int = 3, name: str = ""):
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self.worker, self.critic, self.max_rounds, self.name = _as_step(worker), critic, max_rounds, name

    @trace
    async def run(self, task: str, ctx: Context) -> Outcome:
        runs: list[AgentResult] = []
        current, history = task, []
        for round_ in range(1, self.max_rounds + 1):
            outcome = await self.worker(current, ctx)
            runs += outcome.runs
            if not outcome.ok:
                return replace(outcome, runs=runs, data={**outcome.data, "rounds": round_, "reviews": history})
            verdict = self.critic(task, outcome.answer, ctx)
            if inspect.isawaitable(verdict):
                verdict = await verdict
            history.append({"round": round_, "ok": verdict.ok, "feedback": verdict.feedback})
            log_step(ctx.trace_id, "review", round=round_, ok=verdict.ok, feedback=verdict.feedback[:300])
            if verdict.ok:
                return replace(outcome, runs=runs, data={**outcome.data, "rounds": round_, "reviews": history})
            current = (f"{task}\n\nYour previous answer:\n{outcome.answer}\n\n"
                       f"A reviewer rejected it:\n{verdict.feedback}\nAnswer again, fixing that.")
        return Outcome("rejected_final", None, runs,
                       {"rounds": self.max_rounds, "reviews": history, "last_answer": outcome.answer})


# -- fallback, timeout --------------------------------------------------------


class Fallback(_Block):
    """Tries steps in order; the first that answers wins. A step that
    pauses for approval stops the chain (a human is now involved)."""

    def __init__(self, *steps: StepLike, name: str = ""):
        if not steps:
            raise ValueError("Fallback needs at least one step")
        self.steps, self.name = [_as_step(s) for s in steps], name

    @trace
    async def run(self, task: str, ctx: Context) -> Outcome:
        runs: list[AgentResult] = []
        tried = []
        for i, s in enumerate(self.steps):
            outcome = await s(task, ctx)
            runs += outcome.runs
            tried.append(outcome.status)
            if outcome.ok or outcome.status == "awaiting_approval":
                return replace(outcome, runs=runs, data={**outcome.data, "used": i, "tried": tried})
        return replace(outcome, runs=runs, data={**outcome.data, "used": None, "tried": tried})


class Timeout(_Block):
    """Wall-time limit around any step; on expiry the inner work is
    cancelled (agent runs save their state as "cancelled")."""

    def __init__(self, inner: StepLike, seconds: float, name: str = ""):
        self.inner, self.seconds, self.name = _as_step(inner), seconds, name

    @trace
    async def run(self, task: str, ctx: Context) -> Outcome:
        try:
            async with asyncio.timeout(self.seconds):
                return await self.inner(task, ctx)
        except TimeoutError:
            log_step(ctx.trace_id, "composite_timeout", seconds=self.seconds)
            return Outcome("timeout", None, data={"seconds": self.seconds})
