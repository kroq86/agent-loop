# agent-runloop

`pip install agent-runloop`, then `import agent_loop`.

An explicit agent loop for tool-using LLM agents, and blocks to compose
agents into routers, waterfalls, parallel fan-outs, handoffs and
review loops.

The model never calls anything. On every turn it returns one JSON
decision: "call this tool with these arguments" or "here is the final
answer". The rest is ordinary code:

```
generate -> parse JSON -> validate decision -> look up tool -> validate arguments
  -> reject duplicates -> check budget, precondition, authorization
  -> execute (timeout) -> feed result back as delimited, untrusted data -> generate ...
```

Every rejection (bad JSON, unknown tool, invalid arguments, duplicate
call, tool exception) becomes an observation the model sees on its next
turn. None of them execute anything.

```bash
pip install agent-runloop         # core: pydantic only
pip install "agent-runloop[trace]"   # + flow-xray call-tree tracing
```

Python 3.11+.

## What the loop does

- **Validated tool calls.** Tool arguments are pydantic models. Invalid
  calls go back to the model as structured errors, never to your code.
- **Human approval for writes.** A tool with `writes=True` never runs on
  the model's word. The run pauses as `awaiting_approval` with the
  proposed call. `resume_agent(run_id, approved=True|False)` continues
  it, from another request or another process. The approve step is a
  compare-and-set, so two approvals cannot both execute.
- **Budgets.** Hard limits per run on model turns, executed tool calls,
  tokens actually used and wall time. They carry across pause and resume.
- **Crash recovery.** The loop saves a checkpoint to SQLite with a lease
  before every model turn. If the process dies, `recover_agent(run_id)`
  continues from the last checkpoint once the lease expires.
- **Authorization and preconditions.** These come from trusted run
  context, never from model output. They are checked when a call is
  proposed (a human is never asked to approve a forbidden action) and
  again at execution.
- **Idempotency.** `current_call_id` gives each tool call a stable id
  (run id + hash of the validated call). A write tool can use it as an
  idempotency key.
- **Prompt-injection hygiene.** Tool output goes back delimited and
  labelled as data, with the original task restated after it. The
  system message starts with a fixed instruction-hierarchy policy.
- **Audit and versions.** One audit row per tool execution. Every result
  records the model and a hash of the whole system prompt (instructions,
  protocol, tool catalog).
- **Live events.** Pass an `asyncio.Queue` to stream steps, for example
  over SSE.

## Quick start

```python
import asyncio
from pydantic import BaseModel
from agent_loop import Budget, SqliteAgentRunStore, ToolSpec, run_agent
from agent_loop.ollama import OllamaBackend

class SearchArgs(BaseModel):
    query: str

async def search(args: SearchArgs, trace_id: str):
    return [{"doc_id": "refunds", "text": "Refunds take 5 to 7 business days."}]

result = asyncio.run(run_agent(
    "How long does a refund take?",
    "You are a support agent. Answer only from search results.",
    [ToolSpec("search_knowledge_base", "Search the knowledge base.", SearchArgs, search)],
    backend=OllamaBackend("llama3.2:3b"),
    run_store=SqliteAgentRunStore("runs.sqlite3"),
    budget=Budget(max_iterations=4, max_tool_calls=4, max_tokens=8000, max_seconds=60),
))
print(result.status, result.answer, result.usage)
```

`result.status` is one of `answered`, `awaiting_approval`, `max_iterations`,
`budget_exceeded`, `timeout`, `rejected_final`, `model_unavailable` or
`cancelled`. `result.steps` lists every step, including rejected ones.

### Human approval

```python
TICKET = ToolSpec("create_ticket", "Open a support ticket.", TicketArgs, create_ticket, writes=True,
                  authorize=lambda ctx: ctx.get("role") == "customer")

r = await run_agent(task, instructions, [SEARCH, TICKET], backend=b, run_store=store, context={"role": "customer"})
if r.status == "awaiting_approval":
    print(r.pending_call)                       # {"tool": "create_ticket", "arguments": {...}}
    r = await resume_agent(r.run_id, True, [SEARCH, TICKET], store, backend=b)
```

### Other hooks

- `final_check(answer, citations, steps) -> reason | None` accepts or
  rejects the final answer, for example "every citation must be a
  doc_id a tool returned".
- `prefetch=[("search_knowledge_base", {"query": task})]` runs read
  tools before the first model turn.
- `ToolSpec(precondition=...)` is a deterministic business rule checked
  on a proposed call.

## Composing agents

Every block is a step, `async (task, ctx) -> Outcome`, so blocks nest
freely.

| Block | What it does |
|---|---|
| `Agent` | One run of the loop: instructions, tools, model, budget, final check. |
| `Sequence` | Waterfall: each step's answer becomes part of the next task. Stops at the first step that does not answer. |
| `Router` | A classifier picks one route. The classifier is any function (keyword rules, embeddings) or `LLMClassifier`, which constrains the model to the route names. |
| `Parallel` | Branches run concurrently. Results are merged with `join_answers`, `majority_vote`, your own function, or an aggregator agent. |
| `Handoff` | The source agent gets a `handoff_to_<target>` tool. The target sees only the structured note, not the conversation. At most one hop. |
| `Refine` | Evaluator-optimizer: the worker answers, a critic reviews, the worker revises, up to `max_rounds`. The critic is a rule function or `LLMCritic`. |
| `Fallback` | Tries steps in order; the first that answers wins. |
| `Timeout` | A wall-time limit around any step. |
| `step(fn)` | Any `async (task, ctx) -> str \| Outcome` function as a step. |

```python
from agent_loop import (Agent, Context, Fallback, Handoff, LLMClassifier, LLMCritic, Parallel, Refine, Router,
                        Sequence, Timeout, majority_vote)

billing   = Agent("billing", "You are the billing agent. Answer only from search results.", [SEARCH])
technical = Agent("technical", "You are technical support. Answer only from search results.", [SEARCH])
writer    = Agent("writer", "Rewrite the previous result as one polite sentence.")

pipeline = Timeout(
    Router(
        {"billing": Sequence(billing, writer),                       # waterfall inside a route
         "technical": Refine(technical, LLMCritic("Must give concrete steps."), max_rounds=2)},
        LLMClassifier({"billing": "payments, refunds, plans", "technical": "bugs and errors"}),
        default="technical",
    ),
    seconds=120,
)

ctx = Context(backend=OllamaBackend("llama3.2:3b"), run_store=SqliteAgentRunStore("runs.sqlite3"),
              context={"subject": "user-42"})
outcome = await pipeline("When will my refund arrive?", ctx)
print(outcome.status, outcome.answer, outcome.data["route"], len(outcome.runs))
```

`Outcome` has `status`, `answer`, `runs` (every `AgentResult` inside, in
order) and `data`, which records what each block decided: the route, the
votes, review rounds, which fallback answered.

`examples/patterns.py` runs every block against a local model and
captures each run as a flow-xray trace.

### Limits

- When an agent inside a composite pauses for approval, the composite
  stops and returns `awaiting_approval`. Resuming that agent run does
  not continue the rest of the composite.
- Token and tool budgets apply per agent run. Use `Timeout` for a
  deadline across a whole composite.
- `majority_vote` compares exact answers, so use it for labels and
  numbers, not free text.
- The run store is SQLite: one host, or a shared file system with
  working file locks.
- Recovering after a crash during an approved write re-executes that
  write. It is safe only if the tool is idempotent across processes
  (`current_call_id`).

## Backends

The loop needs any object with:

```python
async def generate(self, messages, format=None, options=None, trace_id=None) -> str
```

`format` is a JSON schema the reply must follow. A backend should report
tokens with `agent_loop.usage.add(prompt, completion)`, or `max_tokens`
is not enforced. Two are included.

**OpenAI-compatible APIs** (`pip install "agent-runloop[openai]"`): OpenAI,
Azure OpenAI, and servers that speak the same protocol (vLLM, LM Studio,
OpenRouter, llama.cpp server, Ollama's `/v1`).

```python
from agent_loop.openai import AzureOpenAIBackend, OpenAIBackend

OpenAIBackend("gpt-4.1-mini")                                    # OPENAI_API_KEY
OpenAIBackend("llama3.2:3b", base_url="http://127.0.0.1:11434/v1", api_key="unused")
AzureOpenAIBackend("my-deployment", endpoint="https://<resource>.openai.azure.com")  # AZURE_OPENAI_API_KEY
AzureOpenAIBackend("my-deployment", endpoint=..., token_provider=get_token)        # Entra ID, refreshed per call
```

- **Retries.** Only what can succeed on a second try is retried:
  connection errors, timeouts, 429 and 5xx. The wait honours
  `Retry-After` and `retry-after-ms`, otherwise it is exponential backoff
  with jitter. Any other 4xx fails at once with `ModelRequestError`.
- **Circuit breaker.** After `breaker_threshold` consecutive failed calls,
  every call fails fast with `ModelUnavailable` for `breaker_reset_s`.
  Then one trial call decides whether it closes. The breaker is per
  backend instance and per process.
- **Structured output.** The loop's schema is sent as a `json_schema`
  response format. Use `structured="json_object"` for servers that only
  support JSON mode, or `"none"`.
- **Tokens.** Usage from each response feeds the run's token budget and
  the flow-xray trace.

Inside a run, an unavailable model ends the run as `model_unavailable`
with its state saved. It does not crash the loop.

**Ollama** (`agent_loop.ollama.OllamaBackend`) is a dependency-free
reference backend with no retries.

## Observability

- **flow-xray**: with `agent-runloop[trace]` installed, `run_agent`, each tool
  execution, each composite block and each model call become nodes in a
  flow-xray call tree, with inputs, outputs, tokens and latency.
- **Step events**: by default they go to the `agent_loop` logger at
  DEBUG. To send them to your own logging and trace-id binding:

```python
from agent_loop import hooks
hooks.configure(log=my_log_step, bind_trace=my_bind_trace)   # log(trace_id, step, **fields)
```

## License

MIT


### Async SQLite persistence (0.3.0)

`run_agent`, `resume_agent` and `recover_agent` await the store's `asave`,
`aload`, `acheckpoint`, `aclaim_for_resume`, `aclaim_orphaned`, and `aaudit`
operations. Use `await acancel_agent(run_id, store)` from async callers;
`cancel_agent` and every synchronous store method remain available for sync code.
Store construction initializes the schema synchronously: construct before serving
requests, or use `await run_sqlite(SqliteAgentRunStore, path)` during async startup.

`agent_loop.sqlite_io.run_sqlite(callable, *args, **kwargs)` runs a **whole**
operation on a separate two-thread SQLite pool. Open, commit/rollback and close
connections inside that callable; never pass a connection or cursor across threads.
Its admission limit is two submitted operations per event loop. Trace, usage and
idempotency context variables are copied into the worker. Each call is traced with
`kind=sqlite` and its operation name when the optional trace integration is installed.

Cancellation (including repeated cancellation while waiting for capacity) waits
for the operation to finish and close its connection, then propagates. A database
error takes precedence over cancellation. A completed terminal save is not overwritten
with `cancelled`; a completed tool observation is recorded before awaiting its audit.
Cancellation during a resume claim can leave its existing lease to expire, without
executing the approved tool. No lease, schema or approval CAS policy changes here.
Drain request tasks before stopping the event loop. Forced loop teardown, process
crashes and SIGKILL are outside this completion guarantee; SQLite rollback/recovery
and existing run leases still apply. This does not guarantee every node is under 50 ms
or remove synchronous CPU work.

Verification: `python -m unittest discover -s tests -v`. Tests use real SQLite
transactions and held write locks, including queued/repeated cancellation, write
failure, worker connection lifetime, cancellation during checkpoint/final save/audit,
and two competing approvals. These are runtime regressions, not a universal refinement
proof of the durable run protocol. Existing sync APIs remain the matched baseline;
the SQL statements and serialized state schema are unchanged.
