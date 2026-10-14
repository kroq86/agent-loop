"""Token meters. A backend reports what each model call actually used
(`usage.add(prompt_tokens, completion_tokens)`, from the server's
response); token budgets read the totals. Meters nest: every active meter
in the current context counts the call, so a per-run budget inside a
workflow budget inside an eval all see it. A ContextVar, so concurrent
runs in one process never share one.

A backend that never calls `add` reports 0 tokens: `Budget.max_tokens`
is then not enforced."""
from contextlib import contextmanager
from contextvars import ContextVar

_meters: ContextVar[tuple[dict, ...]] = ContextVar("agent_loop_usage_meters", default=())


@contextmanager
def metered():
    meter = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    token = _meters.set(_meters.get() + (meter,))
    try:
        yield meter
    finally:
        _meters.reset(token)


def add(prompt_tokens: int, completion_tokens: int) -> None:
    for meter in _meters.get():
        meter["prompt_tokens"] += prompt_tokens
        meter["completion_tokens"] += completion_tokens
        meter["calls"] += 1
