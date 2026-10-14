"""Where the loop's step events go, and how a run's trace id is bound
for the code it calls (tools, backends).

By default events go to the standard `logging` logger "agent_loop" at
DEBUG level as one JSON object each, and binding a trace id only sets
`current_trace_id()`. An application plugs in its own:

    from agent_loop import hooks
    hooks.configure(log=my_log_step, bind_trace=my_bind_trace)

`log(trace_id, step, **fields)` must not raise and should not block
(it is called on the event loop). `bind_trace(trace_id)` returns a
context manager entered around each run segment.
"""
from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from typing import Any

_logger = logging.getLogger("agent_loop")
_current: ContextVar[str | None] = ContextVar("agent_loop_trace_id", default=None)


def new_trace_id() -> str:
    return str(uuid.uuid4())


def current_trace_id() -> str | None:
    return _current.get()


def _default_log(trace_id: str | None, step: str, **fields: Any) -> None:
    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug(json.dumps({"trace_id": trace_id, "step": step, **fields}, default=str))


@contextmanager
def _default_bind(trace_id: str):
    yield trace_id


_log: Callable[..., None] = _default_log
_bind: Callable[[str], AbstractContextManager] = _default_bind


def configure(*, log: Callable[..., None] | None = None,
              bind_trace: Callable[[str], AbstractContextManager] | None = None) -> None:
    """Replace the event sink and/or the trace binding; None restores the default."""
    global _log, _bind
    _log = log or _default_log
    _bind = bind_trace or _default_bind


def log_step(trace_id: str | None, step: str, **fields: Any) -> None:
    try:
        _log(trace_id or _current.get(), step, **fields)
    except Exception:
        pass  # observability must never break the run


@contextmanager
def bind_trace(trace_id: str):
    token = _current.set(trace_id)
    try:
        with _bind(trace_id):
            yield trace_id
    finally:
        _current.reset(token)
