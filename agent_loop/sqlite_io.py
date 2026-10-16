"""Await whole synchronous SQLite operations without blocking the event loop.

Connections/cursors must be created, consumed and closed inside the callable.
The pool is separate from trace persistence and the default executor. Admission
is bounded per event loop (two submitted operations); callers wait for capacity.
Cancellation drains even a waiting operation before propagating. A database
error takes precedence over cancellation, so a failed commit stays visible.
Graceful shutdown must drain callers before closing the loop; SIGKILL can lose
an in-flight transaction. No cross-process durability beyond SQLite is promised.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import partial
import threading
from typing import Callable, ParamSpec, TypeVar
import weakref

from agent_loop._trace import trace

P = ParamSpec("P")
T = TypeVar("T")
_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sqlite-io")
_LOCK = threading.Lock()
_LIMITS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _limit() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    with _LOCK:
        ref = _LIMITS.get(loop)
        limit = ref() if ref else None
        if limit is None:
            limit = asyncio.Semaphore(2)
            _LIMITS[loop] = weakref.ref(limit)
        return limit


@trace
async def run_sqlite(function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
    """Await one whole operation, including transaction exit and connection close.

    Context variables (trace, usage, call identity) travel to the worker. No
    mutable arguments may be changed by the caller until this await finishes.
    There is no detached/drop queue. Repeated caller cancellation cannot abandon
    a write; forced event-loop shutdown remains outside this guarantee.
    """
    trace.meta(kind="sqlite", operation=getattr(function, "__qualname__", type(function).__name__))
    call = partial(function, *args, **kwargs)
    async def execute():
        try:
            async with _limit():
                result = await asyncio.get_running_loop().run_in_executor(_POOL, copy_context().run, call)
            return result, None
        except Exception as exc:
            # Retrieve errors even if shield's waiter was cancelled in the
            # same turn (also avoids asyncio 3.14's shield exception log).
            return None, exc
    operation = asyncio.create_task(execute())
    cancellation = None
    while True:
        try:
            result, error = await asyncio.shield(operation)
            break
        except asyncio.CancelledError as exc:
            if operation.cancelled():
                raise
            cancellation = exc
    if error is not None:
        raise error
    if cancellation is not None:
        raise cancellation
    return result
