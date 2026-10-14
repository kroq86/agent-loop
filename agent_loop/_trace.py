"""flow-xray tracing when it is installed (`pip install agent-loop[trace]`),
otherwise a no-op with the same surface."""
from __future__ import annotations

try:
    from flow_xray import trace
except ImportError:  # pragma: no cover - depends on the environment
    class _NoTrace:
        def __call__(self, fn):
            return fn

        def meta(self, **fields) -> None:
            pass

    trace = _NoTrace()  # type: ignore[assignment]
