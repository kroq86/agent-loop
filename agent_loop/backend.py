"""The one interface the loop needs from a model."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    """`generate` returns the model's reply text for a chat `messages`
    list. `format` is a JSON schema the reply must follow (the loop always
    passes one); `options` carries sampling settings. A backend should
    raise after its own retries when the model is unavailable, and report
    token usage with `agent_loop.usage.add` for token budgets to work.
    An optional `model` attribute names the model in run versions."""

    async def generate(self, messages: list[dict], format: str | dict | None = None,
                       options: dict | None = None, trace_id: str | None = None) -> str: ...
