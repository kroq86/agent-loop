"""The one interface the loop needs from a model."""
from __future__ import annotations

import json
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    """`generate` returns the model's reply text for a chat `messages`
    list. `format` is a JSON schema the reply must follow (the loop always
    passes one); `options` carries sampling settings. A backend should
    raise after its own retries when the model is unavailable, and report
    token usage with `agent_loop.usage.add` for token budgets to work.
    An optional `model` attribute names the model in run versions."""
    # options["num_predict"] is a positive completion cap, tightened to
    # the reported remaining run tokens before each call. Backends must
    # honour it. Prompt tokens are only known after provider reporting,
    # so the total prompt+completion threshold still has a post-call check.

    async def generate(self, messages: list[dict], format: str | dict | None = None,
                       options: dict | None = None, trace_id: str | None = None) -> str: ...


BUDGET_POLICY_VERSION = "2"  # reported usage + estimated prompt admission + completion cap


def estimate_prompt_tokens(messages: list[dict], format: str | dict | None = None) -> int:
    """Fixed chars/4 estimate with role/block framing and request schema.

    Inspired by DeepSeek Harness's token-meter heuristic. This prices the
    explicit request only; it is not exact tokenization or an upper bound
    on a provider's hidden template, tokenizer or multimodal processing.
    """
    def price(value) -> int:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return (len(text) + 3) // 4

    total = 0
    for message in messages:
        total += price(message.get("content", "")) + 4  # role framing
        if message.get("role") != "system":
            total += 4  # content block framing
        extra = {k: v for k, v in message.items() if k not in ("role", "content")}
        if extra:
            total += price(extra) + 4
    if format is not None:
        total += price(format) + 4
    return total
