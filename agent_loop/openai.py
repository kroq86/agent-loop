"""Backends for OpenAI-compatible chat APIs: OpenAI, Azure OpenAI, and the
many servers that speak the same protocol (vLLM, LM Studio, OpenRouter,
Ollama's /v1, llama.cpp server, ...).

    pip install "agent-runloop[openai]"        # adds httpx

    OpenAIBackend("gpt-4.1-mini")                                   # OPENAI_API_KEY
    OpenAIBackend("llama3.2:3b", base_url="http://127.0.0.1:11434/v1", api_key="unused")
    AzureOpenAIBackend("my-deployment", endpoint="https://<resource>.openai.azure.com")
                                                                    # AZURE_OPENAI_API_KEY,
                                                                    # or token_provider= for Entra ID

Failure handling:

    retry    only what can succeed on a second try: connection errors,
             timeouts, 429 and 5xx. The wait honours Retry-After /
             retry-after-ms, otherwise exponential backoff with jitter,
             capped at `max_delay_s`. Any other 4xx (bad request, auth,
             unknown model) fails at once with ModelRequestError.
    breaker  per backend instance: after `breaker_threshold` consecutive
             failed calls every call fails fast with ModelUnavailable for
             `breaker_reset_s`, then one trial call decides. Per process:
             several processes keep separate breakers.

Structured output: the loop passes a JSON schema in `format`. It is sent
as `response_format` "json_schema" (default), "json_object" (servers that
only know JSON mode; the loop's prompt still spells out the shape), or not
at all ("none").

Token usage from the response is reported to `agent_loop.usage`, so
`Budget.max_tokens` is enforced.
"""
from __future__ import annotations

import asyncio
import email.utils
import os
import random
import time
from collections.abc import Callable
from typing import Literal

from agent_loop import usage
from agent_loop._trace import trace
from agent_loop.hooks import log_step

try:
    import httpx
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError('agent_loop.openai needs httpx: pip install "agent-runloop[openai]"') from e

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
StructuredMode = Literal["json_schema", "json_object", "none"]


class ModelUnavailable(Exception):
    """The server could not serve the call after retries, or the circuit
    breaker is open. Distinct from a request the server rejected."""


class ModelRequestError(Exception):
    """The server answered and refused the request (4xx other than 429), or
    the model refused to answer. Retrying the same request will not help."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class CircuitBreaker:
    def __init__(self, threshold: int = 5, reset_s: float = 30.0):
        self.threshold, self.reset_s = threshold, reset_s
        self.failures = 0
        self.opened_at: float | None = None

    def check(self, target: str) -> None:
        if self.opened_at is not None and time.monotonic() - self.opened_at < self.reset_s:
            raise ModelUnavailable(f"circuit open for {target} ({self.failures} consecutive failures)")

    def success(self) -> None:
        self.failures, self.opened_at = 0, None

    def failure(self, target: str) -> None:
        self.failures += 1
        if self.opened_at is not None or self.failures >= self.threshold:
            if self.opened_at is None:
                log_step(None, "circuit_opened", target=target, failures=self.failures)
            self.opened_at = time.monotonic()  # opened, or a failed trial call keeps it open


def _retry_after_s(headers: httpx.Headers) -> float | None:
    """Retry-After in seconds or as an HTTP date; retry-after-ms (OpenAI, Azure)."""
    if (ms := headers.get("retry-after-ms")) is not None:
        try:
            return float(ms) / 1000
        except ValueError:
            pass
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            return max(0.0, email.utils.parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


# Loop options that have an OpenAI equivalent; the rest are dropped.
_OPTION_NAMES = {"temperature": "temperature", "seed": "seed", "top_p": "top_p",
                 "num_predict": "max_tokens", "max_tokens": "max_tokens", "stop": "stop"}


class OpenAIBackend:
    def __init__(self, model: str, *, base_url: str | None = None, api_key: str | None = None,
                 structured: StructuredMode = "json_schema", timeout_s: float = 60.0, max_attempts: int = 3,
                 base_delay_s: float = 0.5, max_delay_s: float = 20.0, breaker_threshold: int = 5,
                 breaker_reset_s: float = 30.0, extra_body: dict | None = None,
                 extra_headers: dict[str, str] | None = None):
        self.model = model
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.structured = structured
        self.timeout_s, self.max_attempts = timeout_s, max_attempts
        self.base_delay_s, self.max_delay_s = base_delay_s, max_delay_s
        self.breaker = CircuitBreaker(breaker_threshold, breaker_reset_s)
        self.extra_body, self.extra_headers = dict(extra_body or {}), dict(extra_headers or {})
        # httpx clients are bound to the event loop they were made on.
        self._clients: dict[int, httpx.AsyncClient] = {}

    # -- request shape (overridden by AzureOpenAIBackend) -----------------

    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _target(self) -> str:
        return self.base_url

    # ---------------------------------------------------------------------

    def _client(self) -> httpx.AsyncClient:
        loop = id(asyncio.get_running_loop())
        client = self._clients.get(loop)
        if client is None or client.is_closed:
            client = self._clients[loop] = httpx.AsyncClient(
                timeout=self.timeout_s, limits=httpx.Limits(max_connections=20, max_keepalive_connections=10))
        return client

    def _payload(self, messages: list[dict], format: str | dict | None, options: dict | None) -> dict:
        payload: dict = {"model": self.model, "messages": messages}
        for key, value in (options or {}).items():
            if key in _OPTION_NAMES:
                payload[_OPTION_NAMES[key]] = value
        if format and self.structured == "json_schema" and isinstance(format, dict):
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": str(format.get("title") or "reply"),
                                                          "schema": format}}
        elif format and self.structured != "none":
            payload["response_format"] = {"type": "json_object"}
        out = {**payload, **self.extra_body}
        caps = [(options or {}).get(key) for key in ("num_predict", "max_tokens")]
        caps = [cap for cap in caps if cap is not None]
        if caps:
            # Deployment extras may tighten, never undo the caller's cap.
            for field in ("max_tokens", "max_completion_tokens"):
                if field in out:
                    out[field] = min(out[field], *caps)
        return out

    @trace
    async def generate(self, messages: list[dict], format: str | dict | None = None,
                       options: dict | None = None, trace_id: str | None = None) -> str:
        target = self._target()
        self.breaker.check(target)
        payload = self._payload(messages, format, options)
        headers = {**self._auth_headers(), **self.extra_headers}
        last_error = ""
        for attempt in range(self.max_attempts):
            wait: float | None = None
            try:
                response = await self._client().post(self._url(), json=payload, headers=headers)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_error = f"{type(e).__name__}: {e}"
            else:
                if response.status_code < 400:
                    self.breaker.success()
                    return self._read(response.json())
                if response.status_code not in RETRYABLE_STATUS:
                    # The server said no: not an availability problem, so no
                    # retry and the breaker is not tripped.
                    raise ModelRequestError(f"HTTP {response.status_code}: {response.text[:300]}",
                                            response.status_code)
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                wait = _retry_after_s(response.headers)
            if attempt + 1 < self.max_attempts:
                delay = min(self.max_delay_s, wait if wait is not None
                            else random.uniform(0, self.base_delay_s * 2 ** attempt) + self.base_delay_s)
                log_step(trace_id, "model_call_retry", target=target, model=self.model, attempt=attempt + 1,
                         of=self.max_attempts, error=last_error, sleep_s=round(delay, 3))
                await asyncio.sleep(delay)
        self.breaker.failure(target)
        log_step(trace_id, "model_call_failed", target=target, model=self.model, attempts=self.max_attempts,
                 error=last_error)
        raise ModelUnavailable(f"{self.model} at {target}: {last_error} after {self.max_attempts} attempts")

    def _read(self, data: dict) -> str:
        u = data.get("usage") or {}
        prompt, completion = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
        usage.add(prompt, completion)
        trace.meta(model=data.get("model") or self.model, prompt_tokens=prompt, completion_tokens=completion,
                   total_tokens=prompt + completion)
        message = (data.get("choices") or [{}])[0].get("message") or {}
        if message.get("content") is None:
            raise ModelRequestError(f"no content in reply (refusal: {message.get('refusal')!r})")
        return message["content"]


class AzureOpenAIBackend(OpenAIBackend):
    """Azure OpenAI: the deployment name is in the URL, the version is a
    query parameter, and auth is an `api-key` header or an Entra ID bearer
    token (`token_provider() -> str`, e.g. from azure-identity's
    get_bearer_token_provider; called per request, so it can refresh)."""

    def __init__(self, deployment: str, *, endpoint: str | None = None, api_key: str | None = None,
                 api_version: str | None = None, token_provider: Callable[[], str] | None = None, **kwargs):
        endpoint = (endpoint or os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
        if not endpoint:
            raise ValueError("Azure OpenAI needs endpoint= or AZURE_OPENAI_ENDPOINT")
        super().__init__(deployment, base_url=endpoint, api_key="", **kwargs)
        self.deployment = deployment
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21"
        self.azure_key = api_key if api_key is not None else os.getenv("AZURE_OPENAI_API_KEY")
        self.token_provider = token_provider
        if not self.azure_key and token_provider is None:
            raise ValueError("Azure OpenAI needs api_key=, AZURE_OPENAI_API_KEY, or token_provider=")

    def _url(self) -> str:
        return (f"{self.base_url}/openai/deployments/{self.deployment}/chat/completions"
                f"?api-version={self.api_version}")

    def _auth_headers(self) -> dict[str, str]:
        if self.token_provider is not None:
            return {"Authorization": f"Bearer {self.token_provider()}"}
        return {"api-key": self.azure_key}

    def _target(self) -> str:
        return f"{self.base_url}/{self.deployment}"
