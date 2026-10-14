"""A minimal Ollama backend with no dependencies (stdlib HTTP in a worker
thread). One request per call, with a timeout and no retries: for
production put retries / a circuit breaker in your own backend; any
object with the `LLMBackend.generate` signature works."""
from __future__ import annotations

import asyncio
import json
import os
import urllib.request

from agent_loop import usage
from agent_loop._trace import trace


class OllamaBackend:
    def __init__(self, model: str = "llama3.2:3b", base_url: str | None = None, timeout_s: float = 120.0):
        self.model = model
        self.base_url = (base_url or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        if "://" not in self.base_url:
            self.base_url = f"http://{self.base_url}"
        self.timeout_s = timeout_s

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(f"{self.base_url}/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read())

    @trace
    async def generate(self, messages: list[dict], format: str | dict | None = None,
                       options: dict | None = None, trace_id: str | None = None) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False}
        if format:
            payload["format"] = format
        if options:
            payload["options"] = options
        data = await asyncio.to_thread(self._post, payload)
        prompt, completion = data.get("prompt_eval_count", 0), data.get("eval_count", 0)
        usage.add(prompt, completion)
        trace.meta(model=self.model, prompt_tokens=prompt, completion_tokens=completion,
                   total_tokens=prompt + completion)
        return data["message"]["content"]
