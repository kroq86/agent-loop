"""Estimated prompt admission and reported-token caps; no model service."""
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pydantic import BaseModel

class Backend:
    model = "contract-fixture"

    def __init__(self, replies, spend=5):
        self.replies, self.spend, self.caps = iter(replies), spend, []

    async def generate(self, messages, **kwargs):
        from agent_loop import usage
        self.caps.append(kwargs["options"]["num_predict"])
        usage.add(self.spend - 1, 1)
        return next(self.replies)


FINAL = json.dumps({"action": "final", "answer": "done"})


class TokenBudgetTests(unittest.IsolatedAsyncioTestCase):
    @patch("agent_loop.loop.estimate_prompt_tokens", return_value=10)
    async def test_loop_tightens_cap_after_reported_usage(self, estimate):
        from agent_loop.loop import Budget, run_agent
        backend = Backend(["invalid json", FINAL])
        result = await run_agent("test", "test", [], backend=backend, budget=Budget(max_tokens=20))
        self.assertEqual(result.status, "answered")
        self.assertEqual(backend.caps, [10, 5])
        self.assertEqual(result.usage["tokens"], 10)


    @patch("agent_loop.loop.estimate_prompt_tokens", return_value=1)
    async def test_exhausted_loop_never_calls_the_model_again(self, estimate):
        from agent_loop.loop import Budget, run_agent
        for limit, spend, wanted_calls in ((0, 5, 0), (5, 5, 1), (4, 5, 1)):
            with self.subTest(limit=limit):
                backend = Backend(["invalid json", FINAL], spend=spend)
                result = await run_agent("test", "test", [], backend=backend, budget=Budget(max_tokens=limit))
                self.assertEqual(result.status, "budget_exceeded")
                self.assertEqual(len(backend.caps), wanted_calls)


    @patch("agent_loop.loop.estimate_prompt_tokens", return_value=10)
    async def test_approval_continues_with_the_saved_token_remainder(self, estimate):
        from agent_loop.loop import Budget, ToolSpec, resume_agent, run_agent
        from agent_loop.runs import SqliteAgentRunStore

        class Args(BaseModel):
            pass

        async def handler(args, trace_id):
            return "saved"

        tools = [ToolSpec("write", "test write", Args, handler, writes=True)]
        backend = Backend([json.dumps({"action": "call_tool", "tool": "write", "arguments": {}}), FINAL])
        with tempfile.TemporaryDirectory() as tmp:
            store = SqliteAgentRunStore(str(Path(tmp) / "runs.db"))
            first = await run_agent("test", "test", tools, backend=backend, run_store=store,
                                    budget=Budget(max_tokens=20))
            self.assertEqual(first.status, "awaiting_approval")
            result = await resume_agent(first.run_id, True, tools, store, backend)
        self.assertEqual(result.status, "answered")
        self.assertEqual(backend.caps, [10, 5])


    @patch("agent_loop.loop.estimate_prompt_tokens", return_value=10)
    async def test_recovery_uses_checkpoint_usage_for_the_cap(self, estimate):
        from agent_loop.loop import Budget, _RunState, recover_agent
        from agent_loop.runs import SqliteAgentRunStore
        state = _RunState(run_id="orphan", trace_id="recover-cap", task="test", max_iterations=4,
                          messages=[{"role": "user", "content": "test"}], context={}, iteration=1,
                          budget=dataclasses.asdict(Budget(max_tokens=20)),
                          usage={"tokens": 7, "tool_calls": 0, "seconds": 0})
        backend = Backend([FINAL])
        with tempfile.TemporaryDirectory() as tmp:
            store = SqliteAgentRunStore(str(Path(tmp) / "runs.db"), resume_lease_s=0)
            await store.acheckpoint(state.run_id, state.to_dict())
            result = await recover_agent(state.run_id, [], store, backend)
        self.assertEqual(result.status, "answered")
        self.assertEqual(backend.caps, [3])


    def test_prompt_estimate_includes_framing_and_schema(self):
        from agent_loop.backend import estimate_prompt_tokens
        self.assertEqual(estimate_prompt_tokens([{"role": "user", "content": "12345678"}]), 10)
        self.assertEqual(estimate_prompt_tokens([{"role": "system", "content": "12345678"}]), 6)
        self.assertEqual(estimate_prompt_tokens([{"role": "user", "content": "12345678"}], {"type": "object"}), 19)


    async def test_prompt_admission_stops_before_the_model(self):
        from agent_loop.loop import Budget, run_agent
        backend = Backend([FINAL])
        result = await run_agent("x" * 1600, "test", [], backend=backend, budget=Budget(max_tokens=100))
        self.assertEqual(result.status, "budget_exceeded")
        self.assertEqual(result.usage["tokens"], 0)
        self.assertEqual(backend.caps, [])
        self.assertIn("estimated prompt", result.steps[-1].detail)


if __name__ == "__main__":
    unittest.main()
