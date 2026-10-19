"""A caller obligation blocks `final` until the required tool has succeeded."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pydantic import BaseModel


class Backend:
    model = "contract-fixture"

    def __init__(self, replies):
        self.replies = iter(replies)

    async def generate(self, messages, **kwargs):
        from agent_loop import usage
        usage.add(1, 1)
        return next(self.replies)


FINAL = json.dumps({"action": "final", "answer": "done"})
CALL = json.dumps({"action": "call_tool", "tool": "write", "arguments": {}})


def unmet(steps):
    if any(s.tool == "write" and s.kind in ("approval_requested", "tool_call") for s in steps):
        return None
    return "write has not succeeded"


class ObligationTests(unittest.IsolatedAsyncioTestCase):
    @patch("agent_loop.loop.estimate_prompt_tokens", return_value=1)
    async def test_final_continues_until_the_write_is_accepted(self, estimate):
        from agent_loop.loop import Budget, ToolSpec, run_agent
        from agent_loop.runs import SqliteAgentRunStore

        class Args(BaseModel):
            pass

        async def handler(args, trace_id):
            return "saved"

        tools = [ToolSpec("write", "test write", Args, handler, writes=True)]
        backend = Backend([FINAL, CALL])
        with tempfile.TemporaryDirectory() as tmp:
            store = SqliteAgentRunStore(str(Path(tmp) / "runs.db"))
            result = await run_agent("test", "test", tools, backend=backend, run_store=store,
                                      budget=Budget(max_iterations=4, max_tokens=100), obligation=unmet)
        self.assertEqual(result.status, "awaiting_approval")
        self.assertIsNone(result.answer)
        self.assertEqual([s.kind for s in result.steps], ["completion_refused", "approval_requested"])

    @patch("agent_loop.loop.estimate_prompt_tokens", return_value=1)
    async def test_exhausted_obligation_is_not_answered(self, estimate):
        from agent_loop.loop import Budget, run_agent
        backend = Backend([FINAL, FINAL])
        result = await run_agent("test", "test", [], backend=backend,
                                  budget=Budget(max_iterations=2, max_tokens=100), obligation=unmet)
        self.assertEqual(result.status, "obligation_unmet")
        self.assertIsNone(result.answer)
        self.assertEqual([s.kind for s in result.steps], ["completion_refused", "completion_refused"])

    @patch("agent_loop.loop.estimate_prompt_tokens", return_value=1)
    async def test_citation_rejection_stays_terminal(self, estimate):
        from agent_loop.loop import Budget, run_agent
        backend = Backend([FINAL, FINAL])
        result = await run_agent("test", "test", [], backend=backend,
                                  budget=Budget(max_iterations=4, max_tokens=100),
                                  final_check=lambda answer, citations, steps: "missing citation")
        self.assertEqual(result.status, "rejected_final")
        self.assertEqual(len(result.steps), 1)
