"""agent-runloop: an explicit, durable agent loop and blocks to compose agents.

The model proposes one JSON decision per turn; this code validates and
executes it. Write tools pause for human approval; runs checkpoint to
SQLite and can be resumed or recovered in another process.
"""
from importlib.metadata import PackageNotFoundError, version

from agent_loop import hooks, usage
from agent_loop.backend import LLMBackend
from agent_loop.compose import (
    Agent, Context, Fallback, Handoff, HandoffNote, LLMClassifier, LLMCritic, Outcome, Parallel, Refine, Router,
    Sequence, Step, Timeout, Verdict, chain_task, join_answers, majority_vote, step,
)
from agent_loop.loop import (
    AgentDecision, AgentResult, AgentStep, Budget, EventSink, FinalCheck, Obligation, Status, ToolSpec, ToolTimeout,
    acancel_agent, cancel_agent, current_call_id, recover_agent, resume_agent, run_agent, tool_error,
)
from agent_loop.runs import SqliteAgentRunStore

try:
    __version__ = version("agent-runloop")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0"

__all__ = [
    "Agent", "AgentDecision", "AgentResult", "AgentStep", "Budget", "Context", "EventSink", "Fallback", "FinalCheck", "Obligation",
    "Handoff", "HandoffNote", "LLMBackend", "LLMClassifier", "LLMCritic", "Outcome", "Parallel", "Refine", "Router",
    "Sequence", "SqliteAgentRunStore", "Status", "Step", "Timeout", "ToolSpec", "ToolTimeout", "Verdict", "acancel_agent", "cancel_agent",
    "chain_task", "current_call_id", "hooks", "join_answers", "majority_vote", "recover_agent", "resume_agent",
    "run_agent", "step", "tool_error", "usage",
]
