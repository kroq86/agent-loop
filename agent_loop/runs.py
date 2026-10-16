"""Durable agent-run state, so a run paused for human approval survives
the request (and the process) that started it.

One row per run: the serialized loop state plus a status column. The
only transition that matters for safety is awaiting_approval -> resuming,
done as a compare-and-set UPDATE: two approve calls for the same run
cannot both win, so an approved write executes at most once from here.

Winning the claim takes a lease (`resume_lease_s`). A process that dies
after the claim leaves the run in "resuming"; once the lease expires the
run can be claimed again, so it is not stuck forever.

Limit, stated: if the dead owner had already executed the approved tool,
the next owner executes it again. That is only safe for tools that are
idempotent across processes: a write tool should use
`agent_loop.current_call_id` as an idempotency key stored where the
side effect is (a key kept in process memory does not survive the crash).

`tool_audit` takes `subject` and `tenant` from the run's context when the
caller put them there.
"""
from contextlib import closing
import json
import sqlite3
import time
from datetime import datetime, timezone

from agent_loop.sqlite_io import run_sqlite

RESUME_LEASE_S = 120.0


class SqliteAgentRunStore:
    def __init__(self, path: str, resume_lease_s: float = RESUME_LEASE_S):
        self.path = path
        self.resume_lease_s = resume_lease_s
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    state TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            # Migration, not a fresh schema: a database created before the
            # lease existed gets the column instead of failing on it later.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(agent_runs)")}
            if "lease_until" not in columns:
                conn.execute("ALTER TABLE agent_runs ADD COLUMN lease_until REAL")
            # Audit trail: one row per tool execution attempt, append-only.
            conn.execute("""CREATE TABLE IF NOT EXISTS tool_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, run_id TEXT NOT NULL,
                trace_id TEXT, subject TEXT, tenant TEXT, tool TEXT NOT NULL, tool_version TEXT,
                arguments TEXT NOT NULL, status TEXT NOT NULL, duration_ms REAL, error TEXT)""")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def save(self, run_id: str, status: str, state: dict) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO agent_runs (run_id, status, state, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET status = excluded.status, state = excluded.state, "
                "updated_at = excluded.updated_at",
                (run_id, status, json.dumps(state, default=str), datetime.now(timezone.utc).isoformat()),
            )

    def load(self, run_id: str) -> tuple[str, dict] | None:
        with closing(self._connect()) as conn, conn:
            row = conn.execute("SELECT status, state FROM agent_runs WHERE run_id = ?", (run_id,)).fetchone()
        return None if row is None else (row[0], json.loads(row[1]))

    def checkpoint(self, run_id: str, state: dict) -> None:
        """Save a running run's state and (re)take its lease -- a heartbeat."""
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO agent_runs (run_id, status, state, updated_at, lease_until) "
                "VALUES (?, 'running', ?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET status = 'running', "
                "state = excluded.state, updated_at = excluded.updated_at, lease_until = excluded.lease_until",
                (run_id, json.dumps(state, default=str), datetime.now(timezone.utc).isoformat(),
                 time.time() + self.resume_lease_s))

    def claim_orphaned(self, run_id: str) -> bool:
        """running with an expired lease (its process died) -> resuming."""
        now = time.time()
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "UPDATE agent_runs SET status = 'resuming', lease_until = ?, updated_at = ? "
                "WHERE run_id = ? AND status = 'running' AND lease_until < ?",
                (now + self.resume_lease_s, datetime.now(timezone.utc).isoformat(), run_id, now))
            return cur.rowcount == 1

    def claim_for_resume(self, run_id: str) -> bool:
        """Compare-and-set awaiting_approval -> resuming, or take over a
        "resuming" run whose lease expired. True only for the one caller
        that won."""
        now = time.time()
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "UPDATE agent_runs SET status = 'resuming', lease_until = ?, updated_at = ? "
                "WHERE run_id = ? AND (status = 'awaiting_approval' "
                "OR (status = 'resuming' AND lease_until < ?))",
                (now + self.resume_lease_s, datetime.now(timezone.utc).isoformat(), run_id, now),
            )
            return cur.rowcount == 1

    def audit(self, state, tool: str, tool_version: str, arguments: dict, status: str,
              duration_s: float, error: str | None) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO tool_audit (at, run_id, trace_id, subject, tenant, tool, tool_version, arguments, "
                "status, duration_ms, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), state.run_id, state.trace_id,
                 state.context.get("subject"), state.context.get("tenant"), tool, tool_version,
                 json.dumps(arguments, default=str), status, round(duration_s * 1000, 1), error))

    def cancel_if_waiting(self, run_id: str) -> bool:
        """awaiting_approval -> cancelled, compare-and-set like the claim."""
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "UPDATE agent_runs SET status = 'cancelled', updated_at = ? "
                "WHERE run_id = ? AND status = 'awaiting_approval'",
                (datetime.now(timezone.utc).isoformat(), run_id))
            return cur.rowcount == 1

    async def asave(self, run_id: str, status: str, state: dict) -> None:
        await run_sqlite(self.save, run_id, status, state)

    async def aload(self, run_id: str) -> tuple[str, dict] | None:
        return await run_sqlite(self.load, run_id)

    async def acheckpoint(self, run_id: str, state: dict) -> None:
        await run_sqlite(self.checkpoint, run_id, state)

    async def aclaim_orphaned(self, run_id: str) -> bool:
        return await run_sqlite(self.claim_orphaned, run_id)

    async def aclaim_for_resume(self, run_id: str) -> bool:
        return await run_sqlite(self.claim_for_resume, run_id)

    async def aaudit(self, state, tool: str, tool_version: str, arguments: dict, status: str,
                     duration_s: float, error: str | None) -> None:
        await run_sqlite(self.audit, state, tool, tool_version, arguments, status, duration_s, error)

    async def acancel_if_waiting(self, run_id: str) -> bool:
        return await run_sqlite(self.cancel_if_waiting, run_id)
