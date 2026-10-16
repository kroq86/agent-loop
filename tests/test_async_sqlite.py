"""Real SQLite lock, cancellation and trace regression tests; no model service."""
import asyncio
from contextlib import closing
from contextvars import ContextVar
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest

from pydantic import BaseModel
from agent_loop import SqliteAgentRunStore, ToolSpec, run_agent, resume_agent
from agent_loop.sqlite_io import run_sqlite


async def wait_for(event):
    async with asyncio.timeout(3):
        while not event.is_set():
            await asyncio.sleep(.001)


class AsyncSqliteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = SqliteAgentRunStore(str(Path(self.tmp.name) / 'runs.db'))

    async def test_real_write_lock_cancellation_drains_and_closes_in_worker(self):
        context = ContextVar('test_context', default=None)
        context.set('trace-and-call-id')
        entered, closed = threading.Event(), threading.Event()
        main_thread = threading.get_ident()
        lock = sqlite3.connect(self.store.path)
        self.addCleanup(lock.close)
        lock.execute('BEGIN IMMEDIATE')
        def write():
            self.assertNotEqual(main_thread, threading.get_ident())
            self.assertEqual(context.get(), 'trace-and-call-id')
            entered.set()
            try:
                self.store.save('id', 'answered', {'detail': 'full-result'})
            finally:
                closed.set()
        task = asyncio.create_task(run_sqlite(write))
        await wait_for(entered)
        ticks = 0
        for _ in range(10):
            await asyncio.sleep(.002)
            ticks += 1
        task.cancel()
        await asyncio.sleep(.01)
        task.cancel()
        self.assertFalse(task.done())
        lock.rollback()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(closed.is_set())
        self.assertEqual(ticks, 10)
        self.assertEqual(self.store.load('id'), ('answered', {'detail': 'full-result'}))

    async def test_cancel_queued_operation_still_finishes(self):
        entered = [threading.Event(), threading.Event()]
        release = threading.Event()
        self.addCleanup(release.set)
        tasks = [asyncio.create_task(run_sqlite(lambda e=e: (e.set(), release.wait(3)))) for e in entered]
        try:
            for e in entered:
                await wait_for(e)
            queued = asyncio.create_task(self.store.asave('queued', 'answered', {}))
            await asyncio.sleep(.01)
            queued.cancel()
            await asyncio.sleep(.01)
            self.assertFalse(queued.done())
        finally:
            release.set()
            await asyncio.gather(*tasks)
        with self.assertRaises(asyncio.CancelledError):
            await queued
        self.assertEqual((await self.store.aload('queued'))[0], 'answered')

    async def test_write_error_is_not_hidden_by_cancellation(self):
        entered, release = threading.Event(), threading.Event()
        def fail():
            entered.set()
            release.wait(3)
            raise sqlite3.OperationalError('disk full')
        task = asyncio.create_task(run_sqlite(fail))
        await wait_for(entered)
        task.cancel()
        release.set()
        with self.assertRaisesRegex(sqlite3.OperationalError, 'disk full'):
            await task

    async def test_final_commit_is_not_overwritten_by_cancel(self):
        started, release = threading.Event(), threading.Event()
        store = self.store
        original = store.save
        def save(*args):
            started.set()
            release.wait(3)
            original(*args)
        store.save = save
        class Backend:
            async def generate(self, *args, **kwargs):
                return '{"action":"final","answer":"full answer"}'
        task = asyncio.create_task(run_agent('q', '', [], Backend(), run_store=store))
        await wait_for(started)
        task.cancel()
        await asyncio.sleep(.01)
        task.cancel()
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        with closing(sqlite3.connect(store.path)) as conn:
            row = conn.execute('select status,state from agent_runs').fetchone()
        self.assertEqual(row[0], 'answered')
        self.assertEqual([s['kind'] for s in json.loads(row[1])['steps']], ['final'])

    async def test_completed_tool_result_survives_cancel_during_audit(self):
        started, release = threading.Event(), threading.Event()
        original = self.store.audit
        def audit(*args):
            started.set()
            release.wait(3)
            original(*args)
        self.store.audit = audit
        class Args(BaseModel):
            pass
        async def handler(*args):
            return {'full': 'x' * 10_000}
        class Backend:
            async def generate(self, *args, **kwargs):
                return '{"action":"call_tool","tool":"read","arguments":{}}'
        task = asyncio.create_task(run_agent('q', '', [ToolSpec('read', '', Args, handler)], Backend(), run_store=self.store))
        await wait_for(started)
        task.cancel()
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        with closing(sqlite3.connect(self.store.path)) as conn:
            status, state = conn.execute('select status,state from agent_runs').fetchone()
            self.assertEqual(conn.execute('select status from tool_audit').fetchone()[0], 'ok')
        self.assertEqual(status, 'cancelled')
        self.assertEqual(json.loads(state)['steps'][0]['output'], {'full': 'x' * 10_000})

    async def test_two_approvals_have_one_winner(self):
        class Args(BaseModel):
            pass
        effects = []
        async def handler(*args):
            effects.append(1)
            return {'saved': True}
        class Backend:
            def __init__(self, reply): self.reply = reply
            async def generate(self, *args, **kwargs): return self.reply
        tools = [ToolSpec('write', '', Args, handler, writes=True)]
        result = await run_agent('q', '', tools, Backend('{"action":"call_tool","tool":"write","arguments":{}}'), run_store=self.store)
        resumed = await asyncio.gather(*(resume_agent(result.run_id, True, tools, self.store, Backend('{"action":"final","answer":"ok"}')) for _ in range(2)), return_exceptions=True)
        self.assertEqual(len(effects), 1)
        self.assertEqual(sum(isinstance(r, RuntimeError) for r in resumed), 1)
        self.assertEqual((await self.store.aload(result.run_id))[0], 'answered')

    async def test_connections_are_created_and_closed_in_worker(self):
        from unittest.mock import patch
        main_thread = threading.get_ident()
        opened, closed = [], []
        class Connection(sqlite3.Connection):
            def close(self):
                closed.append(threading.get_ident())
                super().close()
        original = sqlite3.connect
        def connect(*args, **kwargs):
            opened.append(threading.get_ident())
            self.assertNotEqual(main_thread, threading.get_ident())
            return original(*args, **kwargs, factory=Connection)
        with patch('sqlite3.connect', connect):
            await self.store.acheckpoint('id', {})
            await self.store.aload('id')
            await self.store.asave('id', 'awaiting_approval', {})
            self.assertTrue(await self.store.aclaim_for_resume('id'))
            self.assertFalse(await self.store.acancel_if_waiting('id'))
        self.assertEqual(opened, closed)
        self.assertEqual(len(opened), 5)

    async def test_cancel_during_checkpoint_never_calls_model(self):
        started, release = threading.Event(), threading.Event()
        original = self.store.checkpoint
        def checkpoint(*args):
            started.set()
            release.wait(3)
            original(*args)
        self.store.checkpoint = checkpoint
        class Backend:
            async def generate(self, *args, **kwargs):
                raise AssertionError('cancelled before model call')
        task = asyncio.create_task(run_agent('q', '', [], Backend(), run_store=self.store))
        await wait_for(started)
        task.cancel()
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        with closing(sqlite3.connect(self.store.path)) as conn:
            self.assertEqual(conn.execute('select status from agent_runs').fetchone()[0], 'cancelled')

if __name__ == '__main__':
    unittest.main()
