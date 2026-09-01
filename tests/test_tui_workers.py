"""Regression tests for the TUI proof-run worker machinery.

The parallel runner spawns one Textual ``@work(thread=True)`` worker per
problem, bounded by a semaphore. These tests pin the behaviour that is not
covered anywhere else (the real prove loop needs an LLM/Lean endpoint, so we
stub ``_prove_one``): that a run finishes, that concurrency respects the
worker count, and that stopping marks pending problems as skipped without
hanging.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from agent.tui import ProverApp


def _result(proved: bool = True) -> SimpleNamespace:
    return SimpleNamespace(proved=proved, proof="theorem ok" if proved else "")


def _problems(n: int) -> list[dict]:
    return [
        {"id": f"p{i}", "statement": f"theorem t{i} : True := by trivial",
         "difficulty": "custom"}
        for i in range(n)
    ]


async def _wait_done(app: ProverApp, pilot, timeout: float = 8.0) -> bool:
    """Poll until the active run finishes. Returns True if it did."""
    deadline = time.monotonic() + timeout
    while app._run_active and time.monotonic() < deadline:
        await pilot.pause(0.02)
    return not app._run_active


@pytest.mark.anyio
async def test_parallel_run_proves_all_problems() -> None:
    app = ProverApp(parallel=2)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        problems = _problems(4)
        calls: list[str] = []

        def fake(problem: dict) -> None:
            calls.append(problem["id"])
            time.sleep(0.03)
            app.call_from_thread(app._finish_row, problem["id"], "proved", _result())

        app._prove_one = fake  # type: ignore[method-assign]
        app._start_run(problems)
        assert await _wait_done(app, pilot)
        assert sorted(calls) == sorted(p["id"] for p in problems)
        assert app.counts["proved"] == 4
        assert not app._run_active


@pytest.mark.anyio
async def test_parallel_concurrency_is_bounded_by_workers() -> None:
    app = ProverApp(parallel=2)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        problems = _problems(6)
        lock = threading.Lock()
        active = 0
        max_active = 0

        def fake(problem: dict) -> None:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            app.call_from_thread(app._finish_row, problem["id"], "proved", _result())

        app._prove_one = fake  # type: ignore[method-assign]
        app._start_run(problems)
        assert await _wait_done(app, pilot)
        # Never more concurrent proves than the configured worker count.
        assert max_active <= 2
        assert app.counts["proved"] == 6


@pytest.mark.anyio
async def test_stop_marks_pending_problems_skipped() -> None:
    app = ProverApp(parallel=2)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        problems = _problems(6)
        app._filtered_problems = problems
        app._search_text = "p"
        await app._render_problem_list(problems)
        await pilot.pause()

        def fake(problem: dict) -> None:
            if app._stop_flag:
                return
            time.sleep(0.2)
            if app._stop_flag:
                return
            app.call_from_thread(app._finish_row, problem["id"], "proved", _result())

        app._prove_one = fake  # type: ignore[method-assign]
        app._start_run(problems)
        await pilot.pause(0.03)
        app.action_stop()
        assert await _wait_done(app, pilot)
        assert app.counts["proved"] + app.counts["stopped"] == 6
        assert app.counts["stopped"] >= 1
        assert not app._run_active
