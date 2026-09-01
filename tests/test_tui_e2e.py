"""End-to-end TUI pilot tests — drive the running app with real keypresses.

Unlike the unit tests (which call methods directly), these mount the live
``ProverApp`` and simulate keystrokes through Textual's ``Pilot``, verifying
the dynamic surfaces: starting a proof with ``p`` and streaming events into the
panels, search filtering, the slash-command palette, and screen navigation.

``prove()`` needs an LLM/Lean endpoint, so ``_prove_one`` is stubbed to emit
synthetic events through the real ``_probe_event`` -> ``_render_event`` path —
that exercises the live event-streaming machinery headlessly.

Note: the app-level single-letter bindings (``p``/``l``/``v``) are non-priority,
so the test focuses the problem list first (the mount-time trust flow focuses
the search bar, which consumes letters).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from textual.selection import SELECT_ALL
from textual.widgets import Input, ListView

from agent.tui import (
    LeaderboardScreen,
    MessageScreen,
    ProblemRow,
    ProverApp,
    SelectableRichLog,
    SessionsScreen,
)

PROBLEMS = [
    {"id": f"p{i}", "statement": f"theorem t{i} : 1 + {i} = {1 + i} := by omega",
     "difficulty": "trivial"}
    for i in range(3)
]


def _events(pid: str) -> list[dict]:
    return [
        {"event": "start", "problem_id": pid, "statement": f"theorem {pid} : True := by trivial"},
        {"event": "llm_request", "step": 1},
        {"event": "llm_response", "step": 1, "tokens": 10, "body": "  trivial"},
        {"event": "result", "proved": True, "steps": 1, "seconds": 0.5,
         "tokens": 10, "cost": 0.0, "session_id": None, "stopped": False},
    ]


async def _mount(app: ProverApp, pilot, problems=PROBLEMS) -> None:
    app.problems = list(problems)
    app._filtered_problems = list(problems)
    app._search_text = ""
    await app._render_problem_list(problems)
    await pilot.pause()


async def _focus_list(app: ProverApp, pilot) -> None:
    app.query_one("#problems-list", ListView).focus()
    await pilot.pause()


def _text(widget) -> str:
    sel = widget.get_selection(SELECT_ALL)
    return (sel[0] if sel else "") or ""


async def _wait_done(app: ProverApp, pilot, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while app._run_active and time.monotonic() < deadline:
        await pilot.pause(0.02)
    return not app._run_active


def _stub_prove_one(app: ProverApp) -> None:
    def fake_prove_one(problem: dict) -> None:
        # Match the real prove-loop: on_event fires _probe_event on the worker
        # thread (it marshals to the UI thread itself); the finish handler is
        # marshalled explicitly.
        for ev in _events(problem["id"]):
            app._probe_event(problem["id"], ev)
        app.call_from_thread(app._finish_row, problem["id"], "proved",
                             SimpleNamespace(proved=True, proof="  trivial"))

    app._prove_one = fake_prove_one  # type: ignore[method-assign]


@pytest.mark.anyio
async def test_prove_selected_streams_events_into_panels() -> None:
    app = ProverApp(parallel=1)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _mount(app, pilot)
        _stub_prove_one(app)
        await _focus_list(app, pilot)
        await pilot.press("p")  # prove selected problem
        assert await _wait_done(app, pilot)
        await pilot.pause()  # flush the queued _finish_row on the UI thread
        await pilot.pause()
        assert app.counts["proved"] == 1
        assert not app._run_active
        assert "PROVED" in _text(app.query_one("#log", SelectableRichLog))


@pytest.mark.anyio
async def test_prove_marks_row_proved_and_writes_proof_panel() -> None:
    app = ProverApp(parallel=1)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _mount(app, pilot)
        _stub_prove_one(app)
        await _focus_list(app, pilot)
        await pilot.press("p")
        assert await _wait_done(app, pilot)
        await pilot.pause()  # flush the queued _finish_row on the UI thread
        await pilot.pause()
        row = next(r for r in app.query(ProblemRow) if r.problem["id"] == "p0")
        assert row.status == "proved"
        assert "trivial" in _text(app.query_one("#proof", SelectableRichLog))


@pytest.mark.anyio
async def test_search_filters_problem_list() -> None:
    app = ProverApp(parallel=1)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _mount(app, pilot)
        await _focus_list(app, pilot)
        await pilot.press("/")  # focus the search bar
        await pilot.pause()
        await pilot.press("p", "1")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        rows = list(app.query(ProblemRow))
        assert [r.problem["id"] for r in rows] == ["p1"]


@pytest.mark.anyio
async def test_command_palette_and_message_modal() -> None:
    app = ProverApp(parallel=1)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _mount(app, pilot)
        await _focus_list(app, pilot)
        await pilot.press("ctrl+k")  # open slash-command palette
        await pilot.pause()
        prompt = app.query_one("#prompt", Input)
        assert prompt.value.startswith("/")
        prompt.value = "/status"
        prompt.cursor_position = len(prompt.value)
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, MessageScreen)
        await pilot.press("q")
        await pilot.pause()


@pytest.mark.anyio
async def test_screen_navigation_keyboard_shortcuts() -> None:
    app = ProverApp(parallel=1)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _mount(app, pilot)
        await _focus_list(app, pilot)
        await pilot.press("l")  # leaderboard
        await pilot.pause()
        assert isinstance(app.screen, LeaderboardScreen)
        await pilot.press("q")
        await pilot.pause()
        await _focus_list(app, pilot)
        await pilot.press("v")  # sessions browser
        await pilot.pause()
        assert isinstance(app.screen, SessionsScreen)
        await pilot.press("q")
        await pilot.pause()
        assert app.screen is not None
