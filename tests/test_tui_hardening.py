"""TUI hardening tests: /model provider wiring, corrupt-JSON guards, the
session browser (rename/delete/live-switch), the navigable proof-diff screen,
and the pure helpers behind them.

Headless Textual (App.run_test) where interaction matters; plain unit tests
for the pure parts (CommandResult parsing, _compute_diff, session labels).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from textual.selection import SELECT_ALL
from textual.widgets import DataTable, Input, OptionList

from agent.commands import CommandResult, create_default_command_registry
from agent.session_manager import SessionManager, SessionRecord
from agent.tui import (
    LeaderboardScreen,
    ProofDiffScreen,
    ProverApp,
    SelectableRichLog,
    SessionRenameScreen,
    SessionsScreen,
    build_session_label,
    load_problems,
)

# --------------------------------------------------------------------------- pure


def _write_session(sessions_dir: Path, sid: str, *, proved: bool = True,
                   stopped: bool = False) -> Path:
    """Write a minimal session transcript that list_sessions()/read_session pick up."""
    path = sessions_dir / f"{sid}.jsonl"
    path.write_text("\n".join([
        json.dumps({"event": "start", "problem_id": "p1",
                    "statement": "theorem t : True := by trivial"}),
        json.dumps({"event": "result", "proved": proved, "stopped": stopped,
                    "steps": 1, "seconds": 1.0, "tokens": 10,
                    "prompt_tokens": 4, "completion_tokens": 6}),
    ]))
    return path


def test_model_command_splits_provider() -> None:
    class Session:
        @property
        def model(self) -> str:
            return "old"

        @property
        def session_dir(self) -> Path:
            return Path(".")

        @property
        def session_ids(self) -> list[str]:
            return []

        @property
        def problems_total(self) -> int:
            return 0

        @property
        def counts(self) -> dict[str, int]:
            return {}

        @property
        def n_workers(self) -> int:
            return 1

        @property
        def max_workers(self) -> int:
            return 16

        @property
        def is_running(self) -> bool:
            return False

        @property
        def thinking_level(self) -> str:
            return "off"

        @property
        def current_session_id(self) -> str | None:
            return None

    registry = create_default_command_registry()
    result = registry.execute(Session(), "/model deepseek-r1@openrouter")
    assert result.set_model_choice == "deepseek-r1"
    assert result.set_model_provider == "openrouter"
    # Without @provider the provider part stays unset.
    plain = registry.execute(Session(), "/model gpt-4o")
    assert plain.set_model_choice == "gpt-4o"
    assert plain.set_model_provider is None


def test_compute_diff_produces_unified_diff() -> None:
    app = ProverApp()
    old = "theorem t : True := by\n  trivial\n"
    new = "theorem t : True := by\n  exact trivial\n"
    diff = app._compute_diff(old, new)
    assert diff.startswith("--- old")
    assert "+++ new" in diff
    assert "-  trivial" in diff
    assert "+  exact trivial" in diff


def test_build_session_label() -> None:
    rec = SessionRecord(id="s1", path="/x/s1.jsonl", problem_id="p1", model="?",
                        title="my proof")
    recs = [
        {"event": "start", "problem_id": "p1"},
        {"event": "result", "proved": True, "steps": 3, "seconds": 2.0,
         "tokens": 12},
    ]
    label = build_session_label(Path("/x/s1.jsonl"), rec, recs)
    assert "s1" in label
    assert "p1" in label
    assert "steps=3" in label
    assert "my proof" in label
    assert label.startswith("✓")


# --------------------------------------------------------------------------- corrupt JSON


def test_load_problems_raises_on_corrupt_file(monkeypatch, tmp_path) -> None:
    bad = tmp_path / "problems.json"
    bad.write_text("{ not valid json")
    monkeypatch.setattr("agent.tui.PROBLEMS_FILE", bad)
    with pytest.raises(ValueError, match="unreadable"):
        load_problems()


def test_load_problems_empty_when_missing(monkeypatch, tmp_path) -> None:
    missing = tmp_path / "nope.json"
    monkeypatch.setattr("agent.tui.PROBLEMS_FILE", missing)
    assert load_problems() == []


@pytest.mark.anyio
async def test_corrupt_problems_does_not_crash_app(monkeypatch, tmp_path) -> None:
    bad = tmp_path / "problems.json"
    bad.write_text("{ nope")
    monkeypatch.setattr("agent.tui.PROBLEMS_FILE", bad)
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # App is still up, not a crash screen.
        assert app.screen is not None
        selected = app.query_one("#log", SelectableRichLog).get_selection(SELECT_ALL)
        assert selected is not None and "unreadable" in selected[0]


@pytest.mark.anyio
async def test_corrupt_leaderboard_shows_error_not_crash(monkeypatch, tmp_path) -> None:
    (tmp_path / "leaderboard.json").write_text("{ not json")
    monkeypatch.setattr("agent.tui.REPO", tmp_path)
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(LeaderboardScreen())
        await pilot.pause()
        assert isinstance(app.screen, LeaderboardScreen)
        table = app.screen.query_one(DataTable)
        row = table.get_row_at(0)
        assert "unreadable" in " ".join(row)


# --------------------------------------------------------------------------- /model provider wiring


@pytest.mark.anyio
async def test_model_command_applies_provider(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("PROVER_MODEL", raising=False)
    from agent import models

    app = ProverApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._apply_command(CommandResult(
            handled=True, set_model_choice="deepseek-r1",
            set_model_provider="openrouter",
        ))
        await pilot.pause()
        profile = models.profile_for("deepseek-r1")
        assert profile is not None
        # Bound to the openrouter provider's endpoint, not the legacy default.
        assert profile.base_url == "https://openrouter.ai/api/v1"


@pytest.mark.anyio
async def test_model_command_unknown_provider_notifies(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("PROVER_MODEL", raising=False)
    from agent import models

    app = ProverApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._apply_command(CommandResult(
            handled=True, set_model_choice="deepseek-r1",
            set_model_provider="does-not-exist",
        ))
        await pilot.pause()
        # No profile was bound.
        assert models.profile_for("deepseek-r1") is None


# --------------------------------------------------------------------------- SessionsScreen


def _make_sessions_screen_app(tmp_path, monkeypatch) -> ProverApp:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    _write_session(sessions_dir, "20260101-000001-alpha", proved=True)
    _write_session(sessions_dir, "20260101-000002-beta", proved=False)
    monkeypatch.setenv("PROVER_SESSIONS_DIR", str(sessions_dir))
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    return ProverApp()


async def _open_sessions(app, pilot, index: int = 0) -> None:
    await pilot.pause()
    app.action_sessions()
    await pilot.pause()
    opt_list = app.screen.query_one(OptionList)
    opt_list.highlighted = index
    await pilot.pause()


@pytest.mark.anyio
async def test_sessions_rename(monkeypatch, tmp_path) -> None:
    app = _make_sessions_screen_app(tmp_path, monkeypatch)
    async with app.run_test(size=(120, 40)) as pilot:
        await _open_sessions(app, pilot, index=1)  # alpha
        await pilot.press("r")
        await pilot.pause()
        assert isinstance(app.screen, SessionRenameScreen)
        app.screen.query_one("#rename-input", Input).value = "renamed proof"
        await pilot.press("enter")
        await pilot.pause()
        record = SessionManager().get("20260101-000001-alpha")
        assert record is not None and record.title == "renamed proof"
        # Back on the sessions browser.
        assert isinstance(app.screen, SessionsScreen)
        await pilot.press("q")
        await pilot.pause()


@pytest.mark.anyio
async def test_sessions_delete(monkeypatch, tmp_path) -> None:
    app = _make_sessions_screen_app(tmp_path, monkeypatch)
    async with app.run_test(size=(120, 40)) as pilot:
        await _open_sessions(app, pilot, index=1)  # alpha
        await pilot.press("d")
        await pilot.pause()
        from agent.tui import ConfirmScreen

        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("y")
        await pilot.pause()
        sessions_dir = Path(app.session_dir)
        assert not (sessions_dir / "20260101-000001-alpha.jsonl").exists()
        assert (sessions_dir / "20260101-000002-beta.jsonl").exists()
        assert SessionManager().get("20260101-000001-alpha") is None
        await pilot.press("q")
        await pilot.pause()


@pytest.mark.anyio
async def test_sessions_delete_cancel_keeps_session(monkeypatch, tmp_path) -> None:
    app = _make_sessions_screen_app(tmp_path, monkeypatch)
    async with app.run_test(size=(120, 40)) as pilot:
        await _open_sessions(app, pilot, index=1)  # alpha
        await pilot.press("d")
        await pilot.pause()
        from agent.tui import ConfirmScreen

        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("n")
        await pilot.pause()
        sessions_dir = Path(app.session_dir)
        assert (sessions_dir / "20260101-000001-alpha.jsonl").exists()
        assert SessionManager().get("20260101-000001-alpha") is not None
        # Back on the sessions browser after cancelling.
        assert isinstance(app.screen, SessionsScreen)
        await pilot.press("q")
        await pilot.pause()


@pytest.mark.anyio
async def test_sessions_live_switch(monkeypatch, tmp_path) -> None:
    app = _make_sessions_screen_app(tmp_path, monkeypatch)
    async with app.run_test(size=(120, 40)) as pilot:
        await _open_sessions(app, pilot, index=0)  # beta
        await pilot.press("s")
        await pilot.pause()
        assert app._active_session_id == "20260101-000002-beta"
        assert app.current_session_id == "20260101-000002-beta"


@pytest.mark.anyio
async def test_sessions_enter_replays(monkeypatch, tmp_path) -> None:
    app = _make_sessions_screen_app(tmp_path, monkeypatch)
    async with app.run_test(size=(120, 40)) as pilot:
        await _open_sessions(app, pilot)
        await pilot.press("enter")
        await pilot.pause()
        from agent.tui import ReplayScreen

        assert isinstance(app.screen, ReplayScreen)
        await pilot.press("q")
        await pilot.pause()


# --------------------------------------------------------------------------- proof diff


@pytest.mark.anyio
async def test_proof_diff_opens_navigable_screen(monkeypatch, tmp_path) -> None:
    lean_tmp = tmp_path / "lean" / "tmp"
    lean_tmp.mkdir(parents=True)
    (lean_tmp / "Prover_p1.lean").write_text("current body")
    monkeypatch.setattr("agent.tui.REPO", tmp_path)
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._get_last_proof_body = lambda pid: "old body"  # type: ignore[method-assign]
        app._show_proof_diff("p1", "new body")
        await pilot.pause()
        assert isinstance(app.screen, ProofDiffScreen)
        # Navigation keys are handled without raising.
        await pilot.press("j", "k", "space", "0", "G")
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()


@pytest.mark.anyio
async def test_proof_diff_empty_diff_does_not_push(monkeypatch, tmp_path) -> None:
    lean_tmp = tmp_path / "lean" / "tmp"
    lean_tmp.mkdir(parents=True)
    (lean_tmp / "Prover_p1.lean").write_text("same")
    monkeypatch.setattr("agent.tui.REPO", tmp_path)
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._get_last_proof_body = lambda pid: "same"  # type: ignore[method-assign]
        app._show_proof_diff("p1", "same")
        await pilot.pause()
        assert not isinstance(app.screen, ProofDiffScreen)
