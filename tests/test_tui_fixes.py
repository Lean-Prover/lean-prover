"""Regression + feature tests for the TUI hardening pass.

Covers the empty-list/action-pool consistency fix, the PROVER_THEME env
override, the confirmation modals, the /settings screen, the ProveScreen
placeholder guard, and the dead-pref wiring (window title, bell, paste
threshold). All tests drive the real ProverApp headlessly via Textual's Pilot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from textual.events import Paste
from textual.widgets import Button, Checkbox, DataTable, Input, ListView, OptionList, TextArea

from agent import models as models_mod
from agent.tui import (
    ConfirmScreen,
    LeaderboardScreen,
    ModelsScreen,
    ProverApp,
    ProofDiffScreen,
    PromptsScreen,
    ProveScreen,
    SettingsScreen,
    build_session_label,
)
from agent.tui_render import ProblemRow

PROBLEMS = [
    {"id": f"thm{i}", "statement": f"theorem thm{i} : True := by trivial",
     "difficulty": "easy"}
    for i in range(1, 6)
]


def _write_problems(tmp_path: Path) -> Path:
    path = tmp_path / "problems.json"
    path.write_text(json.dumps(PROBLEMS))
    return path


# ------------------------------------------------------------- active pool


@pytest.mark.anyio
async def test_active_pool_returns_full_pool_without_search() -> None:
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.problems = list(PROBLEMS)
        app._search_text = ""
        app._filtered_problems = []
        assert app._active_pool() == PROBLEMS
        # A search narrows the pool.
        app._search_text = "thm3"
        app._filtered_problems = [PROBLEMS[2]]
        assert app._active_pool() == [PROBLEMS[2]]


@pytest.mark.anyio
async def test_vim_queue_works_without_search() -> None:
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.problems = list(PROBLEMS)
        app._search_text = ""
        app._filtered_problems = []  # the empty-search state
        app._run_active = True  # queue as a follow-up instead of starting a run
        await app._render_problem_list(PROBLEMS)
        app.query_one("#problems-list", ListView).index = 2
        app.action_vim_queue()
        assert app._queued_prompts == [PROBLEMS[2]["statement"]]


@pytest.mark.anyio
async def test_run_remaining_uses_active_pool_and_proved_ids() -> None:
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.problems = list(PROBLEMS)
        app._search_text = ""
        app._filtered_problems = []
        app._proved_ids = {"thm1", "thm3"}
        app._run_problems = lambda problems, workers: None  # type: ignore[method-assign]
        app._start_run = lambda problems: app._queued_prompts.extend(
            p["statement"] for p in problems
        )  # type: ignore[method-assign]
        app.action_run_remaining()
        # thm1/thm3 already proved → the other three run.
        assert app._queued_prompts == [
            PROBLEMS[1]["statement"], PROBLEMS[3]["statement"], PROBLEMS[4]["statement"]
        ]


@pytest.mark.anyio
async def test_status_remaining_uses_active_pool() -> None:
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.problems = list(PROBLEMS)
        app._search_text = ""
        app._filtered_problems = []
        app.counts = {"proved": 1, "failed": 0, "stopped": 0}
        assert "remaining 4" in app._status_text()


@pytest.mark.anyio
async def test_mount_renders_full_problem_list(monkeypatch, tmp_path) -> None:
    problems_file = _write_problems(tmp_path)
    monkeypatch.setattr("agent.tui.PROBLEMS_FILE", problems_file)
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        # Wait for the trust flow + full-list render.
        for _ in range(50):
            await pilot.pause(0.02)
            if app.query_one("#problems-list", ListView).children:
                break
        rows = app.query_one("#problems-list", ListView).children
        assert len(rows) == len(PROBLEMS)
        assert all(isinstance(r, ProblemRow) for r in rows)


# ------------------------------------------------------------- theme override


@pytest.mark.anyio
async def test_prover_theme_env_override_wins(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PROVER_THEME", "prover-light")
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        assert app.theme == "prover-light"


# ------------------------------------------------------------- confirmations


@pytest.mark.anyio
async def test_quit_while_running_confirms(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app._run_active = True
        app.action_quit_app()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        # Cancel keeps the app alive.
        await pilot.press("n")
        await pilot.pause()
        assert not isinstance(app.screen, ConfirmScreen)


@pytest.mark.anyio
async def test_quit_idle_exits_directly(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        exits: list[bool] = []
        app.exit = lambda *a, **k: exits.append(True)  # type: ignore[method-assign]
        app._run_active = False
        app.action_quit_app()
        assert exits == [True]


@pytest.mark.anyio
async def test_models_delete_confirms(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("PROVER_MODEL", raising=False)
    models_mod.save_store(active="m1", profiles=[
        models_mod.ModelProfile(name="m1"),
        models_mod.ModelProfile(name="m2"),
    ])
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.action_models()
        await pilot.pause()
        opt_list = app.screen.query_one(OptionList)
        opt_list.highlighted = 0  # m1
        await pilot.pause()
        app.screen.action_delete_profile()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        # Cancel keeps both profiles.
        await pilot.press("n")
        await pilot.pause()
        assert len(models_mod.load_profiles()) == 2
        # Confirm deletes m1.
        app.screen.action_delete_profile()
        await pilot.pause()
        await pilot.press("y")
        await pilot.pause()
        assert [p.name for p in models_mod.load_profiles()] == ["m2"]


# ------------------------------------------------------------- settings screen


@pytest.mark.anyio
async def test_settings_command_opens_settings_screen(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", Input)
        prompt.focus()
        prompt.value = "/settings"
        await pilot.pause()
        res = pilot.press("enter")
        if hasattr(res, "__await__"):
            await res
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.anyio
async def test_settings_save_persists_toggles(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.action_settings()
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)
        app.screen.query_one("#s-copy", Checkbox).value = True
        app.screen.query_one("#s-bell", Checkbox).value = False
        app.screen.action_save()
        await pilot.pause()
        assert app.tui_settings.auto_copy_selection is True
        assert app.tui_settings.terminal_bell is False
        from agent.tui_settings import load_tui_settings

        persisted = load_tui_settings()
        assert persisted.auto_copy_selection is True
        assert persisted.terminal_bell is False


# ------------------------------------------------------------- prove placeholder


@pytest.mark.anyio
async def test_prove_screen_empty_guard_does_not_submit_placeholder(monkeypatch,
                                                                    tmp_path) -> None:
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.push_screen(ProveScreen())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ProveScreen)
        # Untouched TextArea must be empty (placeholder is not real content).
        assert screen.query_one("#prove-input").text.strip() == ""
        screen.action_submit()
        await pilot.pause()
        # Still on the prove screen — the empty guard fired, nothing dismissed.
        assert isinstance(app.screen, ProveScreen)


# ------------------------------------------------------------- dead-pref wiring


@pytest.mark.anyio
async def test_terminal_title_gated_by_window_title_setting(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        calls: list[bool] = []
        app._terminal_title.update = lambda *a, **k: calls.append(True)  # type: ignore[method-assign]
        from dataclasses import replace

        app.tui_settings = replace(app.tui_settings, window_title_updates=False)
        app._sync_terminal_title()
        assert calls == []
        app.tui_settings = replace(app.tui_settings, window_title_updates=True)
        app._sync_terminal_title()
        assert len(calls) == 1


@pytest.mark.anyio
async def test_large_paste_warns(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        notifs: list[tuple] = []
        app.notify = lambda msg, **k: notifs.append((msg, k))  # type: ignore[method-assign]
        from dataclasses import replace

        app.tui_settings = replace(app.tui_settings, large_paste_threshold=10)
        prompt = app.query_one("#prompt")
        prompt.focus()
        await pilot.pause()
        await app.on_event(Paste("x" * 50))
        assert any("Large paste" in msg for msg, _ in notifs)


@pytest.mark.anyio
async def test_finish_custom_row_syncs_terminal_title(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        calls: list[bool] = []
        app._sync_terminal_title = lambda: calls.append(True)  # type: ignore[method-assign]
        app._drain_queued_prompts = lambda: None  # type: ignore[method-assign]
        result = type("R", (), {"proved": True, "proof": "by trivial"})()
        app._finish_custom_row("custom", "proved", result)
        assert calls == [True]
        assert "custom" in app._proved_ids


# ------------------------------------------------------------- diff stacking


@pytest.mark.anyio
async def test_proof_diff_does_not_stack(monkeypatch, tmp_path) -> None:
    lean_tmp = tmp_path / "lean" / "tmp"
    lean_tmp.mkdir(parents=True)
    (lean_tmp / "Prover_p1.lean").write_text("new body")
    monkeypatch.setattr("agent.tui.REPO", tmp_path)
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app._get_last_proof_body = lambda pid: "old body"  # type: ignore[method-assign]
        app._show_proof_diff("p1", "new body")
        await pilot.pause()
        assert isinstance(app.screen, ProofDiffScreen)
        # A second accepted response must not stack another modal.
        app._show_proof_diff("p1", "newer body")
        await pilot.pause()
        stack = [s for s in app.screen_stack if isinstance(s, ProofDiffScreen)]
        assert len(stack) == 1
        await pilot.press("q")
        await pilot.pause()


# ------------------------------------------------------------- second batch


async def _submit(pilot, text: str) -> None:
    from textual.widgets import Input

    prompt = pilot.app.query_one("#prompt", Input)
    prompt.focus()
    prompt.value = text
    await pilot.pause()
    res = pilot.press("enter")
    if hasattr(res, "__await__"):
        await res
    await pilot.pause()


@pytest.mark.anyio
async def test_custom_problem_id_always_suffixed(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        captured: list[dict] = []
        app._start_run = lambda problems: captured.extend(problems)  # type: ignore[method-assign]
        app._prove_statement("theorem foo : True := by trivial")
        app._prove_statement("theorem bar : True := by trivial")
        assert [p["id"] for p in captured] == ["foo-1", "bar-2"]


@pytest.mark.anyio
async def test_prompt_history_recalls_submissions(monkeypatch, tmp_path) -> None:
    from textual.widgets import Input

    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, "/workers 4")
        await _submit(pilot, "/clear")
        prompt = app.query_one("#prompt", Input)
        prompt.focus()
        await pilot.pause()
        await pilot.press("up")
        await pilot.pause()
        assert prompt.value == "/clear"
        await pilot.press("up")
        await pilot.pause()
        assert prompt.value == "/workers 4"
        await pilot.press("down")
        await pilot.pause()
        assert prompt.value == "/clear"
        await pilot.press("down")
        await pilot.pause()
        assert prompt.value == ""  # back to the original draft


@pytest.mark.anyio
async def test_message_screen_copy_button(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        copied: list[str] = []
        app.copy_to_clipboard = lambda t: copied.append(t)  # type: ignore[method-assign]
        app._show_command_message("hello body")
        await pilot.pause()
        from agent.tui import MessageScreen

        assert isinstance(app.screen, MessageScreen)
        app.screen.query_one("#copy", Button).press()
        await pilot.pause()
        assert copied and "hello body" in copied[0]
        await pilot.press("q")
        await pilot.pause()


@pytest.mark.anyio
async def test_leaderboard_sorted_and_refreshable(monkeypatch, tmp_path) -> None:
    board = tmp_path / "leaderboard.json"
    board.write_text(json.dumps([
        {"name": "low", "score": 10, "total": 100, "tiers": {}, "date": "a"},
        {"name": "high", "score": 90, "total": 100, "tiers": {}, "date": "b"},
    ]))
    monkeypatch.setattr("agent.tui.REPO", tmp_path)
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.push_screen(LeaderboardScreen())
        await pilot.pause()
        table = app.screen.query_one(DataTable)
        # Sorted by score descending → "high" is row 1.
        assert "high" in " ".join(table.get_row_at(0))
        assert "low" in " ".join(table.get_row_at(1))
        # Refresh picks up a changed file.
        board.write_text(json.dumps([
            {"name": "new", "score": 50, "total": 100, "tiers": {}, "date": "c"},
        ]))
        await pilot.press("r")
        await pilot.pause()
        assert "new" in " ".join(table.get_row_at(0))
        await pilot.press("q")
        await pilot.pause()


@pytest.mark.anyio
async def test_sessions_search_filters(monkeypatch, tmp_path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "20260101-000001-alpha.jsonl").write_text(
        '{"event": "start", "problem_id": "a"}\n')
    (sessions_dir / "20260101-000002-beta.jsonl").write_text(
        '{"event": "start", "problem_id": "b"}\n')
    monkeypatch.setenv("PROVER_SESSIONS_DIR", str(sessions_dir))
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.action_sessions()
        await pilot.pause()
        from agent.tui import SessionsScreen

        assert isinstance(app.screen, SessionsScreen)
        opt_list = app.screen.query_one(OptionList)
        assert len(app.screen._ids) == 2
        # Type into the filter box.
        app.screen.query_one("#sess-search", Input).value = "beta"
        app.screen.on_input_changed(
            Input.Changed(app.screen.query_one("#sess-search", Input), "beta")
        )
        await pilot.pause()
        assert app.screen._ids == ["20260101-000002-beta"]
        assert opt_list.option_count == 1
        await pilot.press("q")
        await pilot.pause()


# ------------------------------------------------------------- third batch


@pytest.mark.anyio
async def test_prompts_search_filters(monkeypatch, tmp_path) -> None:
    from agent.prompt_templates import PromptTemplate

    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    templates = [
        PromptTemplate(name="brief", path=tmp_path / "brief.md",
                       content="Prove tersely", description="Concise proof"),
        PromptTemplate(name="detailed", path=tmp_path / "detailed.md",
                       content="Prove fully", description="Long form"),
    ]
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.push_screen(PromptsScreen(templates))
        await pilot.pause()
        assert isinstance(app.screen, PromptsScreen)
        app.screen.query_one("#prompts-search", Input).value = "brief"
        app.screen.on_input_changed(
            Input.Changed(app.screen.query_one("#prompts-search", Input), "brief")
        )
        await pilot.pause()
        opt_list = app.screen.query_one(OptionList)
        assert opt_list.option_count == 1
        assert app.screen.query_one(OptionList).get_option_at_index(0).id == "brief"
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.anyio
async def test_models_search_filters_and_current_resolves(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("PROVER_MODEL", raising=False)
    models_mod.save_store(active="m1", profiles=[
        models_mod.ModelProfile(name="m1"),
        models_mod.ModelProfile(name="m2"),
    ])
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.action_models()
        await pilot.pause()
        from agent.tui import ModelsScreen

        assert isinstance(app.screen, ModelsScreen)
        app.screen.query_one("#models-search", Input).value = "m2"
        app.screen.on_input_changed(
            Input.Changed(app.screen.query_one("#models-search", Input), "m2")
        )
        await pilot.pause()
        opt_list = app.screen.query_one(OptionList)
        assert opt_list.option_count == 1
        opt_list.highlighted = 0
        await pilot.pause()
        # _current must resolve the filtered row to the right profile.
        assert app.screen._current().name == "m2"
        await pilot.press("q")
        await pilot.pause()


@pytest.mark.anyio
async def test_sessions_pagination_indicator(monkeypatch, tmp_path) -> None:
    from agent.tui import SessionsScreen

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True)
    for i in range(3):
        (sessions_dir / f"20260101-00000{i}-s{i}.jsonl").write_text(
            '{"event": "start", "problem_id": "p"}\n')
    monkeypatch.setenv("PROVER_SESSIONS_DIR", str(sessions_dir))
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(SessionsScreen, "MAX_ROWS", 2)
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.action_sessions()
        await pilot.pause()
        assert isinstance(app.screen, SessionsScreen)
        count = str(app.screen.query_one("#sess-count").render())
        assert "2 of 3 sessions" in count
        assert "showing first 2" in count
        await pilot.press("q")
        await pilot.pause()


def test_diff_markup_colorizes() -> None:
    from agent.tui import ProofDiffScreen

    markup = ProofDiffScreen._diff_markup(
        "--- old\n+++ new\n@@ -1,2 +1,2 @@\n-old\n+new\n context\n"
    )
    assert "[red]-old[/red]" in markup
    assert "[green]+new[/green]" in markup
    assert "[cyan]@@ -1,2 +1,2 @@[/cyan]" in markup
    assert "[bold cyan]+++ new[/bold cyan]" in markup


def test_session_label_includes_timestamp(tmp_path) -> None:
    from agent.session_manager import SessionRecord

    sp = tmp_path / "s1.jsonl"
    sp.write_text("x")
    rec = SessionRecord(id="s1", path=str(sp), problem_id="p1", model="?",
                        title="")
    recs = [{"event": "start", "problem_id": "p1"},
            {"event": "result", "proved": True, "steps": 3, "seconds": 2.0}]
    label = build_session_label(sp, rec, recs)
    assert "[" in label and "]" in label  # a timestamp is present
    assert label.startswith("✓")


@pytest.mark.anyio
async def test_question_mark_opens_hotkeys(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        # Focus the problem list so ? isn't typed into the search box.
        app.query_one("#problems-list", ListView).focus()
        await pilot.pause()
        await pilot.press("?")
        await pilot.pause()
        from agent.tui import MessageScreen

        assert isinstance(app.screen, MessageScreen)
        assert "ctrl+k" in app.screen._body
        await pilot.press("q")
        await pilot.pause()


# ------------------------------------------------------------- fourth batch


@pytest.mark.anyio
async def test_prompt_template_edit_saves(monkeypatch, tmp_path) -> None:
    from agent.prompt_templates import PromptTemplate
    from agent.tui import PromptEditScreen

    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    template_path = tmp_path / "brief.md"
    template_path.write_text("original content")
    template = PromptTemplate(name="brief", path=template_path,
                              content="original content")
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.push_screen(PromptsScreen([template]))
        await pilot.pause()
        app.screen.query_one(OptionList).highlighted = 0
        await pilot.press("e")
        await pilot.pause()
        assert isinstance(app.screen, PromptEditScreen)
        app.screen.query_one("#prompt-edit-input", TextArea).text = "edited content"
        app.screen.action_save()
        await pilot.pause()
        assert template_path.read_text() == "edited content"
        await pilot.press("q")
        await pilot.pause()


def test_aggregate_usage_by_model() -> None:
    from agent.session_usage import (
        RequestUsage,
        SessionUsage,
        aggregate_usage_by_model,
        render_usage_by_model,
    )

    r1 = RequestUsage(1, "t", "model-a", 10, 5, 15, 0.01)
    r2 = RequestUsage(2, "t", "model-a", 20, 5, 25, 0.02)
    r3 = RequestUsage(3, "t", "model-b", 30, 5, 35, 0.03)
    by = aggregate_usage_by_model([
        SessionUsage(requests=(r1,)),
        SessionUsage(requests=(r2, r3)),
    ])
    assert by["model-a"].total_tokens == 40
    assert by["model-b"].total_tokens == 35
    text = render_usage_by_model(by)
    assert "model-a" in text and "model-b" in text
    assert "by model" in text


@pytest.mark.anyio
async def test_usage_models_opens_message(monkeypatch, tmp_path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "20260101-000001-a.jsonl").write_text(
        '{"event": "start", "problem_id": "p"}\n'
        '{"event": "llm_response", "model": "qwen", "prompt_tokens": 10,'
        ' "completion_tokens": 5, "tokens": 15}\n'
    )
    monkeypatch.setenv("PROVER_SESSIONS_DIR", str(sessions_dir))
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, "/usage models")
        from agent.tui import MessageScreen

        assert isinstance(app.screen, MessageScreen)
        assert "by model" in app.screen._body
        assert "qwen" in app.screen._body
        await pilot.press("q")
        await pilot.pause()


@pytest.mark.anyio
async def test_sessions_delete_all(monkeypatch, tmp_path) -> None:
    from agent.tui import ConfirmScreen, SessionsScreen

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "20260101-000001-alpha.jsonl").write_text('{"event":"start"}\n')
    (sessions_dir / "20260101-000002-beta.jsonl").write_text('{"event":"start"}\n')
    monkeypatch.setenv("PROVER_SESSIONS_DIR", str(sessions_dir))
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.action_sessions()
        await pilot.pause()
        assert isinstance(app.screen, SessionsScreen)
        await pilot.press("ctrl+a")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("y")
        await pilot.pause()
        assert app.screen._all_paths == {}
        assert len(list(Path(app.session_dir).glob("*.jsonl"))) == 0
        await pilot.press("q")
        await pilot.pause()


# ------------------------------------------------------------- fifth batch


@pytest.mark.anyio
async def test_prompt_history_persists_across_apps(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, "/workers 4")
        await _submit(pilot, "/clear")
    # A fresh app in the same config dir should reload the history.
    app2 = ProverApp()
    async with app2.run_test(size=(140, 40)) as pilot2:
        await pilot2.pause()
        assert app2._prompt_history == ["/workers 4", "/clear"]


@pytest.mark.anyio
async def test_prompt_template_create(monkeypatch, tmp_path) -> None:
    from agent.prompt_templates import prompt_templates_dirs
    from agent.tui import PromptNewScreen

    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.push_screen(PromptsScreen([]))
        await pilot.pause()
        await pilot.press("a")
        await pilot.pause()
        assert isinstance(app.screen, PromptNewScreen)
        app.screen.query_one("#new-name", Input).value = "brief"
        app.screen.action_submit()
        await pilot.pause()
        target = prompt_templates_dirs()[0] / "brief.md"
        assert target.exists()
        assert "Prove {{ arguments }}" in target.read_text()
        await pilot.press("q")
        await pilot.pause()


@pytest.mark.anyio
async def test_prompt_template_delete(monkeypatch, tmp_path) -> None:
    from agent.prompt_templates import PromptTemplate
    from agent.tui import ConfirmScreen

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True)
    template_path = prompts_dir / "brief.md"
    template_path.write_text("content")
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    template = PromptTemplate(name="brief", path=template_path, content="content")
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.push_screen(PromptsScreen([template]))
        await pilot.pause()
        app.screen.query_one(OptionList).highlighted = 0
        await pilot.press("d")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("y")
        await pilot.pause()
        assert not template_path.exists()
        await pilot.press("q")
        await pilot.pause()


# ------------------------------------------------------------- sixth batch


@pytest.mark.anyio
async def test_inspect_problem_shows_statement(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.problems = list(PROBLEMS)
        app._search_text = ""
        app._filtered_problems = []
        await app._render_problem_list(PROBLEMS)
        app.query_one("#problems-list", ListView).index = 1
        app._get_last_proof_body = lambda pid: None  # type: ignore[method-assign]
        app.action_inspect_problem()
        await pilot.pause()
        from agent.tui import MessageScreen

        assert isinstance(app.screen, MessageScreen)
        assert PROBLEMS[1]["statement"] in app.screen._body
        await pilot.press("q")
        await pilot.pause()


@pytest.mark.anyio
async def test_copy_statement(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.problems = list(PROBLEMS)
        app._search_text = ""
        app._filtered_problems = []
        await app._render_problem_list(PROBLEMS)
        app.query_one("#problems-list", ListView).index = 0
        copied: list[str] = []
        app.copy_to_clipboard = lambda t: copied.append(t)  # type: ignore[method-assign]
        app.action_copy_statement()
        assert copied == [PROBLEMS[0]["statement"]]


@pytest.mark.anyio
async def test_status_includes_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        assert app._status_model in app._status_text()


@pytest.mark.anyio
async def test_ctrl_r_reload(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        called: list[bool] = []
        app._reload_resources = lambda: called.append(True)  # type: ignore[method-assign]
        app.query_one("#problems-list", ListView).focus()
        await pilot.pause()
        await pilot.press("ctrl+r")
        await pilot.pause()
        assert called == [True]
