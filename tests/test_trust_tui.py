"""Tests for the TUI trust modal and gating — ported from
huggingface/tau tests (test_project_trust.py modal tests), adapted to
prover's TrustScreen and PROVER_TRUST env policy."""

from __future__ import annotations

import asyncio

from textual.widgets import OptionList

from agent.project_trust import (
    CanonicalProjectPath,
    ProjectTrustRequest,
    ProjectTrustStore,
    ProtectedResourceSummary,
)
from agent.tui import ProverApp, TrustScreen


def _fake_request(tmp_path) -> ProjectTrustRequest:
    from agent.project_trust import canonicalize_project_path

    canonical = canonicalize_project_path(tmp_path)
    summary = ProtectedResourceSummary(
        cwd=canonical,
        categories=("problems",),
        counts={"problems": 1},
        sample_paths=(tmp_path / "benchmark" / "problems.json",),
    )
    return ProjectTrustRequest(canonical, summary, inherited_entry=None)


async def _wait_for_modal(pilot) -> TrustScreen:
    for _ in range(100):
        for screen in pilot.app.screen_stack:
            if isinstance(screen, TrustScreen) and screen.children:
                return screen
        await pilot.pause(0.05)
    raise AssertionError("trust modal never appeared")


def test_trust_modal_lists_all_choices_and_esc_cancels(tmp_path) -> None:
    async def scenario() -> None:
        app = ProverApp()
        async with app.run_test() as pilot:
            screen = TrustScreen(_fake_request(tmp_path))
            choice: list[object] = []

            async def on_result(result: object | None) -> None:
                choice.append(result)

            app.push_screen(screen, on_result)
            await pilot.pause()
            options = screen.query_one("#trust-list", OptionList)
            assert [str(o.prompt) for o in options.options] == [
                "trust-exact", "trust-parent", "trust-run",
                "decline-exact", "decline-run",
            ]
            screen.dismiss(None)
            await pilot.pause()
            assert choice == [None]

    asyncio.run(scenario())


def test_trust_modal_select_triggers_choice(tmp_path) -> None:
    async def scenario() -> None:
        app = ProverApp()
        async with app.run_test() as pilot:
            screen = TrustScreen(_fake_request(tmp_path))
            choice: list[object] = []

            async def on_result(result: object | None) -> None:
                choice.append(result)

            app.push_screen(screen, on_result)
            await pilot.pause()
            screen.dismiss("trust-exact")
            await pilot.pause()
            assert choice == ["trust-exact"]

    asyncio.run(scenario())


def test_trust_modal_option_selection_returns_choice_id(tmp_path) -> None:
    """A real OptionList selection must dismiss with the choice id.

    Regression: options were built without ids, so every selection hit
    `dismiss(None)` — in ask mode trust could never be granted interactively.
    """
    async def scenario() -> None:
        app = ProverApp()
        async with app.run_test() as pilot:
            screen = TrustScreen(_fake_request(tmp_path))
            choice: list[object] = []

            async def on_result(result: object | None) -> None:
                choice.append(result)

            app.push_screen(screen, on_result)
            await pilot.pause()
            options = screen.query_one("#trust-list", OptionList)
            assert [o.id for o in options.options] == [
                "trust-exact", "trust-parent", "trust-run",
                "decline-exact", "decline-run",
            ]
            options.focus()
            options.highlighted = 2
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert choice == ["trust-run"]

    asyncio.run(scenario())


def test_untrusted_project_blocks_problem_loading(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PROVER_TRUST", "never")
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path))

    async def scenario() -> None:
        app = ProverApp()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause(0.1)
            await pilot.pause()
            assert app.problems == []
            assert app._trust_resolution is not None
            assert not app._trust_resolution.trusted
            app.exit()

    asyncio.run(scenario())


def test_ask_mode_persists_approved_trust_and_loads_problems(tmp_path, monkeypatch) -> None:
    store_dir = tmp_path / "cfg"
    monkeypatch.setenv("PROVER_TRUST", "ask")
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(store_dir))

    async def scenario() -> None:
        app = ProverApp()
        async with app.run_test(size=(140, 40)) as pilot:
            trust = await _wait_for_modal(pilot)
            trust.dismiss("trust-exact")
            for _ in range(50):
                if app._trust_resolution is not None:
                    break
                await pilot.pause(0.05)
            assert app.problems, "problems must load after trusting the project"
            store = ProjectTrustStore()
            decision = store.nearest(CanonicalProjectPath(app._trust_summary.cwd.value))
            assert decision is not None and decision.decision == "trusted"
            app.exit()

    asyncio.run(scenario())


def test_ask_mode_decline_blocks_loading(tmp_path, monkeypatch) -> None:
    store_dir = tmp_path / "cfg"
    monkeypatch.setenv("PROVER_TRUST", "ask")
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(store_dir))

    async def scenario() -> None:
        app = ProverApp()
        async with app.run_test(size=(140, 40)) as pilot:
            trust = await _wait_for_modal(pilot)
            trust.dismiss("decline-run")
            for _ in range(50):
                if app._trust_resolution is not None:
                    break
                await pilot.pause(0.05)
            assert app.problems == []
            assert not app._trust_resolution.trusted
            app.exit()

    asyncio.run(scenario())