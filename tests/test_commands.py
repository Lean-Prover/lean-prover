"""Slash command tests — ported from huggingface/tau tests/test_commands.py.

Same pattern as tau: hand-written FakeSession implementing the CommandSession
protocol, no monkeypatching; assert CommandResult flags directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.commands import CommandResult, create_default_command_registry

MAX_WORKERS = 16


class FakeSession:
    """Minimal CommandSession stand-in (tau's FakeSession analogue)."""

    def __init__(self, tmp_path: Path, sessions: list[str] | None = None) -> None:
        self._tmp_path = tmp_path
        self._sessions = sessions or []

    @property
    def model(self) -> str:
        return "Qwen/Qwen3.8-27B"

    @property
    def session_dir(self) -> Path:
        return self._tmp_path

    @property
    def session_ids(self) -> list[str]:
        return list(self._sessions)

    @property
    def problems_total(self) -> int:
        return 100

    @property
    def counts(self) -> dict[str, int]:
        return {"proved": 66, "failed": 34, "stopped": 0}

    @property
    def n_workers(self) -> int:
        return 4

    @property
    def max_workers(self) -> int:
        return MAX_WORKERS

    @property
    def is_running(self) -> bool:
        return False

    @property
    def thinking_level(self) -> str:
        return "off"

    @property
    def current_session_id(self) -> str | None:
        return self._sessions[0] if self._sessions else None


@pytest.fixture
def registry():
    return create_default_command_registry()


@pytest.fixture
def session(tmp_path):
    return FakeSession(tmp_path, sessions=["20260101-000000-sq_nonneg"])


def test_non_slash_input_is_not_handled(registry, session) -> None:
    assert registry.execute(session, "hello").handled is False
    assert registry.execute(session, "").handled is False


def test_unknown_command_is_not_handled_with_message(registry, session) -> None:
    result = registry.execute(session, "/nope")
    assert result.handled is False
    assert "Unknown command" in (result.message or "")


def test_quit_requests_exit(registry, session) -> None:
    assert registry.execute(session, "/quit").exit_requested is True
    assert registry.execute(session, "/exit").exit_requested is True  # alias
    assert registry.execute(session, "/q").exit_requested is True  # alias


def test_clear_requests_clear(registry, session) -> None:
    assert registry.execute(session, "/clear").clear_requested is True


def test_stop_when_idle_says_nothing_running(registry, session) -> None:
    result = registry.execute(session, "/stop")
    assert result.handled is True
    assert result.stop_requested is False
    assert "Nothing is running" in (result.message or "")


def test_stop_when_running_requests_stop(registry, tmp_path) -> None:
    class Running(FakeSession):
        @property
        def is_running(self) -> bool:
            return True

    result = registry.execute(Running(tmp_path), "/stop")
    assert result.stop_requested is True


def test_run_when_idle_requests_run(registry, session) -> None:
    assert registry.execute(session, "/run").run_requested is True


def test_prove_without_args_opens_editor(registry, session) -> None:
    result = registry.execute(session, "/prove")
    assert result.prove_requested is True
    assert result.prove_statement is None


def test_prove_with_inline_statement(registry, session) -> None:
    stmt = "theorem t : 1 + 1 = 2 := by norm_num"
    result = registry.execute(session, f"/prove {stmt}")
    assert result.prove_requested is True
    assert result.prove_statement == stmt


def test_prove_while_running_is_refused(registry) -> None:
    class Running(FakeSession):
        @property
        def is_running(self) -> bool:
            return True

    result = registry.execute(Running(Path(".")), "/prove x")
    assert result.prove_requested is False
    assert "Already running" in (result.message or "")


def test_workers_shows_current_without_args(registry, session) -> None:
    result = registry.execute(session, "/workers")
    assert result.workers_requested is None
    assert "4" in (result.message or "")


def test_workers_sets_value(registry, session) -> None:
    result = registry.execute(session, "/workers 8")
    assert result.workers_requested == 8


def test_workers_rejects_out_of_range(registry, session) -> None:
    for bad in ("0", "17", "-1"):
        assert registry.execute(session, f"/workers {bad}").workers_requested is None


def test_workers_rejects_garbage(registry, session) -> None:
    result = registry.execute(session, "/workers many")
    assert result.workers_requested is None
    assert "Usage" in (result.message or "")


def test_workers_alias_parallel_missing(registry, session) -> None:
    assert registry.get("parallel") is None


def test_resume_without_args_requests_picker(registry, session) -> None:
    assert registry.execute(session, "/resume").sessions_picker_requested is True


def test_resume_with_known_id_replays(registry, session) -> None:
    result = registry.execute(session, "/resume 20260101-000000-sq_nonneg")
    assert result.replay_session_id == "20260101-000000-sq_nonneg"


def test_resume_with_unknown_id_errors(registry, session) -> None:
    result = registry.execute(session, "/resume nope")
    assert result.replay_session_id is None
    assert "not found" in (result.message or "")


def test_export_requires_path(registry, session) -> None:
    assert registry.execute(session, "/export").export_requested is False
    result = registry.execute(session, "/export /tmp/log.txt")
    assert result.export_requested is True
    assert result.export_destination == Path("/tmp/log.txt")


def test_leaderboard_alias_board(registry, session) -> None:
    assert registry.execute(session, "/leaderboard").leaderboard_requested is True
    assert registry.execute(session, "/board").leaderboard_requested is True


def test_status_lists_key_state(registry, session) -> None:
    result = registry.execute(session, "/status")
    msg = result.message or ""
    assert "Qwen/Qwen3.8-27B" in msg
    assert "proved:  66" in msg
    assert "100" in msg


def test_model_shows_active_model(registry, session) -> None:
    assert "Qwen/Qwen3.8-27B" in (registry.execute(session, "/model").message or "")


def test_models_requests_profile_manager(registry, session) -> None:
    result = registry.execute(session, "/models")
    assert result.handled
    assert result.models_requested


def test_system_shows_proof_loop_system_prompt(registry, session) -> None:
    msg = registry.execute(session, "/system").message or ""
    assert "Lean 4 theorem prover" in msg


def test_help_lists_all_commands(registry, session) -> None:
    msg = registry.execute(session, "/help").message or ""
    for name in ("/quit", "/help", "/prove", "/run", "/stop", "/workers",
                 "/resume", "/branch", "/export", "/leaderboard", "/model",
                 "/models", "/system", "/hotkeys", "/status", "/clear", "/theme"):
        assert name in msg, name


def test_hotkeys_lists_bindings(registry, session) -> None:
    msg = registry.execute(session, "/hotkeys").message or ""
    assert "prove selected" in msg
    assert "quit" in msg


def test_registry_rejects_duplicates() -> None:
    from agent.commands import SlashCommand

    def handler(ctx) -> CommandResult:
        return CommandResult(handled=True)

    registry = create_default_command_registry()
    with pytest.raises(ValueError):
        registry.register(SlashCommand("quit", "", "", handler))
    with pytest.raises(ValueError):
        registry.register(SlashCommand("new", "", "", handler, aliases=("exit",)))


def test_branch_requires_session_id(registry, session) -> None:
    result = registry.execute(session, "/branch")
    assert result.branch_requested is False
    assert "Usage" in (result.message or "")


def test_branch_with_known_id_resumes(registry, session) -> None:
    result = registry.execute(session, "/branch 20260101-000000-sq_nonneg")
    assert result.branch_requested is True
    assert result.replay_session_id == "20260101-000000-sq_nonneg"
    assert result.branch_at is None


def test_branch_with_turn_truncates(registry, session) -> None:
    result = registry.execute(session, "/branch 20260101-000000-sq_nonneg 3")
    assert result.branch_requested is True
    assert result.branch_at == 3


def test_branch_rejects_unknown_session(registry, session) -> None:
    result = registry.execute(session, "/branch nope")
    assert result.branch_requested is False
    assert "not found" in (result.message or "")


def test_branch_rejects_bad_turn(registry, session) -> None:
    result = registry.execute(session, "/branch 20260101-000000-sq_nonneg xyz")
    assert result.branch_requested is False
    assert "Usage" in (result.message or "")


def test_theme_shows_current_and_available(registry, session) -> None:
    result = registry.execute(session, "/theme")
    msg = result.message or ""
    assert "prover-dark" in msg
    assert result.theme is None


def test_theme_sets_theme(registry, session) -> None:
    result = registry.execute(session, "/theme prover-light")
    assert result.theme == "prover-light"


def test_theme_rejects_unknown(registry, session) -> None:
    result = registry.execute(session, "/theme nope")
    assert result.theme is None
    assert "Unknown theme" in (result.message or "")


def test_thinking_shows_current_level(registry, session) -> None:
    result = registry.execute(session, "/thinking")
    msg = result.message or ""
    assert "Thinking: off" in msg
    assert "minimal" in msg and "xhigh" in msg


def test_thinking_sets_valid_level(registry, session) -> None:
    result = registry.execute(session, "/thinking high")
    assert result.thinking_level == "high"
    assert "set to high" in (result.message or "")


def test_thinking_alias_think(registry, session) -> None:
    assert registry.execute(session, "/think low").thinking_level == "low"


def test_thinking_rejects_unknown_level(registry, session) -> None:
    result = registry.execute(session, "/thinking turbo")
    assert result.thinking_level is None
    assert "Unknown thinking mode" in (result.message or "")


def test_status_shows_thinking_level(registry, session) -> None:
    assert "thinking off" in (registry.execute(session, "/status").message or "")


def test_registry_command_count(registry) -> None:
    """35 built-ins (26 prover incl. settings + 9 Tau-parity: tree, fork,
    login, logout, skills, contexts, tools, stats, update)."""
    assert len(registry.list_commands()) == 35


def test_command_result_defaults(registry, session) -> None:
    r = CommandResult(handled=True)
    assert r.exit_requested is False
    assert r.message is None
    assert r.workers_requested is None


def test_new_requests_fresh_session(registry, session) -> None:
    result = registry.execute(session, "/new")
    assert result.new_session_requested is True


def test_compact_requests_summary(registry, session) -> None:
    result = registry.execute(session, "/compact focus on divisibility")
    assert result.compact_summary == "focus on divisibility"


def test_compact_without_args(registry, session) -> None:
    result = registry.execute(session, "/compact")
    assert result.compact_summary is None
    assert result.handled is True


def test_name_shows_current_title(registry, session) -> None:
    result = registry.execute(session, "/name")
    assert "Current session" in (result.message or "")
    assert result.rename_requested is False


def test_name_without_session(registry) -> None:
    no_sessions = FakeSession(Path("."), sessions=[])
    result = registry.execute(no_sessions, "/name")
    assert "No recorded session" in (result.message or "")


def test_name_requests_rename(registry, session) -> None:
    result = registry.execute(session, "/name my-clean-proof")
    assert result.rename_requested is True
    assert result.rename_session_id == "20260101-000000-sq_nonneg"
    assert result.rename_title == "my-clean-proof"


def test_name_rejects_multiline(registry, session) -> None:
    result = registry.execute(session, "/name bad\nname")
    assert result.rename_requested is False
    assert "single line" in (result.message or "")


def test_new_refuses_while_running(registry, session) -> None:
    class _Run(FakeSession):
        @property
        def is_running(self) -> bool:
            return True

    result = registry.execute(_Run(session._tmp_path, session._sessions), "/new")
    assert result.new_session_requested is False
    assert "stop" in (result.message or "")


# --------------------------------------------------------------------------- /permissions


def test_permissions_no_args_shows_usage(registry, session) -> None:
    result = registry.execute(session, "/permissions")
    assert result.permissions_requested is True
    assert result.permissions_action is None
    assert "Usage" in (result.message or "")


def test_permissions_list(registry, session) -> None:
    result = registry.execute(session, "/permissions list")
    assert result.permissions_requested is True
    assert result.permissions_action == "list"


def test_permissions_mode_valid(registry, session) -> None:
    result = registry.execute(session, "/permissions mode yolo")
    assert result.permissions_requested is True
    assert result.permissions_action == "mode"
    assert result.permissions_mode == "yolo"


def test_permissions_mode_invalid(registry, session) -> None:
    result = registry.execute(session, "/permissions mode turbo")
    assert result.permissions_action is None
    assert "Invalid mode" in (result.message or "")


def test_permissions_revoke(registry, session) -> None:
    result = registry.execute(session, "/permissions revoke abc123")
    assert result.permissions_action == "revoke"
    assert result.permissions_tool == "abc123"


def test_permissions_revoke_needs_id(registry, session) -> None:
    result = registry.execute(session, "/permissions revoke")
    assert result.permissions_action is None


def test_permissions_remember_allow(registry, session) -> None:
    result = registry.execute(session, "/permissions remember allow prove_theorem")
    assert result.permissions_action == "remember"
    assert result.permissions_tool == "prove_theorem"
    assert result.permissions_allow is True
    assert result.permissions_pattern == ""


def test_permissions_remember_deny_with_pattern(registry, session) -> None:
    result = registry.execute(session, "/permissions remember deny prove_theorem hard")
    assert result.permissions_action == "remember"
    assert result.permissions_tool == "prove_theorem"
    assert result.permissions_allow is False
    assert result.permissions_pattern == "hard"


def test_permissions_remember_bad_verb(registry, session) -> None:
    result = registry.execute(session, "/permissions remember maybe prove_theorem")
    assert result.permissions_action is None
    assert "Usage" in (result.message or "")


def test_permissions_alias_perm(registry, session) -> None:
    result = registry.execute(session, "/perm list")
    assert result.permissions_requested is True
    assert result.permissions_action == "list"
