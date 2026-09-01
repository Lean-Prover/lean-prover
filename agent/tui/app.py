"""The Textual dashboard app (extracted from the former ``agent/tui.py``).

``ProverApp`` is the interactive proof-agent dashboard: problem browser, live
proof watch, slash-command bar, model/theme/session managers. Proof-run and
event-rendering logic lives in ``agent.tui.handlers.ProverRunMixin`` (mixed in
here); the modal screens live in ``agent.tui.screens``; styling is in
``agent/tui/app.tcss`` loaded via ``CSS_PATH``.
"""

from __future__ import annotations

import os
import re
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import Reactive
from textual.widgets import (
    Footer,
    Header,
    Input,
    ListView,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)

from agent import session as sess
from agent.autocomplete import command_completions
from agent.commands import CommandResult, create_default_command_registry
from agent.models import ModelProfile
from agent.session_manager import SessionManager
from agent.terminal_notification import TerminalNotificationController
from agent.terminal_title import TerminalTitleController
from agent.themes import (
    TuiTheme,
    available_tui_theme_names,
    get_tui_theme,
    textual_theme_variables,
    theme_css_variables,
)
from agent.tui import MAX_WORKERS, load_problems
from agent.tui.handlers import ProverRunMixin
from agent.tui.screens import (
    LeaderboardScreen,
    MessageScreen,
    ModelsScreen,
    PromptsScreen,
    ProveScreen,
    ReplayScreen,
    SessionsScreen,
    TrustScreen,
    WorkersScreen,
)
from agent.tui_render import PaneDivider, ProblemRow, PromptInput, SelectableRichLog
from agent.tui_settings import (  # noqa: F401 — re-exported for agent.tui callers
    TuiKeybindings,
    TuiSettings,
    load_tui_settings,
    resolved_theme,
    save_tui_settings,
    tui_settings_path,
)


class ProverApp(ProverRunMixin, App):
    """Interactive proof agent dashboard."""

    TITLE = "lean-prover — Lean 4 proof agent"
    CSS_PATH: ClassVar[str] = "app.tcss"
    # Worker count for parallel proof runs (Textual Reactive: assignment
    # centralizes the value; callers still refresh the status bar explicitly).
    n_workers: Reactive[int] = 1
    # Mirrors tau: the session modal always allows selecting (its text is the
    # copy target); the main screen follows ALLOW_SELECT which is toggled off
    # while a proof run mutates the transcript.
    # Non-priority single-letter bindings so they don't hijack typing in the
    # prompt bar: they still fire when the problem list is focused because
    # ListView does not consume plain letters, but an Input widget does.
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("p", "prove_selected", "Prove"),
        Binding("c", "custom_prove", "Custom"),
        Binding("r", "run_remaining", "Run rest"),
        Binding("w", "set_workers", "Workers"),
        Binding("s", "stop", "Stop"),
        Binding("v", "sessions", "Sessions"),
        Binding("l", "leaderboard", "Leaderboard"),
        Binding("m", "open_models", "Models", show=False),
        Binding("ctrl+o", "open_models", "Models", show=False),
        Binding("ctrl+space", "complete_prompt", show=False),
        Binding("ctrl+k", "open_command_palette", "Commands", show=False),
        Binding("ctrl+e", "edit_queued_message", "Edit queued", show=False),
        Binding("q", "quit_app", "Quit"),
        Binding("j", "vim_down", "Down", show=False),
        Binding("k", "vim_up", "Up", show=False),
        Binding("space", "vim_queue", "Queue", show=False),
        Binding("0", "vim_home", "Home", show=False),
        Binding("G", "vim_end", "End", show=False),
        Binding("/", "focus_search", "Search", show=False),
        Binding("escape", "blur_search", "Blur", show=False),
    ]

    def __init__(self, parallel: int = 1, tui_settings: TuiSettings | None = None) -> None:
        super().__init__()
        self.problems: list[dict] = []
        self._filtered_problems: list[dict] = []
        self._search_text: str = ""
        self._total_problems_in_file: int = 0
        self.n_workers = min(max(parallel, 1), MAX_WORKERS)
        self.tui_settings = tui_settings or load_tui_settings()
        self._supports_pyperclip: bool | None = None
        self._terminal_title = TerminalTitleController()
        # Reset the tab title even if the process dies abnormally (the Textual
        # on_unmount restore only runs on graceful shutdown).
        import atexit

        atexit.register(self._terminal_title.restore)
        self._active_session_id: str | None = None
        self._custom_seq = 0
        self._queued_prompts: list[str] = []
        self.counts = {"proved": 0, "failed": 0, "stopped": 0}
        self.command_registry = create_default_command_registry()
        self._live_tokens: int = 0
        self._live_cost: float = 0.0
        self._run_active: bool = False
        self._compact_warning_shown: set[str] = set()  # track per-session
        mode = os.environ.get("PROVER_NOTIFICATION", self.tui_settings.notification)
        self._terminal_notification = TerminalNotificationController(mode)

    def get_theme_variables(self) -> dict[str, str]:
        """CSS variables from the active theme (tau's get_theme_variable_defaults)."""
        return theme_css_variables(self._active_theme())

    @property
    def resolved_theme(self) -> TuiTheme:
        """Active theme with tau-dark fallback (tau's TuiSettings.resolved_theme)."""
        try:
            return get_tui_theme(self.tui_settings.theme)
        except KeyError:
            return get_tui_theme("prover-dark")

    def _active_theme(self) -> TuiTheme:
        return self.resolved_theme

    def _register_prover_themes(self) -> None:
        """Register every available theme with Textual's theme system."""
        from textual.theme import Theme

        for name in available_tui_theme_names():
            palette = get_tui_theme(name)
            css_vars = textual_theme_variables(palette)
            self.register_theme(Theme(
                name=name,
                primary=palette.accent,
                secondary=palette.border,
                accent=palette.accent,
                warning=palette.warn,
                error=palette.error,
                success=palette.success,
                foreground=palette.screen_text,
                background=palette.screen_background,
                surface=palette.chrome_background,
                panel=palette.sidebar_background,
                dark=palette.dark,
                variables=css_vars,
            ))
        dark = self.tui_settings.theme == "prover-dark" or self.resolved_theme.dark
        self.dark = dark

    # ------------------------------------------------------ CommandSession protocol

    @property
    def model(self) -> str:
        from agent import llm

        return llm.model()

    @property
    def session_dir(self) -> Path:
        return sess.sessions_dir()

    @property
    def session_ids(self) -> list[str]:
        return [sp.stem for sp in sess.list_sessions()]

    @property
    def current_session_id(self) -> str | None:
        """The most recently updated recorded session (tau's active session),
        or the explicitly live-switched one when set."""
        if self._active_session_id is not None:
            if any(sp.stem == self._active_session_id for sp in sess.list_sessions()):
                return self._active_session_id
            self._active_session_id = None  # switched session was deleted
        sessions = sess.list_sessions()
        return sessions[0].stem if sessions else None

    @property
    def session_manager(self) -> SessionManager:
        return SessionManager()

    @property
    def problems_total(self) -> int:
        return len(self.problems)

    @property
    def is_running(self) -> bool:
        return self._run_active

    @property
    def max_workers(self) -> int:
        return MAX_WORKERS

    @property
    def thinking_level(self) -> str:
        from agent import thinking as thinking_mod

        return self.tui_settings.thinking_level or thinking_mod.thinking_level_from_env()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="problems"):
                yield Input(
                    placeholder="search problems (type + Enter to load)...",
                    id="search-bar",
                )
                problems_list = ListView(id="problems-list")
                yield problems_list
                yield Static("", id="problems-hint")
            yield PaneDivider(id="pane-divider")
            with Vertical(id="side"), TabbedContent(initial="tab-log"):
                with TabPane("Log", id="tab-log"):
                    yield SelectableRichLog(id="log", wrap=False, markup=True)
                with TabPane("Goals", id="tab-goals"):
                    yield SelectableRichLog(id="goals", wrap=False, markup=True)
                with TabPane("Errors", id="tab-errors"):
                    yield SelectableRichLog(id="errors", wrap=False, markup=True)
                with TabPane("Proof", id="tab-proof"):
                    yield SelectableRichLog(id="proof", wrap=False, markup=True)
        with Vertical(id="prompt-box"):
            yield Static("", id="prompt-completions")
            yield PromptInput(placeholder="type a slash command (/help) + Enter", id="prompt")
        yield Static(self._status_text(), id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self._register_prover_themes()
        with suppress(Exception):
            self.theme = self.tui_settings.theme
        self.query_one("#problems").styles.width = self.tui_settings.problem_pane_width
        self._apply_thinking_level()
        self._sync_terminal_title()
        mode = os.environ.get("PROVER_TRUST", "always")
        self._trust_summary = None
        self._trust_resolution = None
        self.run_worker(self._mount_trust_flow(mode), name="project-trust")

    # -------------------------------------------------------------- search

    def action_focus_search(self) -> None:
        """Focus the search bar (/, or automatic on mount)."""
        self.query_one("#search-bar", Input).focus()

    def action_blur_search(self) -> None:
        """Blur the search bar (escape)."""
        self.query_one("#prompt", PromptInput).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search-bar":
            await self._filter_problems(event.value)
            return
        if event.input.id != "prompt":
            return
        text = event.value.strip()
        event.input.clear()
        self._hide_completions()
        if not text:
            return
        if text.startswith("/"):
            try:
                result = self.command_registry.execute(self, text)
                if not result.handled and result.message:
                    self.notify(result.message, severity="warning")
                if result.handled:
                    self._apply_command(result)
                    return
            except Exception:  # noqa: BLE001 — command isolation boundary
                # A command handler that raises (file/session I/O, etc.) must
                # never tear down the Textual app — surface it and continue.
                import traceback

                self._log(f"[red]command error:[/red] {traceback.format_exc()}")
                self.notify("Command failed — see the log for details",
                            severity="error")
                return
            # Unknown slash command: fall back to prompt-template expansion
            # (tau parity; built-in commands keep precedence over templates).
            from agent.prompt_templates import (
                expand_prompt_template_command,
                load_prompt_templates,
            )

            expanded = expand_prompt_template_command(text, load_prompt_templates())
            if expanded is not None:
                text = expanded.strip()
            else:
                self.notify(f"Unknown command: {text.split()[0]}", severity="warning")
                return
        # Plain text: queue as a follow-up proof while a run is active
        # (tau's queue_follow_up), else treat as an inline theorem.
        if self._run_active:
            self._queued_prompts.append(text)
            self._log(f"[dim]queued:[/dim] {text[:60]}")
            self.notify(
                f"Queued ({len(self._queued_prompts)} pending). "
                "ctrl+e to edit the last one back.",
                severity="information",
            )
            return
        self.notify("Treat as `/prove`? Press p/c or use /prove <statement>.",
                    severity="information")

    async def _filter_problems(self, query: str) -> None:
        """Filter problems by query and render the list."""
        self._search_text = query.strip().lower()
        if not self._search_text:
            self._filtered_problems = []
            await self._render_problem_list([])
            return
        q = self._search_text
        results = [
            p for p in self.problems
            if q in p.get("id", "").lower()
            or q in p.get("statement", "").lower()
            or q in p.get("difficulty", "").lower()
        ]
        self._filtered_problems = results
        await self._render_problem_list(results)
        self._log(f"[dim]found {len(results)} problems matching '{query}'[/dim]")

    async def _render_problem_list(self, problems: list[dict]) -> None:
        """Render the problem list ListView."""
        list_view = self.query_one("#problems-list", ListView)
        # ListView.clear() removes ListItem children asynchronously; await it
        # so re-filters never leave stale rows behind the new ones. Hints live
        # in a separate Static so the list stays a pure List<ProblemRow> and
        # ListView indices map 1:1 onto the filtered pool.
        await list_view.clear()
        hint = self.query_one("#problems-hint", Static)
        if not problems:
            hint.update("[dim]type /bench to load benchmark, "
                        "or use /import-standard to import a dataset[/dim]")
            hint.display = True
            return
        if len(problems) > 100:  # cap at 100 for performance
            hint.update(f"[dim]... and {len(problems) - 100} more[/dim]")
            hint.display = True
        else:
            hint.display = False
        for p in problems[:100]:
            list_view.append(ProblemRow(p))
        self._refresh_status()

    # --------------------------------------------------------------- vim nav

    def action_vim_down(self) -> None:
        """Move cursor down in problem list (j)."""
        list_view = self.query_one("#problems-list", ListView)
        idx = list_view.index
        if idx is not None and idx < len(list_view.children) - 1:
            list_view.scroll_to_item(idx + 1)

    def action_vim_up(self) -> None:
        """Move cursor up in problem list (k)."""
        list_view = self.query_one("#problems-list", ListView)
        idx = list_view.index
        if idx is not None and idx > 0:
            list_view.scroll_to_item(idx - 1)

    def action_vim_home(self) -> None:
        """Jump to top of problem list (0)."""
        list_view = self.query_one("#problems-list", ListView)
        list_view.scroll_to_item(0)

    def action_vim_end(self) -> None:
        """Jump to bottom of problem list (G)."""
        list_view = self.query_one("#problems-list", ListView)
        list_view.scroll_to_item(len(list_view.children) - 1)

    def action_vim_queue(self) -> None:
        """Queue the selected problem for proof (space)."""
        list_view = self.query_one("#problems-list", ListView)
        idx = list_view.index
        if idx is None or idx >= len(self._filtered_problems):
            return
        problem = self._filtered_problems[idx]
        self._log(f"[dim]queued: {problem['id']}[/dim]")
        if not self._run_active:
            self._start_run([problem])
        else:
            self._queued_prompts.append(problem["statement"])
        self.notify(f"Queued: {problem['id']}")

    def action_open_models(self) -> None:
        """Open model profiles manager (m or Ctrl+O)."""
        self.action_models()

    def _apply_thinking_level(self) -> None:
        """Push the persisted/configured thinking level into agent.thinking."""
        from agent import thinking as thinking_mod

        level = self.tui_settings.thinking_level or thinking_mod.thinking_level_from_env()
        thinking_mod.set_thinking_level(level)

    async def _mount_trust_flow(self, mode: str) -> None:
        """Resolve project-input trust for the repo (tau parity; the TUI
        defaults to always so headless/tests are unaffected, and
        PROVER_TRUST=ask enables the interactive modal)."""
        self._register_prover_themes()
        with suppress(Exception):
            self.theme = self.tui_settings.theme
        self._trust_summary, self._trust_resolution = (
            await self._resolve_project_trust(mode)
        )
        if not self._trust_resolution or not self._trust_resolution.trusted:
            self.problems = []
            self._filtered_problems = []
            self._log("[red]project inputs untrusted — problems not loaded; "
                      "set PROVER_TRUST=always to allow[/red]")
            self._emit_reload_summary(problems=0)
            return
        # Lazy load: don't populate the list yet; show search box instead
        try:
            self.problems = load_problems()
        except ValueError as exc:
            self.problems = []
            self._filtered_problems = []
            self._log(f"[red]{exc}[/red]")
            self._emit_reload_summary(problems=0)
            return
        self._total_problems_in_file = len(self.problems)
        self._filtered_problems = []
        if not self.problems:
            self._log("[dim]benchmark/problems.json not found — type a search to find problems[/dim]")
            self._emit_reload_summary(problems=0)
            return
        # Focus search bar for immediate typing
        search_input = self.query_one("#search-bar", Input)
        search_input.focus()
        model = self.model
        self._log(f"model: [cyan]{model}[/cyan] — "
                  f"[dim]{len(self.problems)} problems available. "
                  f"Type to search, Enter to load.[/dim]")
        self._log("[dim]j/k navigate · space queue · p prove · /models config[/dim]")
        self._emit_reload_summary(problems=0)

    def _emit_reload_summary(self, problems: int) -> None:
        """Log a Pi-style before/after resource reload summary (tau parity)."""
        from agent.reload import (
            ReloadSnapshot,
            build_reload_summary,
            take_reload_snapshot,
        )

        before = getattr(self, "_reload_before", None)
        if not isinstance(before, ReloadSnapshot):
            return
        after = take_reload_snapshot()
        after = replace(after, problems=problems)
        summary = build_reload_summary(before, after)
        self._log(f"reload: {summary.render()}")

    async def _resolve_project_trust(self, mode: str):
        """Resolve project-input trust for the repo (tau parity; the TUI
        defaults to always so headless/tests are unaffected, and
        PROVER_TRUST=ask enables the interactive modal)."""
        from agent.project_trust import (
            ProjectTrustCoordinator,
            ProjectTrustStore,
            format_trust_diagnostic,
        )

        async def ask(request):
            return await self.push_screen_wait(
                TrustScreen(request)
            ) if self._trust_user_visible() else None

        if mode == "ask":
            default = "ask"
            interactive = self._trust_user_visible()
        else:
            default = "always" if mode == "always" else "never"
            interactive = False
        store = ProjectTrustStore()
        try:
            import agent.tui as _tui

            summary, resolution = await ProjectTrustCoordinator(store).resolve(
                _tui.REPO, default=default, interactive=interactive, prompt=ask
            )
        except Exception as exc:  # noqa: BLE001 - trust must never crash the TUI
            self._log(f"[red]project trust failed closed: {exc}[/red]")
            summary, resolution = None, None
        if summary is not None and resolution is not None:
            self._log(f"[dim]{format_trust_diagnostic(summary, resolution)}[/dim]")
        return summary, resolution

    def _trust_user_visible(self) -> bool:
        """Whether an interactive trust prompt can reach the user."""
        try:
            return bool(os.environ.get("TERM") or os.environ.get("PROVER_TRUST") == "ask")
        except Exception:  # noqa: BLE001
            return False

    def _status_text(self) -> str:
        done = self.counts["proved"] + self.counts["failed"] + self.counts["stopped"]
        remaining = max(0, len(self._filtered_problems) - done)
        state = "RUNNING" if self._run_active else "idle"
        cost_str = f" cost≈${self._live_cost:.4f}" if self._live_cost > 0 else ""
        tokens_str = f" tokens={self._live_tokens}" if self._live_tokens > 0 else ""
        return (f"{state:<7} workers={self.n_workers} · thinking={self.thinking_level} · "
                f"proved {self.counts['proved']} · "
                f"failed {self.counts['failed']} · stopped {self.counts['stopped']} · "
                f"remaining {remaining}{tokens_str}{cost_str}")

    def _refresh_status(self) -> None:
        self.query_one("#status-bar", Static).update(self._status_text())

    def _log(self, msg: str) -> None:
        self.query_one("#log", RichLog).write(msg)

    # ------------------------------------------------------------- slash commands

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "prompt":
            return
        self._refresh_completions(event.input.value)

    def _refresh_completions(self, text: str) -> None:
        widget = self.query_one("#prompt-completions", Static)
        items = command_completions(self.command_registry, text)
        if not items:
            self._hide_completions()
            return
        widget.update(
            "\n".join(f"  {cmd:<14} {desc}" for cmd, desc in items)
        )
        widget.styles.display = "block"

    def _hide_completions(self) -> None:
        widget = self.query_one("#prompt-completions", Static)
        widget.styles.display = "none"

    def action_complete_prompt(self) -> None:
        """Tab: complete the current slash-command prefix."""
        prompt = self.query_one("#prompt", Input)
        if prompt is not self.focused:
            return
        items = command_completions(self.command_registry, prompt.value)
        if items:
            prompt.value = items[0][0] + " "
            prompt.cursor_position = len(prompt.value)
            self._hide_completions()

    def action_open_command_palette(self) -> None:
        """Ctrl+k: open the slash-command palette in the prompt (tau parity)."""
        prompt = self.query_one("#prompt", Input)
        prompt.focus()
        prompt.value = "/"
        prompt.cursor_position = len(prompt.value)
        self._refresh_completions(prompt.value)

    def action_edit_queued_message(self) -> bool:
        """Ctrl+e: move the latest queued prompt back into the input (tau parity)."""
        if not self._queued_prompts:
            return False
        prompt = self.query_one("#prompt", Input)
        if prompt.value.strip():
            return False
        prompt.value = self._queued_prompts.pop()
        prompt.cursor_position = len(prompt.value)
        self._log("[dim]edited queued prompt back into input[/dim]")
        return True

    def _new_session(self) -> None:
        """Start fresh: reset statuses, counts and panels (tau's _new_session)."""
        self.counts = {"proved": 0, "failed": 0, "stopped": 0}
        self._live_tokens = 0
        self._live_cost = 0.0
        self._compact_warning_shown = set()
        self._custom_seq = 0
        self._queued_prompts = []
        self._active_session_id = None
        for row in self.query(ProblemRow):
            row.set_status("pending")
        for wid in ("#log", "#goals", "#errors", "#proof"):
            widget = self.query_one(wid)
            if hasattr(widget, "clear"):
                widget.clear()
        self._refresh_status()
        self._sync_terminal_title()
        self._log("[bold]── new session[/bold] — statuses reset")

    def _request_compaction(self, instructions: str | None) -> None:
        """Manual compaction request. The loop compacts deterministically on its
        own, so from the dashboard this is informational (tau's compact flag)."""
        if self._run_active:
            self._log("[dim]compaction: waiting for the active run "
                      "(auto-compaction is already active inside the loop)[/dim]")
            return
        msg = ("History compaction is automatic inside prove(): after 18 turns, "
               "old attempts fold into a failed-attempts summary.")
        if instructions:
            msg = f"{msg}\n(per-run instructions: {instructions})"
        self._show_command_message(msg)

    def _rename_session(self, session_id: str, title: str) -> None:
        """Apply a /name rename to a recorded session (tau's touch_session)."""
        record = SessionManager().rename(session_id, title)
        if record is None:
            self.notify(f"Session not found: {session_id}", severity="error")
        else:
            self._log(f"session [cyan]{session_id}[/cyan] renamed to "
                      f"[bold]{title}[/bold]")

    def _apply_command(self, result: CommandResult) -> None:
        """Apply a CommandResult's flags (tau's TUI dispatch order)."""
        if result.message:
            if "\n" in result.message:
                self._show_command_message(result.message)
            else:
                self.notify(result.message)
        if result.clear_requested:
            self.query_one("#log", RichLog).clear()
        if result.stop_requested:
            self.action_stop()
        if result.run_requested:
            self.action_run_remaining()
        if result.prove_requested:
            if result.prove_statement:
                self._prove_statement(result.prove_statement)
            else:
                self.action_custom_prove()
        if result.workers_requested is not None:
            self.n_workers = result.workers_requested
            self._refresh_status()
        if result.sessions_picker_requested:
            self.action_sessions()
        if result.branch_requested and result.replay_session_id:
            self._branch_run(result.replay_session_id, result.branch_at)
        elif result.replay_session_id:
            self._replay_by_id(result.replay_session_id)
        if result.leaderboard_requested:
            self.action_leaderboard()
        if result.prompts_requested:
            self.action_prompts()
        if result.reload_requested:
            self._reload_resources()
        if result.theme:
            self._set_theme(result.theme)
        if result.thinking_level:
            self._set_thinking(result.thinking_level)
        if result.export_requested and result.export_destination:
            self._export_log(result.export_destination)
        if result.usage_requested:
            if result.message:
                self._log(f"[dim]{result.message}[/dim]")
            self._show_usage(result.usage_session_id)
        if result.new_session_requested:
            self._new_session()
        if result.compact_summary is not None:
            self._request_compaction(result.compact_summary)
        if result.rename_requested and result.rename_session_id:
            self._rename_session(result.rename_session_id, result.rename_title or "")
        if result.models_requested:
            self.action_models()
        if result.tree_picker_requested:
            self._show_session_tree()
        if result.fork_requested:
            self._fork_session(result.fork_path)
        if result.set_model_choice:
            self._switch_model(result.set_model_choice, result.set_model_provider)
        if result.switch_session_id:
            self._switch_session(result.switch_session_id)
        if result.login_requested and result.login_provider:
            self._login_provider(result.login_provider)
        if result.logout_requested and result.login_provider:
            self._logout_provider(result.login_provider)
        if result.skills_requested:
            self._show_skills()
        if result.contexts_requested:
            self._show_contexts()
        if result.tools_requested:
            self._show_tools()
        if result.update_check_requested:
            self._check_update()
        if result.permissions_requested:
            self._handle_permissions(result)
        if result.exit_requested:
            self.exit()

    # --------------------------------------------------------- tau commands

    def _show_session_tree(self) -> None:
        """/tree — render the current session's event records as a flat tree."""
        current = self.current_session_id
        if current is None:
            self.notify("No recorded session yet — prove something first.", severity="warning")
            return
        records = sess.read_session(self.session_dir / f"{current}.jsonl")
        lines = [f"session [cyan]{current}[/cyan] ({len(records)} records)", ""]
        for rec in records:
            ev = rec.get("event", "?")
            step = rec.get("step", "")
            extra = ""
            if ev == "result":
                extra = " PROVED" if rec.get("proved") else " failed"
            lines.append(f"  {len(lines) - 1}. {ev}{f' step={step}' if step != '' else ''}{extra}")
        self.push_screen(MessageScreen("session tree", "\n".join(lines[:200])))

    def _fork_session(self, fork_path: str | None) -> None:
        """/fork — re-run the current session from an earlier step
        (prover has no entry tree; fork maps onto /branch <id> <turn>)."""
        current = self.current_session_id
        if current is None:
            self.notify("No recorded session to fork.", severity="warning")
            return
        turn = None
        if fork_path is not None:
            try:
                turn = max(0, int(fork_path))
            except ValueError:
                self.notify("Usage: /fork [step-number]", severity="warning")
                return
        self._branch_run(current, turn)

    def _switch_model(self, name: str, provider: str | None = None) -> None:
        """/model <name>[@provider] — activate a model profile, optionally bound
        to a named provider's endpoint."""
        from agent import models as models_mod
        from agent.provider_config import provider_config_from_name

        if provider:
            pc = provider_config_from_name(provider)
            if pc is None:
                self.notify(f"No provider named {provider}.", severity="warning")
                return
            profiles = models_mod.load_profiles()
            existing = next((p for p in profiles if p.name == name), None)
            profile = ModelProfile(
                name=name,
                label=existing.label if existing else "",
                base_url=pc.base_url or "",
                api_key=pc.api_key or "",
                context_window=(existing.context_window if existing
                                else pc.max_context_window),
                cost_in=existing.cost_in if existing else None,
                cost_out=existing.cost_out if existing else None,
            )
            profiles = [p for p in profiles if p.name != name] + [profile]
            models_mod.save_store(active=name, profiles=profiles)
            self._log(f"model set to [bold]{name}[/bold] via [cyan]{provider}[/cyan]")
        else:
            profile = models_mod.profile_for(name)
            if profile is None:
                self.notify(f"No profile named {name} — use /models to add one.",
                            severity="warning")
                return
            models_mod.save_store(active=name)
            self._log(f"model set to [bold]{name}[/bold]")
        self._refresh_status()

    def _switch_session(self, session_id: str) -> None:
        """Make a recorded session the active/current one (live session switch)."""
        if not any(sp.stem == session_id for sp in sess.list_sessions()):
            self.notify(f"No recorded session named {session_id}.", severity="warning")
            return
        self._active_session_id = session_id
        self._refresh_status()
        self._log(f"active session switched to [bold]{session_id}[/bold]")
        self.notify(f"Active session: {session_id}")

    def _login_provider(self, provider: str) -> None:
        """/login — run the provider's OAuth flow in a background worker."""
        self._log(f"starting [bold]{provider}[/bold] sign-in…")
        self._login_worker(provider)

    @work(thread=True)
    def _login_worker(self, provider: str) -> None:
        import asyncio

        if provider == "openai-codex":
            from agent.oauth.codex import login_openai_codex

            credential = login_openai_codex()
        else:
            if provider == "anthropic":
                from agent.oauth.anthropic import login_anthropic as login_async
            else:
                from agent.oauth.github_copilot import (
                    login_github_copilot as login_async,
                )
            credential = asyncio.run(login_async())
        if credential is None:
            self.call_from_thread(self.notify, "sign-in did not complete", severity="error")
        else:
            self.call_from_thread(self.notify, f"signed in to {provider}")

    def _logout_provider(self, provider: str) -> None:
        """/logout — remove stored credentials for a provider."""
        from agent.credentials import FileCredentialStore

        if FileCredentialStore().delete(provider):
            self.notify(f"removed {provider} credentials")
        else:
            self.notify(f"no stored credentials for {provider}")

    def _show_skills(self) -> None:
        """/skills — list discovered skills."""
        from agent.skills import load_skills

        skills = load_skills()
        body = "\n".join(f"- {s.name}: {s.description}" for s in skills) or "(none discovered)"
        self.push_screen(MessageScreen("skills", body))

    def _show_contexts(self) -> None:
        """/contexts — list project context files."""
        from pathlib import Path as _P

        from agent.context import discover_project_context_with_diagnostics

        files, _ = discover_project_context_with_diagnostics(_P.cwd())
        body = "\n".join(f"- {f['path']} ({len(f['content'])} chars)" for f in files) or "(none found)"
        self.push_screen(MessageScreen("project contexts", body))

    def _show_tools(self) -> None:
        """/tools — list the tools bound to the prove loop."""
        from agent.prover_loop import LEAN_DIR
        from agent.tools import default_tools

        names = [t.name for t in default_tools(LEAN_DIR)]
        from agent.coding_tools import create_coding_tools

        names += [t["name"] for t in create_coding_tools()]
        self.push_screen(MessageScreen("tools", "\n".join(f"- {n}" for n in names)))

    def _check_update(self) -> None:
        """/update — check PyPI for a newer release (off the UI thread)."""
        self._update_worker()

    @work(thread=True, name="update-check")
    def _update_worker(self) -> None:
        from agent.update_check import check_for_updates

        info = check_for_updates()
        if info.error:
            self.call_from_thread(
                self.notify, f"update check failed: {info.error}",
                severity="warning")
        elif info.is_update_available:
            self.call_from_thread(
                self.notify,
                f"update available: {info.current_version} -> {info.latest_version}")
        else:
            self.call_from_thread(
                self.notify, f"lean-prover {info.current_version} is up to date")

    def _handle_permissions(self, result: CommandResult) -> None:
        """/permissions — list, remember, revoke or set baseline mode (fx ACL)."""
        from agent.permissions import VALID_MODES, PermissionStore

        store = PermissionStore()
        action = result.permissions_action

        if action == "list" or action is None:
            rules = store.rules()
            lines = [f"mode: {store.mode}  (" + ", ".join(VALID_MODES) + ")",
                     f"rules: {len(rules)}", ""]
            for rid, r in sorted(rules.items()):
                dec = "allow" if r.get("allow") else "deny"
                pat = f" when args contain {r['pattern']!r}" if r.get("pattern") else ""
                lines.append(f"  {rid}  {dec}  {r['tool']}{pat}")
            if not rules:
                lines.append("  (no explicit rules — tools open by default)")
            self.push_screen(MessageScreen("permissions", "\n".join(lines)))
            return

        if action == "remember":
            try:
                rid = store.add_rule(
                    result.permissions_tool or "", result.permissions_pattern or "",
                    bool(result.permissions_allow), note="via /permissions",
                )
                store.save()
            except ValueError as exc:
                self.notify(f"permissions: {exc}", severity="warning")
                return
            verb = "allow" if result.permissions_allow else "deny"
            self._log(f"permissions: {verb} {result.permissions_tool} → [{rid}]")
            return

        if action == "revoke":
            rid = result.permissions_tool or ""
            if store.remove_rule(rid):
                store.save()
                self._log(f"permissions: revoked rule [{rid}]")
            else:
                self.notify(f"permissions: no rule with id {rid!r}", severity="warning")
            return

        if action == "mode":
            mode = result.permissions_mode or ""
            if mode not in VALID_MODES:
                self.notify(f"permissions: invalid mode {mode!r}", severity="warning")
                return
            store.mode = mode
            store.save()
            self._log(f"permissions: baseline mode → {mode}")
            return

    def _show_command_message(self, body: str) -> None:
        """Show command output in a modal (tau's _show_command_message)."""
        title = "command output"
        self.push_screen(MessageScreen(title, body))

    def _set_theme(self, name: str) -> None:
        """Apply a theme by name (tau's /theme wiring)."""
        try:
            get_tui_theme(name)
        except KeyError:
            self.notify(f"Unknown theme: {name}", severity="error")
            return
        self.tui_settings = replace(self.tui_settings, theme=name)
        save_tui_settings(self.tui_settings)
        with suppress(Exception):
            self.theme = name
        self.dark = get_tui_theme(name).dark

    def _set_thinking(self, level: str) -> None:
        """Apply a thinking level (tau's /thinking wiring, persisted)."""
        from agent import thinking as thinking_mod

        try:
            thinking_mod.set_thinking_level(level)
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        if self.tui_settings.thinking_level != level:
            self.tui_settings = replace(self.tui_settings, thinking_level=level)
            save_tui_settings(self.tui_settings)
        self._refresh_status()

    def _show_usage(self, session_arg: str | None) -> None:
        from agent.session_manager import SessionManager
        from agent.session_usage import collect_session_usage, render_usage_dashboard

        model = self.model
        sm = SessionManager()
        records = {r.id: r for r in sm.list_sessions()}
        if session_arg is None or session_arg == "all":
            if not records:
                self.notify("No sessions recorded.", severity="error")
                return
            lines = ["", "Usage — all sessions", "=" * 48]
            for sid, rec in records.items():
                usage = collect_session_usage(sess.read_session(Path(rec.path)), model)
                lines.append("")
                lines.append(f"### {sid}")
                lines.append(render_usage_dashboard(usage))
            self._show_command_message("\n".join(lines))
        else:
            sid = session_arg
            rec = records.get(sid)
            if rec is None:
                self.notify(f"Session not found: {sid}", severity="error")
                return
            usage = collect_session_usage(sess.read_session(Path(rec.path)), model)
            self._show_command_message(render_usage_dashboard(usage))

    def _export_log(self, destination: Path) -> None:
        from agent.session_export import export_session

        session_id = self.current_session_id
        if session_id is not None:
            entries = sess.read_session(self.session_dir / f"{session_id}.jsonl")
            if entries:
                try:
                    export_session(
                        entries,
                        destination,
                        title=f"Prover session {session_id}",
                        source=str(self.session_dir / f"{session_id}.jsonl"),
                    )
                except OSError as exc:
                    self.notify(f"Export failed: {exc}", severity="error")
                    return
                self.notify(f"Session exported to {destination}")
                return
        from textual.selection import SELECT_ALL

        log = self.query_one("#log", SelectableRichLog)
        text = log.get_selection(SELECT_ALL)
        try:
            destination.write_text((text[0] if text else "") + "\n")
        except OSError as exc:
            self.notify(f"Export failed: {exc}", severity="error")
            return
        self.notify(f"Log exported to {destination}")

    # ---------------------------------------------------------------- actions

    def _selected_problem(self) -> dict | None:
        problems_list = self.query_one(ListView)
        idx = problems_list.index
        if idx is None:
            idx = 0  # default to first row when nothing is focused yet
        # The list renders the *filtered* pool; index into the same pool the
        # rows came from, otherwise an active search maps p onto the wrong
        # problem (or past the 100-row cap, onto an arbitrary one).
        pool = self._filtered_problems if self._search_text else self.problems
        if not (0 <= idx < len(pool)):
            return None
        return pool[idx]

    def action_prove_selected(self) -> None:
        if self._run_active:
            self.notify("Already running — press s to stop first.", severity="warning")
            return
        problem = self._selected_problem()
        if problem is None:
            return
        self._start_run([problem])

    def action_custom_prove(self) -> None:
        if self._run_active:
            self.notify("Already running — press s to stop first.", severity="warning")
            return

        def got(code: str | None) -> None:
            if not code:
                return
            m = re.search(r"\b(?:theorem|lemma|example)\s+(\w+)", code)
            base = m.group(1) if m else "custom"
            self._custom_seq += 1
            problem = {
                "id": f"{base}-{self._custom_seq}" if self._custom_seq > 1 else base,
                "statement": code,
                "difficulty": "custom",
            }
            self._log(f"\n[bold magenta]custom: {problem['id']}[/bold magenta]")
            self._start_run([problem])

        self.push_screen(ProveScreen(), got)

    def action_set_workers(self) -> None:
        def got(n: int | None) -> None:
            if n is None:
                return
            self.n_workers = n
            self._log(f"workers set to {n}")
            self._refresh_status()

        self.push_screen(WorkersScreen(), got)

    def action_run_remaining(self) -> None:
        if self._run_active:
            self.notify("Already running — press s to stop first.", severity="warning")
            return
        rows = self.query(ProblemRow)
        todo = [r.problem for r in rows if r.status not in ("proved",)]
        if not todo:
            self.notify("Nothing left to run.", severity="information")
            return
        self._start_run(todo)

    def action_stop(self) -> None:
        if not self._run_active:
            return
        self._stop_flag = True
        self.notify("Stopping after current step…", severity="warning")
        self._log("[yellow]stop requested[/yellow]")

    def action_sessions(self) -> None:
        def opened(path: Path | None) -> None:
            if path is not None:
                self.push_screen(ReplayScreen(path))

        self.push_screen(SessionsScreen(), opened)

    def action_leaderboard(self) -> None:
        self.push_screen(LeaderboardScreen())

    def action_models(self) -> None:
        """Open the model-profiles manager (agent/models.py store)."""
        self.push_screen(ModelsScreen())

    def action_prompts(self) -> None:
        """Pick and insert a prompt template into the bar (tau's /prompts)."""
        from agent.prompt_templates import load_prompt_templates

        templates = load_prompt_templates()
        if not templates:
            self.notify(
                "No prompt templates in ~/.prover/prompts or .prover/prompts.",
                severity="warning",
            )
            return
        self.push_screen(PromptsScreen(templates), callback=self._prompt_template_chosen)

    def _reload_resources(self) -> None:
        """Re-run discovery: trust, problems, themes (tau's /reload)."""
        from agent.reload import take_reload_snapshot

        mode = os.environ.get("PROVER_TRUST", "always")
        self._reload_before = take_reload_snapshot()
        self._reload_before = replace(self._reload_before, problems=len(self.problems))
        self.run_worker(self._mount_trust_flow(mode), name="project-trust-reload")

    def _prompt_template_chosen(self, name: str | None) -> None:
        if not name:
            return
        prompt = self.query_one("#prompt", PromptInput)
        prompt.value = f"/{name} "
        prompt.cursor_position = len(prompt.value)
        prompt.focus()

    def action_quit_app(self) -> None:
        self.exit()

    def on_unmount(self) -> None:
        self._terminal_title.restore()

    async def on_event(self, event: events.Event) -> None:
        """Reroute drops/pastes that target the prompt (tau file-drop parity).

        Textual dispatches every MRO handler for a message, so an
        `_on_paste` override on the input would double-insert (our spaced
        version plus the built-in verbatim one).  Instead intercept here,
        before the built-in forwarding runs, and hand the text to the
        prompt's `insert_pasted_text` when focus is the prompt or nothing.
        """
        if isinstance(event, events.Paste):
            prompt = self.query_one_optional("#prompt", PromptInput)
            if prompt is not None and (self.focused is None or self.focused is prompt):
                event.stop()
                prompt.insert_pasted_text(event.text)
                return
        await super().on_event(event)


def main(parallel: int = 1) -> None:
    ProverApp(parallel=parallel).run()


if __name__ == "__main__":
    main()
