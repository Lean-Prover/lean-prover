"""Textual TUI for prover — browse problems, watch live proof attempts,
prove custom theorems, run benchmarks in parallel, replay past sessions.

Run with: `prover tui [-p/--parallel N]`
"""

from __future__ import annotations

import functools
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

from rich.markup import escape
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.selection import Selection
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    ListItem,
    ListView,
    OptionList,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.widgets.option_list import Option

from . import session as sess
from .autocomplete import command_completions
from .commands import CommandResult, create_default_command_registry
from .file_drop import normalize_dropped_paths
from .loop import prove
from .models import ModelProfile
from .session_manager import SessionManager
from .session_stats import calculate_session_stats
from .terminal_notification import TerminalNotificationController
from .terminal_title import TerminalTitleController
from .themes import (
    TuiTheme,
    available_tui_theme_names,
    get_tui_theme,
    textual_theme_variables,
    theme_css_variables,
)

REPO = Path(__file__).resolve().parent.parent
PROBLEMS_FILE = REPO / "benchmark" / "problems.json"

STATUS_MARK = {"pending": "·", "running": "▶", "proved": "✔", "failed": "✘", "stopped": "◼"}
TIER_COLOR = {"trivial": "green", "easy": "cyan", "medium": "yellow",
              "hard": "red", "custom": "magenta"}

MAX_WORKERS = 16


@dataclass(frozen=True)
class TuiSettings:
    """TUI preferences (tau TuiSettings parity, subset used by prover), persisted to disk."""

    auto_copy_selection: bool = False
    theme: str = "prover-dark"
    notification: str = "auto"  # "auto" | "bell" | "off"
    thinking_level: str = ""    # "" = resolve from env (agent/thinking.py)
    problem_pane_width: int = 46
    large_paste_threshold: int = 2000
    terminal_bell: bool = True
    window_title_updates: bool = True

    def to_json(self) -> dict:
        return {
            "auto_copy_selection": self.auto_copy_selection,
            "theme": self.theme,
            "notification": self.notification,
            "thinking_level": self.thinking_level,
            "problem_pane_width": self.problem_pane_width,
            "large_paste_threshold": self.large_paste_threshold,
            "terminal_bell": self.terminal_bell,
            "window_title_updates": self.window_title_updates,
        }

    @classmethod
    def from_json(cls, data: dict) -> TuiSettings:
        base = cls()
        from .thinking import normalize_thinking_level

        thinking = str(data.get("thinking_level") or "")
        if thinking:
            try:
                thinking = normalize_thinking_level(thinking)
            except ValueError:
                thinking = ""
        notif = str(data.get("notification") or base.notification)
        if notif not in ("auto", "bell", "off"):
            notif = base.notification
        pane = int(data.get("problem_pane_width") or base.problem_pane_width)
        pane = min(max(pane, PaneDivider.MIN_WIDTH), PaneDivider.MAX_WIDTH)
        threshold = int(data.get("large_paste_threshold") or base.large_paste_threshold)
        threshold = max(200, threshold)
        return cls(
            auto_copy_selection=bool(data.get("auto_copy_selection", base.auto_copy_selection)),
            theme=str(data.get("theme") or base.theme),
            notification=notif,
            thinking_level=thinking,
            problem_pane_width=pane,
            large_paste_threshold=threshold,
            terminal_bell=bool(data.get("terminal_bell", base.terminal_bell)),
            window_title_updates=bool(data.get("window_title_updates", base.window_title_updates)),
        )


@dataclass(frozen=True)
class TuiKeybindings:
    """Configurable keybindings (tau TuiKeybindings parity)."""

    submit: str = "enter"
    interrupt: str = "escape"
    interrupt_force: str = "ctrl+c"
    toggle_sidebar: str = "ctrl+b"
    focus_input: str = "ctrl+g"


def resolved_theme(settings: TuiSettings) -> str:
    """The effective theme name (env override wins, tau resolved_theme parity)."""
    import os

    return os.environ.get("PROVER_THEME") or settings.theme


def tui_settings_path() -> Path:
    """Where TUI preferences persist (tau's tui_settings_path, ~/.prover/tui.json)."""
    from .paths import ProverPaths

    return ProverPaths().config_dir / "tui.json"


def load_tui_settings() -> TuiSettings:
    """Load persisted TUI preferences, falling back to defaults."""
    path = tui_settings_path()
    if not path.exists():
        return TuiSettings()
    try:
        return TuiSettings.from_json(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, TypeError):
        return TuiSettings()


def save_tui_settings(settings: TuiSettings) -> None:
    """Persist TUI preferences as JSON (tau's save_tui_settings)."""
    path = tui_settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings.to_json(), indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def load_problems() -> list[dict]:
    if not PROBLEMS_FILE.exists():
        return []
    try:
        return json.loads(PROBLEMS_FILE.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"benchmark/problems.json is unreadable: {exc}") from exc


def _put(log: RichLog, text: str, style: str = "") -> None:
    """Write a plain-text line (no markup interpretation of the content)."""
    log.write(Text(text, style=style)) if style else log.write(Text(text))


class SelectableRichLog(RichLog):
    """A RichLog whose plain text can be selected (ported from tau's
    TranscriptMessageWidget.get_selection). RichLog renders content to Strips
    and exposes no get_selection, so we track the plain-text lines ourselves
    and let Textual's Selection machinery extract from them.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._plain_lines: list[str] = []

    def write(self, content, *args, **kwargs):
        if isinstance(content, Text):
            plain = content.plain
        elif isinstance(content, str) and self.markup:
            plain = Text.from_markup(content).plain
        else:
            plain = escape(str(content)) if not isinstance(content, str) else content
        self._plain_lines.extend(plain.splitlines() or [""])
        return super().write(content, *args, **kwargs)

    def clear(self):
        self._plain_lines.clear()
        return super().clear()

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        text = "\n".join(self._plain_lines)
        selected = selection.extract(text)
        if not selected:
            return None
        return selected, "\n"


class PromptInput(Input):
    """Single-line prompt with (single) file-drop insertion (tau Parity:
    terminal drops arrive as a Paste; the app reroutes them here via
    `insert_pasted_text` so the normalized paths land with separating
    whitespace)."""

    def _insert_dropped_paths(self, inserted: str) -> None:
        """Insert dropped paths at the cursor, separated from surrounding text."""
        position = self.cursor_position
        before = self.value[:position]
        after = self.value[position:]
        if before and not before[-1].isspace():
            inserted = f" {inserted}"
        if not after or not after[0].isspace():
            inserted = f"{inserted} "
        self.insert_text_at_cursor(inserted)

    def insert_pasted_text(self, text: str) -> None:
        """Insert pasted text that Textual could not deliver to this widget.

        Used for drops that arrive while the terminal is unfocused, where no
        default paste handler runs (tau's insert_pasted_text parity).
        """
        dropped = normalize_dropped_paths(text)
        if dropped is not None:
            self._insert_dropped_paths(dropped)
            return
        self.insert_text_at_cursor(text)


def render_event(ev: dict, pid: str, log: RichLog, goals: RichLog,
                 errors: RichLog, proof: RichLog, tag: str = "",
                 skip_failed_hammers: bool = False) -> None:
    """Render one prove() event record into the given panels.

    Shared by the live view and the session-replay screen (agent/events.py
    defines the record shapes).
    """
    t = ev.get("event")
    if t == "start":
        goals.clear()
        errors.clear()
        proof.clear()
        _put(log, f"{tag}── {pid}  {str(ev.get('statement', ''))[:90]}", "bold")
    elif t == "hammer":
        if ev.get("ok"):
            _put(log, f"{tag}  hammer {ev['i']}/{ev['total']}: `{ev['tactic']}` ✓ PROVED ∎",
                 "bold green")
        elif not skip_failed_hammers:
            _put(log, f"{tag}  hammer {ev['i']}/{ev['total']}: `{ev['tactic']}` ✗", "dim")
    elif t == "llm_start":
        _put(log, f"{tag}  no hammer worked → LLM repair loop", "yellow")
    elif t == "build":
        if ev.get("ok"):
            return
        report = str(ev.get("report") or "(no detailed report)")
        errors.clear()
        errors.write(escape(report))
        summary = str(ev.get("summary", ""))[:80]
        _put(log, f"{tag}  [step {ev['step']}] {ev['diagnostics']} diagnostics — {summary}")
    elif t == "goals":
        goals.clear()
        goals.write(escape(str(ev.get("goals", ""))))
    elif t == "llm_request":
        _put(log, f"{tag}  step {ev['step']}: asking LLM…", "cyan")
    elif t == "llm_response":
        _put(log, f"{tag}  step {ev['step']}: LLM replied ({ev.get('tokens', '?')} tokens)")
        proof.clear()
        proof.write(escape(str(ev.get("body", "")) or "(empty)"))
    elif t == "llm_error":
        _put(log, f"{tag}  step {ev['step']}: {str(ev.get('error', ''))[:120]}", "red")
    elif t == "result":
        if ev.get("stopped"):
            _put(log, f"{tag}  stopped by user ({ev.get('seconds', 0):.1f}s)", "yellow")
        elif ev.get("proved"):
            _put(log, f"{tag}  PROVED ∎ ({ev.get('steps')} steps, {ev.get('seconds', 0):.1f}s)",
                 "bold green")
            proof.write("\n\n" + escape("── final file ──\n"))
        else:
            _put(log, f"{tag}  FAILED after {ev.get('steps')} steps ({ev.get('seconds', 0):.1f}s)",
                 "bold red")
        sid = ev.get("session_id")
        if sid:
            _put(log, f"{tag}  session: ~/.prover/sessions/{sid}.jsonl", "dim")


class ProblemRow(ListItem):
    """One benchmark problem with a live status marker."""

    def __init__(self, problem: dict) -> None:
        super().__init__()
        self.problem = problem
        self.status = "pending"

    def _render_str(self) -> str:
        mark = STATUS_MARK[self.status]
        tier = self.problem.get("difficulty", "?")
        color = TIER_COLOR.get(tier, "white")
        return f"{mark} [{color}]{tier:>6}[/{color}] {self.problem['id']}"

    def render(self) -> Text:
        return Text.from_markup(self._render_str())

    def set_status(self, status: str) -> None:
        self.status = status
        self.refresh()


class PaneDivider(Static):
    """Draggable divider between the problem list and the side panels.

    Textual 8.2.8 ships no Split container, so the drag is implemented here:
    left-drag resizes the `#problems` pane (changes are persisted to
    ~/.prover/tui.json on release), double-click resets it to the default.
    """

    ALLOW_SELECT = False
    DEFAULT_WIDTH = 46
    MIN_WIDTH = 20
    MAX_WIDTH = 400
    MIN_SIDE_WIDTH = 40

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dragging = False
        self._drag_start_x = 0.0
        self._drag_start_width = 0

    def _problems(self) -> Vertical:
        return self.app.query_one("#problems", Vertical)

    def _max_width(self) -> int:
        terminal = self.app.size.width
        return max(self.MIN_WIDTH, terminal - self.MIN_SIDE_WIDTH)

    def _set_width(self, width: int) -> None:
        width = min(max(int(width), self.MIN_WIDTH), self._max_width())
        self._problems().styles.width = width
        self.app.tui_settings = replace(
            self.app.tui_settings, problem_pane_width=width
        )

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button != 0 or self.app is None:
            return
        self._dragging = True
        self._drag_start_x = event._screen_x
        current = self._problems().styles.width
        self._drag_start_width = int(getattr(current, "value", current))
        self.add_class("-dragging")
        self.capture_mouse(True)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if not self._dragging:
            return
        width = self._drag_start_width + int(event._screen_x - self._drag_start_x)
        self._set_width(width)

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if not self._dragging:
            return
        self._dragging = False
        self.remove_class("-dragging")
        self.capture_mouse(False)
        save_tui_settings(self.app.tui_settings)
        self.app._refresh_status()

    def on_click(self, event: events.Click) -> None:
        if event.button != 0 or event.chain < 2 or self.app is None:
            return
        self._set_width(self.DEFAULT_WIDTH)
        save_tui_settings(self.app.tui_settings)
        self.app._refresh_status()
        self.app.notify("Pane width reset to default.")


class LeaderboardScreen(ModalScreen[None]):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "dismiss(None)", "Close"),
        Binding("escape", "dismiss(None)", "Close"),
    ]

    def compose(self) -> ComposeResult:
        table = DataTable(id="board-table")
        yield table

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("#", "name", "score", "tiers", "date")
        board_file = REPO / "leaderboard.json"
        try:
            board = json.loads(board_file.read_text()) if board_file.exists() else []
        except (json.JSONDecodeError, OSError) as exc:
            table.add_row("—", f"(unreadable leaderboard.json: {exc})", "", "", "")
            return
        if not board:
            table.add_row("—", "(empty)", "run `prover leaderboard --run`", "", "")
            return
        for i, e in enumerate(board, 1):
            tiers = " ".join(
                f"{t}:{v['proved']}/{v['total']}" for t, v in sorted(e.get("tiers", {}).items())
            )
            table.add_row(str(i), e.get("name", "?"), f"{e.get('score',0)}/{e.get('total','?')}",
                          tiers, e.get("date", ""))


class ProofDiffScreen(ModalScreen[None]):
    """Interactive, navigable diff of the last proof file written for a problem.

    j/k scroll, space page, 0/G jump to top/bottom, q/escape close.
    """

    CSS = """
    ProofDiffScreen { width: 100%; height: 100%; }
    #diff-title { height: 1; background: $panel; padding: 0 1; }
    #diff-body { height: 1fr; padding: 0 1; }
    """
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "dismiss(None)", "Close"),
        Binding("escape", "dismiss(None)", "Close"),
        Binding("j", "scroll_down", "Down", show=False),
        Binding("k", "scroll_up", "Up", show=False),
        Binding("space", "page_down", "Page down", show=False),
        Binding("0", "top", "Top", show=False),
        Binding("G", "bottom", "Bottom", show=False),
    ]

    def __init__(self, problem_id: str, diff: str) -> None:
        super().__init__()
        self._problem_id = problem_id
        self._diff = diff

    def compose(self) -> ComposeResult:
        yield Static(
            f"[bold]proof diff — {self._problem_id}[/bold] "
            f"[dim](j/k scroll · space page · 0/G ends · q close)[/dim]",
            id="diff-title",
        )
        with VerticalScroll(id="diff-body"):
            yield Static(escape(self._diff))

    def _body(self) -> VerticalScroll:
        return self.query_one("#diff-body", VerticalScroll)

    def action_scroll_down(self) -> None:
        self._body().scroll_down()

    def action_scroll_up(self) -> None:
        self._body().scroll_up()

    def action_page_down(self) -> None:
        self._body().scroll_page_down()

    def action_top(self) -> None:
        self._body().scroll_home()

    def action_bottom(self) -> None:
        self._body().scroll_end()


class MessageScreen(ModalScreen[None]):
    """Output modal for slash commands (tau's CommandOutputScreen analogue).

    Its content is the copy target regardless of the global auto-copy setting
    — same rule tau applies to the session modal.
    """

    auto_copy_selection: bool = True

    CSS = """
    MessageScreen { align: center middle; }
    #command-output {
        width: 76; max-width: 90; height: auto; max-height: 70%;
        background: $panel; border: round $primary; padding: 1 2;
    }
    #command-output-body { height: auto; }
    """
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "dismiss(None)", "Close"),
        Binding("escape", "dismiss(None)", "Close"),
    ]

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        with Vertical(id="command-output"):
            yield Static(f"[bold]{self._title}[/bold]")
            yield Static(self._body, id="command-output-body")


class ProveScreen(ModalScreen[str | None]):
    """Free-form theorem input."""

    CSS = """
    ProveScreen { align: center middle; }
    #prove-box {
        width: 90; height: 80%;
        background: $panel; border: round $primary; padding: 1 2;
    }
    #prove-input { height: 1fr; margin-bottom: 1; scrollbar-size: 1 1; }
    #prove-buttons { height: auto; dock: bottom; align-horizontal: right; }
    #prove-buttons Button { margin-left: 2; }
    """
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+enter", "submit", "Prove", priority=True),
    ]

    PLACEHOLDER = "theorem sq_nonneg (x : ℤ) : 0 ≤ x ^ 2 := by sorry"

    def compose(self) -> ComposeResult:
        with Vertical(id="prove-box"):
            yield Static("[bold]prove a custom theorem[/bold] "
                         "[dim](statement with := by, may end in sorry)[/dim]")
            yield TextArea(self.PLACEHOLDER, id="prove-input", soft_wrap=True)
            with Horizontal(id="prove-buttons"):
                yield Button("Prove", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok":
            self.action_submit()
        else:
            self.dismiss(None)

    def action_submit(self) -> None:
        code = self.query_one("#prove-input", TextArea).text.strip()
        if not code:
            self.notify("Enter a theorem statement first.", severity="warning")
            return
        self.dismiss(code)

    def action_cancel(self) -> None:
        self.dismiss(None)


class WorkersScreen(ModalScreen[int | None]):
    """Prompt for the number of parallel workers."""

    CSS = """
    WorkersScreen { align: center middle; }
    #workers-box {
        width: 60; height: 7;
        background: $panel; border: round $primary; padding: 1 2;
    }
    """
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="workers-box"):
            yield Static(f"parallel workers [dim](1–{MAX_WORKERS})[/dim]")
            yield Input(value=str(self.app.n_workers), type="integer", id="workers-input")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        try:
            n = int(event.value)
        except ValueError:
            self.notify("Enter a number.", severity="warning")
            return
        self.dismiss(min(max(n, 1), MAX_WORKERS))

    def action_cancel(self) -> None:
        self.dismiss(None)


class ReplayScreen(ModalScreen[None]):
    """Replay a recorded JSONL session.

    Like tau's session modal, this screen always allows copying selections —
    the session text is the copy target regardless of the global setting.
    """

    auto_copy_selection: bool = True

    CSS = """
    ReplayScreen { width: 100%; height: 100%; }
    #replay-title { height: 1; background: $panel; padding: 0 1; }
    #replay-side { width: 1fr; }
    """
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "dismiss(None)", "Close"),
        Binding("escape", "dismiss(None)", "Close"),
        Binding("j", "raw", "Raw JSON"),
    ]

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.recs: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Static("", id="replay-title")
        with Horizontal(id="main"):
            yield SelectableRichLog(id="rlog", wrap=False)
            with Vertical(id="replay-side"), TabbedContent(initial="tab-rproof"):
                with TabPane("Goals", id="tab-rgoals"):
                    yield SelectableRichLog(id="rgoals", wrap=False)
                with TabPane("Errors", id="tab-rerrors"):
                    yield SelectableRichLog(id="rerrors", wrap=False)
                with TabPane("Proof", id="tab-rproof"):
                    yield SelectableRichLog(id="rproof", wrap=False)

    def on_mount(self) -> None:
        self.recs = sess.read_session(self.path)
        start = next((r for r in self.recs if r.get("event") == "start"), {})
        pid = start.get("problem_id") or self.path.stem
        self.query_one("#replay-title", Static).update(
            f"replay: {self.path.stem}  ·  {str(start.get('statement', ''))[:80]}  "
            f"[dim]({len(self.recs)} events, j=raw)[/dim]"
        )
        for rec in self.recs:
            render_event(rec, pid,
                         self.query_one("#rlog", RichLog),
                         self.query_one("#rgoals", RichLog),
                         self.query_one("#rerrors", RichLog),
                         self.query_one("#rproof", RichLog),
                         skip_failed_hammers=True)

    def action_raw(self) -> None:
        log = self.query_one("#rlog", RichLog)
        log.write(Text("── raw records ──", style="bold"))
        for rec in self.recs:
            log.write(json.dumps(rec, ensure_ascii=False, default=str))


def build_session_label(sp: Path, rec, recs: list[dict]) -> str:
    """Build the one-line SessionsScreen label for a recorded session."""
    start = next((r for r in recs if r.get("event") == "start"), {})
    result = next((r for r in recs if r.get("event") == "result"), {})
    mark = "✓" if result.get("proved") else ("◼" if result.get("stopped") else "✘")
    pid = start.get("problem_id") or "?"
    steps = result.get("steps", "?")
    secs = result.get("seconds", "?")
    stats = calculate_session_stats(recs)
    title = rec.title if rec is not None and rec.title else ""
    label = (f"{mark} {sp.stem:<42} {pid!s:<26} steps={steps} {secs}s"
             f" {stats.total_tokens}t")
    if title:
        label += f"  [{title[:40]}]"
    return label


class SessionRenameScreen(ModalScreen[str | None]):
    """Prompt for a new session title (rename-inline for SessionsScreen)."""

    CSS = """
    SessionRenameScreen { align: center middle; }
    #rename-box {
        width: 72; height: 7;
        background: $panel; border: round $primary; padding: 1 2;
    }
    """
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, session_id: str, title: str = "") -> None:
        super().__init__()
        self._session_id = session_id
        self._title = title

    def compose(self) -> ComposeResult:
        with Vertical(id="rename-box"):
            yield Static(f"rename session [cyan]{self._session_id}[/cyan]")
            yield Input(value=self._title, id="rename-input")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def action_cancel(self) -> None:
        self.dismiss(None)


class SessionsScreen(ModalScreen[Path | None]):
    """Browse ~/.prover/sessions — replay (Enter), rename (r), delete (d),
    or make the highlighted session the active/current one (s)."""

    CSS = """
    SessionsScreen { align: center middle; }
    #sess-list {
        width: 90; height: 80%;
        background: $panel; border: round $primary; padding: 1 2;
    }
    """
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "open_selected", "Replay", show=False),
        Binding("r", "rename_selected", "Rename", show=False),
        Binding("d", "delete_selected", "Delete", show=False),
        Binding("s", "switch_selected", "Switch", show=False),
        Binding("q", "dismiss(None)", "Close"),
        Binding("escape", "dismiss(None)", "Close"),
    ]

    MAX_ROWS = 50

    def __init__(self) -> None:
        super().__init__()
        self._paths: dict[str, Path] = {}
        self._ids: list[str] = []

    def compose(self) -> ComposeResult:
        yield OptionList(id="sess-list")
        yield Static("Enter replay · r rename · d delete · s switch active · q close",
                     classes="hint")

    def on_mount(self) -> None:
        self._reload()

    def _reload(self) -> None:
        opt_list = self.query_one(OptionList)
        opt_list.clear_options()
        self._paths = {}
        self._ids = []
        sessions = sess.list_sessions()[: self.MAX_ROWS]
        if not sessions:
            opt_list.add_option(Option(f"(no sessions in {sess.sessions_dir()})", id=None))
            return
        records = {r.id: r for r in SessionManager().list_sessions()}
        for sp in sessions:
            rec = records.get(sp.stem)
            recs = sess.read_session(sp)
            opt_list.add_option(Option(build_session_label(sp, rec, recs), id=sp.stem))
            self._paths[sp.stem] = sp
            self._ids.append(sp.stem)

    def _highlighted_id(self) -> str | None:
        idx = self.query_one(OptionList).highlighted
        if idx is None or not (0 <= idx < len(self._ids)):
            return None
        return self._ids[idx]

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            self.dismiss(self._paths.get(event.option.id))
        else:
            self.dismiss(None)

    def action_open_selected(self) -> None:
        sid = self._highlighted_id()
        if sid is not None:
            self.dismiss(self._paths.get(sid))

    def action_rename_selected(self) -> None:
        sid = self._highlighted_id()
        if sid is None:
            return
        rec = SessionManager().get(sid)
        current = rec.title if rec is not None else ""
        self.app.push_screen(SessionRenameScreen(sid, current), self._on_renamed)

    def _on_renamed(self, title: str | None) -> None:
        sid = self._highlighted_id()
        if title and sid:
            SessionManager().rename(sid, title)
            self.notify(f"Session renamed to {title}")
            self._reload()

    def action_delete_selected(self) -> None:
        sid = self._highlighted_id()
        if sid is None:
            return
        if SessionManager().delete(sid):
            self.notify(f"Deleted session {sid}")
            if getattr(self.app, "_active_session_id", None) == sid:
                self.app._active_session_id = None
            self._reload()
        else:
            self.notify(f"Session not found: {sid}", severity="error")

    def action_switch_selected(self) -> None:
        sid = self._highlighted_id()
        if sid is None:
            return
        from .commands import CommandResult

        self.app._apply_command(CommandResult(
            handled=True, switch_session_id=sid,
        ))
        self.dismiss(None)


class TrustScreen(ModalScreen[str | None]):
    """First-run project-input trust decision (tau's trust modal, trimmed).

    Protected inputs are listed by category/count only — the modal never
    displays their contents.
    """

    CSS = """
    TrustScreen { align: center middle; }
    #trust-box {
        width: 92; height: 80%;
        background: $panel; border: round $primary; padding: 1 2;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel"),
    ]

    CHOICES = ("trust-exact", "trust-parent", "trust-run",
               "decline-exact", "decline-run")

    def __init__(self, request: object) -> None:
        super().__init__()
        self._request = request

    def compose(self) -> ComposeResult:
        from .project_trust import ProjectTrustRequest

        request = self._request
        assert isinstance(request, ProjectTrustRequest)
        counts = ", ".join(
            f"[dim]{category}[/dim] {request.resources.counts[category]}"
            for category in request.resources.categories
        )
        inherited = (
            f"\ninherits [cyan]{request.inherited_entry.decision}[/cyan] "
            f"from {request.inherited_entry.path.value}"
            if request.inherited_entry is not None
            else ""
        )
        with Vertical(id="trust-box"):
            yield Static(
                "[bold]Trust project inputs?[/bold]\n"
                f"{request.cwd.value}{inherited}\n"
                f"protected: {counts}\n"
                "[dim]project trust is an input-loading guard, not a sandbox[/dim]\n"
            )
            yield OptionList(*[Option(c, id=c) for c in self.CHOICES],
                             id="trust-list")
            yield Static("↑/↓ select · Enter decide · esc cancel",
                         classes="hint")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            self.dismiss(event.option.id)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class PromptsScreen(ModalScreen[str | None]):
    """Pick a markdown prompt template (tau's /prompts picker)."""

    CSS = """
    PromptsScreen { align: center middle; }
    #prompts-list {
        width: 90; height: 80%;
        background: $panel; border: round $primary; padding: 1 2;
    }
    """
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "dismiss(None)", "Close"),
    ]

    def __init__(self, templates: list) -> None:
        super().__init__()
        self._templates = templates

    def compose(self) -> ComposeResult:
        yield OptionList(id="prompts-list")

    def on_mount(self) -> None:
        opt_list = self.query_one(OptionList)
        if not self._templates:
            opt_list.add_option(Option("(no prompt templates found)", id=None))
            return
        for template in self._templates:
            label = template.name
            if template.description:
                label += f"  [dim]— {template.description[:50]}[/dim]"
            opt_list.add_option(Option(label, id=template.name))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)


class ModelsScreen(ModalScreen[None]):
    """Browse/select/add/edit/delete model profiles (agent/models.py store).

    Enter selects the highlighted profile as the active model; `a` adds a new
    one, `e` edits the highlighted one, `d` deletes it. All changes persist
    to ~/.prover/models.json and take effect immediately (llm.client() and
    llm.model() read the store live).
    """

    CSS = """
    ModelsScreen { align: center middle; }
    #models-box {
        width: 92; height: 80%;
        background: $panel; border: round $primary; padding: 1 2;
    }
    #models-title { height: 1; }
    #models-list { height: 1fr; }
    """
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "select_profile", "Select", show=False),
        Binding("a", "add_profile", "Add", show=False),
        Binding("e", "edit_profile", "Edit", show=False),
        Binding("d", "delete_profile", "Delete", show=False),
        Binding("q", "dismiss(None)", "Close"),
        Binding("escape", "dismiss(None)", "Close"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._active = ""
        self._profiles: list = []

    def compose(self) -> ComposeResult:
        with Vertical(id="models-box"):
            yield Static("", id="models-title")
            yield OptionList(id="models-list")
            yield Static("Enter select · a add · e edit · d delete · q close",
                         classes="hint")

    def on_mount(self) -> None:
        self._reload()

    def _reload(self) -> None:
        from . import models as models_mod

        self._profiles = models_mod.load_profiles()
        self._active = models_mod.resolved_model_name()
        title = (f"[bold]model profiles[/bold] — active: "
                 f"[cyan]{self._active or '(none)'}[/cyan]")
        self.query_one("#models-title", Static).update(title)
        opt_list = self.query_one(OptionList)
        opt_list.clear_options()
        if not self._profiles:
            opt_list.add_option(Option("(no model profiles — press a to add one)", id=None))
            return
        for profile in self._profiles:
            mark = "▶" if profile.name == self._active else " "
            endpoint = profile.base_url or "(env endpoint)"
            label = f"  [dim]— {profile.display}[/dim]" if profile.display != profile.name else ""
            opt_list.add_option(Option(
                f"{mark} {profile.name}{label}  [dim]{endpoint}[/dim]",
                id=profile.name,
            ))

    def _highlighted(self) -> int:
        return self.query_one(OptionList).highlighted or 0

    def _current(self) -> ModelProfile | None:
        idx = self._highlighted()
        if not (0 <= idx < len(self._profiles)):
            return None
        return self._profiles[idx]

    def action_select_profile(self) -> None:
        profile = self._current()
        if profile is None:
            return
        from . import models as models_mod

        models_mod.save_store(active=profile.name, profiles=self._profiles)
        self.notify(f"Active model: {profile.name}")
        self._reload()

    def action_add_profile(self) -> None:
        self.app.push_screen(ModelFormScreen(), self._on_form_result)

    def action_edit_profile(self) -> None:
        profile = self._current()
        if profile is None:
            return
        self.app.push_screen(ModelFormScreen(profile), self._on_form_result)

    def _on_form_result(self, profile: ModelProfile | None) -> None:
        if profile is None:
            return
        from . import models as models_mod

        idx = next((i for i, p in enumerate(self._profiles)
                    if p.name == profile.name), -1)
        if idx >= 0:
            self._profiles[idx] = profile
        else:
            self._profiles.append(profile)
        models_mod.save_store(active=self._active, profiles=self._profiles)
        self.notify(f"Saved profile: {profile.name}")
        self._reload()

    def action_delete_profile(self) -> None:
        profile = self._current()
        if profile is None:
            return
        from . import models as models_mod

        self._profiles = [p for p in self._profiles if p.name != profile.name]
        active = self._active if self._active != profile.name else ""
        models_mod.save_store(active=active, profiles=self._profiles)
        self.notify(f"Deleted profile: {profile.name}")
        self._reload()


class ModelFormScreen(ModalScreen[ModelProfile | None]):
    """Add/edit a model profile (name + optional endpoint/overrides)."""

    CSS = """
    ModelFormScreen { align: center middle; }
    #model-form {
        width: 76; height: auto;
        background: $panel; border: round $primary; padding: 1 2;
    }
    #model-form Input { margin-bottom: 1; }
    #model-form-buttons { dock: bottom; height: auto; align-horizontal: right; }
    #model-form-buttons Button { margin-left: 2; }
    """
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+enter", "submit", "Save", priority=True),
    ]

    FIELDS = (
        ("f-name", "model name (the id sent to the endpoint)", False),
        ("f-label", "display label (optional)", False),
        ("f-base", "base URL (empty = env OPENAI_BASE_URL)", False),
        ("f-key", "API key (empty = env OPENAI_API_KEY)", True),
        ("f-ctx", "context window in tokens (optional)", False),
        ("f-in", "prompt cost per 1M tokens, USD (optional)", False),
        ("f-out", "completion cost per 1M tokens, USD (optional)", False),
    )

    def __init__(self, profile: ModelProfile | None = None) -> None:
        super().__init__()
        self._profile = profile

    def compose(self) -> ComposeResult:
        title = "edit model profile" if self._profile else "add model profile"
        yield Static(f"[bold]{title}[/bold]")
        with Vertical(id="model-form"):
            for fid, placeholder, password in self.FIELDS:
                yield Input(
                    placeholder=placeholder,
                    password=password,
                    id=fid,
                    value="",
                )
            with Horizontal(id="model-form-buttons"):
                yield Button("Save", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")
        yield Static("ctrl+enter save · esc cancel", classes="hint")

    def on_mount(self) -> None:
        profile = self._profile
        if profile is None:
            return
        self.query_one("#f-name", Input).value = profile.name
        self.query_one("#f-label", Input).value = profile.label
        self.query_one("#f-base", Input).value = profile.base_url
        self.query_one("#f-key", Input).value = profile.api_key
        if profile.context_window is not None:
            self.query_one("#f-ctx", Input).value = str(profile.context_window)
        if profile.cost_in is not None:
            self.query_one("#f-in", Input).value = str(profile.cost_in)
        if profile.cost_out is not None:
            self.query_one("#f-out", Input).value = str(profile.cost_out)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok":
            self.action_submit()
        else:
            self.dismiss(None)

    def _value(self, fid: str) -> str:
        return self.query_one(f"#{fid}", Input).value.strip()

    def action_submit(self) -> None:
        name = self._value("f-name")
        if not name:
            self.notify("A model name is required.", severity="warning")
            return
        try:
            ctx = int(self._value("f-ctx")) if self._value("f-ctx") else None
            cost_in = float(self._value("f-in")) if self._value("f-in") else None
            cost_out = float(self._value("f-out")) if self._value("f-out") else None
        except ValueError:
            self.notify("Context window and costs must be numbers.", severity="warning")
            return
        self.dismiss(ModelProfile(
            name=name,
            label=self._value("f-label"),
            base_url=self._value("f-base"),
            api_key=self.query_one("#f-key", Input).value,
            context_window=ctx,
            cost_in=cost_in,
            cost_out=cost_out,
        ))

    def action_cancel(self) -> None:
        self.dismiss(None)


class ProverApp(App):
    """Interactive proof agent dashboard."""

    TITLE = "lean-prover — Lean 4 proof agent"
    # Mirrors tau: the session modal always allows selecting (its text is the
    # copy target); the main screen follows ALLOW_SELECT which is toggled off
    # while a proof run mutates the transcript.
    CSS = """
    #main { height: 1fr; }
    #problems { width: 46; }
    #pane-divider {
        width: 2; height: 100%; min-height: 3;
        background: $primary 20%; content-align: center middle;
    }
    #pane-divider:hover, #pane-divider.-dragging { background: $primary; }
    #prompt-box { dock: bottom; height: auto; padding: 0 1; }
    #prompt { height: 1; }
    #prompt-completions { height: auto; max-height: 9; display: none; background: $panel; }
    #status-bar { dock: bottom; height: 1; background: $panel; padding: 0 1; }
    #board-table { width: 100%; height: 100%; }
    #side { width: 1fr; }
    #search-bar { height: 1; margin-bottom: 1; }
    """
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
        from . import llm

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
        from . import thinking as thinking_mod

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
            result = self.command_registry.execute(self, text)
            if not result.handled and result.message:
                self.notify(result.message, severity="warning")
            if result.handled:
                self._apply_command(result)
                return
            # Unknown slash command: fall back to prompt-template expansion
            # (tau parity; built-in commands keep precedence over templates).
            from .prompt_templates import (
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
        from . import thinking as thinking_mod

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
        from .reload import ReloadSnapshot, build_reload_summary, take_reload_snapshot

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
        from .project_trust import (
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
            summary, resolution = await ProjectTrustCoordinator(store).resolve(
                REPO, default=default, interactive=interactive, prompt=ask
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
        return (f"{state:<7} workers={self.n_workers} · thinking={self.thinking_level} · "
                f"proved {self.counts['proved']} · "
                f"failed {self.counts['failed']} · stopped {self.counts['stopped']} · "
                f"remaining {remaining}{cost_str}")

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
        from . import models as models_mod
        from .provider_config import provider_config_from_name

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

        def _run() -> None:
            import asyncio

            if provider == "openai-codex":
                from .oauth.codex import login_openai_codex

                credential = login_openai_codex()
            else:
                if provider == "anthropic":
                    from .oauth.anthropic import login_anthropic as login_async
                else:
                    from .oauth.github_copilot import (
                        login_github_copilot as login_async,
                    )
                credential = asyncio.run(login_async())
            if credential is None:
                self.call_from_thread(self.notify, "sign-in did not complete", severity="error")
            else:
                self.call_from_thread(self.notify, f"signed in to {provider}")

        self.run_worker(_run, thread=True)

    def _logout_provider(self, provider: str) -> None:
        """/logout — remove stored credentials for a provider."""
        from .credentials import FileCredentialStore

        if FileCredentialStore().delete(provider):
            self.notify(f"removed {provider} credentials")
        else:
            self.notify(f"no stored credentials for {provider}")

    def _show_skills(self) -> None:
        """/skills — list discovered skills."""
        from .skills import load_skills

        skills = load_skills()
        body = "\n".join(f"- {s.name}: {s.description}" for s in skills) or "(none discovered)"
        self.push_screen(MessageScreen("skills", body))

    def _show_contexts(self) -> None:
        """/contexts — list project context files."""
        from pathlib import Path as _P

        from .context import discover_project_context_with_diagnostics

        files, _ = discover_project_context_with_diagnostics(_P.cwd())
        body = "\n".join(f"- {f['path']} ({len(f['content'])} chars)" for f in files) or "(none found)"
        self.push_screen(MessageScreen("project contexts", body))

    def _show_tools(self) -> None:
        """/tools — list the tools bound to the prove loop."""
        from .prover_loop import LEAN_DIR
        from .tools import default_tools

        names = [t.name for t in default_tools(LEAN_DIR)]
        from .coding_tools import create_coding_tools

        names += [t["name"] for t in create_coding_tools()]
        self.push_screen(MessageScreen("tools", "\n".join(f"- {n}" for n in names)))

    def _check_update(self) -> None:
        """/update — check PyPI for a newer release (off the UI thread)."""
        from .update_check import check_for_updates

        def worker() -> None:
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

        self.run_worker(worker, thread=True, name="update-check")

    def _handle_permissions(self, result: CommandResult) -> None:
        """/permissions — list, remember, revoke or set baseline mode (fx ACL)."""
        from .permissions import VALID_MODES, PermissionStore

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
        from . import thinking as thinking_mod

        try:
            thinking_mod.set_thinking_level(level)
        except ValueError as exc:
            self.notify(str(exc), severity="error")
            return
        if self.tui_settings.thinking_level != level:
            self.tui_settings = replace(self.tui_settings, thinking_level=level)
            save_tui_settings(self.tui_settings)
        self._refresh_status()

    def _prove_statement(self, statement: str) -> None:
        """Prove an inline theorem from /prove <statement>."""
        m = re.search(r"\b(?:theorem|lemma|example)\s+(\w+)", statement)
        base = m.group(1) if m else "custom"
        self._custom_seq += 1
        problem = {
            "id": f"{base}-{self._custom_seq}" if self._custom_seq > 1 else base,
            "statement": statement,
            "difficulty": "custom",
        }
        self._log(f"\n[bold magenta]custom: {problem['id']}[/bold magenta]")
        self._start_run([problem])

    def _replay_by_id(self, session_id: str) -> None:
        for sp in sess.list_sessions():
            if sp.stem == session_id:
                self.push_screen(ReplayScreen(sp))
                return

    def _branch_run(self, session_id: str, branch_at: int | None) -> None:
        """Re-run a recorded session's theorem, seeded from its history
        (tau: branch_to_entry repoints the leaf; here prove(resume_from=...))."""
        from .session_manager import SessionManager

        if self._run_active:
            self.notify("Already running — press s to stop first.", severity="warning")
            return
        rec = SessionManager().get(session_id)
        if rec is None:
            self.notify(f"Session not found: {session_id}", severity="error")
            return
        records = sess.read_session(Path(rec.path))
        start = next((r for r in records if r.get("event") == "start"), {})
        statement = start.get("statement")
        if not statement:
            self.notify("Session has no recorded statement.", severity="error")
            return
        pid = start.get("problem_id") or "branch"

        def worker() -> None:
            summary = None
            if branch_at is not None and os.environ.get("PROVER_BRANCH_SUMMARY", "1") != "0":
                from .branch_summary import summarize_branch_with_model

                summary = summarize_branch_with_model(records)
                if summary:
                    self.call_from_thread(
                        self._log, "[dim]branch summary generated[/dim]")
                else:
                    self.call_from_thread(
                        self._log,
                        "[dim]branch summary unavailable — continuing with "
                        "recorded history only[/dim]")
            result = prove(
                statement, max_steps=20, verbose=False,
                problem_id=pid, goal_feedback=True,
                on_event=lambda ev: self._probe_event(pid, ev),
                should_stop=lambda: self._stop_flag,
                resume_from=session_id, branch_at=branch_at,
                branch_summary=summary,
            )
            status = ("proved" if result.proved
                      else "stopped" if result.stopped else "failed")
            self.call_from_thread(self._finish_custom_row, pid, status, result)

        self._stop_flag = False
        self._run_active = True
        self._sync_text_selection_state()
        self._refresh_status()
        self._log(f"\n[bold]── branch {session_id}"
                  f"{' @ turn ' + str(branch_at) if branch_at is not None else ''}[/bold]")
        self.run_worker(worker, thread=True, group="prove")

    def _finish_custom_row(self, problem_id: str, status: str, result) -> None:
        """Finish handler for runs not backed by a ProblemRow (branch/custom)."""
        self.counts[status] = self.counts.get(status, 0) + 1
        proof = result.proof if result.proved else ""
        if proof:
            panel = self.query_one("#proof", RichLog)
            panel.clear()
            panel.write(escape(proof))
        self._run_active = False
        self._sync_text_selection_state()
        self._refresh_status()
        self._terminal_notification.notify_turn_finished()
        self._drain_queued_prompts()

    def _show_usage(self, session_arg: str | None) -> None:
        from .session_manager import SessionManager
        from .session_usage import collect_session_usage, render_usage_dashboard

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
        from .session_export import export_session

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

    # ---------------------------------------------------------------- proving

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

    def _start_run(self, problems: list[dict]) -> None:
        self._stop_flag = False
        self._run_active = True
        self._sync_text_selection_state()
        self._refresh_status()
        self.run_worker(functools.partial(self._run_problems, problems, self.n_workers),
                        thread=True, group="prove")

    def _run_problems(self, problems: list[dict], workers: int) -> None:
        """Run problems — sequential or with a thread pool."""
        if workers <= 1:
            for problem in problems:
                if self._stop_flag:
                    self.call_from_thread(self._mark_pending_skipped, problems)
                    break
                self._prove_one(problem)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(self._prove_one, p): p for p in problems}
                for fut in as_completed(futs):
                    if self._stop_flag:
                        for f in futs:
                            f.cancel()
                        break
        self.call_from_thread(self._run_done)

    def _mark_pending_skipped(self, problems: list[dict]) -> None:
        """Mark rows that were still pending when a run stopped as skipped."""
        for problem in problems:
            row = self._row_by_id(problem["id"])
            if row is not None and row.status == "pending":
                row.set_status("stopped")
                self.counts["stopped"] = self.counts.get("stopped", 0) + 1
        self._refresh_status()

    def _prove_one(self, problem: dict) -> None:
        """Prove a single problem (called from a worker thread)."""
        if self._stop_flag:
            return
        pid = problem["id"]
        self.call_from_thread(self._set_row_status, pid, "running")
        result = prove(
            problem["statement"],
            max_steps=20,
            verbose=False,
            problem_id=pid,
            goal_feedback=True,
            on_event=lambda ev, p=pid: self._probe_event(p, ev),
            should_stop=lambda: self._stop_flag,
        )
        status = "proved" if result.proved else ("stopped" if result.stopped else "failed")
        self.call_from_thread(self._finish_row, pid, status, result)

    def _probe_event(self, problem_id: str, ev: dict) -> None:
        """Forward a prove() event to the UI thread."""
        self.call_from_thread(self._render_event, problem_id, ev)
        # Show diff panel on file write events (full-file mode)
        if ev.get("event") == "llm_response" and ev.get("accepted"):
            body = ev.get("body")
            if body and ("\ntheorem" in body or "\nlemma" in body):
                self.call_from_thread(self._show_proof_diff, problem_id, body)

    def _render_event(self, problem_id: str, ev: dict) -> None:
        """Render a prove() event record into the main panels."""
        tag = f"{problem_id[:16]}: " if self.n_workers > 1 else ""
        render_event(ev, problem_id,
                     self.query_one("#log", RichLog),
                     self.query_one("#goals", RichLog),
                     self.query_one("#errors", RichLog),
                     self.query_one("#proof", RichLog),
                     tag=tag)
        # Track live token usage for cost meter
        if ev.get("event") == "llm_response":
            tokens = ev.get("tokens", 0)
            if isinstance(tokens, (int, float)):
                self._live_tokens += int(tokens)
        elif ev.get("event") == "result":
            cost = ev.get("cost", 0.0)
            if isinstance(cost, (int, float)):
                self._live_cost += float(cost)
        # Auto-compact warning: check context window usage
        elif ev.get("event") == "compaction":
            dropped = ev.get("dropped", 0)
            self._log(f"[dim]auto-compacted: dropped {dropped} turns[/dim]")
            if problem_id not in self._compact_warning_shown:
                self._compact_warning_shown.add(problem_id)
                self.notify(
                    f"Session {problem_id} auto-compacted "
                    f"(dropped {dropped} turns to stay within context window)",
                    severity="warning",
                )
        self._refresh_status()

    # ---------------------------------------------------------------- rows/status

    def _row_by_id(self, problem_id: str) -> ProblemRow | None:
        for row in self.query(ProblemRow):
            if row.problem["id"] == problem_id:
                return row
        return None

    def _set_row_status(self, problem_id: str, status: str) -> None:
        row = self._row_by_id(problem_id)
        if row:
            row.set_status(status)

    def _finish_row(self, problem_id: str, status: str, result) -> None:
        self._set_row_status(problem_id, status)
        self.counts[status] = self.counts.get(status, 0) + 1
        proof = result.proof if result.proved else ""
        if proof and self.n_workers <= 1:
            panel = self.query_one("#proof", RichLog)
            panel.clear()
            panel.write(escape(proof))
        self._refresh_status()

    def _run_done(self) -> None:
        self._run_active = False
        self._sync_text_selection_state()
        self._refresh_status()
        self._log(f"[bold]run finished[/bold] — proved {self.counts['proved']} so far")
        self._terminal_notification.notify_turn_finished()
        self._drain_queued_prompts()

    def _drain_queued_prompts(self) -> None:
        """Prove prompts queued while a run was active (tau's queued follow-ups).

        Starts exactly one: when it finishes, _run_done re-enters here, so
        queued prompts prove sequentially instead of racing on the shared
        #proof/#goals panels.
        """
        if not self._queued_prompts:
            return
        text = self._queued_prompts.pop(0)
        self._log(f"[dim]applying queued:[/dim] {text[:60]}")
        self._prove_statement(text)

    # ---------------------------------------------------------------- diff

    def _show_proof_diff(self, problem_id: str, body: str) -> None:
        """Show a diff view of the proof when full-file mode writes a new file."""
        try:
            target = REPO / "lean" / "tmp" / f"Prover_{problem_id}.lean"
            if not target.exists():
                return
            current = target.read_text(encoding="utf-8")
            old = self._get_last_proof_body(problem_id)
            if old is None:
                old = current
            diff = self._compute_diff(old, body)
            if diff:
                self.push_screen(ProofDiffScreen(problem_id, diff))
                self._log(f"[dim]diff for {problem_id}: {len(diff.splitlines())} lines[/dim]")
        except Exception as exc:  # noqa: BLE001 — diff is best-effort
            self._log(f"[dim]diff failed: {exc}[/dim]")

    def _get_last_proof_body(self, problem_id: str) -> str | None:
        """Get the last proof body for a problem from its session."""
        try:
            from . import session as sess_mod
            sessions = list(sess_mod.list_sessions())
            for sp in sessions:
                if sp.stem == problem_id:
                    records = sess_mod.read_session(sp)
                    for rec in reversed(records):
                        if rec.get("event") == "llm_response" and rec.get("body"):
                            return rec["body"]
        except Exception as exc:  # noqa: BLE001
            self._log(f"[dim]failed to get last proof body: {exc}[/dim]")
        return None

    @staticmethod
    def _compute_diff(old: str, new: str) -> str:
        """Compute a simple unified diff between two strings."""
        import difflib
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        return "".join(difflib.unified_diff(old_lines, new_lines,
                                            fromfile="old", tofile="new", n=3))

    # ---------------------------------------------------------------- clipboard

    def _sync_terminal_title(self) -> None:
        """Reflect run state in the terminal tab title (tau parity)."""
        title = f"{len(self.problems)} problems · proved {self.counts.get('proved', 0)}"
        self._terminal_title.update(title, running=self._run_active)

    def on_unmount(self) -> None:
        self._terminal_title.restore()

    def _sync_text_selection_state(self) -> None:
        """Disable native text selection while the transcript is mutating (tau parity)."""
        type(self).ALLOW_SELECT = not self._run_active
        if self._run_active and self.screen_stack:
            with suppress(Exception):
                self.screen.clear_selection()

    def copy_to_clipboard(self, text: str) -> None:
        """Copy text using pyperclip when available, then Textual's fallback (tau parity)."""
        if self._supports_pyperclip is None:
            try:
                import pyperclip  # type: ignore[import-untyped]
            except ImportError:
                self._supports_pyperclip = False
            else:
                if pyperclip is not None:
                    self._supports_pyperclip = True
        if self._supports_pyperclip:
            import pyperclip  # type: ignore[import-untyped]

            with suppress(Exception):
                pyperclip.copy(text)
        super().copy_to_clipboard(text)

    @on(events.TextSelected)
    async def on_text_selected(self) -> None:
        """Optionally copy selected text automatically (tau parity)."""
        active_screen = self.screen
        if not (
            self.tui_settings.auto_copy_selection
            or getattr(active_screen, "auto_copy_selection", False)
        ):
            return
        selection = active_screen.get_selected_text()
        if selection:
            self.copy_to_clipboard(selection)
            self.notify("Copied selection to clipboard.")

    # ---------------------------------------------------------------- actions

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
        from .prompt_templates import load_prompt_templates

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
        from .reload import take_reload_snapshot

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
