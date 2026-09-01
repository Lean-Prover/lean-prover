"""TUI rendering helpers — event rendering, selectable logs, widgets.

Extracted from ``agent/tui.py`` (the TUI monolith). These are the leaf
rendering pieces with no dependency on the ``ProverApp`` class, so they can
live outside the app module without a circular import. ``tui.py`` re-exports
them for existing call sites.
"""

from __future__ import annotations

from dataclasses import replace

from rich.markup import escape
from rich.text import Text
from textual import events
from textual.containers import Vertical
from textual.selection import Selection
from textual.widgets import Input, ListItem, RichLog, Static

from .file_drop import normalize_dropped_paths
from .tui_settings import save_tui_settings

STATUS_MARK = {"pending": "·", "running": "▶", "proved": "✔", "failed": "✘", "stopped": "◼"}
TIER_COLOR = {"trivial": "green", "easy": "cyan", "medium": "yellow",
              "hard": "red", "custom": "magenta"}


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
