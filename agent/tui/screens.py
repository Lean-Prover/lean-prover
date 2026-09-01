"""TUI modal screens (extracted from the former ``agent/tui.py`` monolith).

One screen class per modal, plus ``build_session_label``. These are pushed via
``app.push_screen`` from ``ProverApp`` (see ``agent.tui.app``). Styling lives
in ``agent/tui/app.tcss`` (loaded by the app), not inline class attributes.

``REPO`` / ``MAX_WORKERS`` are read through the ``agent.tui`` package namespace
at call time so tests that ``monkeypatch.setattr("agent.tui.REPO", ...)`` keep
working after the split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Input,
    OptionList,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.widgets.option_list import Option

import agent.tui as _tui
from agent import session as sess
from agent.models import ModelProfile
from agent.session_manager import SessionManager
from agent.session_stats import calculate_session_stats
from agent.tui_render import SelectableRichLog, render_event


class LeaderboardScreen(ModalScreen[None]):
    CSS_PATH: ClassVar[str] = "screens.tcss"
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
        board_file = _tui.REPO / "leaderboard.json"
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
    CSS_PATH: ClassVar[str] = "screens.tcss"
    """Interactive, navigable diff of the last proof file written for a problem.

    j/k scroll, space page, 0/G jump to top/bottom, q/escape close.
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
    CSS_PATH: ClassVar[str] = "screens.tcss"
    """Output modal for slash commands (tau's CommandOutputScreen analogue).

    Its content is the copy target regardless of the global auto-copy setting
    — same rule tau applies to the session modal.
    """

    auto_copy_selection: bool = True

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
    CSS_PATH: ClassVar[str] = "screens.tcss"
    """Free-form theorem input."""

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
    CSS_PATH: ClassVar[str] = "screens.tcss"
    """Prompt for the number of parallel workers."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="workers-box"):
            yield Static(f"parallel workers [dim](1–{_tui.MAX_WORKERS})[/dim]")
            yield Input(value=str(self.app.n_workers), type="integer", id="workers-input")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        try:
            n = int(event.value)
        except ValueError:
            self.notify("Enter a number.", severity="warning")
            return
        self.dismiss(min(max(n, 1), _tui.MAX_WORKERS))

    def action_cancel(self) -> None:
        self.dismiss(None)


class ReplayScreen(ModalScreen[None]):
    CSS_PATH: ClassVar[str] = "screens.tcss"
    """Replay a recorded JSONL session.

    Like tau's session modal, this screen always allows copying selections —
    the session text is the copy target regardless of the global setting.
    """

    auto_copy_selection: bool = True

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
    CSS_PATH: ClassVar[str] = "screens.tcss"
    """Prompt for a new session title (rename-inline for SessionsScreen)."""

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
    CSS_PATH: ClassVar[str] = "screens.tcss"
    """Browse ~/.prover/sessions — replay (Enter), rename (r), delete (d),
    or make the highlighted session the active/current one (s)."""

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
        from agent.commands import CommandResult

        self.app._apply_command(CommandResult(
            handled=True, switch_session_id=sid,
        ))
        self.dismiss(None)


class TrustScreen(ModalScreen[str | None]):
    CSS_PATH: ClassVar[str] = "screens.tcss"
    """First-run project-input trust decision (tau's trust modal, trimmed).

    Protected inputs are listed by category/count only — the modal never
    displays their contents.
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
        from agent.project_trust import ProjectTrustRequest

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
    CSS_PATH: ClassVar[str] = "screens.tcss"
    """Pick a markdown prompt template (tau's /prompts picker)."""

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
    CSS_PATH: ClassVar[str] = "screens.tcss"
    """Browse/select/add/edit/delete model profiles (agent/models.py store).

    Enter selects the highlighted profile as the active model; `a` adds a new
    one, `e` edits the highlighted one, `d` deletes it. All changes persist
    to ~/.prover/models.json and take effect immediately (llm.client() and
    llm.model() read the store live).
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
        from agent import models as models_mod

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
        from agent import models as models_mod

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
        from agent import models as models_mod

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
        from agent import models as models_mod

        self._profiles = [p for p in self._profiles if p.name != profile.name]
        active = self._active if self._active != profile.name else ""
        models_mod.save_store(active=active, profiles=self._profiles)
        self.notify(f"Deleted profile: {profile.name}")
        self._reload()


class ModelFormScreen(ModalScreen[ModelProfile | None]):
    CSS_PATH: ClassVar[str] = "screens.tcss"
    """Add/edit a model profile (name + optional endpoint/overrides)."""

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
