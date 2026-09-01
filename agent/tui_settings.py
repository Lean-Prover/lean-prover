"""TUI preferences — settings, keybindings, and persistence.

Extracted from ``agent/tui.py`` (the TUI monolith) so preferences live in
one small, self-contained module. ``tui.py`` re-exports these names, so
existing ``from agent.tui import TuiSettings`` call sites keep working.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# Pane-width bounds (shared with the PaneDivider widget; kept here so
# TuiSettings.from_json can clamp without importing the widget module).
PANE_MIN_WIDTH = 20
PANE_MAX_WIDTH = 400


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
        pane = min(max(pane, PANE_MIN_WIDTH), PANE_MAX_WIDTH)
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
