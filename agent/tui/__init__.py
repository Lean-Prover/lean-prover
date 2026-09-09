"""TUI package — the interactive proof-agent dashboard.

Formerly a single ``agent/tui.py`` monolith (~2 200 LOC); now split into:

- ``app.py``      — ``ProverApp`` (composition, bindings, command dispatch)
- ``screens.py``  — the modal screens + session-label helper
- ``handlers.py`` — ``ProverRunMixin`` (proof-run + event rendering)
- ``app.tcss``    — the Textual stylesheet (theming)

This ``__init__`` re-exports every public name that ``main.py`` and the test
suite imported from ``agent.tui``, and owns the module-level ``REPO`` /
``PROBLEMS_FILE`` / ``MAX_WORKERS`` constants that tests monkeypatch. ``REPO``
and ``PROBLEMS_FILE`` are defined *before* the submodule imports so the
submodules (which read them at call time via ``agent.tui.REPO``) never hit a
partially-initialized package.
"""

from __future__ import annotations

import json
from pathlib import Path

# Public, monkeypatchable module constants (tests patch these attributes).
REPO = Path(__file__).resolve().parent.parent.parent
PROBLEMS_FILE = REPO / "benchmark" / "problems.json"

MAX_WORKERS = 16


def load_problems() -> list[dict]:
    """Load the benchmark problem set. Raises ValueError on corrupt JSON."""
    if not PROBLEMS_FILE.exists():
        return []
    try:
        return json.loads(PROBLEMS_FILE.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"benchmark/problems.json is unreadable: {exc}") from exc


# Submodule imports come after the constants so ``agent.tui`` is fully
# initialised by the time the app/screens read ``REPO`` / ``MAX_WORKERS``.
from agent.tui.app import ProverApp, main
from agent.tui.handlers import ProverRunMixin
from agent.tui.screens import (
    ConfirmScreen,
    LeaderboardScreen,
    MessageScreen,
    ModelFormScreen,
    ModelsScreen,
    PromptEditScreen,
    PromptNewScreen,
    PromptsScreen,
    ProofDiffScreen,
    ProveScreen,
    ReplayScreen,
    SessionRenameScreen,
    SessionsScreen,
    SettingsScreen,
    TrustScreen,
    WorkersScreen,
    build_session_label,
)
from agent.tui_render import (
    PaneDivider,
    ProblemRow,
    PromptInput,
    SelectableRichLog,
    render_event,
)
from agent.tui_settings import (
    TuiKeybindings,
    TuiSettings,
    load_tui_settings,
    resolved_theme,
    save_tui_settings,
    tui_settings_path,
)

__all__ = [
    "MAX_WORKERS",
    "PROBLEMS_FILE",
    "REPO",
    "ConfirmScreen",
    "LeaderboardScreen",
    "MessageScreen",
    "ModelFormScreen",
    "ModelsScreen",
    "PaneDivider",
    "ProblemRow",
    "PromptEditScreen",
    "PromptInput",
    "PromptNewScreen",
    "PromptsScreen",
    "ProofDiffScreen",
    "ProveScreen",
    "ProverApp",
    "ProverRunMixin",
    "ReplayScreen",
    "SelectableRichLog",
    "SessionRenameScreen",
    "SessionsScreen",
    "SettingsScreen",
    "TrustScreen",
    "TuiKeybindings",
    "TuiSettings",
    "WorkersScreen",
    "build_session_label",
    "load_problems",
    "load_tui_settings",
    "main",
    "render_event",
    "resolved_theme",
    "save_tui_settings",
    "tui_settings_path",
]
