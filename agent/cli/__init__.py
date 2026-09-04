"""Prover CLI command registry (extracted from the former ``agent/main.py``).

Each command family lives in its own module:

- ``prove.py``      — ``prover prove`` / ``prover ask``
- ``bench.py``      — ``prover bench`` (scoring core: ``_prove_one``/``_run_problems``)
- ``leaderboard.py``— ``prover leaderboard`` + provenance/comparability helpers
- ``sessions.py``   — ``prover sessions`` / ``prover usage``
- ``misc.py``       — thin delegators (mcp/formalize/synth/finetune/tui/…)
- ``parser.py``     — the argparse dispatcher (``cli`` / ``_build_parser``)
- ``common.py``     — shared helpers
- ``coding.py``-equivalent lives in ``agent.cli_coding`` (the small coding facade)

``agent.main`` re-exports everything here, so the installed entry point
(``prover = agent.main:cli``) and the test suite's ``agent.main`` monkeypatch
contract are unchanged by the split.
"""

from __future__ import annotations

from .bench import _prove_one, _run_problems, cmd_bench
from .common import _preload_extensions, _warn_model_mismatch
from .leaderboard import (
    BOARD_FILE,
    DEFAULT_SET,
    LEADERBOARD_SOLVER,
    SCHEMA_VERSION,
    _check_frozen_set,
    _current_config,
    _current_statement_hashes,
    _entry_is_comparable,
    _entry_reasons,
    _leaderboard_run,
    _leaderboard_show,
    _report_drift,
    _upsert_board,
    cmd_leaderboard,
    load_board,
    provenance,
)
from .misc import (
    cmd_acp,
    cmd_chat,
    cmd_datagen,
    cmd_export,
    cmd_finetune,
    cmd_formalize,
    cmd_lean_baseline,
    cmd_lean_synth,
    cmd_login,
    cmd_mcp,
    cmd_rpc,
    cmd_rldata,
    cmd_search,
    cmd_synth,
    cmd_tui,
    cmd_update,
    cmd_version,
)
from .parser import _build_parser, cli
from .prove import cmd_ask, cmd_prove
from .sessions import cmd_sessions, cmd_usage

__all__ = [
    "BOARD_FILE",
    "DEFAULT_SET",
    "LEADERBOARD_SOLVER",
    "SCHEMA_VERSION",
    "_build_parser",
    "_check_frozen_set",
    "_current_config",
    "_current_statement_hashes",
    "_entry_is_comparable",
    "_entry_reasons",
    "_leaderboard_run",
    "_leaderboard_show",
    "_preload_extensions",
    "_prove_one",
    "_report_drift",
    "_run_problems",
    "_upsert_board",
    "_warn_model_mismatch",
    "cli",
    "cmd_ask",
    "cmd_acp",
    "cmd_bench",
    "cmd_chat",
    "cmd_datagen",
    "cmd_export",
    "cmd_finetune",
    "cmd_formalize",
    "cmd_leaderboard",
    "cmd_lean_baseline",
    "cmd_lean_synth",
    "cmd_login",
    "cmd_mcp",
    "cmd_prove",
    "cmd_rpc",
    "cmd_rldata",
    "cmd_search",
    "cmd_sessions",
    "cmd_synth",
    "cmd_tui",
    "cmd_update",
    "cmd_usage",
    "cmd_version",
    "load_board",
    "provenance",
]
