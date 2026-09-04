"""CLI entry point — thin facade over the ``agent/cli`` command registry.

The command handlers and the argparse dispatcher moved to ``agent/cli/``
(ARCHITECTURE.md Step H). This module re-exports them so:

- the installed entry point ``prover = agent.main:cli`` keeps working, and
- the test suite's ``agent.main`` monkeypatch contract is unchanged: the CLI
  modules read ``agent.main.<name>`` at call time, so patching e.g.
  ``agent.main.prove`` / ``agent.main._prove_one`` still takes effect.

``prove`` / ``prove_best_of`` are re-exported from ``agent.loop`` (the engine).
"""

from __future__ import annotations

import sys

# Unbuffered stdout so background runs (nohup ... > log) show progress live.
sys.stdout.reconfigure(line_buffering=True)

# All imports are re-exports for the facade; `__all__` below keeps them used.
from .cli import (
    BOARD_FILE,
    DEFAULT_SET,
    LEADERBOARD_SOLVER,
    SCHEMA_VERSION,
    _build_parser,
    _check_frozen_set,
    _current_config,
    _current_statement_hashes,
    _entry_is_comparable,
    _entry_reasons,
    _leaderboard_run,
    _leaderboard_show,
    _preload_extensions,
    _prove_one,
    _report_drift,
    _run_problems,
    _upsert_board,
    _warn_model_mismatch,
    cli,
    cmd_acp,
    cmd_ask,
    cmd_bench,
    cmd_chat,
    cmd_datagen,
    cmd_export,
    cmd_finetune,
    cmd_formalize,
    cmd_leaderboard,
    cmd_lean_baseline,
    cmd_lean_synth,
    cmd_login,
    cmd_mcp,
    cmd_prove,
    cmd_rpc,
    cmd_rldata,
    cmd_search,
    cmd_sessions,
    cmd_synth,
    cmd_tui,
    cmd_update,
    cmd_usage,
    cmd_version,
    load_board,
    provenance,
)
from .loop import prove, prove_best_of

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
    "cmd_acp",
    "cmd_ask",
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
    "prove",
    "prove_best_of",
    "provenance",
]

if __name__ == "__main__":
    cli()
