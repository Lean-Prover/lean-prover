"""Shared helpers for the prover CLI command modules."""

from __future__ import annotations

import os
import sys


def _warn_model_mismatch() -> None:
    """Fast endpoint sanity check: warn when PROVER_MODEL isn't served, so a
    mistyped model name fails in seconds rather than hanging on completions."""
    from .. import llm

    hint = llm.validate_model()
    if hint:
        print(f"[model warn] {hint}", file=sys.stderr)
        print("set PROVER_MODEL to one of the served models.", file=sys.stderr)


def _preload_extensions(extra_paths: list[str]) -> None:
    """Load extension paths ahead of the TUI session (Tau --extensions parity)."""
    from pathlib import Path

    from ..extensions import ExtensionRuntime

    rt = ExtensionRuntime.load(
        paths=[Path(p) for p in extra_paths if Path(p).is_dir()],
        extra_paths=[Path(p) for p in extra_paths if Path(p).is_file()],
    )
    names = [e.name for e in rt.extensions]
    if names and os.environ.get("PROVER_EXTENSIONS_LOADED") is None:
        print(f"extensions loaded: {', '.join(names)}")
    os.environ["PROVER_EXTENSIONS_LOADED"] = "1"
