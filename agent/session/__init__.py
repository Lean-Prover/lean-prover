"""Session entry tree + storage — Tau session/__init__.py port.

Re-exports the typed entry tree, JSONL codec, and storage backends.  Legacy
flat JSONL events (``agent/session.py``) remain the prover-proof path; this
package powers the Tau-fidelity ``CodingSession`` path.
"""

from __future__ import annotations

from .entries import SessionEntry
from .flat import Session, list_sessions, read_session, sessions_dir
from .jsonl import entries_from_json_lines, entry_to_json_line
from .memory import InMemoryStorage
from .storage import JsonlSessionStorage, SessionStorage

__all__ = [
    "InMemoryStorage",
    "JsonlSessionStorage",
    "Session",
    "SessionEntry",
    "SessionStorage",
    "entries_from_json_lines",
    "entry_to_json_line",
    "list_sessions",
    "read_session",
    "sessions_dir",
]
