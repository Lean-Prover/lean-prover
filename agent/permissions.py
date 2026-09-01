"""Per-tool permission ACL — adapted from fx's permission system.

fx gates every sensitive tool through a permission system with baseline modes
(``ask``/``auto``/``yolo``) and *exact, revocable rules under stable IDs*
(``/permissions remember allow|deny <tool> <arguments-json>``).

lean-prover already has per-*project* trust (``agent/project_trust.py``). This
module adds the orthogonal, simpler slice fx models: a per-*tool* exact rule
with a stable, revocable ID. A rule binds a tool name (and optional argument
substring/pattern) to an explicit allow/deny, so you can precisely approve one
tool without granting the whole project.

Persistence: ``~/.prover/permissions.json`` — a JSON object::

    {
      "mode": "ask",                        # ask | auto | yolo
      "rules": {
        "<stable-id>": {
          "tool": "prove_theorem",
          "pattern": "",                    # optional substring of the args JSON
          "allow": true,                    # or false for deny
          "note": "my favourite hammer"
        }
      }
    }

Stable IDs are deterministic hashes of (tool, pattern) so re-adding the same
rule yields the same ID (fx-style revocable, deduplicated rules).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from .paths import ProverPaths

VALID_MODES = ("ask", "auto", "yolo")


def _stable_id(tool: str, pattern: str) -> str:
    raw = f"{tool}\x00{pattern}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


class PermissionStore:
    """Load/save the ACL. Thread-safe via a module-local lock."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or ProverPaths().config_dir / "permissions.json"
        # No permissions.json on disk → the user never opted into the ACL;
        # baseline modes are advisory and the gate stays open (explicit deny
        # rules still block). A file that exists means the mode is enforced.
        self.enforced = self.path.exists()
        self._data: dict[str, Any] = {"mode": "ask", "rules": {}}
        self._dirty = False
        self._read()

    # ---- persistence ----------------------------------------------------
    def _read(self) -> None:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text())
                if isinstance(data, dict):
                    self._data = data
        except (json.JSONDecodeError, OSError):
            self._data = {"mode": "ask", "rules": {}}
        self._data.setdefault("mode", "ask")
        self._data.setdefault("rules", {})

    def save(self) -> None:
        with _LOCK:
            self._write()
        # Persisting the store is the opt-in moment: baseline modes now
        # govern this instance's decisions as well.
        self.enforced = True

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)

    # ---- accessors ------------------------------------------------------
    @property
    def mode(self) -> str:
        return self._data.get("mode", "ask")

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in VALID_MODES:
            raise ValueError(f"invalid permission mode: {value}")
        self._data["mode"] = value

    def rules(self) -> dict[str, dict[str, Any]]:
        return self._data.get("rules", {})

    def add_rule(self, tool: str, pattern: str, allow: bool, note: str = "") -> str:
        """Add (or overwrite) an exact rule, returning its stable ID."""
        tool = tool.strip()
        if not tool:
            raise ValueError("tool name required")
        rid = _stable_id(tool, pattern)
        self._data.setdefault("rules", {})[rid] = {
            "tool": tool,
            "pattern": pattern,
            "allow": bool(allow),
            "note": note,
        }
        return rid

    def remove_rule(self, rid: str) -> bool:
        rules = self._data.setdefault("rules", {})
        return rules.pop(rid, None) is not None

    def get_rule(self, rid: str) -> dict[str, Any] | None:
        return self._data.get("rules", {}).get(rid)

    def lookup(self, tool: str, args: str = "") -> str | None:
        """Resolve the decision for (tool, args): 'allow' | 'deny' | None.

        Exact (tool, pattern) match where pattern is a substring of the
        serialised args decides. Most-specific (longest pattern) wins on ties;
        when two rules share the longest pattern, **deny wins** so an allow
        rule can never mask an equal-or-shorter deny.
        """
        rules = self._data.get("rules", {})
        best_len = -1
        allow_seen = False
        deny_seen = False
        for r in rules.values():
            if r.get("tool") != tool:
                continue
            pattern = r.get("pattern", "")
            if pattern and pattern not in args:
                continue
            length = len(pattern)
            if length > best_len:
                best_len = length
                allow_seen = bool(r.get("allow"))
                deny_seen = not r.get("allow")
            elif length == best_len:
                allow_seen = allow_seen or bool(r.get("allow"))
                deny_seen = deny_seen or not r.get("allow")
        if best_len < 0:
            return None
        return "deny" if deny_seen else "allow"

    def check(self, tool: str, args: str = "") -> str:
        """Full decision: rule override, else baseline mode.

        Returns 'allow' | 'deny'. yolo always allows; ask defaults to deny
        (an explicit rule can still allow); auto defaults to allow.

        When the store was never persisted (no permissions.json) the user has
        not opted in: baseline modes are advisory and only explicit deny
        rules block, so the gate never breaks tooling by default.
        """
        decision = self.lookup(tool, args)
        if decision is not None:
            return decision
        if not self.enforced:
            # No permissions.json: the user never opted into the ACL. Keep the
            # gate open (so tooling keeps working) but never let that be silent.
            if not _warned_no_acl:
                _warned_no_acl.add(True)
                import sys

                print("[permissions warn] no permissions.json — the tool ACL is "
                      "inactive; every tool runs ungated. Run /permissions or "
                      "create ~/.prover/permissions.json to enforce modes.",
                      file=sys.stderr)
            return "allow"
        mode = self.mode
        if mode in ("yolo", "auto"):
            return "allow"
        return "deny"

    def reload(self) -> None:
        """Re-read the store from disk (another instance may have saved)."""
        self.enforced = self.path.exists()
        self._read()


def acl_before_tool_call(store: PermissionStore | None = None):
    """Return an async ``before_tool_call`` hook for the agent loop (step C).

    ``loop.run_agent_loop`` accepts a ``before_tool_call(call) -> (blocked,
    reason)`` callback invoked before each tool runs. The gate re-reads the
    store on every call so rules saved mid-session (e.g. via the TUI's
    ``/permissions remember deny``) apply immediately.

    Semantics follow :meth:`PermissionStore.check`: an explicit rule decides;
    otherwise baseline modes apply when the user has opted in (a persisted
    permissions.json exists). Without one, only explicit deny rules block.
    """
    import json

    if store is None:
        store = PermissionStore()

    async def gate(call):
        # Another instance (TUI /permissions, CLI) may have written rules
        # since this gate was built — pick them up live. The read is a cheap
        # JSON parse of a small file, once per tool call.
        try:
            mtime = store.path.stat().st_mtime_ns if store.path.exists() else None
        except OSError:
            mtime = None
        if mtime != getattr(store, "_loaded_mtime", None):
            store._loaded_mtime = mtime
            store.reload()
        args = json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
        if store.check(call.name, args) == "deny":
            return True, f"permission denied for tool {call.name!r} by ACL"
        return False, None

    return gate


_LOCK = threading.Lock()
_warned_no_acl: set[bool] = set()
