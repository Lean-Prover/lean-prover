"""Lean LSP client: goal-state feedback via `lean --server` + widget RPC.

The Lean 4 language server exposes interactive goal states through the
`Lean.Widget.getInteractiveGoals` RPC procedure (the same one the VS Code
infoview uses). We drive a short-lived server over stdio JSON-RPC so the
repair loop can feed the model the actual open goals, not just error text.

Falls back to None on any failure — the loop must work with errors alone.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Self
from urllib.parse import quote

LSP_TIMEOUT = float(os.environ.get("PROVER_LSP_TIMEOUT", "90"))


def file_uri(path: Path) -> str:
    return "file://" + quote(str(path))


def _tt_str(tt) -> str:
    """Flatten Lean's TaggedText JSON to plain text."""
    if isinstance(tt, dict):
        if "text" in tt:
            return tt["text"]
        if "tag" in tt:
            return _tt_str(tt["tag"][1])
        if "append" in tt:
            return "".join(_tt_str(c) for c in tt["append"])
        if "group" in tt:
            return _tt_str(tt["group"])
    return ""


def format_goals(result: dict) -> str | None:
    """Render a getInteractiveGoals result as an infoview-style string."""
    goals = result.get("goals") or []
    parts = []
    for g in goals:
        lines = []
        for hyp in g.get("hyps", []):
            names = ", ".join(hyp.get("names", []))
            lines.append(f"{names} : {_tt_str(hyp.get('type', {}))}")
        lines.append(g.get("goalPrefix", "⊢") + _tt_str(g.get("type", {})))
        parts.append("\n".join(lines))
    return "\n\n".join(parts) if parts else None


class LeanLSP:
    """One `lean --server` session bound to one file."""

    def __init__(self, lean_dir: Path, lean_file: Path):
        self.lean_dir = lean_dir
        self.lean_file = lean_file
        self.uri = file_uri(lean_file)
        self.version = 1
        self._next_id = 0
        self._opened = False  # didOpen sent for current server process
        self._proc: subprocess.Popen | None = None
        self._lock = threading.RLock()

    def _send(self, obj: dict) -> None:
        assert self._proc and self._proc.stdin
        data = json.dumps(obj).encode()
        self._proc.stdin.write(b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data)
        self._proc.stdin.flush()

    def _request(self, method: str, params: dict, timeout: float | None = None) -> dict | None:
        """Send a request and block until its response arrives. None on failure."""
        timeout = LSP_TIMEOUT if timeout is None else timeout
        self._next_id += 1
        rid = self._next_id
        try:
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        except (OSError, ValueError):
            return None
        return self._wait_for(rid, timeout)

    def _wait_for(self, rid: int, timeout: float) -> dict | None:
        deadline = threading.Timer(timeout, self._kill)
        deadline.start()
        try:
            while True:
                msg = self._read_msg()
                if msg is None:
                    return None
                if msg.get("id") == rid:
                    return msg
        finally:
            deadline.cancel()

    def _read_exact(self, n: int) -> bytes | None:
        """Read exactly `n` bytes; None on EOF or short read (stream lost)."""
        buf = bytearray()
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _read_msg(self) -> dict | None:
        assert self._proc and self._proc.stdout
        headers = {}
        line = self._proc.stdout.readline()
        if not line:
            return None
        while line.strip():
            k, _, v = line.decode("utf-8", "replace").partition(":")
            headers[k.strip()] = v.strip()
            line = self._proc.stdout.readline()
        try:
            n = int(headers.get("Content-Length", "0"))
            if n <= 0:
                return None
            body = self._read_exact(n)
            if body is None:
                return None
            return json.loads(body)
        except (ValueError, json.JSONDecodeError):
            return None

    def _kill(self) -> None:
        """Kill the entire LSP process group.

        `lake env lean --server` spawns `lean --server`, which spawns one
        `lean --worker` (~1GB) per file. Killing only the direct child
        orphans the rest; killing the session group (start_new_session)
        takes the whole tree down.
        """
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except OSError:
                self._proc.kill()
        # The server is gone: the next one must see didOpen again.
        self._opened = False

    def _ensure_started(self) -> bool:
        if self._proc and self._proc.poll() is None:
            return True
        try:
            self._proc = subprocess.Popen(
                ["lake", "env", "lean", "--server"],
                cwd=self.lean_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                start_new_session=True,  # own process group → killpg in close()
            )
        except OSError:
            return False
        # Fresh server process: any didOpen from the previous one is void.
        self._opened = False
        init = self._request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": file_uri(self.lean_dir),
                "capabilities": {},
            },
        )
        if init is None or "error" in init:
            self.close()
            return False
        try:
            self._send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        except OSError:
            self.close()
            return False
        return True

    def open_file(self) -> bool:
        with self._lock:
            if not self._ensure_started():
                return False
            if not self._opened:
                text = self.lean_file.read_text()
                self._send(
                    {
                        "jsonrpc": "2.0",
                        "method": "textDocument/didOpen",
                        "params": {
                            "textDocument": {
                                "uri": self.uri,
                                "languageId": "lean",
                                "version": self.version,
                                "text": text,
                            }
                        },
                    }
                )
                self.version += 1
                self._opened = True
            return True

    def update(self, text: str) -> None:
        """Push new file contents (didChange) so the next query sees them."""
        with self._lock:
            if not self._ensure_started() or not self._opened:
                return
            self._send(
                {
                    "jsonrpc": "2.0",
                    "method": "textDocument/didChange",
                    "params": {
                        "textDocument": {"uri": self.uri, "version": self.version},
                        "contentChanges": [{"text": text}],
                    },
                }
            )
            self.version += 1

    def goals(self, line: int, character: int) -> str | None:
        """Goal state at a 0-indexed LSP position, or None if unavailable."""
        with self._lock:
            if not self.open_file():
                return None
            td = {"uri": self.uri}
            pos = {"line": line, "character": character}
            resp = self._request(
                "$/lean/rpc/connect", {"uri": self.uri}
            )
            if resp is None or "error" in resp:
                return None
            sid = resp.get("result", {}).get("sessionId")
            inner = {"textDocument": td, "position": pos}
            resp = self._request(
                "$/lean/rpc/call",
                {
                    "sessionId": sid,
                    "method": "Lean.Widget.getInteractiveGoals",
                    "textDocument": td,
                    "position": pos,
                    "params": inner,
                },
            )
            if resp is None or "error" in resp or resp.get("result") is None:
                return None
            return format_goals(resp["result"])

    def goal_at_end(self, text: str) -> str | None:
        """Open goal state for `text` (already updated to server).

        Tries, in order: end of last non-empty line, end of the `:= by`
        signature line (the initial goal), so that `sorry` bodies still yield
        a goal state rather than nothing.
        """
        lines = text.splitlines()
        idx = len(lines) - 1
        while idx >= 0 and not lines[idx].strip():
            idx -= 1

        candidates = []
        if idx >= 0:
            candidates.append((idx, len(lines[idx])))
        for i, ln in enumerate(lines):
            if ":=" in ln and "by" in ln:
                candidates.append((i, len(ln)))
        for line, char in candidates:
            goals = self.goals(line, char)
            if goals:
                return goals
        return None

    def run_tactic(
        self,
        text: str,
        tactic: str,
        line: int | None = None,
        character: int | None = None,
    ) -> str | None:
        """Apply `tactic` virtually and return the resulting goal state.

        Uses the Lean infoview's ``Lean.Widget.runTactic`` RPC: the tactic is
        run against a snapshot of the goal at the given position (default: end
        of the last non-empty line) without modifying the file. Returns the
        formatted goals, or None on any failure — callers must treat it as
        best-effort, exactly like `goal_at_end`.

        NOTE: this RPC only exists in Lean ≥ v4.22. On our pinned v4.20.0
        toolchain the server answers "No RPC method 'Lean.Widget.runTactic'
        found" and the method returns None (verified against the real server).
        It becomes usable once the toolchain is bumped.
        """
        with self._lock:
            if not self.open_file():
                return None
            self.update(text)
            lines = text.splitlines()
            if line is None:
                line = len(lines) - 1
                while line >= 0 and not lines[line].strip():
                    line -= 1
            line = max(line, 0)
            if character is None:
                character = len(lines[line]) if line < len(lines) else 0

            pos = {"line": line, "character": character}
            td = {"uri": self.uri}
            resp = self._request("$/lean/rpc/connect", {"uri": self.uri})
            if resp is None or "error" in resp:
                return None
            sid = resp.get("result", {}).get("sessionId")
            params = {"textDocument": td, "position": pos, "tactic": tactic}
            resp = self._request(
                "$/lean/rpc/call",
                {
                    "sessionId": sid,
                    "method": "Lean.Widget.runTactic",
                    "textDocument": td,
                    "position": pos,
                    "params": params,
                },
            )
            if resp is None or "error" in resp or resp.get("result") is None:
                return None
            return format_goals(resp["result"])

    def close(self) -> None:
        with self._lock:
            if self._proc is not None:
                self._kill()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Already SIGKILLed; block to reap the zombie so the PID
                    # is not leaked across many sessions.
                    self._proc.wait()
                self._proc = None
            self._opened = False
            self.version = 1

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: S110, BLE001 — never let finalization raise
            pass
