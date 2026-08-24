"""Unit tests for the LSP runTactic primitive (agent/lsp.py).

The real `lean --server` process is not spun up here: we drive the RPC
framing and goal formatting against a mocked `_request` (and a trivial fake
process for the send/read plumbing).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import lsp


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> lsp.LeanLSP:
    f = tmp_path / "T.lean"
    f.write_text("theorem t (n : ℕ) : n = n := by\n  sorry\n", encoding="utf-8")
    c = lsp.LeanLSP(tmp_path, f)
    # stub the process plumbing so no subprocess is spawned
    c._opened = True
    c._send = lambda _obj: None  # type: ignore[method-assign]
    c._ensure_started = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(lsp, "LSP_TIMEOUT", 5)
    return c


def test_run_tactic_frames_rpc_call(client: lsp.LeanLSP, monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    def fake_request(method: str, params: dict, timeout: float | None = None) -> dict | None:
        calls.append((method, params))
        if method == "$/lean/rpc/connect":
            return {"result": {"sessionId": "sess-1"}}
        if method == "$/lean/rpc/call":
            assert params["method"] == "Lean.Widget.runTactic"
            inner = params["params"]
            assert inner["tactic"] == "rfl"
            assert inner["position"] == {"line": 2, "character": 4}
            return {"result": {"goals": []}}
        return {"error": "unexpected"}

    monkeypatch.setattr(client, "_request", fake_request)
    client.run_tactic("theorem t (n : ℕ) : n = n := by\n  sorry\n", "rfl",
                      line=2, character=4)
    methods = [m for m, _ in calls]
    assert methods == ["$/lean/rpc/connect", "$/lean/rpc/call"]


def test_run_tactic_formats_result_goals(client: lsp.LeanLSP, monkeypatch) -> None:
    def fake_request(method: str, params: dict, timeout: float | None = None) -> dict | None:
        if method == "$/lean/rpc/connect":
            return {"result": {"sessionId": "sess-1"}}
        if method == "$/lean/rpc/call":
            return {"result": {"goals": [{
                "goalPrefix": "⊢",
                "type": {"text": " n + 0 = n"},
                "hyps": [{"names": ["n"], "type": {"text": "ℕ"}}],
            }]}}
        return {"error": "unexpected"}

    monkeypatch.setattr(client, "_request", fake_request)
    out = client.run_tactic("theorem t (n : ℕ) : n + 0 = n := by\n  sorry\n", "omega")
    assert out == "n : ℕ\n⊢ n + 0 = n"


def test_run_tactic_default_position_end_of_file(client: lsp.LeanLSP, monkeypatch) -> None:
    seen: list[dict] = []

    def fake_request(method: str, params: dict, timeout: float | None = None) -> dict | None:
        if method == "$/lean/rpc/connect":
            return {"result": {"sessionId": "sess-1"}}
        if method == "$/lean/rpc/call":
            seen.append(params["params"])
            return {"result": {"goals": []}}
        return {"error": "unexpected"}

    monkeypatch.setattr(client, "_request", fake_request)
    client.run_tactic("theorem t : True := by\n  sorry\n", "trivial")
    # end of the last non-empty line ("  sorry", line 1)
    assert seen[0]["position"] == {"line": 1, "character": 7}


def test_run_tactic_returns_none_on_rpc_error(client: lsp.LeanLSP, monkeypatch) -> None:
    def fake_request(method: str, params: dict, timeout: float | None = None) -> dict | None:
        return {"error": {"code": -32601, "message": "unknown method"}}

    monkeypatch.setattr(client, "_request", fake_request)
    assert client.run_tactic("theorem t : True := by sorry", "trivial") is None


def test_run_tactic_returns_none_on_null_result(client: lsp.LeanLSP, monkeypatch) -> None:
    def fake_request(method: str, params: dict, timeout: float | None = None) -> dict | None:
        if method == "$/lean/rpc/connect":
            return {"result": {"sessionId": "sess-1"}}
        return {"result": None}

    monkeypatch.setattr(client, "_request", fake_request)
    assert client.run_tactic("theorem t : True := by sorry", "trivial") is None


# ------------------------------------------------------------------ framing


class _ShortReadPipe:
    """Pipe stub whose read() returns at most `chunk` bytes per call,
    mimicking kernel-level partial reads on a raw (bufsize=0) pipe."""

    def __init__(self, data: bytes, chunk: int = 3) -> None:
        self._buf = bytearray(data)
        self._chunk = chunk

    def readline(self) -> bytes:
        i = self._buf.find(b"\n")
        if i < 0:
            out = bytes(self._buf)
            self._buf.clear()
            return out
        out = bytes(self._buf[: i + 1])
        del self._buf[: i + 1]
        return out

    def read(self, n: int) -> bytes:
        n = min(n, self._chunk)
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out


def _frame(payload: dict) -> bytes:
    body = json.dumps(payload).encode()
    return b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body


def test_read_msg_reassembles_short_pipe_reads(client: lsp.LeanLSP) -> None:
    """A body delivered in several short reads must be reassembled before
    JSON parsing (regression: single read(n) truncated mid-frame)."""
    msg = {"jsonrpc": "2.0", "id": 7, "result": {"ok": True}}
    client._proc = SimpleNamespace(stdout=_ShortReadPipe(_frame(msg)))
    assert client._read_msg() == msg


def test_read_msg_returns_none_on_eof_mid_body(client: lsp.LeanLSP) -> None:
    payload = json.dumps({"jsonrpc": "2.0", "id": 7}).encode()
    frame = b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload[:5]
    client._proc = SimpleNamespace(stdout=_ShortReadPipe(frame))
    assert client._read_msg() is None


# ------------------------------------------------------- server restart state


def test_ensure_started_resets_opened_for_new_process(
        client: lsp.LeanLSP, monkeypatch) -> None:
    """After a respawn the new server never saw didOpen, so `_opened` must
    reset (regression: stale flag permanently disabled goal feedback)."""
    assert client._opened is True
    # Restore the real method: the client fixture stubs `_ensure_started`.
    monkeypatch.setattr(client, "_ensure_started",
                        lsp.LeanLSP._ensure_started.__get__(client))
    monkeypatch.setattr(client, "_request", lambda *a, **k: {"result": {}})
    proc = SimpleNamespace(pid=4242, poll=lambda: None)
    monkeypatch.setattr(lsp.subprocess, "Popen", lambda *a, **k: proc)
    assert client._ensure_started() is True
    assert client._opened is False


def test_kill_clears_opened_flag(client: lsp.LeanLSP, monkeypatch) -> None:
    """The timeout path kills the process group; `_opened` must not survive."""
    client._opened = True
    client._proc = SimpleNamespace(pid=123, poll=lambda: None)
    monkeypatch.setattr(lsp.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(lsp.os, "killpg", lambda pgid, sig: None)
    client._kill()
    assert client._opened is False
