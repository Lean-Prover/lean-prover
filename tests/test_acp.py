"""Unit tests for the ACP server protocol handling (agent/acp.py).

The serve loop is driven over in-memory stdin/stdout with a faked ``prove``
so the protocol (initialize / session/new / session/prompt / session/close
and session/update streaming) is exercised offline — no LLM, no Lean.
"""

from __future__ import annotations

import io
import json
import sys

from agent import acp


def _result(proved: bool, proof: str = ""):
    from types import SimpleNamespace

    return SimpleNamespace(proved=proved, proof=proof, steps=1)


def _drive(monkeypatch, script, fake_prove):
    import agent.loop as loop

    monkeypatch.setattr(loop, "prove", fake_prove)
    inp = io.StringIO("".join(json.dumps(m) + "\n" for m in script))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdin", inp)
    monkeypatch.setattr(sys, "stdout", out)
    acp.serve()
    return [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]


def test_initialize_negotiates_protocol(monkeypatch) -> None:
    resp = _drive(monkeypatch,
                  [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": 1,
                               "clientInfo": {"name": "test", "version": "0"}}}],
                  lambda *a, **k: _result(False))
    r = resp[0]
    assert r["id"] == 1
    assert r["result"]["protocolVersion"] == 1
    assert r["result"]["agentInfo"]["name"] == "lean-prover"
    assert r["result"]["agentCapabilities"]["sessionCapabilities"]["close"] is True


def test_session_lifecycle_streams_update_and_ends_turn(monkeypatch) -> None:
    def fake_prove(*a, **k):
        return _result(True, "trivial")

    monkeypatch.setattr(acp.uuid, "uuid4", lambda: "fixed-sid")
    resp = _drive(monkeypatch, [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": 1}},
        {"jsonrpc": "2.0", "id": 2, "method": "session/new",
         "params": {"cwd": "/tmp", "mcpServers": []}},
        {"jsonrpc": "2.0", "id": 3, "method": "session/prompt",
         "params": {"sessionId": "fixed-sid", "prompt": [
             {"type": "text", "text": "theorem t : True := by sorry"}]}},
        {"jsonrpc": "2.0", "id": 4, "method": "session/close",
         "params": {"sessionId": "fixed-sid"}},
    ], fake_prove)

    new_resp = next(r for r in resp if r.get("id") == 2)
    assert new_resp["result"]["sessionId"] == "fixed-sid"
    prompt_resp = next(r for r in resp if r.get("id") == 3)
    assert prompt_resp["result"]["stopReason"] == "end_turn"
    updates = [r for r in resp if r.get("method") == "session/update"]
    assert updates
    chunk = updates[-1]["params"]["update"]
    assert chunk["sessionUpdate"] == "agent_message_chunk"
    assert chunk["content"]["type"] == "text"
    assert "PROVED" in chunk["content"]["text"]
    close_resp = next(r for r in resp if r.get("id") == 4)
    assert close_resp["result"] == {}


def test_handle_prompt_cancelled_stop_reason(monkeypatch) -> None:
    import agent.loop as loop

    monkeypatch.setattr(loop, "prove", lambda *a, **k: _result(False))
    stop = acp._handle_prompt("sid", "theorem t : True := by sorry",
                              {"cancelled": True})
    assert stop == "cancelled"


def test_prompt_text_extracts_text_blocks() -> None:
    blocks = [
        {"type": "text", "text": "theorem t : True := by"},
        {"type": "image", "data": "x", "mimeType": "image/png"},
        {"type": "text", "text": "sorry"},
    ]
    assert acp._prompt_text(blocks) == "theorem t : True := by\nsorry"


def test_unknown_method_returns_error(monkeypatch) -> None:
    resp = _drive(monkeypatch,
                  [{"jsonrpc": "2.0", "id": 9, "method": "nope", "params": {}}],
                  lambda *a, **k: _result(False))
    assert resp[0]["error"]["code"] == -32601
