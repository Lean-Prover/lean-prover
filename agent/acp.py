"""ACP (Agent Client Protocol) server — expose `prove` to ACP editors.

Minimal-but-spec-compliant baseline (agentclientprotocol.com): the methods
every ACP agent MUST support — ``initialize``, ``session/new``,
``session/prompt``, ``session/cancel``, ``session/update`` — plus
``session/close``. A prompt is treated as a theorem statement; the agent
proves it and streams the result back as ``agent_message_chunk`` updates,
then answers the prompt with a stop reason.

Scope note: ``session/prompt`` runs ``prove`` synchronously, so a
``session/cancel`` that arrives mid-proof is honored *after* the turn
(stop reason ``cancelled``) rather than interrupting a running Lean/LLM
call. True pre-emption would need the proof loop in a cancellable thread.
"""

from __future__ import annotations

import json
import sys
import uuid

SERVER_INFO = {"name": "lean-prover", "version": "0.2.0"}
PROTOCOL_VERSION = 1

_MALFORMED = object()  # sentinel: line was readable but not valid JSON


def _read_message(stdin) -> dict | None:
    """Next request dict, None on EOF, or _MALFORMED on a bad line."""
    line = stdin.readline()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return _MALFORMED


def _write(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _notify(sid: str, update: dict) -> None:
    _write({"jsonrpc": "2.0", "method": "session/update",
            "params": {"sessionId": sid, "update": update}})


def _agent_chunk(sid: str, mid: str, text: str) -> None:
    _notify(sid, {"sessionUpdate": "agent_message_chunk",
                  "messageId": mid,
                  "content": {"type": "text", "text": text}})


def _prompt_text(blocks) -> str:
    """Concatenate the text blocks of a prompt's ContentBlock[]."""
    parts = []
    for b in blocks or []:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "\n".join(parts).strip()


def _handle_prompt(sid: str, text: str, state: dict) -> str:
    """Prove `text` and stream the outcome; return the stop reason."""
    from .loop import prove

    mid = str(uuid.uuid4())
    _agent_chunk(sid, mid, f"Proving: {text}")
    try:
        r = prove(text, max_steps=20, verbose=False)
    except Exception as e:  # noqa: BLE001 — report tool crashes to the client
        _agent_chunk(sid, mid, f"error: {e}")
        return "end_turn"
    if state.get("cancelled"):
        return "cancelled"
    if r.proved:
        _agent_chunk(sid, mid, "PROVED ∎\n\n" + (r.proof or ""))
    else:
        _agent_chunk(sid, mid, f"FAILED after {r.steps} steps")
    return "end_turn"


def serve() -> int:
    """Run the ACP stdio server until stdin closes."""
    sessions: dict[str, dict] = {}
    stdin = sys.stdin
    while True:
        msg = _read_message(stdin)
        if msg is None:
            return 0
        if msg is _MALFORMED:
            _write({"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": "parse error"}})
            continue
        method = msg.get("method")
        rid = msg.get("id")
        params = msg.get("params") or {}

        if method == "initialize":
            _write({"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "agentCapabilities": {
                    "sessionCapabilities": {"close": True},
                    "promptCapabilities": {
                        "image": False, "audio": False, "embeddedContext": False},
                },
                "agentInfo": SERVER_INFO,
                "authMethods": [],
            }})
            continue

        if method == "session/new":
            sid = str(uuid.uuid4())
            sessions[sid] = {"cancelled": False}
            _write({"jsonrpc": "2.0", "id": rid, "result": {"sessionId": sid}})
            continue

        if method == "session/prompt":
            sid = params.get("sessionId")
            state = sessions.get(sid)
            if state is None:
                _write({"jsonrpc": "2.0", "id": rid,
                        "error": {"code": -32002, "message": f"unknown session {sid}"}})
                continue
            state["cancelled"] = False
            text = _prompt_text(params.get("prompt"))
            stop = _handle_prompt(sid, text, state)
            _write({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": stop}})
            continue

        if method == "session/cancel":
            sid = params.get("sessionId")
            if sid in sessions:
                sessions[sid]["cancelled"] = True
            continue  # notification — no response

        if method == "session/close":
            sessions.pop(params.get("sessionId"), None)
            _write({"jsonrpc": "2.0", "id": rid, "result": {}})
            continue

        if method == "ping":
            _write({"jsonrpc": "2.0", "id": rid, "result": {}})
            continue

        if rid is not None:
            _write({"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32601, "message": f"method not found: {method}"}})


def main() -> None:
    # ACP server must not print anything else to stdout.
    sys.exit(serve())


if __name__ == "__main__":
    main()
