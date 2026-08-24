"""RPC server — Tau rpc.py port (Tau 37a9e43 src/tau_coding/rpc.py), lean-adapted.

Two transports coexist:

* Native RPC mode (Tau/Pi parity): line-delimited JSON frames carrying a
  ``type`` plus ``id``, e.g. ``{"type":"get_state","id":1}`` responds with
  ``{"id":1,"success":true,"state":{...}}``. Frames: ``get_state``
  (session/tool snapshot), ``prove`` (run the lean prove loop headless),
  and ``tools/list``. This is the mode exercised by
  ``echo '{"type":"get_state","id":1}' | prover rpc`` from the Phase 15
  exit criteria.
* ``serve()`` delegates to the MCP stdio server — the JSON-RPC 2.0 surface
  that Claude/opencode speak. This is the default so existing clients are
  unaffected.

``serve()`` and ``serve_rpc()`` are explicit so callers pick the transport.
"""

from __future__ import annotations

import json
import sys

from .mcp import serve as mcp_serve


def _handle_frame(req: dict) -> dict:
    """Route one native RPC frame; returns the response dict."""
    if not isinstance(req, dict):
        return {"id": None, "success": False,
                "error": "frame must be a JSON object"}
    rid = req.get("id")
    ftype = req.get("type", "")

    if ftype == "get_state":
        from .paths import ProverPaths
        from .version import current_version

        paths = ProverPaths()
        try:
            from . import session as sess

            session_ids = [sp.stem for sp in sess.list_sessions()]
        except Exception:  # noqa: BLE001
            session_ids = []
        return {
            "id": rid,
            "success": True,
            "state": {
                "version": current_version(),
                "sessions_dir": str(paths.sessions_dir),
                "sessions": session_ids,
            },
        }

    if ftype == "tools/list":
        from .prover_loop import LEAN_DIR
        from .tools import default_tools

        return {
            "id": rid,
            "success": True,
            "tools": [t.name for t in default_tools(LEAN_DIR)],
        }

    if ftype == "prove":
        from .loop import prove

        params = req.get("params") or {}
        statement = str(params.get("statement", "")).strip()
        if not statement:
            return {"id": rid, "success": False, "error": "statement is required"}
        try:
            r = prove(
                statement,
                max_steps=int(params.get("max_steps", 20)),
                verbose=False,
                problem_id=params.get("problem_id"),
                record_session=False,
            )
            return {
                "id": rid,
                "success": True,
                "proved": r.proved,
                "steps": r.steps,
                "proof": r.proof if r.proved else "",
            }
        except Exception as exc:  # noqa: BLE001
            return {"id": rid, "success": False, "error": str(exc)}

    return {"id": rid, "success": False, "error": f"unknown type: {ftype}"}


def serve_rpc(stdin=None, stdout=None) -> int:
    """Run the native Tau-style RPC loop over stdio until stdin closes."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            out = {"id": None, "success": False, "error": "parse error"}
        else:
            out = _handle_frame(req)
        stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        stdout.flush()
    return 0


def serve() -> int:
    """Entry point: MCP transport by default (existing clients unaffected)."""
    return mcp_serve()
