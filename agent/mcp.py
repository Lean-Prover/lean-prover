"""MCP (Model Context Protocol) server — expose `prove` to any MCP agent.

Zero-dependency implementation of the MCP stdio transport: newline-delimited
JSON-RPC 2.0 (MCP protocol version 2025-03-26). Run via `prover mcp` and
point any MCP client (Claude, opencode, Cursor, ...) at it:

    {"command": "prover", "args": ["mcp"]}

Tools:
- prove_theorem {statement, max_steps?, goal_feedback?} → proof result
- benchmark_score {problems?, max_steps?} → run a subset and report scores
- problems {difficulty?} → list benchmark problems
"""

from __future__ import annotations

import json
import re
import sys

from .loop import Result

SERVER_INFO = {"name": "lean-prover", "version": "0.2.0"}
PROTOCOL_VERSION = "2025-03-26"

TOOLS = [
    {
        "name": "prove_theorem",
        "description": (
            "Write and verify a Lean 4 proof for a theorem statement. "
            "Lean's kernel checks the result — returns verified proof text. "
            "Statement may be given with or without `:= by` / trailing sorry."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "statement": {"type": "string", "description": "Lean theorem statement"},
                "max_steps": {"type": "integer", "default": 20, "minimum": 1},
                "goal_feedback": {"type": "boolean", "default": True},
            },
            "required": ["statement"],
        },
    },
    {
        "name": "benchmark_score",
        "description": (
            "Run part of the 100-problem graded benchmark and report the "
            "score. Useful to gauge agent strength before tackling new proofs."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "problems": {"type": "string", "default": "benchmark/problems.json"},
                "max_steps": {"type": "integer", "default": 20, "minimum": 1},
                "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 100},
            },
        },
    },
    {
        "name": "problems",
        "description": "List benchmark problems, optionally filtered by difficulty tier.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "difficulty": {
                    "type": "string",
                    "enum": ["trivial", "easy", "medium", "hard"],
                },
            },
        },
    },
    {
        "name": "validate_proof",
        "description": (
            "Comparator-style validation: check a Lean file for axiom injection, "
            "statement match and kernel acceptance (Tau RPC-inspired). Uses validate.py."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "lean_code": {"type": "string", "description": "Full Lean file text to validate"},
                "statement": {"type": "string", "description": "Expected theorem signature for match"},
            },
            "required": ["lean_code"],
        },
    },
    {
        "name": "loogle_search",
        "description": "Search Mathlib via Loogle/Moogle (online fallback for retrieval.py keyword index).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Loogle pattern e.g. (?a -> ?b) -> List ?a -> List ?b"},
                "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
    },
]


def _result_json(r: Result, statement: str) -> dict:
    return {
        "proved": r.proved,
        "steps": r.steps,
        "seconds": round(r.seconds, 1),
        "tokens": r.total_tokens,
        "cost_usd": round(r.estimated_cost_usd, 6),
        "proof": r.proof if r.proved else "",
        "statement": statement,
    }


def _validate_args(name: str, args: dict) -> str | None:
    """Validate call args against the tool's inputSchema. None if OK, else error.

    Enforces required fields, JSON types (with a safe int/str coercion for
    clients that send numbers as strings), and the integer minimum/maximum and
    string enum bounds declared in each tool's schema.
    """
    try:
        schema = next(t for t in TOOLS if t["name"] == name)["inputSchema"]
    except (StopIteration, KeyError):
        return f"unknown tool: {name}"
    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if req not in args:
            return f"missing required argument {req!r}"
    for key, spec in props.items():
        if key not in args:
            continue
        value = args[key]
        typ = spec.get("type")
        if typ in ("integer", "number"):
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                return f"argument {key!r} must be a number"
            try:
                num = int(value) if typ == "integer" else float(value)
            except (TypeError, ValueError):
                return f"argument {key!r} must be a number"
            args[key] = num
            if "minimum" in spec and num < spec["minimum"]:
                return f"argument {key!r} must be >= {spec['minimum']}"
            if "maximum" in spec and num > spec["maximum"]:
                return f"argument {key!r} must be <= {spec['maximum']}"
        elif typ == "string":
            if not isinstance(value, str):
                args[key] = str(value)
            if "enum" in spec and args[key] not in spec["enum"]:
                return f"argument {key!r} must be one of {spec['enum']}"
        elif typ == "boolean" and not isinstance(value, bool):
            return f"argument {key!r} must be a boolean"
    return None


def _handle_tool(name: str, args: dict) -> tuple[dict, bool]:
    """Run a tool; return (payload, is_error)."""
    from pathlib import Path

    if name == "prove_theorem":
        from .loop import prove

        statement = args.get("statement", "").strip()
        if not statement:
            return {"error": "statement is required"}, True
        r = prove(
            statement,
            max_steps=int(args.get("max_steps", 20)),
            verbose=False,
            problem_id=re.sub(r"\W+", "_", statement.split()[1])[:40] if len(statement.split()) > 1 else None,
            goal_feedback=bool(args.get("goal_feedback", True)),
        )
        return _result_json(r, statement), False

    if name == "benchmark_score":
        problems_path = Path(args.get("problems", "benchmark/problems.json"))
        if not problems_path.exists():
            return {"error": f"problems file not found: {problems_path}"}, True
        problems = json.loads(problems_path.read_text())[: int(args.get("limit", 5))]
        from .loop import prove

        results = []
        for p in problems:
            r = prove(
                p["statement"],
                max_steps=int(args.get("max_steps", 20)),
                verbose=False,
                problem_id=p["id"],
            )
            results.append({"id": p["id"], "proved": r.proved, "steps": r.steps})
        solved = sum(1 for x in results if x["proved"])
        return {"score": solved, "total": len(results), "results": results}, False

    if name == "problems":
        problems_path = Path("benchmark/problems.json")
        if not problems_path.exists():
            return {"error": "benchmark/problems.json not found"}, True
        problems = json.loads(problems_path.read_text())
        tier = args.get("difficulty")
        if tier:
            problems = [p for p in problems if p["difficulty"] == tier]
        return {"problems": problems}, False

    if name == "validate_proof":
        from pathlib import Path as _P

        from .validate import validate_text

        lean_code = args.get("lean_code", "")
        statement = args.get("statement")
        lean_dir = _P(__file__).resolve().parent.parent / "lean"
        r = validate_text(lean_code, lean_dir, expected_signature=statement)
        return {"ok": r.ok, "reason": r.reason, "axioms_found": r.axioms_found}, r.ok is False

    if name == "loogle_search":
        from .loogle import search_loogle

        try:
            hits = search_loogle(args.get("query", ""), limit=int(args.get("limit", 5)))
            return {"hits": hits}, False
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}, True

    return {"error": f"unknown tool: {name}"}, True


_MALFORMED = object()  # sentinel: line was readable but not valid JSON


def _read_message(stdin) -> dict | None:
    """Next request dict, None on EOF, or _MALFORMED on a bad line.

    Malformed input must not terminate the server (a client that sends one
    bad frame should get an error response, not a dead session).
    """
    line = stdin.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return _MALFORMED
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        return _MALFORMED
    if not isinstance(msg, dict):
        return _MALFORMED
    return msg


def _write_message(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _acl_denies(name: str, args: dict) -> bool:
    """True when an explicit ACL deny rule matches (tool, serialized args).

    MCP tools stay open by default; only an explicit deny rule blocks a call,
    so adding the ACL never breaks working tooling. Exact rules with patterns
    are matched against the JSON-serialised arguments.
    """
    try:
        from .permissions import PermissionStore

        decision = PermissionStore().lookup(name, json.dumps(args, sort_keys=True))
        return decision == "deny"
    except Exception:  # noqa: BLE001 — never let the ACL take the server down
        return False


def serve() -> int:
    """Run the MCP stdio server until stdin closes."""
    stdin = sys.stdin
    while True:
        msg = _read_message(stdin)
        if msg is None:
            return 0
        if msg is _MALFORMED:
            _write_message({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "parse error"},
            })
            continue
        method = msg.get("method")
        rid = msg.get("id")
        params = msg.get("params") or {}

        if method == "initialize":
            _write_message({
                "jsonrpc": "2.0",
                "id": rid,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": SERVER_INFO,
                },
            })
            continue

        if method in ("notifications/initialized", "notifications/cancelled"):
            continue  # notifications: no response

        if method == "tools/list":
            _write_message({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})
            continue

        if method == "tools/call":
            name = params.get("name", "")
            args = params.get("arguments") or {}
            # Per-tool ACL (fx permission slice): a deny rule blocks the call
            # before it runs; ask baseline defaults to deny for sensitive tools.
            if _acl_denies(name, args):
                _write_message({
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(
                            {"error": f"permission denied for tool {name!r}"},
                            indent=2)}],
                        "isError": True,
                    },
                })
                continue
            # Enforce the tool's own schema bounds (required/types/minmax/enum)
            # before dispatch, so out-of-range or malformed args never reach the
            # engine with unvalidated values.
            err = _validate_args(name, args)
            if err is not None:
                _write_message({
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(
                            {"error": err}, indent=2)}],
                        "isError": True,
                    },
                })
                continue
            try:
                payload, is_error = _handle_tool(name, args)
            except Exception as e:  # noqa: BLE001 — report tool crashes as MCP results
                payload, is_error = {"error": str(e)}, True
            _write_message({
                "jsonrpc": "2.0",
                "id": rid,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
                    "isError": is_error,
                },
            })
            continue

        if method == "ping":
            _write_message({"jsonrpc": "2.0", "id": rid, "result": {}})
            continue

        if rid is not None:
            _write_message({
                "jsonrpc": "2.0",
                "id": rid,
                "error": {"code": -32601, "message": f"method not found: {method}"},
            })


def main() -> None:
    # MCP server must not print anything else to stdout.
    sys.exit(serve())


if __name__ == "__main__":
    main()
