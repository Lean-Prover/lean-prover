"""agent.permissions — per-tool ACL + stable IDs tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.permissions import PermissionStore


def _store(tmp_path: Path) -> PermissionStore:
    return PermissionStore(path=tmp_path / "permissions.json")


def test_default_mode_is_ask(tmp_path) -> None:
    s = _store(tmp_path)
    assert s.mode == "ask"
    # Not persisted yet → the user never opted in; only explicit rules block.
    assert s.enforced is False
    assert s.check("prove_theorem") == "allow"
    # Once persisted, the ask baseline denies by default (an explicit
    # rule can still allow).
    s.save()
    s2 = PermissionStore(path=s.path)
    assert s2.enforced is True
    assert s2.check("prove_theorem") == "deny"


def test_mode_validation(tmp_path) -> None:
    s = _store(tmp_path)
    with pytest.raises(ValueError):
        s.mode = "nope"
    s.mode = "auto"
    assert s.check("anything") == "allow"
    s.mode = "yolo"
    assert s.check("anything") == "allow"


def test_rule_lookup_exact(tmp_path) -> None:
    s = _store(tmp_path)
    s.add_rule("prove_theorem", "", allow=True)
    s.save()
    assert s.check("prove_theorem") == "allow"
    # unrelated tool still defaults to deny under an enforced ask baseline
    assert s.check("other_tool") == "deny"


def test_rule_with_pattern(tmp_path) -> None:
    s = _store(tmp_path)
    s.add_rule("prove_theorem", '"difficulty": "easy"', allow=True)
    s.save()
    assert s.check("prove_theorem", '{"difficulty": "easy"}') == "allow"
    assert s.check("prove_theorem", '{"difficulty": "hard"}') == "deny"


def test_most_specific_wins(tmp_path) -> None:
    s = _store(tmp_path)
    s.mode = "auto"
    s.add_rule("prove_theorem", "", allow=False)          # deny all
    s.add_rule("prove_theorem", "hard", allow=True)       # but allow hard
    assert s.check("prove_theorem", "hard problem") == "allow"
    assert s.check("prove_theorem", "easy problem") == "deny"


def test_stable_id_dedup(tmp_path) -> None:
    s = _store(tmp_path)
    rid1 = s.add_rule("prove_theorem", "", allow=True)
    rid2 = s.add_rule("prove_theorem", "", allow=False)  # same (tool, pattern)
    assert rid1 == rid2
    rule = s.get_rule(rid1)
    assert rule["tool"] == "prove_theorem"
    assert rule["allow"] is False


def test_remove_rule(tmp_path) -> None:
    s = _store(tmp_path)
    rid = s.add_rule("prove_theorem", "", allow=True)
    assert s.remove_rule(rid) is True
    assert s.remove_rule(rid) is False
    assert s.get_rule(rid) is None


def test_persist_roundtrip(tmp_path) -> None:
    path = tmp_path / "permissions.json"
    s = PermissionStore(path=path)
    s.mode = "auto"
    rid = s.add_rule("prove_theorem", "easy", allow=True, note="test")
    s.save()

    s2 = PermissionStore(path=path)
    assert s2.mode == "auto"
    assert s2.get_rule(rid)["note"] == "test"
    assert s2.check("prove_theorem", "easy") == "allow"


def test_missing_store_is_fresh(tmp_path) -> None:
    s = _store(tmp_path)
    assert s.rules() == {}
    # No permissions.json → not enforced: baseline modes are advisory and
    # only explicit deny rules block (opt-in ACL).
    assert s.enforced is False
    assert s.check("x") == "allow"


def test_requires_tool_name(tmp_path) -> None:
    s = _store(tmp_path)
    with pytest.raises(ValueError):
        s.add_rule("   ", "", allow=True)


# --------------------------------------------------------------------------- MCP gating


def test_acl_denies_blocks_mcp_tool(monkeypatch, tmp_path) -> None:
    """An explicit deny rule blocks an MCP tool call; ask mode fails closed."""
    from agent import mcp

    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path))
    store = PermissionStore(path=tmp_path / "permissions.json")
    store.add_rule("prove_theorem", "", allow=False)
    store.save()
    assert mcp._acl_denies("prove_theorem", {"statement": "theorem t : True := by trivial"}) is True
    # ask is the default once a store exists: non-allowlisted tools are denied
    # on the MCP surface too (fail-closed), matching the local loop gate.
    assert mcp._acl_denies("problems", {}) is True


def test_acl_denies_open_by_default(monkeypatch, tmp_path) -> None:
    from agent import mcp

    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path))
    assert mcp._acl_denies("prove_theorem", {}) is False


def test_acl_denies_pattern_match(monkeypatch, tmp_path) -> None:
    from agent import mcp

    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path))
    store = PermissionStore(path=tmp_path / "permissions.json")
    store.add_rule("prove_theorem", "hard", allow=False)
    store.mode = "auto"  # non-denied tools stay open
    store.save()
    assert mcp._acl_denies("prove_theorem", {"statement": "a hard theorem"}) is True
    assert mcp._acl_denies("prove_theorem", {"statement": "an easy tree"}) is False


def test_acl_denies_never_crashes_server(monkeypatch, tmp_path) -> None:
    from agent import mcp

    # A corrupt store must fail CLOSED (deny) rather than silently reopening
    # every tool on the MCP surface.
    permissions = tmp_path / "permissions.json"
    permissions.parent.mkdir(parents=True, exist_ok=True)
    permissions.write_text("{not valid json]")
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path))
    assert mcp._acl_denies("prove_theorem", {}) is True


# --------------------------------------------------------------------------- loop ACL gate (step C)


@pytest.mark.anyio
async def test_acl_before_tool_call_blocks_deny_rule(tmp_path) -> None:
    from types import SimpleNamespace

    from agent.permissions import PermissionStore, acl_before_tool_call

    store = PermissionStore(path=tmp_path / "permissions.json")
    store.add_rule("rm_tool", "dangerous", allow=False)
    gate = acl_before_tool_call(store)

    blocked, reason = await gate(
        SimpleNamespace(name="rm_tool", arguments={"path": "dangerous entry"})
    )
    assert blocked is True
    assert "permission denied" in reason

    not_blocked, _ = await gate(SimpleNamespace(name="ls_tool", arguments={}))
    assert not_blocked is False


@pytest.mark.anyio
async def test_acl_before_tool_call_open_by_default(tmp_path) -> None:
    from types import SimpleNamespace

    from agent.permissions import PermissionStore, acl_before_tool_call

    store = PermissionStore(path=tmp_path / "permissions.json")
    gate = acl_before_tool_call(store)
    blocked, _ = await gate(SimpleNamespace(name="anything", arguments={"x": 1}))
    assert blocked is False


@pytest.mark.anyio
async def test_acl_gate_enforces_ask_once_opted_in(tmp_path) -> None:
    """A persisted store with mode=ask denies tools without an allow rule."""
    from types import SimpleNamespace

    from agent.permissions import PermissionStore, acl_before_tool_call

    path = tmp_path / "permissions.json"
    writer = PermissionStore(path=path)
    writer.add_rule("safe_tool", "", allow=True)
    writer.save()  # opts in; baseline mode stays "ask"

    gate = acl_before_tool_call(PermissionStore(path=path))
    blocked, reason = await gate(
        SimpleNamespace(name="unlisted_tool", arguments={})
    )
    assert blocked is True and "permission denied" in reason
    blocked, _ = await gate(SimpleNamespace(name="safe_tool", arguments={}))
    assert blocked is False


@pytest.mark.anyio
async def test_acl_gate_picks_up_rules_written_mid_session(tmp_path) -> None:
    """The gate re-reads the store, so a deny saved by another instance
    (e.g. the TUI's /permissions remember deny) applies immediately."""

    from types import SimpleNamespace

    from agent.permissions import PermissionStore, acl_before_tool_call

    path = tmp_path / "permissions.json"
    gate_store = PermissionStore(path=path)
    gate = acl_before_tool_call(gate_store)

    call = SimpleNamespace(name="prove_theorem", arguments={"statement": "t"})
    blocked, _ = await gate(call)
    assert blocked is False  # nothing denied yet

    # A second instance (TUI) persists a deny rule for this exact tool.
    tui_store = PermissionStore(path=path)
    tui_store.add_rule("prove_theorem", "", allow=False)
    tui_store.save()

    blocked, reason = await gate(call)
    assert blocked is True
    assert "prove_theorem" in reason


# --------------------------------------------------------------------------- rpc/mcp frame hardening


def test_rpc_handle_frame_rejects_non_dict() -> None:
    from agent.rpc import _handle_frame

    out = _handle_frame(["not", "an", "object"])
    assert out["success"] is False
    assert "object" in out["error"]
    assert _handle_frame(42)["success"] is False


def test_rpc_serve_survives_bad_frames() -> None:
    """Valid-JSON non-dict lines get an error response, not a dead server."""
    import io

    from agent.rpc import serve_rpc

    stdin = io.StringIO('{"type":"get_state","id":1}\n[1,2]\n"hi"\n')
    stdout = io.StringIO()
    rc = serve_rpc(stdin=stdin, stdout=stdout)
    assert rc == 0
    lines = [__import__("json").loads(l) for l in stdout.getvalue().splitlines()]
    assert len(lines) == 3
    assert lines[0]["success"] is True and lines[0]["id"] == 1
    assert all(l["success"] is False for l in lines[1:])


def test_mcp_read_message_distinguishes_eof_from_malformed() -> None:
    import io

    from agent.mcp import _MALFORMED, _read_message

    assert _read_message(io.StringIO("")) is None  # EOF
    bad = io.StringIO()
    bad.write("{not json]\n")
    bad.seek(0)
    assert _read_message(bad) is _MALFORMED
    arr = io.StringIO("[1,2]\n")
    assert _read_message(arr) is _MALFORMED  # valid JSON, wrong shape
    empty_line = io.StringIO("\n")
    assert _read_message(empty_line) is _MALFORMED


def test_mcp_server_serves_after_malformed_line() -> None:
    """One bad frame must not kill the session: error reply, then ping works."""
    import io
    import json as _json

    from agent import mcp

    stdin = io.StringIO("garbage\n" + _json.dumps({
        "jsonrpc": "2.0", "id": 2, "method": "ping"}) + "\n")
    stdout = io.StringIO()
    orig_stdin, orig_stdout = mcp.sys.stdin, mcp.sys.stdout
    mcp.sys.stdin, mcp.sys.stdout = stdin, stdout
    try:
        assert mcp.serve() == 0  # stdin exhausted → clean exit
    finally:
        mcp.sys.stdin, mcp.sys.stdout = orig_stdin, orig_stdout
    lines = [_json.loads(l) for l in stdout.getvalue().splitlines()]
    assert len(lines) == 2
    assert lines[0]["error"]["code"] == -32700
    assert lines[1] == {"jsonrpc": "2.0", "id": 2, "result": {}}
