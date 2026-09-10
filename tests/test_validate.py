"""Regression tests for proof-integrity validation (agent/validate.py)."""

from __future__ import annotations

from pathlib import Path

from agent.validate import (
    _find_axiom_injections,
    _sanitize_decl_name,
    audit_file,
)


def test_eval_is_flagged_as_injection() -> None:
    assert "#eval" in _find_axiom_injections("#eval IO.println 1\n")
    assert "#eval" in _find_axiom_injections("theorem t : True := by trivial\n#eval 1+1\n")
    assert _find_axiom_injections("theorem t : True := by trivial\n") == []


def test_axiom_sorry_still_flagged() -> None:
    assert "axiom" in _find_axiom_injections("axiom bad : False\n")
    assert "sorry" in _find_axiom_injections("theorem t : True := by sorry\n")


def test_audit_file_rejects_eval(tmp_path: Path, monkeypatch) -> None:
    import os

    monkeypatch.setenv("PROVER_SKIP_AXIOM_AUDIT", "1")
    f = tmp_path / "P.lean"
    f.write_text("import Mathlib\ntheorem t : True := by trivial\n#eval 1\n")
    # audit_file reads the file; lean_dir only used for kernel audit (skipped)
    r = audit_file(f, tmp_path, expected_signature="theorem t : True")
    assert not r.ok and "#eval" in r.reason
    os.environ.pop("PROVER_SKIP_AXIOM_AUDIT", None)


def test_sanitize_decl_name_blocks_path_traversal() -> None:
    assert "/" not in _sanitize_decl_name("../../etc/passwd")
    assert _sanitize_decl_name("a/b\\c") == "a_b_c"
    assert len(_sanitize_decl_name("x" * 200)) <= 64
    assert _sanitize_decl_name("") == "decl"
