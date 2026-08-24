"""Adoptions from the leanprover org survey: axiom audit, diagnostic
priority order, leaderboard provenance stamps."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import lean
from agent import main as main_mod
from agent.validate import ALLOWED_AXIOMS, audit_axioms

# ---- sorry warning detection across Lean message formats ----

def test_has_sorry_warning_matches_both_quote_styles() -> None:
    assert lean.has_sorry_warning("f.lean:6:8: warning: declaration uses 'sorry'")
    assert lean.has_sorry_warning("f.lean:6:8: warning: declaration uses `sorry`")
    assert not lean.has_sorry_warning("f.lean:1:1: error: unknown identifier")


# ---- kernel-level axiom audit (leanprover/comparator) ----

def test_audit_axioms_allowlist_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "Audit_t.lean"
    f.write_text("theorem t : 1 + 1 = 2 := by rfl\n")
    deps = ", ".join(sorted(ALLOWED_AXIOMS))
    monkeypatch.setattr(
        lean, "check_file",
        lambda *_a, **_k: (True, f"'t' depends on axioms: [{deps}]\n"),
    )
    ok, bad, reason = audit_axioms(f, tmp_path, "t")
    assert ok and bad == [] and reason == ""


def test_audit_axioms_rejects_sorry_ax(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "Audit_t.lean"
    f.write_text("theorem t : 1 + 1 = 2 := by rfl\n")
    monkeypatch.setattr(
        lean, "check_file",
        lambda *_a, **_k: (True, "'t' depends on axioms: [propext, sorryAx]\n"),
    )
    ok, bad, _reason = audit_axioms(f, tmp_path, "t")
    assert not ok and bad == ["sorryAx"]


def test_audit_axioms_no_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "Audit_t.lean"
    f.write_text("theorem t : 1 + 1 = 2 := by rfl\n")
    monkeypatch.setattr(
        lean, "check_file",
        lambda *_a, **_k: (True, "'t' does not depend on any axioms\n"),
    )
    ok, bad, reason = audit_axioms(f, tmp_path, "t")
    assert ok and bad == [] and reason == ""


def test_audit_axioms_missing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "Audit_t.lean"
    f.write_text("theorem t : 1 + 1 = 2 := by rfl\n")
    monkeypatch.setattr(lean, "check_file", lambda *_a, **_k: (True, "(no info)\n"))
    ok, _bad, reason = audit_axioms(f, tmp_path, "t")
    assert not ok and reason == "axiom audit output missing (#print axioms)"


def test_audit_probe_file_is_cleaned_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "Audit_t.lean"
    f.write_text("theorem t : 1 + 1 = 2 := by rfl\n")
    seen: list[Path] = []

    def fake(file: Path, _dir: Path, timeout: int = 60) -> tuple[bool, str]:
        seen.append(file)
        return True, "'t' does not depend on any axioms\n"

    monkeypatch.setattr(lean, "check_file", fake)
    audit_axioms(f, tmp_path, "t")
    assert seen and not seen[0].exists()


# ---- diagnostic fix-priority order (leanprover/skills lean-proof) ----

def test_error_report_orders_syntax_before_goals_before_warnings(tmp_path: Path) -> None:
    src = tmp_path / "fake.lean"
    src.write_text("line1\nline2\nline3\nline4\nline5\n")
    output = "\n".join([
        f"{src.name}:5:1: warning: unused variable x",
        f"{src.name}:3:1: error: unsolved goals",
        f"{src.name}:1:1: error: expected token",
        f"{src.name}:2:1: error: type mismatch",
    ])
    report = lean.error_report(tmp_path, output)
    lines = [ln for ln in report.splitlines() if ".lean:" in ln]
    kinds = [ln.split(": ")[-1] for ln in lines]
    assert kinds == ["expected token", "type mismatch", "unsolved goals", "unused variable x"]


def test_error_report_stable_within_rank(tmp_path: Path) -> None:
    src = tmp_path / "fake.lean"
    src.write_text("x\n")
    output = "\n".join([
        f"{src.name}:2:1: error: type mismatch B",
        f"{src.name}:1:1: error: type mismatch A",
    ])
    report = lean.error_report(tmp_path, output)
    idx_a = report.index("type mismatch A")
    idx_b = report.index("type mismatch B")
    assert idx_b < idx_a


# ---- duplicate-proof-body guard (anti omega-loop) ----

def test_loop_rejects_duplicate_proof_body(tmp_path, monkeypatch):
    """Second identical proof body is bounced with accepted=False and never
    rebuilt; a fresh body after it is written normally."""
    from agent import llm as llm_mod
    from agent import prover_loop as loop

    monkeypatch.setenv("PROVER_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("PROVER_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(llm_mod, "context_window_tokens", lambda: 1_000_000)

    writes: list[str] = []

    def fake_check(_file: Path, _dir: Path, timeout: int = 60) -> tuple[bool, str]:
        try:
            writes.append(_file.read_text())
        except OSError:
            writes.append("")
        return False, "fake.lean:1:1: error: unsolved goals\n  ⊢ ℕ\n"

    monkeypatch.setattr(loop.lean, "check_file", fake_check)
    bodies = iter([
        "```lean\n  simp [Nat.succ_mul]\n```",
        "```lean\n  simp [Nat.succ_mul]\n```",  # exact duplicate → rejected
        "```lean\n  induction n with\n  | zero => simp\n  | succ n ih => simp [ih]\n```",
    ])

    def fake_chat(system, messages, temperature=0.2, model_name=None):
        return llm_mod.LLMResponse(content=next(bodies), prompt_tokens=10,
                                   completion_tokens=5, total_tokens=15)

    monkeypatch.setattr(loop.llm, "chat", fake_chat)

    r = loop.prove("theorem t (n : ℕ) : n + 0 = n := by sorry",
                   max_steps=3, verbose=False, goal_feedback=False,
                   record_session=False, skip_hammers=True,
                   on_event=lambda rec: None)
    accepted = [e for e in r.trace if e["event"] == "llm_response"]
    assert [e["accepted"] for e in accepted] == [True, False, True]
    # the duplicate was never compiled again: only 2 distinct file states
    assert len({w for w in writes if w}) == 2


# ---- compile sandboxing (leanprover/comparator, lean-workbench) ----

def test_sandbox_off_by_default(tmp_path: Path) -> None:
    cmd = ["lake", "env", "lean", "x.lean"]
    assert lean.sandbox_cmd(cmd, tmp_path, mode=None) == cmd
    assert lean.sandbox_cmd(cmd, tmp_path, mode="off") == cmd


def test_sandbox_bwrap_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name: "/usr/bin/bwrap")
    cmd = ["lake", "env", "lean", "x.lean"]
    wrapped = lean.sandbox_cmd(cmd, tmp_path, mode="bwrap")
    assert wrapped[-len(cmd):] == cmd
    assert wrapped[0] == "bwrap"
    assert "--unshare-net" in wrapped
    binds = [(wrapped[i], wrapped[i + 1]) for i in range(len(wrapped)) if wrapped[i] == "--bind"]
    assert ("--bind", str(tmp_path / ".lake")) in binds


def test_sandbox_landrun_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name: "/usr/bin/landrun")
    cmd = ["lake", "env", "lean", "x.lean"]
    wrapped = lean.sandbox_cmd(cmd, tmp_path, mode="landrun")
    assert wrapped[0] == "landrun" and wrapped[-len(cmd):] == cmd
    assert str(tmp_path / ".lake") in wrapped


def test_sandbox_missing_tool_degrades(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name: None)
    lean._warned_missing.clear()
    cmd = ["lake", "env", "lean", "x.lean"]
    assert lean.sandbox_cmd(cmd, tmp_path, mode="bwrap") == cmd  # fail-open + warn


# ---- provenance stamps (leanprover/lean-eval) ----

def test_provenance_fields(tmp_path: Path) -> None:
    problems = tmp_path / "problems.json"
    problems.write_text('[]')
    prov = main_mod.provenance(str(problems))
    assert set(prov) == {"benchmark_commit", "toolchain", "problems_sha256"}
    # repo is a git checkout and pins a toolchain
    assert isinstance(prov["benchmark_commit"], str) and len(prov["benchmark_commit"]) >= 7
    assert prov["toolchain"] and "lean-toolchain" in prov["toolchain"] or ":" in (prov["toolchain"] or "")
    import hashlib

    assert prov["problems_sha256"] == hashlib.sha256(problems.read_bytes()).hexdigest()


def test_provenance_handles_missing_problems_file(tmp_path: Path) -> None:
    prov = main_mod.provenance(str(tmp_path / "missing.json"))
    assert prov["problems_sha256"] is None
