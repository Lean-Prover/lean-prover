from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from . import lean

_AXIOM_RE = re.compile(r"\baxiom\b", re.IGNORECASE)
_SORRY_RE = re.compile(r"\bsorry\b", re.IGNORECASE)
_CONCAT_AXIOM_RE = re.compile(r'"\s*ax\s*"\s*\+\+\s*"\s*iom\s*"', re.IGNORECASE)
_ELAB_BYPASS_RE = re.compile(r"\b(elab|elabCommand|runParserCategory)\b")

# Kernel-level audit allowlist (leanprover/comparator trust model): a proof
# may only depend on Lean's three standard axioms. Anything else — sorryAx
# from an imported lemma, Lean.ofReduceBool via native_decide, a custom
# `axiom` — invalidates the result even though `lake build` exits 0.
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
_AXIOM_DEPS_RE = re.compile(r"depends on axioms:\s*\[([^\]]*)\]")
_NO_AXIOMS_RE = re.compile(r"does not depend on any axioms")
_DECL_NAME_RE = re.compile(r"^\s*(?:theorem|lemma|example)\s+([A-Za-z0-9_'.]+)", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    ok: bool
    reason: str
    output: str
    axioms_found: list[str]


def _find_axiom_injections(text: str) -> list[str]:
    hits: list[str] = []
    if _AXIOM_RE.search(text):
        hits.append("axiom")
    if _SORRY_RE.search(text):
        hits.append("sorry")
    if _CONCAT_AXIOM_RE.search(text):
        hits.append("concat-axiom")
    if _ELAB_BYPASS_RE.search(text) and "axiom" in text.lower():
        hits.append("elab-axiom")
    return hits


def audit_axioms(lean_file: Path, lean_dir: Path, name: str, timeout: int = 60) -> tuple[bool, list[str], str]:
    """Kernel-level axiom audit for one declaration.

    Appends `#print axioms <name>` to the (already compiling) file and reads
    the dependency set Lean reports. Returns (ok, offending_axioms, reason).
    ok means every reported axiom is in ALLOWED_AXIOMS.
    """
    text = lean_file.read_text(errors="replace")
    tmp_dir = lean_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    probe = tmp_dir / f"Audit_{name}.lean"
    probe.write_text(text + f"\n\n#print axioms {name}\n")
    try:
        proved, output = lean.check_file(probe, lean_dir, timeout=timeout)
        if not proved:
            return False, [], "axiom audit check failed"
        if _NO_AXIOMS_RE.search(output):
            return True, [], ""
        m = _AXIOM_DEPS_RE.search(output)
        if not m:
            return False, [], "axiom audit output missing (#print axioms)"
        axioms = [a.strip() for a in m.group(1).split(",") if a.strip()]
        bad = [a for a in axioms if a not in ALLOWED_AXIOMS]
        return (not bad), bad, ""
    finally:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass


def _statement_mismatch(text: str, expected_signature: str) -> str | None:
    """Type-aware statement check: the file's target declaration must carry the
    expected name AND statement/type — not just the name. Returns a reason
    string on mismatch, else None. Best-effort: exact structural match on the
    normalized declaration (name + binders + type), so a renamed or retyped
    theorem is rejected while whitespace/line-break churn is tolerated.
    """
    expected = expected_signature.strip()
    # Drop the proof body (and any trailing `:= by`) so only the declaration
    # remains; a bare signature with no `:= by` is left intact.
    expected = expected.rsplit(":= by", 1)[0].strip()
    # The declaration keyword (theorem/lemma/example/def) is incidental.
    expected = re.sub(r"^(?:theorem|lemma|example|def)\s+", "", expected)
    norm_expected = re.sub(r"\s+", " ", expected).strip()
    if not norm_expected:
        return None
    name = norm_expected.split(" ", 1)[0]
    norm_text = re.sub(r"\s+", " ", text)
    if norm_expected in norm_text:
        return None
    return f"statement mismatch for {name}: expected '{norm_expected}' not found in file"


def audit_file(lean_file: Path, lean_dir: Path, expected_signature: str | None = None, timeout: int = 60) -> ValidationResult:
    """Proof-integrity audit for a file that already compiled clean.

    The cheap regex injection scan and the type-aware statement check always
    run; the expensive kernel `#print axioms` audit runs unless
    PROVER_SKIP_AXIOM_AUDIT=1 (default STRICT ON). Does NOT re-run the base
    compile — callers do that. ``ok`` means no illegal axiom, injection, or
    statement mismatch was found.
    """
    text = lean_file.read_text(errors="replace") if lean_file.exists() else ""
    injections = _find_axiom_injections(text)
    if injections:
        return ValidationResult(ok=False, reason=f"illegal axiom injection: {','.join(injections)}", output=text[:4000], axioms_found=injections)
    if expected_signature:
        reason = _statement_mismatch(text, expected_signature)
        if reason:
            return ValidationResult(ok=False, reason=reason, output=text[:4000], axioms_found=[])
    if os.environ.get("PROVER_SKIP_AXIOM_AUDIT") == "1":
        return ValidationResult(ok=True, reason="kernel audit skipped (PROVER_SKIP_AXIOM_AUDIT=1)", output="", axioms_found=[])
    # Kernel-level audit: the last declaration is the target theorem (see
    # _split_signature in prover_loop.py). Follows dependencies, so sorryAx
    # reaching through helper lemmas is caught too.
    names = _DECL_NAME_RE.findall(text)
    if names:
        ok, bad, reason = audit_axioms(lean_file, lean_dir, names[-1], timeout=timeout)
        if not ok:
            detail = f": {','.join(bad)}" if bad else (f" ({reason})" if reason else "")
            return ValidationResult(ok=False, reason=f"axiom audit failed{detail}", output="", axioms_found=bad)
    return ValidationResult(ok=True, reason="comparator pass", output="", axioms_found=[])


def validate_file(lean_file: Path, lean_dir: Path, expected_signature: str | None = None, timeout: int = 60) -> ValidationResult:
    text = lean_file.read_text(errors="replace") if lean_file.exists() else ""
    injections = _find_axiom_injections(text)
    if injections:
        return ValidationResult(ok=False, reason=f"illegal axiom injection: {','.join(injections)}", output=text[:4000], axioms_found=injections)
    proved, output = lean.check_file(lean_file, lean_dir, timeout=timeout)
    if not proved:
        return ValidationResult(ok=False, reason="lean check failed", output=output, axioms_found=[])
    if "axiom" in output.lower():
        return ValidationResult(ok=False, reason="axiom in lean output", output=output, axioms_found=["axiom-output"])
    # Type-aware statement check + (skippable) kernel axiom audit.
    return audit_file(lean_file, lean_dir, expected_signature=expected_signature, timeout=timeout)


def validate_text(text: str, lean_dir: Path, expected_signature: str | None = None, timeout: int = 60) -> ValidationResult:
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".lean", mode="w", delete=False, dir=str(lean_dir / "tmp")) as f:
        f.write(text)
        tmp_path = Path(f.name)
    try:
        return validate_file(tmp_path, lean_dir, expected_signature, timeout)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
