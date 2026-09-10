"""Autoformalization: natural language → a compilable Lean theorem.

``formalize()`` asks the model to turn a natural-language math statement into
a Lean 4 theorem (ending in ``:= by sorry``), then *verifies* it against the
pinned Mathlib with ``lean.compile_only``. On failure the diagnostics are fed
back and the model retries (bounded). The proof is left as ``sorry`` — proving
it is the agent loop's job.

This is deliberately honest about scope: we check the *statement* type-checks;
we do not claim the statement captures the intended math.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import lean, llm
from .prover_loop import HEADER

SORRY = " := by\n  sorry"

SYSTEM = """You are an expert at formalizing mathematics into Lean 4 / Mathlib.
Given a natural-language mathematical statement, write it as ONE Lean theorem:
    theorem <name> <binders> : <type> := by
      sorry
Rules:
- Output ONLY the theorem declaration, in a single ```lean code block.
- Use only symbols and definitions available in Mathlib.
- Do NOT write a proof — end with `:= by` followed by `sorry`.
- If the input is ambiguous or unformalizable, say so instead of guessing."""

_NAME_RE = re.compile(r"^(theorem|lemma|example)\s+([A-Za-z0-9_'.]+)")
_PROOF_RE = re.compile(r":=\s*by\b")


@dataclass
class FormalizeResult:
    statement: str
    ok: bool  # statement compiles against Mathlib
    attempts: int
    seconds: float
    diagnostics: str = ""
    history: list[dict] = field(default_factory=list)


def _normalize(text: str, name: str) -> str:
    """Extract a clean theorem declaration ending in `:= by\n  sorry`."""
    code = llm.extract_lean_code(text).strip()
    m = _NAME_RE.match(code)
    if not m:
        return code
    # Rename to a fixed fresh name so repeated formalizations never collide.
    code = code.replace(m.group(0), f"theorem prover_formal_{name}", 1)
    if _PROOF_RE.search(code):
        code = code[:_PROOF_RE.search(code).start()].rstrip()
    return code.rstrip() + SORRY


def formalize(
    nl_statement: str,
    max_attempts: int = 4,
    lean_dir: Path | None = None,
    model_name: str | None = None,
    on_event: object | None = None,
) -> FormalizeResult:
    from .loop import LEAN_DIR

    lean_dir = lean_dir or LEAN_DIR
    t0 = time.time()
    history: list[dict] = []
    diagnostics = ""
    statement = ""

    for attempt in range(1, max_attempts + 1):
        user_msg = (
            f"Formalize this statement into a single Lean 4 theorem:\n"
            f"{nl_statement}\n"
            + (f"\nThe previous attempt failed to compile:\n{diagnostics}\n" if attempt > 1 else "")
        )
        history.append({"role": "user", "content": user_msg})
        resp = llm.chat(SYSTEM, history, temperature=0.2, model_name=model_name)
        if resp.content.startswith("[LLM error"):
            diagnostics = resp.content
            history.append({"role": "assistant", "content": "(no response)"})
            continue
        history.append({"role": "assistant", "content": resp.content})
        statement = _normalize(resp.content, f"{attempt}")
        if not statement or not statement.startswith("theorem"):
            diagnostics = "model did not produce a theorem declaration"
            continue

        tmp_dir = lean_dir / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        tmp = tmp_dir / f"Prover_formal_{attempt}.lean"
        tmp.write_text(HEADER + statement + "\n", encoding="utf-8")
        rc, out = lean.compile_only(tmp, lean_dir, timeout=90)
        tmp.unlink(missing_ok=True)
        if rc == 0:
            return FormalizeResult(statement=statement, ok=True, attempts=attempt,
                                   seconds=time.time() - t0, diagnostics="",
                                   history=history)
        diagnostics = lean.error_report(lean_dir, out, max_diags=3)

    return FormalizeResult(statement=statement, ok=False, attempts=max_attempts,
                           seconds=time.time() - t0, diagnostics=diagnostics,
                           history=history)


def main(argv: list[str] | None = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: prover formalize '<natural language statement>'")
        return 2
    r = formalize(" ".join(args))
    if r.ok:
        print(r.statement)
        print(f"\nok: compiles against Mathlib ({r.attempts} attempt(s))")
        return 0
    print(r.statement or "(no statement produced)")
    print(f"\nfailed after {r.attempts} attempt(s): {r.diagnostics[:500]}")
    return 1
