"""Lean-native proof corpus generator (`prover synth-lean`).

Two sources, both verified by real Lean:

1. A baseline report (`benchmark/lean_baseline.json`) — JSONL of
   `{"statement", "tactic", "id", "difficulty"}` entries for exactly the
   statements Lean itself proved.
2. Mathlib-flavored statement templates (`--templates N`) — each template
   is compiled with the hammer chain, and ONLY the statements Lean proved
   are written to the corpus. Counts match what Lean closed.

This is the "expert" half of expert iteration with zero model cost. Python
only aggregates; every check is Lean.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# (id, difficulty, statement) — `statement` ends with `:= by\n  sorry`,
# matching the baseline runner's format. Only hammer-provable statements
# are kept; failures are dropped by the real Lean check.
TEMPLATES: list[tuple[str, str, str]] = [
    ("tpl_add_comm_nat", "trivial", "theorem prover_tpl_add_comm_nat (a b : \u2115) : a + b = b + a := by\n  sorry"),
    ("tpl_add_assoc_nat", "trivial", "theorem prover_tpl_add_assoc_nat (a b c : \u2115) : a + b + c = a + (b + c) := by\n  sorry"),
    ("tpl_add_zero_nat", "trivial", "theorem prover_tpl_add_zero_nat (a : \u2115) : a + 0 = a := by\n  sorry"),
    ("tpl_zero_add_nat", "trivial", "theorem prover_tpl_zero_add_nat (a : \u2115) : 0 + a = a := by\n  sorry"),
    ("tpl_mul_comm_nat", "trivial", "theorem prover_tpl_mul_comm_nat (a b : \u2115) : a * b = b * a := by\n  sorry"),
    ("tpl_mul_assoc_nat", "trivial", "theorem prover_tpl_mul_assoc_nat (a b c : \u2115) : a * b * c = a * (b * c) := by\n  sorry"),
    ("tpl_one_mul_nat", "trivial", "theorem prover_tpl_one_mul_nat (a : \u2115) : 1 * a = a := by\n  sorry"),
    ("tpl_mul_one_nat", "trivial", "theorem prover_tpl_mul_one_nat (a : \u2115) : a * 1 = a := by\n  sorry"),
    ("tpl_mul_add_dist_nat", "easy", "theorem prover_tpl_mul_add_dist_nat (a b c : \u2115) : a * (b + c) = a * b + a * c := by\n  sorry"),
    ("tpl_add_mul_dist_nat", "easy", "theorem prover_tpl_add_mul_dist_nat (a b c : \u2115) : (a + b) * c = a * c + b * c := by\n  sorry"),
    ("tpl_add_self_nat", "trivial", "theorem prover_tpl_add_self_nat (a : \u2115) : a + a = 2 * a := by\n  sorry"),
    ("tpl_two_mul_nat", "trivial", "theorem prover_tpl_two_mul_nat (a : \u2115) : 2 * a = a + a := by\n  sorry"),
    ("tpl_add_comm_int", "trivial", "theorem prover_tpl_add_comm_int (a b : \u2124) : a + b = b + a := by\n  sorry"),
    ("tpl_mul_comm_int", "trivial", "theorem prover_tpl_mul_comm_int (a b : \u2124) : a * b = b * a := by\n  sorry"),
    ("tpl_mul_add_dist_int", "easy", "theorem prover_tpl_mul_add_dist_int (a b c : \u2124) : a * (b + c) = a * b + a * c := by\n  sorry"),
    ("tpl_sub_add_cancel_int", "easy", "theorem prover_tpl_sub_add_cancel_int (a b : \u2124) : a - b + b = a := by\n  sorry"),
    ("tpl_diff_squares_int", "medium", "theorem prover_tpl_diff_squares_int (a b : \u2124) : a ^ 2 - b ^ 2 = (a - b) * (a + b) := by\n  sorry"),
    ("tpl_sq_add_int", "medium", "theorem prover_tpl_sq_add_int (a b : \u2124) : (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2 := by\n  sorry"),
    ("tpl_sq_sub_int", "medium", "theorem prover_tpl_sq_sub_int (a b : \u2124) : (a - b) ^ 2 = a ^ 2 - 2 * a * b + b ^ 2 := by\n  sorry"),
    ("tpl_mul_pow_two", "medium", "theorem prover_tpl_mul_pow_two (a b : \u2124) : (a * b) ^ 2 = a ^ 2 * b ^ 2 := by\n  sorry"),
    ("tpl_mul_comm_pow", "easy", "theorem prover_tpl_mul_comm_pow (a b : \u2124) : a ^ 2 * b ^ 2 = b ^ 2 * a ^ 2 := by\n  sorry"),
    ("tpl_neg_mul_int", "easy", "theorem prover_tpl_neg_mul_int (a b : \u2124) : -(a * b) = (-a) * b := by\n  sorry"),
    ("tpl_mul_neg_int", "easy", "theorem prover_tpl_mul_neg_int (a b : \u2124) : a * (-b) = -(a * b) := by\n  sorry"),
    ("tpl_neg_sq_int", "easy", "theorem prover_tpl_neg_sq_int (a : \u2124) : a ^ 2 = (-a) ^ 2 := by\n  sorry"),
    ("tpl_sub_sub_int", "easy", "theorem prover_tpl_sub_sub_int (a b c : \u2124) : a - (b - c) = a - b + c := by\n  sorry"),
    ("tpl_add_sub_assoc_int", "easy", "theorem prover_tpl_add_sub_assoc_int (a b c : \u2124) : a + (b - c) = a + b - c := by\n  sorry"),
    ("tpl_sub_self_int", "trivial", "theorem prover_tpl_sub_self_int (a : \u2124) : a - a = 0 := by\n  sorry"),
    ("tpl_nat_sub_self", "trivial", "theorem prover_tpl_nat_sub_self (n : \u2115) : n - n = 0 := by\n  sorry"),
    ("tpl_nat_add_sub", "easy", "theorem prover_tpl_nat_add_sub (n m : \u2115) : n + m - n = m := by\n  sorry"),
    ("tpl_le_refl_nat", "trivial", "theorem prover_tpl_le_refl_nat (n : \u2115) : n \u2264 n := by\n  sorry"),
    ("tpl_lt_succ_nat", "trivial", "theorem prover_tpl_lt_succ_nat (n : \u2115) : n < n + 1 := by\n  sorry"),
    ("tpl_le_add_self", "easy", "theorem prover_tpl_le_add_self (n m : \u2115) : n \u2264 n + m := by\n  sorry"),
    ("tpl_add_lt_add", "medium", "theorem prover_tpl_add_lt_add (a b c : \u2115) (h : a < b) : a + c < b + c := by\n  sorry"),
    ("tpl_mul_pos", "easy", "theorem prover_tpl_mul_pos (a b : \u2115) (ha : 0 < a) (hb : 0 < b) : 0 < a * b := by\n  sorry"),
    ("tpl_add_pos", "easy", "theorem prover_tpl_add_pos (a b : \u2115) (ha : 0 < a) (hb : 0 < b) : 0 < a + b := by\n  sorry"),
    ("tpl_pow_pos", "easy", "theorem prover_tpl_pow_pos (n : \u2115) : 0 < 2 ^ n := by\n  sorry"),
    ("tpl_sq_nonneg_int", "easy", "theorem prover_tpl_sq_nonneg_int (a : \u2124) : 0 \u2264 a ^ 2 := by\n  sorry"),
    ("tpl_sqsum_nonneg_int", "medium", "theorem prover_tpl_sqsum_nonneg_int (a b : \u2124) : 0 \u2264 a ^ 2 + b ^ 2 := by\n  sorry"),
    ("tpl_factorial_pos", "easy", "theorem prover_tpl_factorial_pos (n : \u2115) : 0 < n.factorial := by\n  sorry"),
    ("tpl_choose_self", "easy", "theorem prover_tpl_choose_self (n : \u2115) : n.choose n = 1 := by\n  sorry"),
    ("tpl_pow_two_add_comm", "easy", "theorem prover_tpl_pow_two_add_comm (a b : \u2124) : a ^ 2 + b ^ 2 = b ^ 2 + a ^ 2 := by\n  sorry"),
    ("tpl_nat_sub_add", "medium", "theorem prover_tpl_nat_sub_add (n m : \u2115) : n - m + m = n := by\n  sorry"),
]


def _sig(statement: str) -> str:
    from .lean import split_signature

    return split_signature(statement)


def generate_templates(
    lean_dir: Path,
    tmp_dir: Path,
    timeout: int = 120,
) -> tuple[list[dict], list[str]]:
    """Compile every template with the hammer chain; return (proven, failed)."""
    from .lean import check_file
    from .lean_baseline import build_lean_file

    tmp_dir.mkdir(parents=True, exist_ok=True)
    proven: list[dict] = []
    failed: list[str] = []
    t0 = time.time()
    for i, (tid, difficulty, statement) in enumerate(TEMPLATES, 1):
        f = tmp_dir / f"Tpl_{tid}.lean"
        f.write_text(build_lean_file(statement), encoding="utf-8")
        ok, _ = check_file(f, lean_dir, timeout=timeout)
        if ok:
            proven.append({
                "id": tid,
                "difficulty": difficulty,
                "statement": statement,
                "tactic": "prover_finish",
            })
        else:
            failed.append(tid)
        print(f"[{i}/{len(TEMPLATES)}] {tid:<28} {'OK' if ok else 'FAIL'}",
              file=sys.stderr)
    print(f"templates: {len(proven)}/{len(TEMPLATES)} proven in "
          f"{round(time.time() - t0, 1)}s", file=sys.stderr)
    return proven, failed


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def corpus_lines(report: dict) -> list[dict]:
    tactic = report.get("tactic", "prover_finish")
    return [
        {
            "id": s.get("id", f"p{i}"),
            "difficulty": s.get("difficulty", "unknown"),
            "statement": s["statement"],
            "tactic": tactic,
        }
        for i, s in enumerate(report.get("solved_ids", []), 1)
    ]


def write_corpus(report: dict, out: Path) -> int:
    entries = corpus_lines(report)
    return write_lines(entries, out)


def write_lines(entries: list[dict], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return len(entries)


def append_lines(entries: list[dict], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return len(entries)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default="benchmark/lean_baseline.json",
                    help="baseline JSON report (default benchmark/lean_baseline.json)")
    ap.add_argument("--out", default="corpus/lean_proved.jsonl",
                    help="output JSONL path")
    ap.add_argument("--templates", action="store_true",
                    help="generate template statements (proven by real Lean) "
                         "instead of copying a report")
    ap.add_argument("--timeout", type=int, default=120,
                    help="per-template Lean timeout (seconds)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if args.templates:
        from .loop import LEAN_DIR

        proven, failed = generate_templates(LEAN_DIR, LEAN_DIR / "tmp",
                                            timeout=args.timeout)
        n = append_lines(proven, out)
        print(f"corpus: {n} template entries appended to {out}",
              file=sys.stderr)
        if failed:
            print(f"failed (excluded): {' '.join(failed)}", file=sys.stderr)
        return 0

    report = load_report(Path(args.report))
    n = write_corpus(report, out)
    print(f"corpus: {n} Lean-proved entries from report "
          f"'{args.report}' (tactic={report.get('tactic')})")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
