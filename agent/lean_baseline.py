"""Lean-native no-LLM baseline runner (`prover lean-baseline`).

For every problem in a benchmark JSON file we run ONE `lake env lean` per
problem: `import ProverSupport` + the theorem statement + the hammer chain
(`prover_finish`). No LLM, no repair loop, no search — this is the honest
"what does Lean itself solve" floor that the agent must beat.

Output is a per-tier table plus solved/unsolved lists, written to a JSON
report (default `benchmark/lean_baseline.json`). Everything here is Python
only as a driver; every check is Lean.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .lean import check_file

HEADER = "import Mathlib\nimport ProverSupport\n\nopen BigOperators Nat Finset\n\n"
SEARCH_OPTIONS = "set_option maxHeartbeats 0\n\n"


def _signature(statement: str) -> str:
    from .lean import split_signature

    return split_signature(statement)


def build_lean_file(statement: str, tactic: str = "prover_finish",
                    search_budget: int = 1000, search_depth: int = 3) -> str:
    """The single-file check body: header + statement + one native tactic."""
    options = SEARCH_OPTIONS if tactic == "prover_search" else ""
    if tactic == "prover_search" and search_budget != 1000:
        options += f"set_option prover_search.budget {search_budget}\n"
    body = "prover_finish" if tactic == "prover_finish" else f"prover_search {search_depth}"
    return HEADER + options + _signature(statement) + "\n  " + body + "\n"


@dataclass
class Problem:
    id: str
    difficulty: str | None
    statement: str


def load_problems(path: Path) -> list[Problem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("problems", [])
    problems = []
    for p in raw:
        problems.append(Problem(
            id=str(p.get("id", f"p{len(problems) + 1}")),
            difficulty=p.get("difficulty") or p.get("tier") or "unknown",
            statement=p["statement"],
        ))
    return problems


def _safe_id(problem_id: str) -> str:
    """Restrict an untrusted problem id to filesystem-safe characters so it
    can never escape the tmp dir via ``..``/``/`` when used in a filename."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", problem_id)
    return cleaned.strip(". ") or "problem"


def run_baseline(
    problems: list[Problem],
    lean_dir: Path,
    tmp_dir: Path,
    tactic: str = "prover_finish",
    timeout: int = 120,
    search_budget: int = 1000,
    search_depth: int = 3,
) -> dict:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    solved: list[dict] = []
    unsolved: list[dict] = []
    t0 = time.time()
    for i, p in enumerate(problems, 1):
        f = tmp_dir / f"Baseline_{_safe_id(p.id)}.lean"
        f.write_text(build_lean_file(p.statement, tactic, search_budget,
                                     search_depth), encoding="utf-8")
        ok, output = check_file(f, lean_dir, timeout=timeout)
        entry = {
            "id": p.id,
            "difficulty": p.difficulty,
            "statement": p.statement,
            "proved": ok,
            "output": "" if ok else output[-400:],
        }
        (solved if ok else unsolved).append(entry)
        print(f"[{i}/{len(problems)}] {p.id:<28} {'OK' if ok else 'FAIL'}",
              file=sys.stderr)
    return {
        "tactic": tactic,
        "total": len(problems),
        "solved": len(solved),
        "seconds": round(time.time() - t0, 2),
        "search_budget": search_budget if tactic == "prover_search" else None,
        "search_depth": search_depth if tactic == "prover_search" else None,
        "tiers": _tiers(solved, problems),
        "solved_ids": solved,
        "unsolved_ids": unsolved,
    }


def _tiers(solved: list[dict], all_problems: list[Problem]) -> dict[str, dict]:
    totals: dict[str, int] = defaultdict(int)
    for p in all_problems:
        totals[p.difficulty] += 1
    counts: dict[str, int] = defaultdict(int)
    for s in solved:
        counts[s["difficulty"]] += 1
    return {
        d: {"solved": counts[d], "total": totals[d]}
        for d in totals
    }


def print_report(report: dict) -> None:
    print(f"\ntactic:       {report['tactic']}")
    print(f"total:        {report['total']}")
    print(f"solved:       {report['solved']}  ({report['solved'] / report['total']:.0%})")
    print(f"wall time:    {report['seconds']}s")
    print("\nper tier:")
    for tier, v in sorted(report["tiers"].items()):
        pct = v["solved"] / v["total"] if v["total"] else 0
        print(f"  {tier:<10} {v['solved']:>3}/{v['total']:<3} ({pct:.0%})")
    print("\nsolved ids:")
    print("  " + " ".join(s["id"] for s in report["solved_ids"]) or "  (none)")
    print("\nunsolved ids:")
    print("  " + " ".join(s["id"] for s in report["unsolved_ids"]) or "  (none)")


def _source_of(pid: str) -> str:
    """Category of a problem id: 'minif2f_amc12_2001_p2' -> 'amc12'."""
    body = pid
    for prefix in ("minif2f_",):
        if body.startswith(prefix):
            body = body[len(prefix):]
            break
    return body.split("_", 1)[0]


def print_category_breakdown(report: dict) -> None:
    """Per-source solved/total breakdown plus the solved ids in each bucket."""
    by_source: dict[str, dict[str, list[str]]] = {}
    for s in report["solved_ids"]:
        by_source.setdefault(_source_of(s["id"]), {"solved": [], "unsolved": []})["solved"].append(s["id"])
    for s in report["unsolved_ids"]:
        by_source.setdefault(_source_of(s["id"]), {"solved": [], "unsolved": []})["unsolved"].append(s["id"])
    print("\nper source:")
    for src in sorted(by_source):
        bucket = by_source[src]
        n = len(bucket["solved"]) + len(bucket["unsolved"])
        print(f"  {src:<14} {len(bucket['solved']):>3}/{n:<3} "
              f"({len(bucket['solved']) / n:.0%})")
        if bucket["solved"]:
            print(f"      solved: {' '.join(bucket['solved'])}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problems", default="benchmark/problems.json",
                    help="benchmark JSON file (list of {id,difficulty,statement})")
    ap.add_argument("--out", default="benchmark/lean_baseline.json",
                    help="JSON report path")
    ap.add_argument("--tactic", default="prover_finish",
                    help="native tactic to try (prover_finish | prover_search)")
    ap.add_argument("--timeout", type=int, default=120,
                    help="per-problem Lean timeout (seconds)")
    ap.add_argument("--search-budget", type=int, default=1000,
                    help="prover_search node budget (default 1000, e.g. 4000)")
    ap.add_argument("--search-depth", type=int, default=3,
                    help="prover_search depth (default 3, e.g. 4)")
    ap.add_argument("--start", type=int, default=1,
                    help="resume from problem N (1-indexed)")
    ap.add_argument("--report-only", metavar="REPORT_JSON",
                    help="load a finished report and print summary/breakdown")
    args = ap.parse_args(argv)

    from pathlib import Path

    from .loop import LEAN_DIR

    if args.report_only:
        report = json.loads(Path(args.report_only).read_text())
        print_report(report)
        print_category_breakdown(report)
        return 0

    problems = load_problems(Path(args.problems))
    if not problems:
        print("no problems found", file=sys.stderr)
        return 1
    slice_problems = problems[args.start - 1:]

    print(f"baseline: {len(slice_problems)} problems this run, tactic={args.tactic}, "
          f"lean_dir={LEAN_DIR}", file=sys.stderr)
    new_report = run_baseline(slice_problems, LEAN_DIR, LEAN_DIR / "tmp",
                              tactic=args.tactic, timeout=args.timeout,
                              search_budget=args.search_budget,
                              search_depth=args.search_depth)
    out = Path(args.out)

    # Merge this slice into any prior report so `--start N` resumes instead of
    # overwriting the already-solved ids. Newest result wins per id.
    by_id: dict[str, dict] = {}
    prior_seconds = 0.0
    if out.exists():
        try:
            prior = json.loads(out.read_text(encoding="utf-8"))
            for e in prior.get("solved_ids", []):
                by_id[str(e.get("id"))] = e
            for e in prior.get("unsolved_ids", []):
                by_id[str(e.get("id"))] = e
            prior_seconds = float(prior.get("seconds", 0.0))
        except (OSError, json.JSONDecodeError):
            by_id = {}
    for e in new_report["solved_ids"] + new_report["unsolved_ids"]:
        by_id[str(e["id"])] = e
    solved = [e for e in by_id.values() if e["proved"]]
    unsolved = [e for e in by_id.values() if not e["proved"]]
    report = {
        "tactic": new_report["tactic"],
        "total": len(by_id),
        "solved": len(solved),
        "seconds": round(prior_seconds + new_report["seconds"], 2),
        "search_budget": new_report.get("search_budget"),
        "search_depth": new_report.get("search_depth"),
        "tiers": _tiers(solved, problems),
        "solved_ids": sorted(solved, key=lambda e: str(e["id"])),
        "unsolved_ids": sorted(unsolved, key=lambda e: str(e["id"])),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print_report(report)
    print(f"\nreport written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
