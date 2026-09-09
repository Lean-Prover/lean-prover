"""``prover bench`` — run the benchmark suite (the scoring core).

``prove_best_of`` / ``_prove_one`` are read through the ``agent.main`` facade
at call time so tests that monkeypatch them keep working after the CLI split.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from datetime import datetime, timezone
from pathlib import Path

from .common import _lazy_main, _warn_model_mismatch

_m = _lazy_main()
from .leaderboard import (
    LEADERBOARD_SOLVER,
    SCHEMA_VERSION,
    _check_frozen_set,
    _report_drift,
    _resolve_set_name,
    provenance,
)


def _prove_one(p: dict, max_steps: int, idx: int, total: int, goal_feedback: bool = True,
               record_session: bool = True, skip_hammers: bool = False,
               n_attempts: int = 1, full_file: bool = False,
               adaptive_steps: bool = False, quiet: bool = False) -> tuple[dict, int, float]:
    """Prove a single problem. Returns (result_dict, tokens, cost)."""
    if not quiet:
        print(f"[{idx}/{total}] {p['id']}: {p['statement'][:70]}...")
    r = _m.prove_best_of(p["statement"], n_attempts=n_attempts, max_steps=max_steps,
                         verbose=False, problem_id=p["id"],
                         goal_feedback=goal_feedback, record_session=record_session,
                         skip_hammers=skip_hammers, difficulty=p.get("difficulty"),
                         full_file=full_file, adaptive_steps=adaptive_steps)
    from ..problems import statement_hash

    result = {
        "id": p["id"],
        "difficulty": p.get("difficulty"),
        "statement": p["statement"],
        "statement_hash": statement_hash(p["statement"]),
        "proved": r.proved,
        "steps": r.steps,
        "seconds": round(r.seconds, 1),
        "tokens": r.total_tokens,
        "prompt_tokens": r.total_prompt_tokens,
        "completion_tokens": r.total_completion_tokens,
        "cost_usd": round(r.estimated_cost_usd, 6),
        "session": r.session_path,
        "trace": r.trace,
        "attempts": len(r.attempts),
    }
    cost_str = f" cost≈${r.estimated_cost_usd:.6f}" if r.total_tokens else ""
    if not quiet:
        print(f"    -> {'PROVED' if r.proved else 'FAILED'} in {r.steps} steps ({r.total_tokens} tokens{cost_str})")
    return result, r.total_tokens, r.estimated_cost_usd


def _run_problems(
    problems: list[dict],
    max_steps: int,
    *,
    start: int = 1,
    goal_feedback: bool = True,
    record: bool = True,
    no_hammers: bool = False,
    n_attempts: int = 1,
    full_file: bool = False,
    adaptive: bool = False,
    quiet: bool = False,
    parallel: int = 1,
) -> tuple[list[dict], int, float]:
    """Prove a list of problems (parallel or serial), returning
    (results_sorted, total_tokens, total_cost). Shared by ``cmd_bench`` and
    ``_leaderboard_run`` so the two pipelines cannot drift."""
    results: list[dict] = []
    total_tokens = 0
    total_cost = 0.0

    def _run(i: int, p: dict) -> tuple[dict, int, float]:
        return _m._prove_one(p, max_steps, i, len(problems), goal_feedback, record,
                             no_hammers, n_attempts, full_file, adaptive, quiet)

    if parallel and parallel > 1:
        print(f"Running {len(problems)} problems in parallel (workers={parallel})...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = [
                executor.submit(_run, i, p)
                for i, p in enumerate(problems, start)
            ]
            for fut in concurrent.futures.as_completed(futures):
                result, tokens, cost = fut.result()
                results.append(result)
                total_tokens += tokens
                total_cost += cost
    else:
        for i, p in enumerate(problems, start):
            result, tokens, cost = _run(i, p)
            results.append(result)
            total_tokens += tokens
            total_cost += cost

    # Sort results by original problem order.
    id_order = {p["id"]: i for i, p in enumerate(problems)}
    results.sort(key=lambda r: id_order.get(r["id"], 0))
    return results, total_tokens, total_cost


def cmd_bench(args: argparse.Namespace) -> int:
    _warn_model_mismatch()
    from ..problems import active_problems, load_problems

    problems_all = load_problems(args.problems)
    set_name = _resolve_set_name(args.problems, getattr(args, "set", None))
    if set_name is not None:
        drifted = _check_frozen_set(args.problems, set_name, problems_all)
        if drifted:
            _report_drift(drifted, set_name)
            return 1
    start = args.start - 1  # 1-indexed for humans
    problems = [p.as_dict() for p in active_problems(problems_all)[start:]]

    goal_feedback = not args.no_goal_feedback

    from .. import settings

    quiet = bool(settings.get("quiet"))

    results, total_tokens, total_cost = _run_problems(
        problems, args.max_steps,
        start=start + 1, goal_feedback=goal_feedback, record=not args.no_record,
        no_hammers=args.no_hammers, n_attempts=args.n_attempts,
        full_file=args.full_file, adaptive=args.adaptive, quiet=quiet,
        parallel=args.parallel,
    )

    solved = sum(1 for r in results if r["proved"])
    print(f"\nScore: {solved}/{len(results)}")
    print(f"Total tokens: {total_tokens}, estimated cost: ${total_cost:.6f}")
    if getattr(args, "validate", False):
        from ..prover_loop import _resolve_lean_dir
        from ..validate import validate_file

        lean_dir = _resolve_lean_dir()
        validated = 0
        for r in results:
            if r["proved"]:
                lf = lean_dir / "tmp" / f"Prover_{r['id']}.lean"
                if not lf.exists():
                    lf = lean_dir / "src" / "Prover.lean"
                vr = validate_file(lf, lean_dir, expected_signature=r.get("statement", ""))
                if vr.ok:
                    validated += 1
                else:
                    print(f"  validate fail {r['id']}: {vr.reason}")
                    r["proved"] = False
        print(f"Validated: {validated}/{solved} (comparator)")
        solved = sum(1 for r in results if r["proved"])
        print(f"Validated score: {solved}/{len(results)}")
    if args.report:
        Path(args.report).write_text(json.dumps({
            "score": solved, "total": len(results),
            "total_tokens": total_tokens, "total_cost_usd": round(total_cost, 6),
            "max_steps": args.max_steps, "solver": LEADERBOARD_SOLVER,
            "schema_version": SCHEMA_VERSION,
            "results": results, **provenance(args.problems),
        }, indent=2))
        print(f"Report written to {args.report}")
    results_path = getattr(args, "results", None)
    if results_path:
        write_results(results_path, results, args, provenance(args.problems))
    return 0


def write_results(path: str, results: list[dict], args: argparse.Namespace,
                  prov: dict | None = None) -> None:
    """Write a compact, web-friendly per-problem results document.

    Leaner than ``--report``: per-problem entries carry only the fields the
    site's benchmark page renders (no ``trace``/``attempts`` bloat), wrapped
    with run provenance so the page can attribute scores to a model + commit.
    """
    from .. import llm

    try:
        model = llm.model()
    except Exception:  # noqa: BLE001 — model name is best-effort metadata
        model = None
    compact = [
        {
            "id": r["id"],
            "difficulty": r.get("difficulty"),
            "statement": r["statement"],
            "proved": r["proved"],
            "steps": r["steps"],
            "seconds": r["seconds"],
            "tokens": r["tokens"],
        }
        for r in results
    ]
    Path(path).write_text(json.dumps({
        "schema_version": 1,
        "model": model,
        "solver": LEADERBOARD_SOLVER,
        "max_steps": args.max_steps,
        "date": datetime.now(timezone.utc).date().isoformat(),
        "provenance": prov or {},
        "results": compact,
    }, indent=2))
    print(f"Results written to {path}")
