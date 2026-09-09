"""``prover leaderboard`` — record/show benchmark scores.

``BOARD_FILE`` / ``load_board`` / ``_run_problems`` are read through the
``agent.main`` facade at call time so tests that monkeypatch them keep working
after the CLI split.
"""

from __future__ import annotations

import argparse
import json
import os

from agent import main as _m

BOARD_FILE = "leaderboard.json"

SCHEMA_VERSION = 1

LEADERBOARD_SOLVER = "prove_best_of"

DEFAULT_SET = "core"


def _resolve_set_name(problems_path: str, set_name: str | None) -> str | None:
    """Resolve which frozen set to drift-check, or None to skip the check.

    An explicit ``--set`` always wins. Otherwise the default ``core`` set
    only applies when running the core benchmark file; other problem sets
    (MiniF2F, Putnam, FormalQual) are outside the core set's scope, so the
    check is skipped rather than falsely reporting every pin as drifted.
    """
    if set_name:
        return set_name
    from pathlib import Path as _Path
    if _Path(problems_path).resolve() == _Path("benchmark/problems.json").resolve():
        return DEFAULT_SET
    return None


def provenance(problems_path: str | None = None) -> dict:
    """Run-provenance stamps for leaderboard/report entries (lean-eval style).

    Pins the exact inputs so scores stay comparable across weeks:
    - benchmark_commit: git SHA of this repo (None outside a checkout)
    - toolchain: the lean/lean-toolchain pin
    - problems_sha256: digest of the problem set actually run
    """
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parent.parent.parent
    commit = None
    try:
        import subprocess

        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root,
            stderr=subprocess.DEVNULL, text=True,
        ).strip() or None
    except Exception:  # noqa: BLE001, S110 — provenance is best-effort
        pass
    toolchain = None
    try:
        toolchain = (root / "lean" / "lean-toolchain").read_text().strip() or None
    except OSError:
        pass
    problems_sha256 = None
    if problems_path:
        try:
            from ..problems import dir_digest

            problems_sha256 = dir_digest(problems_path)
        except Exception:  # noqa: BLE001 — provenance is best-effort
            from hashlib import sha256 as _sha

            try:
                problems_sha256 = _sha(
                    _Path(problems_path).read_bytes()).hexdigest()
            except OSError:
                pass
    return {
        "benchmark_commit": commit,
        "toolchain": toolchain,
        "problems_sha256": problems_sha256,
    }


def _check_frozen_set(problems_path: str, set_name: str, problems) -> list[str]:
    """Frozen-set drift messages for the eval set, or [] when clean/absent."""
    from pathlib import Path as _Path

    from ..problems import check_frozen_set, load_frozen_set

    sp = _Path(problems_path).parent / "sets" / f"{set_name}.toml"
    if not sp.exists():
        return []
    pinned = load_frozen_set(sp)
    return [
        f"{m.id} (frozen rev {m.set_revision}, current {m.actual_revision})"
        for m in check_frozen_set(pinned, problems)
    ]


def _report_drift(drifted: list[str], set_name: str) -> None:
    import sys

    print(f"[frozen set '{set_name}'] refusing to run: {len(drifted)} pinned "
          "problem(s) drifted from their frozen revision", file=sys.stderr)
    for line in drifted[:20]:
        print(f"  {line}", file=sys.stderr)
    print("  regenerate the frozen set or fix the manifests, then re-run.",
          file=sys.stderr)


def _current_statement_hashes(problems_path: str) -> dict[str, str]:
    """{problem_id: statement_hash} for the active problems at a path."""
    from ..problems import active_problems, load_problems, statement_hash

    try:
        probs = active_problems(load_problems(problems_path))
    except (OSError, ValueError):
        return {}
    return {pr.id: statement_hash(pr.statement) for pr in probs}


def _current_config(args: argparse.Namespace) -> dict:
    """The config an entry must match to be comparable with the current run."""
    prov = provenance(args.problems)
    return {
        "solver": LEADERBOARD_SOLVER,
        "max_steps": args.max_steps,
        "problems_sha256": prov.get("problems_sha256"),
        "toolchain": prov.get("toolchain"),
        "statement_hashes": _current_statement_hashes(args.problems),
    }


def _entry_reasons(entry: dict, config: dict) -> list[str]:
    """Incompatibility reasons making an entry non-comparable; [] => comparable."""
    reasons: list[str] = []
    if entry.get("schema_version") != SCHEMA_VERSION:
        reasons.append(f"schema_version={entry.get('schema_version')!r} (need {SCHEMA_VERSION})")
    for key in ("max_steps", "solver", "problems_sha256", "toolchain"):
        if entry.get(key) != config.get(key):
            reasons.append(f"{key}={entry.get(key)!r} (need {config.get(key)!r})")
    if entry.get("problems_sha256") == config.get("problems_sha256"):
        hashes = entry.get("statement_hashes") or {}
        current = config.get("statement_hashes") or {}
        drifted = sorted(pid for pid, h in hashes.items() if current.get(pid) != h)
        if drifted:
            reasons.append(f"{len(drifted)} statement(s) drifted: {', '.join(drifted[:5])}")
    return reasons


def _entry_is_comparable(entry: dict, config: dict) -> bool:
    """Whether an entry is comparable with the current run (no reasons)."""
    return not _entry_reasons(entry, config)


def _upsert_board(board: list[dict], entry: dict, config: dict) -> list[dict]:
    """Append a run, superseding any older comparable run of the same name so
    names stay unique; sort by score descending. Uses the same comparability
    rule as the leaderboard view, so a run that would show as stale is never
    silently superseded (and vice-versa)."""
    name = entry.get("name")
    kept = [
        e for e in board
        if not (e.get("name") == name and _entry_is_comparable(e, config))
    ]
    kept.append(entry)
    kept.sort(key=lambda e: e.get("score", 0), reverse=True)
    return kept


def load_board() -> list[dict]:
    from pathlib import Path

    p = Path(_m.BOARD_FILE)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _leaderboard_run(args: argparse.Namespace) -> int:
    from datetime import datetime, timezone
    from pathlib import Path

    from ..problems import active_problems, load_problems

    problems_all = load_problems(args.problems)
    set_name = _resolve_set_name(args.problems, getattr(args, "set", None))
    if set_name is not None:
        drifted = _check_frozen_set(args.problems, set_name, problems_all)
        if drifted:
            _report_drift(drifted, set_name)
            return 1

    problems = [p.as_dict() for p in active_problems(problems_all)]
    results, _tokens, _cost = _m._run_problems(problems, args.max_steps, quiet=True)

    score = sum(1 for r in results if r["proved"])
    by_tier: dict[str, dict] = {}
    for r in results:
        t = by_tier.setdefault(r.get("difficulty", "?"), {"proved": 0, "total": 0})
        t["total"] += 1
        t["proved"] += int(r["proved"])

    prov = provenance(args.problems)
    config = _current_config(args)
    name = args.name or os.environ.get("PROVER_MODEL", "unknown")
    board = _m.load_board()
    prev = next(
        (e for e in board if e.get("name") == name and _entry_is_comparable(e, config)),
        None,
    )
    now = datetime.now(timezone.utc)
    entry = {
        "name": name,
        "score": score,
        "total": len(results),
        "tiers": by_tier,
        "max_steps": args.max_steps,
        "solver": LEADERBOARD_SOLVER,
        "schema_version": SCHEMA_VERSION,
        "result_version": (prev.get("result_version", 0) if prev else 0) + 1,
        "date": now.strftime("%Y-%m-%d"),
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "statement_hashes": {r["id"]: r["statement_hash"] for r in results},
        **prov,
    }
    board = _upsert_board(board, entry, config)
    Path(_m.BOARD_FILE).write_text(json.dumps(board, indent=2) + "\n")
    print(f"\nRecorded: {entry['name']} {score}/{len(results)} → {_m.BOARD_FILE}")
    results_path = getattr(args, "results", None)
    if results_path:
        from .bench import write_results

        write_results(results_path, results, args, prov)
    return 0


def _leaderboard_show(args: argparse.Namespace) -> int:
    board = _m.load_board()
    if not board:
        print("Leaderboard is empty. Run `prover leaderboard --run` after a benchmark.")
        return 0
    config = _current_config(args)
    comparable, stale = [], []
    for e in board:
        reasons = _entry_reasons(e, config)
        (comparable if not reasons else stale).append((e, reasons))

    print(f"{'#':>2} {'name':<28} {'score':>5}  tiers")
    for i, (e, _r) in enumerate(comparable, 1):
        tiers = " ".join(f"{t}:{v['proved']}/{v['total']}"
                         for t, v in sorted(e.get("tiers", {}).items()))
        print(f"{i:>2} {e.get('name','?'):<28} {e.get('score',0):>3}/{e.get('total','?')}  {tiers}")
    if stale:
        print(f"\n{len(stale)} stale / non-comparable entry(ies) excluded from ranking:")
        for e, reasons in stale:
            print(f"  - {e.get('name','?')}: {', '.join(reasons)}")
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    if args.run:
        return _leaderboard_run(args)
    return _leaderboard_show(args)
