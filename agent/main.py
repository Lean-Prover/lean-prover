"""CLI entry point: `prover prove ...` / `prover bench ...`."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys

# Unbuffered stdout so background runs (nohup ... > log) show progress live.
sys.stdout.reconfigure(line_buffering=True)

from .loop import prove, prove_best_of


def _warn_model_mismatch() -> None:
    """Fast endpoint sanity check: warn when PROVER_MODEL isn't served, so a
    mistyped model name fails in seconds rather than hanging on completions."""
    import sys

    from . import llm

    hint = llm.validate_model()
    if hint:
        print(f"[model warn] {hint}", file=sys.stderr)
        print("set PROVER_MODEL to one of the served models.", file=sys.stderr)


def cmd_prove(args: argparse.Namespace) -> int:
    from .rendering import create_event_renderer

    _warn_model_mismatch()
    mode = args.output
    if mode in ("json", "transcript"):
        # Stream output through a renderer instead of the default text summary.
        renderer = create_event_renderer(mode)
        r = prove(args.statement, max_steps=args.max_steps,
                  goal_feedback=not args.no_goal_feedback,
                  record_session=not args.no_record,
                  on_event=renderer.render, verbose=False)
        ok = renderer.finish()
        return 0 if ok and r.proved else 1

    print(f"Proving:\n{args.statement}\n")
    if args.n_attempts > 1:
        r = prove_best_of(args.statement, n_attempts=args.n_attempts,
                          max_steps=args.max_steps,
                          goal_feedback=not args.no_goal_feedback,
                          record_session=not args.no_record,
                          full_file=args.full_file, adaptive_steps=args.adaptive)
        print(f"best-of-{len(r.attempts)}: proved={r.proved} attempts={len(r.attempts)}")
    else:
        r = prove(args.statement, max_steps=args.max_steps,
                  goal_feedback=not args.no_goal_feedback,
                  record_session=not args.no_record,
                  full_file=args.full_file, adaptive_steps=args.adaptive)
    print(f"\nproved={r.proved} steps={r.steps} time={r.seconds:.1f}s")
    print(f"tokens: {r.total_tokens} (prompt={r.total_prompt_tokens}, completion={r.total_completion_tokens}) cost≈${r.estimated_cost_usd:.6f}")
    if r.session_path:
        print(f"session: {r.session_path}")
    if r.proved:
        print("\n" + r.proof)
    return 0 if r.proved else 1


def _prove_one(p: dict, max_steps: int, idx: int, total: int, goal_feedback: bool = True,
               record_session: bool = True, skip_hammers: bool = False,
               n_attempts: int = 1, full_file: bool = False,
               adaptive_steps: bool = False, quiet: bool = False) -> tuple[dict, int, float]:
    """Prove a single problem. Returns (result_dict, tokens, cost)."""
    if not quiet:
        print(f"[{idx}/{total}] {p['id']}: {p['statement'][:70]}...")
    r = prove_best_of(p["statement"], n_attempts=n_attempts, max_steps=max_steps,
                      verbose=False, problem_id=p["id"],
                      goal_feedback=goal_feedback, record_session=record_session,
                      skip_hammers=skip_hammers, difficulty=p.get("difficulty"),
                      full_file=full_file, adaptive_steps=adaptive_steps)
    from .problems import statement_hash

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


def cmd_bench(args: argparse.Namespace) -> int:
    from pathlib import Path

    _warn_model_mismatch()
    from .problems import active_problems, load_problems

    problems_all = load_problems(args.problems)
    drifted = _check_frozen_set(args.problems, getattr(args, "set", DEFAULT_SET),
                                problems_all)
    if drifted:
        _report_drift(drifted, getattr(args, "set", DEFAULT_SET))
        return 1
    start = args.start - 1  # 1-indexed for humans
    problems = [p.as_dict() for p in active_problems(problems_all)[start:]]
    results = []
    total_tokens = 0
    total_cost = 0.0

    goal_feedback = not args.no_goal_feedback

    from . import settings

    quiet = bool(settings.get("quiet"))

    if args.parallel and args.parallel > 1:
        print(f"Running {len(problems)} problems in parallel (workers={args.parallel})...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = [
                executor.submit(_prove_one, p, args.max_steps, i, len(problems),
                                goal_feedback, args.record, args.no_hammers, args.n_attempts,
                                args.full_file, args.adaptive, quiet)
                for i, p in enumerate(problems, start + 1)
            ]
            for fut in concurrent.futures.as_completed(futures):
                result, tokens, cost = fut.result()
                results.append(result)
                total_tokens += tokens
                total_cost += cost
    else:
        for i, p in enumerate(problems, start + 1):
            result, tokens, cost = _prove_one(p, args.max_steps, i, len(problems),
                                              goal_feedback, args.record, args.no_hammers,
                                              args.n_attempts, args.full_file, args.adaptive,
                                              quiet)
            results.append(result)
            total_tokens += tokens
            total_cost += cost

    # Sort results by original problem order
    id_order = {p["id"]: i for i, p in enumerate(problems)}
    results.sort(key=lambda r: id_order.get(r["id"], 0))

    solved = sum(1 for r in results if r["proved"])
    print(f"\nScore: {solved}/{len(results)}")
    print(f"Total tokens: {total_tokens}, estimated cost: ${total_cost:.6f}")
    if getattr(args, "validate", False):
        from pathlib import Path as _P

        from .validate import validate_file
        lean_dir = _P(__file__).resolve().parent.parent / "lean"
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
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp import serve

    return serve()


def cmd_formalize(args: argparse.Namespace) -> int:
    from .formalize import formalize

    r = formalize(args.statement, max_attempts=args.max_attempts,
                  model_name=args.model or None)
    if r.ok:
        print(r.statement)
        print(f"\nok: compiles against Mathlib ({r.attempts} attempt(s))")
        return 0
    print(r.statement or "(no statement produced)")
    print(f"\nfailed after {r.attempts} attempt(s): {r.diagnostics[:500]}")
    return 1


def cmd_synth(args: argparse.Namespace) -> int:
    from .synth import main as synth_main

    return synth_main(["--out", args.out, "--count", str(args.count),
                       "--max-steps", str(args.max_steps),
                       "--problems", args.problems,
                       *(["--seeds", args.seeds] if args.seeds else []),
                       *(["--no-hammers"] if args.no_hammers else []),
                       *(["--model", args.model] if args.model else [])])


def cmd_lean_baseline(args: argparse.Namespace) -> int:
    from .lean_baseline import main as baseline_main

    rc = baseline_main(["--problems", args.problems, "--out", args.out,
                        "--tactic", args.tactic,
                        "--timeout", str(args.timeout),
                        "--start", str(args.start)])
    if getattr(args, "validate", False) and rc == 0:
        import json as _j
        from pathlib import Path as _P

        from .validate import validate_file
        lean_dir = _P(__file__).resolve().parent.parent / "lean"
        try:
            data = _j.loads(_P(args.out).read_text())
            results = data.get("results", data) if isinstance(data, dict) else data
            if isinstance(results, dict) and "results" in data:
                results = data["results"]
            for r in results if isinstance(results, list) else []:
                if r.get("proved"):
                    lf = lean_dir / "tmp" / f"Prover_{r.get('id','')}.lean"
                    if not lf.exists():
                        lf = lean_dir / "src" / "Prover.lean"
                    vr = validate_file(lf, lean_dir, expected_signature=r.get("statement",""))
                    if not vr.ok:
                        print(f"  validate fail {r.get('id')}: {vr.reason}")
        except Exception as e:  # noqa: BLE001
            print(f"validate error: {e}")
    return rc


def cmd_lean_synth(args: argparse.Namespace) -> int:
    from .synth_lean import main as synth_main

    extra = ["--templates"] if args.templates else []
    return synth_main(["--report", args.report, "--out", args.out,
                       "--timeout", str(args.timeout), *extra])


def cmd_datagen(args: argparse.Namespace) -> int:
    from .datagen import main as datagen_main

    argv = ["--corpus", args.corpus, "--out", args.out]
    for r in args.report:
        argv += ["--report", r]
    return datagen_main(argv)


def cmd_finetune(args: argparse.Namespace) -> int:
    from .finetune import main as finetune_main

    argv: list[str] = []
    if args.fidelity:
        argv += ["--fidelity", "--chat", args.chat, "--chat-out", args.chat_out,
                 "--report", args.report, "--timeout", str(args.timeout)]
    else:
        argv += ["--prepare", "--sft", args.sft, "--chat", args.chat,
                 "--launcher", args.launcher]
    return finetune_main(argv)


def cmd_tui(args: argparse.Namespace) -> int:
    try:
        from .tui import main as tui_main
    except ImportError:
        print("TUI requires the optional 'textual' dependency: pip install 'lean-prover[tui]'")
        return 1
    extensions = getattr(args, "extensions", []) or []
    if extensions:
        _preload_extensions(extensions)
    tui_main(parallel=args.parallel)
    return 0


def _preload_extensions(extra_paths: list[str]) -> None:
    """Load extension paths ahead of the TUI session (Tau --extensions parity)."""
    from pathlib import Path

    from .extensions import ExtensionRuntime

    rt = ExtensionRuntime.load(
        paths=[Path(p) for p in extra_paths if Path(p).is_dir()],
        extra_paths=[Path(p) for p in extra_paths if Path(p).is_file()],
    )
    names = [e.name for e in rt.extensions]
    if names and os.environ.get("PROVER_EXTENSIONS_LOADED") is None:
        print(f"extensions loaded: {', '.join(names)}")
    os.environ["PROVER_EXTENSIONS_LOADED"] = "1"


def cmd_usage(args: argparse.Namespace) -> int:
    """Print the token/cost dashboard (/usage parity, Batch 4.3)."""
    from pathlib import Path

    from . import session
    from .session_manager import SessionManager
    from .session_usage import collect_session_usage, render_usage_dashboard

    model = os.environ.get("PROVER_MODEL") or None
    arg = (args.id or "").strip().lower()

    if not arg or arg == "all":
        # Aggregate across recorded sessions.
        manager = SessionManager()
        records: list[dict] = []
        count = 0
        for rec in manager.list_sessions():
            if not rec.path or not Path(rec.path).exists():
                continue
            records.extend(session.read_session(Path(rec.path)))
            count += 1
        if not records:
            print(f"No sessions in {session.sessions_dir()}")
            return 0
        usage = collect_session_usage(records, model=model)
        print(f"Usage across {count} session(s)")
        print(render_usage_dashboard(usage))
        return 0

    # Single session by id (allow suffix match, like cmd_sessions).
    path = None
    for sp in session.list_sessions():
        if sp.stem == arg or str(sp).endswith(arg):
            path = sp
            break
    if path is None:
        print(f"session not found: {args.id}")
        return 1
    usage = collect_session_usage(session.read_session(path), model=model)
    print(f"Usage for {path.stem}")
    print(render_usage_dashboard(usage))
    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    from . import session
    from .events import format as fmt_event
    from .session_stats import calculate_session_stats

    sessions = session.list_sessions()
    if not sessions:
        print(f"No sessions in {session.sessions_dir()}")
        return 0

    if args.id:
        path = None
        for sp in sessions:
            if sp.stem == args.id or str(sp).endswith(args.id):
                path = sp
                break
        if path is None:
            print(f"session not found: {args.id}")
            return 1
        for rec in session.read_session(path):
            line = fmt_event(rec)
            if line:
                print(line)
            elif args.raw:
                print(json.dumps(rec, ensure_ascii=False))
        return 0

    # list mode
    if args.limit:
        sessions = sessions[: args.limit]
    for sp in sessions:
        recs = session.read_session(sp)
        start = next((r for r in recs if r.get("event") == "start"), {})
        result = next((r for r in recs if r.get("event") == "result"), {})
        stats = calculate_session_stats(recs)
        status = "✓" if result.get("proved") else ("◼" if result.get("stopped") else "✘")
        pid = start.get("problem_id") or "?"
        steps = result.get("steps", "?")
        secs = result.get("seconds", "?")
        cost_str = f" cost≈${stats.estimated_cost:.2f}" if stats.estimated_cost else ""
        print(f"{status} {sp.stem:<48} {pid!s:<28} steps={steps} {secs}s "
              f"{stats.total_tokens} tokens{cost_str}")
    return 0


BOARD_FILE = "leaderboard.json"
SCHEMA_VERSION = 1
LEADERBOARD_SOLVER = "prove_best_of"
DEFAULT_SET = "core"


def provenance(problems_path: str | None = None) -> dict:
    """Run-provenance stamps for leaderboard/report entries (lean-eval style).

    Pins the exact inputs so scores stay comparable across weeks:
    - benchmark_commit: git SHA of this repo (None outside a checkout)
    - toolchain: the lean/lean-toolchain pin
    - problems_sha256: digest of the problem set actually run
    """
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parent.parent
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
            from .problems import dir_digest

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

    from .problems import check_frozen_set, load_frozen_set

    sp = _Path(problems_path).parent / "sets" / f"{set_name}.toml"
    if not sp.exists():
        return []
    pinned = load_frozen_set(sp)
    return [
        f"{m.id} (frozen rev {m.set_revision}, current {m.actual_revision})"
        for m in check_frozen_set(pinned, problems)
    ]


def _report_drift(drifted: list[str], set_name: str) -> None:
    print(f"[frozen set '{set_name}'] refusing to run: {len(drifted)} pinned "
          "problem(s) drifted from their frozen revision", file=sys.stderr)
    for line in drifted[:20]:
        print(f"  {line}", file=sys.stderr)
    print("  regenerate the frozen set or fix the manifests, then re-run.",
          file=sys.stderr)


def _current_statement_hashes(problems_path: str) -> dict[str, str]:
    """{problem_id: statement_hash} for the active problems at a path."""
    from .problems import active_problems, load_problems, statement_hash

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


def _same_config(entry: dict, config: dict) -> bool:
    return all(
        entry.get(key) == config.get(key)
        for key in ("max_steps", "solver", "problems_sha256", "toolchain")
    )


def _upsert_board(board: list[dict], entry: dict, config: dict) -> list[dict]:
    """Append a run, superseding any older run of the same name+config so names
    stay unique; sort by score descending."""
    name = entry.get("name")
    kept = [
        e for e in board
        if not (e.get("name") == name and _same_config(e, config))
    ]
    kept.append(entry)
    kept.sort(key=lambda e: e.get("score", 0), reverse=True)
    return kept


def _leaderboard_run(args: argparse.Namespace) -> int:
    from datetime import datetime, timezone
    from pathlib import Path

    from .problems import active_problems, load_problems

    problems_all = load_problems(args.problems)
    drifted = _check_frozen_set(args.problems, getattr(args, "set", DEFAULT_SET),
                                problems_all)
    if drifted:
        _report_drift(drifted, getattr(args, "set", DEFAULT_SET))
        return 1

    problems = [p.as_dict() for p in active_problems(problems_all)]
    results = []
    for i, p in enumerate(problems, 1):
        result, _tokens, _cost = _prove_one(p, args.max_steps, i, len(problems))
        results.append(result)

    score = sum(1 for r in results if r["proved"])
    by_tier: dict[str, dict] = {}
    for r in results:
        t = by_tier.setdefault(r.get("difficulty", "?"), {"proved": 0, "total": 0})
        t["total"] += 1
        t["proved"] += int(r["proved"])

    prov = provenance(args.problems)
    config = {
        "max_steps": args.max_steps,
        "solver": LEADERBOARD_SOLVER,
        "problems_sha256": prov.get("problems_sha256"),
        "toolchain": prov.get("toolchain"),
    }
    name = args.name or os.environ.get("PROVER_MODEL", "unknown")
    board = load_board()
    prev = next(
        (e for e in board if e.get("name") == name and _same_config(e, config)),
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
    Path(BOARD_FILE).write_text(json.dumps(board, indent=2) + "\n")
    print(f"\nRecorded: {entry['name']} {score}/{len(results)} → {BOARD_FILE}")
    return 0


def _leaderboard_show(args: argparse.Namespace) -> int:
    board = load_board()
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


def cmd_login(args: argparse.Namespace) -> int:
    from . import cli

    return cli.cmd_login(args)


def cmd_update(args: argparse.Namespace) -> int:
    from . import cli

    return cli.cmd_update(args)


def cmd_rpc(args: argparse.Namespace) -> int:
    """Native Tau-style RPC stdio loop (`prover rpc`).

    Frames: {"type":"get_state","id":1} -> {"id":1,"success":true,"state":{...}}
    Use `prover mcp` for the JSON-RPC 2.0 / MCP surface instead.
    """
    from .rpc import serve_rpc

    return serve_rpc()


def cmd_version(args: argparse.Namespace) -> int:
    from . import cli

    return cli.cmd_version(args)


def cmd_chat(args: argparse.Namespace) -> int:
    from . import cli

    return cli.cmd_chat(args)


def cmd_export(args: argparse.Namespace) -> int:
    from . import cli

    return cli.cmd_export(args)


def load_board() -> list[dict]:
    from pathlib import Path

    p = Path(BOARD_FILE)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def cmd_ask(args: argparse.Namespace) -> int:
    """Single theorem request, script-friendly.

    fx's ``fx ask`` contract: non-interactive by default, JSON output stays on
    stdout (parseable), any human prose goes to stderr so stdout stays clean.
    """
    import json as _json

    r = prove(
        args.statement,
        max_steps=args.max_steps,
        goal_feedback=not args.no_goal_feedback,
        record_session=not args.no_record,
        full_file=args.full_file,
        adaptive_steps=bool(getattr(args, "adaptive", False)),
    )
    payload = {
        "proved": r.proved,
        "steps": r.steps,
        "seconds": round(r.seconds, 1),
        "tokens": r.total_tokens,
        "prompt_tokens": r.total_prompt_tokens,
        "completion_tokens": r.total_completion_tokens,
        "cost_usd": round(r.estimated_cost_usd, 6),
        "session": r.session_path,
        "proof": r.proof if r.proved else None,
    }
    print(_json.dumps(payload, ensure_ascii=False))
    # Human prose never pollutes stdout when --json is being consumed.
    print(
        f"ask: proved={r.proved} steps={r.steps} time={r.seconds:.1f}s "
        f"tokens={r.total_tokens} cost≈${r.estimated_cost_usd:.6f}",
        file=sys.stderr,
    )
    return 0 if r.proved else 1


def cli(argv: list[str] | None = None) -> None:
    ap = _build_parser()

    args = ap.parse_args(argv)
    if args.cmd is None:
        return cmd_tui(argparse.Namespace(parallel=1))
    sys.exit(args.fn(args))


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser; resolved settings set option defaults.

    Defaults come from :func:`agent.settings.get`, the single documented
    precedence chain (env > project settings > user settings > .prover.json
    repo defaults > built-in). Split out from ``cli`` so tests can inspect the
    parsed defaults without running a command (which needs a live endpoint).
    """
    from . import settings

    proj_max_steps = int(settings.get("max_steps"))
    proj_workers = int(settings.get("workers"))

    ap = argparse.ArgumentParser(
        prog="prover",
        description="Lean 4 proof agent. Run `prover` with no arguments to open the interactive TUI.",
    )
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("prove", help="prove a single theorem")
    p.add_argument("statement", help="Lean theorem statement (with proof or sorry)")
    p.add_argument("--max-steps", type=int, default=proj_max_steps)
    p.add_argument("--n-attempts", type=int, default=1,
                   help="best-of-N: run up to N independent attempts (temperature ramp)")
    p.add_argument("--full-file", action="store_true",
                   help="let the model write the whole Lean file (helpers/imports); "
                        "the theorem statement is still enforced")
    p.add_argument("--adaptive", action="store_true",
                   help="extend the step budget when the last step made progress")
    p.add_argument("--no-goal-feedback", action="store_true",
                   help="disable LSP goal-state feedback")
    p.add_argument("--no-record", action="store_true",
                   help="disable JSONL session recording")
    p.add_argument("--output", choices=["text", "json", "transcript"], default="text",
                   help="output mode: text (default summary), json event stream, or transcript")
    p.set_defaults(fn=cmd_prove)

    a = sub.add_parser("ask", help="single theorem request, JSON on stdout (script-friendly)")
    a.add_argument("statement", help="Lean theorem statement (with proof or sorry)")
    a.add_argument("--max-steps", type=int, default=proj_max_steps)
    a.add_argument("--full-file", action="store_true")
    a.add_argument("--adaptive", action="store_true")
    a.add_argument("--no-goal-feedback", action="store_true")
    a.add_argument("--no-record", action="store_true")
    a.set_defaults(fn=cmd_ask)

    b = sub.add_parser("bench", help="run the benchmark suite")
    b.add_argument("--problems", default="benchmark/problems.json")
    b.add_argument("--set", default=DEFAULT_SET,
                   help="frozen set to drift-check against (benchmark/sets/<name>.toml)")
    b.add_argument("--max-steps", type=int, default=proj_max_steps)
    b.add_argument("--start", type=int, default=1, help="resume from problem N (1-indexed)")
    b.add_argument("--report", default=None)
    b.add_argument("--parallel", type=int, default=proj_workers, help="number of parallel workers (default: repo/1, sequential)")
    b.add_argument("--no-goal-feedback", action="store_true",
                   help="disable LSP goal-state feedback")
    b.add_argument("--no-hammers", action="store_true",
                   help="skip the hammer pre-pass (for retries of known failures)")
    b.add_argument("--n-attempts", type=int, default=1,
                   help="best-of-N: up to N independent attempts per problem")
    b.add_argument("--full-file", action="store_true",
                   help="let the model write whole files (helpers/imports) per problem")
    b.add_argument("--adaptive", action="store_true",
                   help="extend the step budget when the last step made progress")
    b.add_argument("--validate", action="store_true",
                    help="Comparator-style validation: check proofs for axiom injection + statement match")
    b.add_argument("--no-record", action="store_false", dest="record",
                    help="disable JSONL session recording")
    b.set_defaults(fn=cmd_bench, record=True)

    m = sub.add_parser("mcp", help="run the MCP (Model Context Protocol) stdio server")
    m.set_defaults(fn=cmd_mcp)

    f = sub.add_parser("formalize", help="autoformalize a natural-language statement to Lean")
    f.add_argument("statement", help="natural-language math statement to formalize")
    f.add_argument("--max-attempts", type=int, default=4,
                   help="max formalization+compile attempts (default 4)")
    f.add_argument("--model", default=None, help="model to use (default: PROVER_MODEL)")
    f.set_defaults(fn=cmd_formalize)

    sy = sub.add_parser("synth-data", help="generate synthetic proof data (JSONL corpus)")
    sy.add_argument("--out", default="synth.jsonl", help="output JSONL (train file derived)")
    sy.add_argument("--count", type=int, default=20, help="number of seed statements")
    sy.add_argument("--seeds", default=None, help="file with one statement per line")
    sy.add_argument("--problems", default="benchmark/problems.json", help="seed problems JSON")
    sy.add_argument("--max-steps", type=int, default=15)
    sy.add_argument("--model", default=None, help="model name (default: PROVER_MODEL)")
    sy.add_argument("--no-hammers", action="store_true", help="skip hammer pre-pass")
    sy.set_defaults(fn=cmd_synth)

    lbn = sub.add_parser("lean-baseline",
                         help="no-LLM baseline: how many problems Lean itself solves "
                              "(one `lake env lean` per problem)")
    lbn.add_argument("--problems", default="benchmark/problems.json")
    lbn.add_argument("--out", default="benchmark/lean_baseline.json")
    lbn.add_argument("--tactic", default="prover_finish",
                     help="native tactic: prover_finish (hammers) | prover_search (bounded search)")
    lbn.add_argument("--timeout", type=int, default=120)
    lbn.add_argument("--start", type=int, default=1, help="resume from problem N (1-indexed)")
    lbn.add_argument("--validate", action="store_true",
                     help="Comparator validation after baseline (axiom injection check)")
    lbn.set_defaults(fn=cmd_lean_baseline)

    sl = sub.add_parser("synth-lean",
                        help="write a Lean-proved corpus JSONL (from a baseline "
                             "report or from verified templates)")
    sl.add_argument("--report", default="benchmark/lean_baseline.json")
    sl.add_argument("--out", default="corpus/lean_proved.jsonl")
    sl.add_argument("--templates", action="store_true",
                    help="compile template statements with the hammer chain "
                         "and keep only the ones Lean proves")
    sl.add_argument("--timeout", type=int, default=120)
    sl.set_defaults(fn=cmd_lean_synth)

    dg = sub.add_parser("datagen",
                        help="generate (statement, tactic) expert data for SFT/RL "
                             "from the Lean-verified corpus + baseline reports")
    dg.add_argument("--corpus", default="corpus/lean_proved.jsonl")
    dg.add_argument("--report", action="append", default=[],
                    help="JSON report (repeatable); default: all committed baseline "
                         "reports under benchmark/")
    dg.add_argument("--out", default="benchmark/train_sft.jsonl")
    dg.set_defaults(fn=cmd_datagen)

    ft = sub.add_parser("finetune",
                        help="LoRA data prep (--prepare, default) or Lean fidelity "
                             "certification of training data (--fidelity)")
    ft.add_argument("--fidelity", action="store_true",
                    help="re-prove every training entry with real Lean; "
                         "write certified subset + report")
    ft.add_argument("--sft", default="benchmark/train_sft.jsonl")
    ft.add_argument("--chat", default="benchmark/train_chat.jsonl")
    ft.add_argument("--chat-out", default="benchmark/train_chat_fidelity.jsonl")
    ft.add_argument("--report", default="benchmark/fidelity_report.json")
    ft.add_argument("--launcher", default="benchmark/finetune_lora.sh")
    ft.add_argument("--timeout", type=int, default=120)
    ft.set_defaults(fn=cmd_finetune)

    t = sub.add_parser("tui", help="interactive terminal UI (browse problems, watch proofs)")
    t.add_argument("-p", "--parallel", type=int, default=proj_workers,
                   help="number of parallel proof workers (default: repo/1)")
    t.add_argument("--extensions", action="append", default=[],
                   help="extension path (file or dir), repeatable (Tau --extensions parity)")
    t.set_defaults(fn=cmd_tui)

    lg = sub.add_parser("login", help="sign in to a provider (openai-codex|anthropic|github-copilot)")
    lg.add_argument("provider", choices=["openai-codex", "anthropic", "github-copilot"])
    lg.set_defaults(fn=cmd_login)

    up = sub.add_parser("update", help="check for a newer lean-prover release")
    up.add_argument("--install", action="store_true")
    up.add_argument("--check", action="store_true", help="force a fresh PyPI check")
    up.add_argument("--version", default=None, help="pin a specific version")
    up.set_defaults(fn=cmd_update)

    vr = sub.add_parser("version", help="print the current version")
    vr.set_defaults(fn=cmd_version)

    rp = sub.add_parser("rpc", help="run the native RPC JSONL stdio loop")
    rp.set_defaults(fn=cmd_rpc)

    ch = sub.add_parser("chat", help="open a CodingSession REPL (Tau chat parity)")
    ch.add_argument("--model", default="gpt-4o")
    ch.add_argument("--provider", default="")
    ch.add_argument("--cwd", default=None)
    ch.add_argument("--session-id", default=None)
    ch.set_defaults(fn=cmd_chat)

    ex = sub.add_parser("export", help="export a recorded session transcript")
    ex.add_argument("session")
    ex.add_argument("--output", default=None)
    ex.add_argument("--format", default=None, choices=["html", "jsonl", "md", "markdown"])
    ex.set_defaults(fn=cmd_export)

    s = sub.add_parser("sessions", help="list/inspect recorded proof sessions (~/.prover/sessions)")
    s.add_argument("id", nargs="?", default=None,
                   help="session id (filename stem) to show in detail")
    s.add_argument("--limit", type=int, default=20, help="max sessions to list")
    s.add_argument("--raw", action="store_true", help="also dump raw JSON records")
    s.set_defaults(fn=cmd_sessions)

    u = sub.add_parser("usage", help="token/cost dashboard for one or all sessions (/usage parity)")
    u.add_argument("id", nargs="?", default=None,
                   help="session id (or 'all' / empty for all sessions)")
    u.set_defaults(fn=cmd_usage)

    lb = sub.add_parser("leaderboard", help="record a benchmark score / show the board")
    lb.add_argument("--run", action="store_true",
                    help="run the benchmark first, then record the score")
    lb.add_argument("--problems", default="benchmark/problems.json")
    lb.add_argument("--set", default=DEFAULT_SET,
                   help="frozen set to drift-check against (benchmark/sets/<name>.toml)")
    lb.add_argument("--max-steps", type=int, default=proj_max_steps)
    lb.add_argument("--name", default=None, help="entry name (default: model or env)")
    lb.add_argument("--show", action="store_true", help="only show current leaderboard")
    lb.set_defaults(fn=cmd_leaderboard)

    return ap


if __name__ == "__main__":
    cli()
