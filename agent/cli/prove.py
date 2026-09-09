"""``prover prove`` and ``prover ask`` commands.

The engine (``prove`` / ``prove_best_of``) is read through the ``agent.main``
facade at call time so tests that monkeypatch ``agent.main.prove`` /
``agent.main.prove_best_of`` keep working after the CLI split.
"""

from __future__ import annotations

import argparse
import json as _json
import sys

from .common import _lazy_main, _warn_model_mismatch

_m = _lazy_main()


def cmd_prove(args: argparse.Namespace) -> int:
    from ..rendering import create_event_renderer

    _warn_model_mismatch()
    mode = args.output
    if mode in ("json", "transcript"):
        # Stream output through a renderer instead of the default text summary.
        renderer = create_event_renderer(mode)
        r = _m.prove(args.statement, max_steps=args.max_steps,
                     goal_feedback=not args.no_goal_feedback,
                     record_session=not args.no_record,
                     full_file=args.full_file, adaptive_steps=args.adaptive,
                     on_event=renderer.render, verbose=False)
        ok = renderer.finish()
        return 0 if ok and r.proved else 1

    print(f"Proving:\n{args.statement}\n")
    if args.n_attempts > 1:
        r = _m.prove_best_of(args.statement, n_attempts=args.n_attempts,
                             max_steps=args.max_steps,
                             goal_feedback=not args.no_goal_feedback,
                             record_session=not args.no_record,
                             full_file=args.full_file, adaptive_steps=args.adaptive)
        print(f"best-of-{len(r.attempts)}: proved={r.proved} attempts={len(r.attempts)}")
    else:
        r = _m.prove(args.statement, max_steps=args.max_steps,
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


def cmd_ask(args: argparse.Namespace) -> int:
    """Single theorem request, script-friendly.

    fx's ``fx ask`` contract: non-interactive by default, JSON output stays on
    stdout (parseable), any human prose goes to stderr so stdout stays clean.
    """
    _warn_model_mismatch()
    r = _m.prove(
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
