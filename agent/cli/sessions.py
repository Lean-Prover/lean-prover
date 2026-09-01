"""``prover sessions`` and ``prover usage`` commands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .. import session
from ..session_manager import SessionManager
from ..session_stats import calculate_session_stats
from ..session_usage import collect_session_usage, render_usage_dashboard


def cmd_sessions(args: argparse.Namespace) -> int:
    from ..events import format as fmt_event

    # Startup recovery: mark interrupted (killed) runs that were left
    # 'running' as failed so the list reflects reality.
    SessionManager().recover_interrupted()
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


def cmd_usage(args: argparse.Namespace) -> int:
    """Print the token/cost dashboard (/usage parity, Batch 4.3)."""
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
