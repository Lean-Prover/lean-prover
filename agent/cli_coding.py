"""Coding CLI — Tau cli.py port (Tau 37a9e43 src/tau_coding/cli.py), lean-adapted (dispatch layer).

A thin argparse entry point for the Tau-style coding facade:

    prover-coding chat [--model m] [--provider p] [--session-id id]  coding session REPL
    prover-coding version                                           print current version
    prover-coding login <provider>                                  OAuth sign-in
    prover-coding update [--install]                                release check
    prover-coding rpc                                               RPC/MCP stdio server
    prover-coding export <session.jsonl> [--format f]               transcript export

The full prover CLI (prove/bench/tui/…) lives in ``agent/cli/``; this module is
the small *coding* facade that main.py delegates login/update/version/chat/
export to.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .version import current_version


def cmd_version(args: argparse.Namespace) -> int:
    print(f"lean-prover {current_version()}")
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    from .update_check import check_for_updates
    from .updater import run_updater

    info = check_for_updates(force=args.check)
    if info.error:
        print(f"update check failed: {info.error}", file=sys.stderr)
        return 1
    if info.is_update_available:
        print(f"update available: {info.current_version} -> {info.latest_version}")
        if args.install:
            result = run_updater(version=args.version)
            print(f"{result.command} (exit {result.returncode})")
            print(result.output)
            return 0 if result.ok else 1
        print("run with --install to upgrade")
        return 0
    print(f"lean-prover {info.current_version} is up to date")
    return 0


def cmd_login(args: argparse.Namespace) -> int:
    import asyncio

    provider = args.provider.lower()
    if provider == "openai-codex":
        from .oauth.codex import login_openai_codex

        credential = login_openai_codex()  # blocking browser flow
    elif provider in ("anthropic", "github-copilot"):
        if provider == "anthropic":
            from .oauth.anthropic import login_anthropic as login_async
        else:
            from .oauth.github_copilot import login_github_copilot as login_async
        credential = asyncio.run(login_async())
    else:
        print(f"unknown provider: {provider} (openai-codex|anthropic|github-copilot)", file=sys.stderr)
        return 2
    if credential is None:
        print("sign-in did not complete", file=sys.stderr)
        return 1
    print(f"signed in to {provider}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .session.flat import read_session
    from .session_export import default_session_export_path, export_session

    session_path = Path(args.session)
    if not session_path.exists():
        print(f"session not found: {session_path}", file=sys.stderr)
        return 2
    # Prover sessions are flat event dicts; typed entry files round-trip via jsonl too.
    records = read_session(session_path)
    output = Path(args.output) if args.output else default_session_export_path(session_path)
    export_session(records, output, format=args.format, source=str(session_path))
    print(f"wrote {output}")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """Minimal CodingSession REPL over the harness (Tau cli chat parity)."""
    import asyncio

    from .coding_session import CodingSession, CodingSessionConfig

    async def run() -> int:
        config = CodingSessionConfig(
            model=args.model,
            provider_name=args.provider or "",
            cwd=Path(args.cwd or Path.cwd()),
            session_id=args.session_id,
        )
        session = await CodingSession.load(config)
        print(f"session {session.session_id} · model {session.model_choice.model}")
        try:
            while True:
                try:
                    text = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return 0
                if not text:
                    continue
                if text in ("/quit", "/exit"):
                    return 0
                async for event in session.prompt(text):
                    if hasattr(event, "message") and getattr(event.message, "text", ""):
                        pass  # text streamed through events; TUI owns display
        except KeyboardInterrupt:
            return 0

    return asyncio.run(run())


def cmd_rpc(args: argparse.Namespace) -> int:
    from .rpc import serve

    return serve()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("prover-coding", description="Lean prover coding facade (Tau CLI parity)")
    sub = ap.add_subparsers(dest="command", required=True)

    c = sub.add_parser("chat", help="open a coding session REPL")
    c.add_argument("--model", default="gpt-4o")
    c.add_argument("--provider", default="")
    c.add_argument("--cwd", default=None)
    c.add_argument("--session-id", default=None)
    c.set_defaults(func=cmd_chat)

    v = sub.add_parser("version", help="print the current version")
    v.set_defaults(func=cmd_version)

    u = sub.add_parser("update", help="check for updates")
    u.add_argument("--install", action="store_true")
    u.add_argument("--check", action="store_true", help="force a fresh PyPI check")
    u.add_argument("--version", default=None, help="pin a specific version")
    u.set_defaults(func=cmd_update)

    l = sub.add_parser("login", help="sign in to a provider")
    l.add_argument("provider", choices=["openai-codex", "anthropic", "github-copilot"])
    l.set_defaults(func=cmd_login)

    e = sub.add_parser("export", help="export a recorded session transcript")
    e.add_argument("session")
    e.add_argument("--output", default=None)
    e.add_argument("--format", default=None, choices=["html", "jsonl", "md", "markdown"])
    e.set_defaults(func=cmd_export)

    r = sub.add_parser("rpc", help="run the RPC/MCP stdio server")
    r.set_defaults(func=cmd_rpc)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
