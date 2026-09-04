"""Prover CLI dispatcher + argparse builder.

``cli()`` and ``_build_parser()`` are the (formerly god-class) dispatcher.
They read every command handler through the ``agent.main`` facade at call time
so tests that monkeypatch ``agent.main.cmd_tui`` etc. keep working.
"""

from __future__ import annotations

import argparse
import sys

from agent import main as _m


def cli(argv: list[str] | None = None) -> None:
    ap = _build_parser()

    args = ap.parse_args(argv)
    if args.cmd is None:
        return _m.cmd_tui(argparse.Namespace(parallel=1))
    sys.exit(args.fn(args))


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser; resolved settings set option defaults.

    Defaults come from :func:`agent.settings.get`, the single documented
    precedence chain (env > project settings > user settings > .prover.json
    repo defaults > built-in). Split out from ``cli`` so tests can inspect the
    parsed defaults without running a command (which needs a live endpoint).
    """
    from .. import settings

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
    p.set_defaults(fn=_m.cmd_prove)

    a = sub.add_parser("ask", help="single theorem request, JSON on stdout (script-friendly)")
    a.add_argument("statement", help="Lean theorem statement (with proof or sorry)")
    a.add_argument("--max-steps", type=int, default=proj_max_steps)
    a.add_argument("--full-file", action="store_true")
    a.add_argument("--adaptive", action="store_true")
    a.add_argument("--no-goal-feedback", action="store_true")
    a.add_argument("--no-record", action="store_true")
    a.set_defaults(fn=_m.cmd_ask)

    b = sub.add_parser("bench", help="run the benchmark suite")
    b.add_argument("--problems", default="benchmark/problems.json")
    b.add_argument("--set", default=None,
                   help="frozen set to drift-check against (benchmark/sets/<name>.toml); "
                        "defaults to 'core' only when running the core benchmark")
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
    b.add_argument("--no-record", action="store_true",
                   help="disable JSONL session recording")
    b.set_defaults(fn=_m.cmd_bench)

    m = sub.add_parser("mcp", help="run the MCP (Model Context Protocol) stdio server")
    m.set_defaults(fn=_m.cmd_mcp)

    a = sub.add_parser("acp", help="run the ACP (Agent Client Protocol) stdio server for editors")
    a.set_defaults(fn=_m.cmd_acp)

    f = sub.add_parser("formalize", help="autoformalize a natural-language statement to Lean")
    f.add_argument("statement", help="natural-language math statement to formalize")
    f.add_argument("--max-attempts", type=int, default=4,
                   help="max formalization+compile attempts (default 4)")
    f.add_argument("--model", default=None, help="model to use (default: PROVER_MODEL)")
    f.set_defaults(fn=_m.cmd_formalize)

    sy = sub.add_parser("synth-data", help="generate synthetic proof data (JSONL corpus)")
    sy.add_argument("--out", default="synth.jsonl", help="output JSONL (train file derived)")
    sy.add_argument("--count", type=int, default=20, help="number of seed statements")
    sy.add_argument("--seeds", default=None, help="file with one statement per line")
    sy.add_argument("--problems", default="benchmark/problems.json", help="seed problems JSON")
    sy.add_argument("--max-steps", type=int, default=15)
    sy.add_argument("--model", default=None, help="model name (default: PROVER_MODEL)")
    sy.add_argument("--no-hammers", action="store_true", help="skip hammer pre-pass")
    sy.set_defaults(fn=_m.cmd_synth)

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
    lbn.set_defaults(fn=_m.cmd_lean_baseline)

    sl = sub.add_parser("synth-lean",
                        help="write a Lean-proved corpus JSONL (from a baseline "
                             "report or from verified templates)")
    sl.add_argument("--report", default="benchmark/lean_baseline.json")
    sl.add_argument("--out", default="corpus/lean_proved.jsonl")
    sl.add_argument("--templates", action="store_true",
                    help="compile template statements with the hammer chain "
                         "and keep only the ones Lean proves")
    sl.add_argument("--timeout", type=int, default=120)
    sl.set_defaults(fn=_m.cmd_lean_synth)

    dg = sub.add_parser("datagen",
                        help="generate (statement, tactic) expert data for SFT/RL "
                             "from the Lean-verified corpus + baseline reports")
    dg.add_argument("--corpus", default="corpus/lean_proved.jsonl")
    dg.add_argument("--report", action="append", default=[],
                    help="JSON report (repeatable); default: all committed baseline "
                         "reports under benchmark/")
    dg.add_argument("--out", default="benchmark/train_sft.jsonl")
    dg.set_defaults(fn=_m.cmd_datagen)

    rd = sub.add_parser("rl-data",
                        help="generate GRPO-ready verifier-reward rollouts "
                             "(prompt + per-attempt completions + binary Lean reward)")
    rd.add_argument("--problems", default="benchmark/problems.json")
    rd.add_argument("--out", default="benchmark/train_grpo.jsonl")
    rd.add_argument("--n-attempts", type=int, default=3,
                    help="rollouts (attempts) per problem")
    rd.add_argument("--max-steps", type=int, default=20)
    rd.add_argument("--limit", type=int, default=None,
                    help="only process first N problems")
    rd.set_defaults(fn=_m.cmd_rldata)

    sr = sub.add_parser("search",
                        help="beam/tree search over the repair loop (progress-pruned frontier)")
    sr.add_argument("--problems", default="benchmark/problems.json")
    sr.add_argument("--beam", type=int, default=3, help="beam width (frontier size)")
    sr.add_argument("--max-steps", type=int, default=20)
    sr.add_argument("--limit", type=int, default=None)
    sr.add_argument("--out", default="benchmark/search_report.json")
    sr.set_defaults(fn=_m.cmd_search)

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
    ft.set_defaults(fn=_m.cmd_finetune)

    t = sub.add_parser("tui", help="interactive terminal UI (browse problems, watch proofs)")
    t.add_argument("-p", "--parallel", type=int, default=proj_workers,
                   help="number of parallel proof workers (default: repo/1)")
    t.add_argument("--extensions", action="append", default=[],
                   help="extension path (file or dir), repeatable (Tau --extensions parity)")
    t.set_defaults(fn=_m.cmd_tui)

    lg = sub.add_parser("login", help="sign in to a provider (openai-codex|anthropic|github-copilot)")
    lg.add_argument("provider", choices=["openai-codex", "anthropic", "github-copilot"])
    lg.set_defaults(fn=_m.cmd_login)

    up = sub.add_parser("update", help="check for a newer lean-prover release")
    up.add_argument("--install", action="store_true")
    up.add_argument("--check", action="store_true", help="force a fresh PyPI check")
    up.add_argument("--version", default=None, help="pin a specific version")
    up.set_defaults(fn=_m.cmd_update)

    vr = sub.add_parser("version", help="print the current version")
    vr.set_defaults(fn=_m.cmd_version)

    rp = sub.add_parser("rpc", help="run the native RPC JSONL stdio loop")
    rp.set_defaults(fn=_m.cmd_rpc)

    ch = sub.add_parser("chat", help="open a CodingSession REPL (Tau chat parity)")
    ch.add_argument("--model", default="gpt-4o")
    ch.add_argument("--provider", default="")
    ch.add_argument("--cwd", default=None)
    ch.add_argument("--session-id", default=None)
    ch.set_defaults(fn=_m.cmd_chat)

    ex = sub.add_parser("export", help="export a recorded session transcript")
    ex.add_argument("session")
    ex.add_argument("--output", default=None)
    ex.add_argument("--format", default=None, choices=["html", "jsonl", "md", "markdown"])
    ex.set_defaults(fn=_m.cmd_export)

    s = sub.add_parser("sessions", help="list/inspect recorded proof sessions (~/.prover/sessions)")
    s.add_argument("id", nargs="?", default=None,
                   help="session id (filename stem) to show in detail")
    s.add_argument("--limit", type=int, default=20, help="max sessions to list")
    s.add_argument("--raw", action="store_true", help="also dump raw JSON records")
    s.set_defaults(fn=_m.cmd_sessions)

    u = sub.add_parser("usage", help="token/cost dashboard for one or all sessions (/usage parity)")
    u.add_argument("id", nargs="?", default=None,
                   help="session id (or 'all' / empty for all sessions)")
    u.set_defaults(fn=_m.cmd_usage)

    lb = sub.add_parser("leaderboard", help="record a benchmark score / show the board")
    lb.add_argument("--run", action="store_true",
                    help="run the benchmark first, then record the score")
    lb.add_argument("--problems", default="benchmark/problems.json")
    lb.add_argument("--set", default=None,
                    help="frozen set to drift-check against (benchmark/sets/<name>.toml); "
                         "defaults to 'core' only when running the core benchmark")
    lb.add_argument("--max-steps", type=int, default=proj_max_steps)
    lb.add_argument("--name", default=None, help="entry name (default: model or env)")
    lb.add_argument("--show", action="store_true", help="only show current leaderboard")
    lb.set_defaults(fn=_m.cmd_leaderboard)

    return ap
