"""Miscellaneous prover CLI commands — thin delegators to their submodules."""

from __future__ import annotations

import argparse

from .common import _preload_extensions


def cmd_mcp(args: argparse.Namespace) -> int:
    from ..mcp import serve

    return serve()


def cmd_acp(args: argparse.Namespace) -> int:
    from ..acp import serve

    return serve()


def cmd_formalize(args: argparse.Namespace) -> int:
    from ..formalize import formalize

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
    from ..synth import main as synth_main

    return synth_main(["--out", args.out, "--count", str(args.count),
                       "--max-steps", str(args.max_steps),
                       "--problems", args.problems,
                       *(["--seeds", args.seeds] if args.seeds else []),
                       *(["--no-hammers"] if args.no_hammers else []),
                       *(["--model", args.model] if args.model else [])])


def cmd_lean_baseline(args: argparse.Namespace) -> int:
    from ..lean_baseline import main as baseline_main

    rc = baseline_main(["--problems", args.problems, "--out", args.out,
                        "--tactic", args.tactic,
                        "--timeout", str(args.timeout),
                        "--start", str(args.start)])
    if getattr(args, "validate", False) and rc == 0:
        import json as _j
        from pathlib import Path as _P

        from ..prover_loop import _resolve_lean_dir
        from ..validate import validate_file

        lean_dir = _resolve_lean_dir()
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
    from ..synth_lean import main as synth_main

    extra = ["--templates"] if args.templates else []
    return synth_main(["--report", args.report, "--out", args.out,
                       "--timeout", str(args.timeout), *extra])


def cmd_datagen(args: argparse.Namespace) -> int:
    from ..datagen import main as datagen_main

    argv = ["--corpus", args.corpus, "--out", args.out]
    for r in args.report:
        argv += ["--report", r]
    return datagen_main(argv)


def cmd_rldata(args: argparse.Namespace) -> int:
    from ..rldata import main as rldata_main

    argv = ["--problems", args.problems, "--out", args.out,
            "--n-attempts", str(args.n_attempts),
            "--max-steps", str(args.max_steps)]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    return rldata_main(argv)


def cmd_search(args: argparse.Namespace) -> int:
    from ..search import main as search_main

    argv = ["--problems", args.problems, "--out", args.out,
            "--beam", str(args.beam), "--max-steps", str(args.max_steps)]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    return search_main(argv)


def cmd_finetune(args: argparse.Namespace) -> int:
    from ..finetune import main as finetune_main

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
        from ..tui import main as tui_main
    except ImportError:
        print("TUI requires the optional 'textual' dependency: pip install 'lean-prover[tui]'")
        return 1
    extensions = getattr(args, "extensions", []) or []
    if extensions:
        _preload_extensions(extensions)
    tui_main(parallel=args.parallel)
    return 0


def cmd_rpc(args: argparse.Namespace) -> int:
    """Native Tau-style RPC stdio loop (`prover rpc`).

    Frames: {"type":"get_state","id":1} -> {"id":1,"success":true,"state":{...}}
    Use `prover mcp` for the JSON-RPC 2.0 / MCP surface instead.
    """
    from ..rpc import serve_rpc

    return serve_rpc()


def cmd_login(args: argparse.Namespace) -> int:
    from ..cli_coding import cmd_login as _cmd

    return _cmd(args)


def cmd_update(args: argparse.Namespace) -> int:
    from ..cli_coding import cmd_update as _cmd

    return _cmd(args)


def cmd_version(args: argparse.Namespace) -> int:
    from ..cli_coding import cmd_version as _cmd

    return _cmd(args)


def cmd_chat(args: argparse.Namespace) -> int:
    from ..cli_coding import cmd_chat as _cmd

    return _cmd(args)


def cmd_export(args: argparse.Namespace) -> int:
    from ..cli_coding import cmd_export as _cmd

    return _cmd(args)
