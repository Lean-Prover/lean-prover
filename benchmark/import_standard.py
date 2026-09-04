#!/usr/bin/env python3
"""Import standard theorem-proving benchmarks into prover format.

Supported sources:

* **MiniF2F** Lean 4 port (https://github.com/yangky11/miniF2F-lean4):
  ``theorem ... := by sorry`` with a uniform preamble
  (``set_option maxHeartbeats 0`` + ``open BigOperators ...``).
* **PutnamBench** (https://github.com/trishullab/PutnamBench): 672 Putnam
  problems in ``lean4/src/putnam_<year>_<a|b><n>.lean``; ~350 files define an
  ``abbrev putnam_*_solution := sorry`` scaffold before the theorem, which we
  keep as part of the statement.
* **FormalQualBench** (https://github.com/math-inc/FormalQualBench): 23
  graduate-qualifying theorems, one ``theorem MainTheorem`` per
  ``FormalQualBench/<Name>/Main.lean`` with expert scaffolding definitions
  (namespaces preserved; theorem renamed to ``prover_MainTheorem``).

Every imported statement is re-checked against the *pinned* Mathlib in
``../lean`` (our toolchain, Lean v4.33.0 — PutnamBench pins v4.27.0 and
FormalQualBench pins v4.28.0), so a statement that no longer type-checks on
our stack is flagged ``"compiles": false`` with the first diagnostics, instead
of being silently dropped. That keeps the benchmark honest: scores are only
meaningful over the subset that compiles on our pinned Mathlib.

Usage::

    python benchmark/import_standard.py minif2f \
        --src <dir containing MiniF2F/> --split test --verify
    python benchmark/import_standard.py putnam \
        --src <PutnamBench checkout> --verify
    python benchmark/import_standard.py formalqual \
        --src <FormalQualBench checkout> --verify
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import lean
from agent.loop import HEADER, LEAN_DIR

SORRY = " := by\n  sorry"
NAME_RE = re.compile(r"^theorem\s+([A-Za-z0-9_']+)")
VERIFY_TAG = "minif2f_import"


def parse_problem_file(path: Path) -> dict:
    """Parse one MiniF2F problem file into prover-format problem dict.

    Keeps the preamble (set_option / open) so the statement type-checks with
    our header; drops ``import`` lines (re-added by the agent header) and any
    comment lines. Renames the theorem with the ``prover_`` prefix to mirror
    gen_problems.py and guarantee no collision with a Mathlib declaration.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    th_idx = [i for i, l in enumerate(lines) if l.strip().startswith("theorem ")]
    if len(th_idx) != 1:
        raise ValueError(f"{path.name}: expected exactly one theorem, found {len(th_idx)}")
    th_idx = th_idx[0]

    preamble = []
    in_block = False
    for l in lines[:th_idx]:
        s = l.strip()
        if in_block:
            if "-/" in s:
                in_block = False
            continue
        if s.startswith("/-"):
            if "-/" not in s:
                in_block = True
            continue
        if not s or s.startswith(("import ", "--")):
            continue
        preamble.append(s)

    sig_lines = []
    for l in lines[th_idx:]:
        sig_lines.append(l.rstrip())
        if ":=" in l:
            break

    m = NAME_RE.match(sig_lines[0])
    if not m:
        raise ValueError(f"{path.name}: cannot parse theorem name from {sig_lines[0]!r}")
    name = m.group(1)

    sig = "\n".join(sig_lines)
    sig = sig[: sig.index(":=")].rstrip()
    sig = sig.replace(f"theorem {name}", f"theorem prover_{name}", 1)

    parts = []
    if preamble:
        parts.append("\n".join(preamble))
    parts.append(sig + SORRY)

    return {
        "id": f"minif2f_{name}",
        "difficulty": "",  # filled in by import_minif2f
        "statement": "\n\n".join(parts),
        "source": "minif2f",
        "source_name": name,
    }


def import_minif2f(src_dir: Path, split: str, limit: int | None = None) -> list[dict]:
    """Import every problem in ``<src_dir>/MiniF2F/<Split>/*.lean``."""
    subdir = src_dir / "MiniF2F" / split.capitalize()
    if not subdir.is_dir():
        raise FileNotFoundError(f"expected split directory at {subdir}")
    files = sorted(subdir.glob("*.lean"))
    if limit is not None:
        files = files[:limit]
    problems = []
    for f in files:
        p = parse_problem_file(f)
        p["difficulty"] = f"minif2f_{split}"
        problems.append(p)
    return problems


# ---------------------------------------------------------------- PutnamBench

PUTNAM_FILE_RE = re.compile(r"^putnam_(\d{4})_[ab](\d+)$")


def parse_putnam_file(path: Path) -> dict:
    """Parse one PutnamBench problem file into prover-format problem dict.

    Upstream layout: imports, an optional ``abbrev putnam_*_solution := sorry``
    scaffold (the known answer, kept verbatim — the theorem explicitly
    references it), an informal-statement docstring, and exactly one theorem.
    We drop imports (re-added by the agent header) and the docstring, keep
    ``open``/``set_option`` lines and the solution scaffold, rename the
    theorem with the ``prover_`` prefix, and append the canonical sorry.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    th_idx = [i for i, l in enumerate(lines) if l.strip().startswith("theorem ")]
    if len(th_idx) != 1:
        raise ValueError(f"{path.name}: expected exactly one theorem, found {len(th_idx)}")
    th_idx = th_idx[0]

    preamble = []
    in_block = False
    for l in lines[:th_idx]:
        s = l.strip()
        if in_block:
            if "-/" in s:
                in_block = False
            continue
        if s.startswith("/-"):
            if "-/" not in s:
                in_block = True
            continue
        if not s or s.startswith(("import ", "--", "#")):
            continue
        if re.match(r"^(theorem|lemma|example)\s", s):
            raise ValueError(f"{path.name}: unexpected declaration before theorem: {s!r}")
        preamble.append(l.rstrip())

    sig_lines = []
    for l in lines[th_idx:]:
        sig_lines.append(l.rstrip())
        if ":=" in l:
            break

    m = NAME_RE.match(sig_lines[0])
    if not m:
        raise ValueError(f"{path.name}: cannot parse theorem name from {sig_lines[0]!r}")
    name = m.group(1)
    if not PUTNAM_FILE_RE.match(name):
        raise ValueError(f"{path.name}: unexpected theorem name {name!r}")

    sig = "\n".join(sig_lines)
    sig = sig[: sig.index(":=")].rstrip()
    sig = sig.replace(f"theorem {name}", f"theorem prover_{name}", 1)

    parts = []
    if preamble:
        parts.append("\n".join(preamble).strip("\n"))
    parts.append(sig + SORRY)

    year = PUTNAM_FILE_RE.match(name).group(1)
    return {
        "id": name,  # already namespaced: putnam_1962_a2
        "difficulty": f"putnam_{year}",
        "statement": "\n\n".join(parts),
        "source": "putnambench",
        "source_name": name,
    }


def import_putnam(src_dir: Path, limit: int | None = None) -> list[dict]:
    """Import every problem in ``<src_dir>/lean4/src/putnam_*.lean``."""
    subdir = src_dir / "lean4" / "src"
    if not subdir.is_dir():
        raise FileNotFoundError(f"expected lean4/src directory at {subdir}")
    files = sorted(f for f in subdir.glob("putnam_*.lean") if PUTNAM_FILE_RE.match(f.stem))
    if not files:
        raise FileNotFoundError(f"no putnam_*.lean problems found in {subdir}")
    if limit is not None:
        files = files[:limit]
    return [parse_putnam_file(f) for f in files]


# ----------------------------------------------------------- FormalQualBench

FQ_NAME_RE = re.compile(r"^theorem\s+MainTheorem\b")


def parse_formalqual_file(path: Path) -> dict:
    """Parse one FormalQualBench problem into prover-format problem dict.

    Upstream layout: ``FormalQualBench/<Name>/Main.lean`` with scaffolding
    definitions inside ``namespace <Name>`` and exactly one
    ``theorem MainTheorem ... := by sorry``. We drop only ``import`` lines,
    keep every definition in its namespace (the signature references them),
    rename ``MainTheorem`` to ``prover_MainTheorem``, and append the
    canonical sorry.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    th_idx = [i for i, l in enumerate(lines) if FQ_NAME_RE.match(l.strip())]
    if len(th_idx) != 1:
        raise ValueError(f"{path.name}: expected exactly one MainTheorem, found {len(th_idx)}")
    th_idx = th_idx[0]

    preamble = [
        l.rstrip() for l in lines[:th_idx]
        if not l.strip().startswith("import ")
    ]
    while preamble and not preamble[0].strip():
        preamble.pop(0)
    while preamble and not preamble[-1].strip():
        preamble.pop()

    sig_lines = []
    for l in lines[th_idx:]:
        sig_lines.append(l.rstrip())
        if ":=" in l:
            break

    sig = "\n".join(sig_lines)
    sig = sig[: sig.index(":=")].rstrip()
    sig = sig.replace("theorem MainTheorem", "theorem prover_MainTheorem", 1)

    parts = []
    if preamble:
        parts.append("\n".join(preamble))
    parts.append(sig + SORRY)

    name = path.parent.name
    if not name or name in (".", ".."):
        raise ValueError(f"{path}: problem dir name not derivable")
    return {
        "id": f"formalqual_{name}",
        "difficulty": "formalqual",
        "statement": "\n\n".join(parts),
        "source": "formalqualbench",
        "source_name": name,
    }


def import_formalqual(src_dir: Path, limit: int | None = None) -> list[dict]:
    """Import every problem in ``<src_dir>/FormalQualBench/<Name>/Main.lean``."""
    subdir = src_dir / "FormalQualBench"
    if not subdir.is_dir():
        raise FileNotFoundError(f"expected FormalQualBench directory at {subdir}")
    files = sorted(subdir.glob("*/Main.lean"))
    if not files:
        raise FileNotFoundError(f"no <Name>/Main.lean problems found in {subdir}")
    if limit is not None:
        files = files[:limit]
    return [parse_formalqual_file(f) for f in files]


def verify(problems: list[dict], lean_dir: Path, timeout: int = 180) -> list[dict]:
    """Type-check every statement against our pinned Mathlib.

    Tolerates ``sorry`` (compile_only): ``compiles`` is true iff the file
    exits 0. Non-compiling entries keep their first diagnostics so nothing is
    silently dropped. Returns a new list; input is not mutated.
    """
    # compile_only runs `lake env lean <file>` with cwd=lean_dir, so the
    # per-problem temp file must be absolute (or resolved) — a relative
    # lean_dir otherwise yields a path that doesn't exist from that cwd.
    lean_dir = Path(lean_dir).resolve()
    checked = []
    tmp_dir = lean_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    for i, p in enumerate(problems):
        tmp = tmp_dir / f"{VERIFY_TAG}_{i}.lean"
        tmp.write_text(HEADER + p["statement"] + "\n", encoding="utf-8")
        rc, out = lean.compile_only(tmp, lean_dir, timeout=timeout)
        tmp.unlink(missing_ok=True)
        p2 = dict(p)
        p2["compiles"] = rc == 0
        p2["error"] = "" if rc == 0 else lean.error_report(lean_dir, out, max_diags=3)
        checked.append(p2)
    return checked


def _write(path: Path, problems: list[dict]) -> None:
    path.write_text(json.dumps(problems, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cmd_minif2f(args: argparse.Namespace) -> int:
    problems = import_minif2f(Path(args.src), args.split, limit=args.limit)
    print(f"parsed {len(problems)} problems ({args.split})")

    if args.verify:
        problems = verify(problems, Path(args.lean_dir), timeout=args.timeout)
        ok = sum(1 for p in problems if p["compiles"])
        bad = [p["id"] for p in problems if not p["compiles"]]
        print(f"compile check: {ok}/{len(problems)} type-check on our Mathlib")
        if bad:
            print(f"non-compiling ({len(bad)}): {', '.join(bad[:10])}{' ...' if len(bad) > 10 else ''}")

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _write(dest, problems)
    print(f"wrote {dest}")
    return 0


def cmd_putnam(args: argparse.Namespace) -> int:
    problems = import_putnam(Path(args.src), limit=args.limit)
    print(f"parsed {len(problems)} problems (putnambench)")

    if args.verify:
        problems = verify(problems, Path(args.lean_dir), timeout=args.timeout)
        ok = sum(1 for p in problems if p["compiles"])
        bad = [p["id"] for p in problems if not p["compiles"]]
        print(f"compile check: {ok}/{len(problems)} type-check on our Mathlib")
        if bad:
            print(f"non-compiling ({len(bad)}): {', '.join(bad[:10])}{' ...' if len(bad) > 10 else ''}")

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _write(dest, problems)
    print(f"wrote {dest}")
    return 0


def cmd_formalqual(args: argparse.Namespace) -> int:
    problems = import_formalqual(Path(args.src), limit=args.limit)
    print(f"parsed {len(problems)} problems (formalqualbench)")

    if args.verify:
        problems = verify(problems, Path(args.lean_dir), timeout=args.timeout)
        ok = sum(1 for p in problems if p["compiles"])
        bad = [p["id"] for p in problems if not p["compiles"]]
        print(f"compile check: {ok}/{len(problems)} type-check on our Mathlib")
        if bad:
            print(f"non-compiling ({len(bad)}): {', '.join(bad[:10])}{' ...' if len(bad) > 10 else ''}")

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _write(dest, problems)
    print(f"wrote {dest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    mf = sub.add_parser("minif2f", help="import MiniF2F Lean4 port")
    mf.add_argument("--src", required=True, help="directory containing MiniF2F/")
    mf.add_argument("--split", choices=["test", "valid"], default="test")
    mf.add_argument("--limit", type=int, default=None, help="only import first N problems")
    mf.add_argument("--dest", default=None, help="output JSON (default benchmark/minif2f_<split>.json)")
    mf.add_argument("--verify", action="store_true", help="type-check against our pinned Mathlib")
    mf.add_argument("--lean-dir", default=str(LEAN_DIR), help="lean project dir")
    mf.add_argument("--timeout", type=int, default=180, help="per-file lean timeout (s)")
    mf.set_defaults(fn=cmd_minif2f)

    pn = sub.add_parser("putnam", help="import PutnamBench (trishullab/PutnamBench)")
    pn.add_argument("--src", required=True, help="PutnamBench checkout (containing lean4/src)")
    pn.add_argument("--limit", type=int, default=None, help="only import first N problems")
    pn.add_argument("--dest", default=str(Path(__file__).parent / "putnam.json"),
                    help="output JSON (default benchmark/putnam.json)")
    pn.add_argument("--verify", action="store_true", help="type-check against our pinned Mathlib")
    pn.add_argument("--lean-dir", default=str(LEAN_DIR), help="lean project dir")
    pn.add_argument("--timeout", type=int, default=180, help="per-file lean timeout (s)")
    pn.set_defaults(fn=cmd_putnam)

    fq = sub.add_parser("formalqual", help="import FormalQualBench (math-inc/FormalQualBench)")
    fq.add_argument("--src", required=True, help="FormalQualBench checkout (containing FormalQualBench/)")
    fq.add_argument("--limit", type=int, default=None, help="only import first N problems")
    fq.add_argument("--dest", default=str(Path(__file__).parent / "formalqual.json"),
                    help="output JSON (default benchmark/formalqual.json)")
    fq.add_argument("--verify", action="store_true", help="type-check against our pinned Mathlib")
    fq.add_argument("--lean-dir", default=str(LEAN_DIR), help="lean project dir")
    fq.add_argument("--timeout", type=int, default=180, help="per-file lean timeout (s)")
    fq.set_defaults(fn=cmd_formalqual)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "minif2f" and args.dest is None:
        args.dest = str(Path(__file__).parent / f"minif2f_{args.split}.json")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
