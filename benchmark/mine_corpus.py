"""Mine (statement, tactic-script) pairs from Lean projects via leagent.

Wraps leanprover/leagent's `lean_extract` binary:
  1. resolves the binary (PROVER_LEAN_EXTRACT, default ~/.prover/bin/lean_extract)
  2. derives LD_LIBRARY_PATH from the project's pinned toolchain
     (~/.elan/toolchains/leanprover--lean4---<v>/lib/lean) so
     libleanshared.so resolves without polluting the environment
  3. refuses to run when the binary's Lean version tag does not match the
     pinned toolchain (the lakefile of a newer Mathlib will not parse)
  4. converts the extracted JSONL into corpus rows compatible with
     agent/retrieval.corpus_append's schema and appends them to the corpus

Setup (one-time):
    elan toolchain install <version>          # matching libleanshared.so
    gh release download v0.1.1 -R leanprover/leagent \
        -p "lean_extract-v<version>" -O ~/.prover/bin/lean_extract
    chmod +x ~/.prover/bin/lean_extract

Usage:
    python -m benchmark.mine_corpus --modules Mathlib --out /tmp/mined
    python -m benchmark.mine_corpus --modules ProverSupport --append-corpus
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def toolchain_version(lean_dir: Path | None = None) -> str:
    """Pinned version from lean/lean-toolchain, e.g. 'v4.33.0'."""
    p = (lean_dir or REPO_ROOT / "lean") / "lean-toolchain"
    return p.read_text().strip().removeprefix("leanprover/lean4:")


def default_binary() -> Path:
    env = os.environ.get("PROVER_LEAN_EXTRACT")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
    return Path(xdg) / "prover" / "bin" / "lean_extract"


def library_path_for(version: str) -> Path:
    v = version if version.startswith("v") else f"v{version}"
    return Path.home() / ".elan" / "toolchains" / f"leanprover--lean4---{v}" / "lib" / "lean"


def extract(binary: Path, modules: list[str], out_dir: Path,
            lean_dir: Path | None = None, jobs: int = 4,
            timeout: int = 3600) -> list[Path]:
    """Run lean_extract; returns the JSONL files it produced under out_dir."""
    version = toolchain_version(lean_dir)
    lib = library_path_for(version)
    if not lib.exists():
        raise SystemExit(
            f"toolchain {version} not installed (`elan toolchain install {version}`)")
    marker = re.search(r"v(\d+\.\d+\.\d+)", binary.name)
    if marker and f"v{marker.group(1)}" != version:
        raise SystemExit(
            f"binary targets Lean {marker.group(1)} but project pins {version}; "
            "download a matching release asset")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary), "--output", str(out_dir), "--jobs", str(jobs),
        *(["--source-root", str(lean_dir)] if lean_dir else []),
    ]
    for m in modules:
        cmd.extend(["--modules", m])
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (
        f"{lib}{os.pathsep}{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH")
        else str(lib))
    subprocess.run(cmd, cwd=lean_dir or REPO_ROOT, env=env, check=True,
                   timeout=timeout)
    return sorted(out_dir.glob("*.jsonl"))


def mined_to_rows(jsonl: Path) -> list[dict]:
    """Convert leagent extraction JSONL to corpus-shaped rows.

    Keeps only declarations that carry a real tactic proof body; native-tactic
    tags and non-theorems are dropped (same filter as retrieval._proof_body).
    """
    rows: list[dict] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = rec.get("name") or rec.get("declName") or ""
        signature = rec.get("signature") or rec.get("type") or ""
        proof = rec.get("proof") or rec.get("tactics") or ""
        kind = rec.get("kind") or rec.get("declKind") or "theorem"
        if kind not in ("theorem", "lemma", "example"):
            continue
        text = proof.strip()
        if not signature or not text or text in ("prover_finish", "prover_search"):
            continue
        head = re.sub(r":=\s*by\s*$", "", signature.strip()).strip()
        head = re.sub(r":=\s*$", "", head).strip()
        if not head:
            continue
        rows.append({"id": f"mined_{name}", "difficulty": "mined",
                     "statement": f"{head} := by\n{text}"})
    return rows


def append_to_corpus(rows: list[dict], corpus: Path) -> int:
    seen = set()
    if corpus.exists():
        for line in corpus.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(json.loads(line).get("id", ""))
            except json.JSONDecodeError:
                pass
    added = 0
    with corpus.open("a", encoding="utf-8") as fh:
        for row in rows:
            if row["id"] in seen:
                continue
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen.add(row["id"])
            added += 1
    return added


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modules", nargs="+", required=True)
    ap.add_argument("--out", default="/tmp/prover-mined")
    ap.add_argument("--binary", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--append-corpus", action="store_true",
                    help="merge mined statements into the live corpus")
    args = ap.parse_args(argv)

    binary = args.binary or default_binary()
    if not binary.exists():
        raise SystemExit(f"lean_extract not found at {binary} (see module docstring)")
    jsonls = extract(binary, args.modules, Path(args.out),
                     jobs=args.jobs, timeout=args.timeout)
    print(f"{len(jsonls)} extraction file(s): "
          + ", ".join(str(j) for j in jsonls))
    if args.append_corpus:
        rows: list[dict] = []
        for j in jsonls:
            rows.extend(mined_to_rows(j))
        n = append_to_corpus(rows, REPO_ROOT / "corpus" / "lean_proved.jsonl")
        print(f"appended {n}/{len(rows)} new rows to corpus/lean_proved.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
