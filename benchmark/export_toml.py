"""One-time migration: benchmark/problems.json -> per-problem TOML manifests.

Usage:
    python -m benchmark.export_toml problems.json benchmark/problems
    python -m benchmark.export_toml problems.json benchmark/problems --set core

Writes one <id>.toml per problem, then (with --set) freezes the whole set at
its current revisions into benchmark/sets/<name>.toml.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def toml_quote(s: str) -> str:
    return '"""' + s.replace("\\", "\\\\").replace('"""', '\\"""') + '"""'


def problem_to_toml(p: dict) -> str:
    lines = [
        f"id = {toml_quote(p['id'])}",
        f"difficulty = {toml_quote(p['difficulty'])}",
        f"statement_revision = {int(p.get('statement_revision', 1))}",
    ]
    if p.get("tags"):
        lines.append("tags = [" + ", ".join(toml_quote(t) for t in p["tags"]) + "]")
    if p.get("source"):
        lines.append(f"source = {toml_quote(p['source'])}")
    if not p.get("visible", True):
        lines.append("visible = false")
    if p.get("status", "active") != "active":
        lines.append(f"status = {toml_quote(p['status'])}")
    # statement LAST: multi-line literal keeps the trailing `sorry` readable.
    lines.append(f"statement = {toml_quote(p['statement'])}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_path")
    ap.add_argument("out_dir")
    ap.add_argument("--set", dest="set_name", default=None,
                    help="also write a frozen set manifest sets/<name>.toml")
    args = ap.parse_args(argv)

    problems = json.loads(Path(args.json_path).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in problems:
        (out / f"{p['id']}.toml").write_text(problem_to_toml(p))
    print(f"wrote {len(problems)} manifests to {out}")

    if args.set_name:
        sets_dir = out.parent / "sets"
        sets_dir.mkdir(exist_ok=True)
        entries = "\n".join(
            f"{{ id = \"{p['id']}\", revision = {int(p.get('statement_revision', 1))} }},"
            for p in problems
        )
        (sets_dir / f"{args.set_name}.toml").write_text(
            "# Frozen eval set: append-only once published. A revision here that\n"
            "# no longer matches its problem invalidates old scores loudly.\n"
            f"name = \"{args.set_name}\"\n"
            "frozen = true\n"
            "problems = [\n" + entries + "\n]\n"
        )
        print(f"frozen set: {sets_dir / (args.set_name + '.toml')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
