"""Problem-set loading — lean-eval-style per-problem manifests.

Two formats, one entry point:

- legacy: a single JSON file (list of {id, difficulty, statement}) —
  still the default at benchmark/problems.json and what gen_problems.py
  writes.
- manifests: a directory of per-problem TOML files (lean-eval's
  manifests/problems/<id>.toml). One file per problem means adding or
  editing a problem never conflicts, and each carries its own provenance:
  statement_revision, status, visible, tags, source.

Every problem gets a content hash of its normalized statement; a frozen
set (benchmark/sets/*.toml) pins (id, revision) pairs so old scores are
invalidated loudly when a statement drifts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import tomllib

DIFFICULTIES = ("trivial", "easy", "medium", "hard")


@dataclass(frozen=True, slots=True)
class Problem:
    id: str
    difficulty: str
    statement: str
    statement_revision: int = 1
    status: str = "active"
    visible: bool = True
    tags: list[str] = field(default_factory=list)
    source: str = ""

    def as_dict(self) -> dict:
        """Legacy JSON shape (loop/bench consumers expect plain dicts)."""
        return {"id": self.id, "difficulty": self.difficulty, "statement": self.statement}


def statement_hash(statement: str) -> str:
    """Stable content identity for a statement (comparator convention)."""
    return hashlib.sha256(statement.strip().encode() + b"\n").hexdigest()


def _from_dict(d: dict) -> Problem:
    for key in ("id", "difficulty", "statement"):
        if not d.get(key):
            raise ValueError(f"problem missing required field {key!r}: {d}")
    if d["difficulty"] not in DIFFICULTIES:
        raise ValueError(f"problem {d['id']}: unknown difficulty {d['difficulty']!r}")
    rev = d.get("statement_revision", 1)
    if not isinstance(rev, int) or rev < 1:
        raise ValueError(f"problem {d['id']}: statement_revision must be a positive int")
    return Problem(
        id=d["id"],
        difficulty=d["difficulty"],
        statement=d["statement"],
        statement_revision=rev,
        status=d.get("status", "active"),
        visible=bool(d.get("visible", True)),
        tags=list(d.get("tags", [])),
        source=d.get("source", ""),
    )


def load_problems(path: str | Path) -> list[Problem]:
    """Load problems from a legacy JSON file or a TOML manifest directory."""
    p = Path(path)
    if p.is_dir():
        problems = [
            _from_dict({"id": f.stem, **tomllib.loads(f.read_text())})
            for f in sorted(p.glob("*.toml"))
        ]
    elif p.suffix == ".json":
        raw = json.loads(p.read_text())
        if not isinstance(raw, list):
            raise ValueError(f"{p}: expected a list of problems")
        problems = [_from_dict(d) for d in raw]
    else:
        raise ValueError(f"{p}: not a problems directory or .json file")
    ids = [pr.id for pr in problems]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate problem ids: {', '.join(dupes)}")
    return problems


def active_problems(problems: list[Problem]) -> list[Problem]:
    """Problems eligible for runs: status 'active' and visible."""
    return [pr for pr in problems if pr.status == "active" and pr.visible]


# ---- frozen sets ----

@dataclass(frozen=True, slots=True)
class FrozenSetMismatch:
    id: str
    set_revision: int
    actual_revision: int


def load_frozen_set(path: str | Path) -> dict[str, int]:
    """Read a frozen set manifest: {problem_id: pinned statement_revision}."""
    data = tomllib.loads(Path(path).read_text())
    out: dict[str, int] = {}
    for entry in data.get("problems", []):
        out[entry["id"]] = int(entry.get("revision", 1))
    return out


def check_frozen_set(pinned: dict[str, int], problems: list[Problem]) -> list[FrozenSetMismatch]:
    """Return every pin whose current revision drifted from the frozen one."""
    by_id = {pr.id: pr.statement_revision for pr in problems}
    missing = [FrozenSetMismatch(i, rev, -1) for i, rev in pinned.items() if i not in by_id]
    drifted = [
        FrozenSetMismatch(i, rev, by_id[i])
        for i, rev in pinned.items()
        if i in by_id and by_id[i] != rev
    ]
    return sorted(missing + drifted, key=lambda m: m.id)


def dir_digest(path: str | Path) -> str | None:
    """Digest of a problem set: JSON bytes, or sorted per-file digests."""
    import hashlib

    p = Path(path)
    try:
        if p.is_dir():
            h = hashlib.sha256()
            for f in sorted(p.glob("*.toml")):
                h.update(f.name.encode())
                h.update(f.read_bytes())
            return h.hexdigest()
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None
