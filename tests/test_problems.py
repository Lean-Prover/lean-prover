"""Per-problem TOML manifests + frozen sets (lean-eval adoptions)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.problems import (
    Problem,
    check_frozen_set,
    dir_digest,
    load_frozen_set,
    load_problems,
)

P = {"id": "t1", "difficulty": "easy", "statement": "theorem t : 1 = 1 := by\n  sorry"}


def _write_manifest(d: Path, pid: str, body: str) -> Path:
    f = d / f"{pid}.toml"
    f.write_text(body)
    return f


# ---- loading ----

def test_load_legacy_json(tmp_path: Path) -> None:
    f = tmp_path / "problems.json"
    f.write_text(json.dumps([P]))
    assert load_problems(f) == [Problem(**P)]


def test_load_manifest_dir_sorted(tmp_path: Path) -> None:
    for pid in ("b2", "a1"):
        _write_manifest(tmp_path, pid,
                        f'id = "{pid}"\ndifficulty = "trivial"\nstatement = """s{pid}"""\n')
    got = load_problems(tmp_path)
    assert [p.id for p in got] == ["a1", "b2"]
    assert got[0].as_dict() == {"id": "a1", "difficulty": "trivial", "statement": "sa1"}


def test_manifest_fields_roundtrip(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "x", (
        'id = "x"\ndifficulty = "medium"\nstatement_revision = 3\n'
        'tags = ["induction", "nat"]\nsource = "mathlib"\nvisible = false\n'
        'statement = """theorem x : 2 = 2 := by\n  sorry"""'
    ))
    p = load_problems(tmp_path)[0]
    assert p.statement_revision == 3
    assert p.tags == ["induction", "nat"]
    assert p.source == "mathlib"
    assert p.visible is False


def test_rejects_bad_difficulty_and_dupes(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "d", 'id = "d"\ndifficulty = "insane"\nstatement = "s"\n')
    with pytest.raises(ValueError, match="unknown difficulty"):
        load_problems(tmp_path)
    d2 = tmp_path / "dupes"
    d2.mkdir()
    for fname in ("one", "two"):
        _write_manifest(d2, fname, 'id = "same"\ndifficulty = "trivial"\nstatement = "s"\n')
    with pytest.raises(ValueError, match="duplicate problem ids"):
        load_problems(d2)


def test_id_comes_from_filename(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "filename-id", 'difficulty = "hard"\nstatement = "s"\n')
    assert load_problems(tmp_path)[0].id == "filename-id"


# ---- frozen sets ----

def test_frozen_set_detects_drift(tmp_path: Path) -> None:
    problems = [
        Problem(id="keep", difficulty="trivial", statement="s"),
        Problem(id="drifted", difficulty="trivial", statement="s",
                statement_revision=2),
    ]
    pinned = {"keep": 1, "drifted": 1, "gone": 1}
    mism = check_frozen_set(pinned, problems)
    assert [(m.id, m.set_revision, m.actual_revision) for m in mism] == [
        ("drifted", 1, 2), ("gone", 1, -1)]


def test_frozen_set_toml_loading(tmp_path: Path) -> None:
    f = tmp_path / "core.toml"
    f.write_text('problems = [\n { id = "a", revision = 1 },\n { id = "b" }\n]\n')
    assert load_frozen_set(f) == {"a": 1, "b": 1}


# ---- digests ----

def test_dir_digest_stable_and_content_sensitive(tmp_path: Path) -> None:
    d = tmp_path / "problems"
    d.mkdir()
    _write_manifest(d, "a", 'difficulty = "trivial"\nstatement = "s"\n')
    d1 = dir_digest(d)
    assert d1 == dir_digest(d)
    _write_manifest(d, "a", 'difficulty = "trivial"\nstatement = "s2"\n')
    assert dir_digest(d) != d1
    assert dir_digest(tmp_path / "nope") is None
