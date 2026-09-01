"""leaderboard.json schema validation — keeps the public site from breaking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REQUIRED_TOP_LEVEL = {"name", "score", "total", "tiers", "max_steps", "date"}
REQUIRED_TIER = {"proved", "total"}
ALLOWED_TIER_NAMES = {"trivial", "easy", "medium", "hard"}


@pytest.fixture(scope="module")
def board():
    return json.loads(Path("leaderboard.json").read_text())


def test_board_is_a_list(board) -> None:
    assert isinstance(board, list)


def test_entries_have_required_fields(board) -> None:
    for entry in board:
        missing = REQUIRED_TOP_LEVEL - set(entry)
        assert not missing, f"entry {entry.get('name')!r} missing fields: {missing}"


def test_scores_are_consistent(board) -> None:
    for entry in board:
        tier_sum = sum(t["proved"] for t in entry["tiers"].values())
        assert entry["score"] == tier_sum, (
            f"{entry['name']}: score {entry['score']} != tier sum {tier_sum}"
        )
        total_sum = sum(t["total"] for t in entry["tiers"].values())
        assert entry["total"] == total_sum, (
            f"{entry['name']}: total {entry['total']} != tier total sum {total_sum}"
        )


def test_tier_proved_within_total(board) -> None:
    for entry in board:
        for tier, counts in entry["tiers"].items():
            assert set(counts) >= REQUIRED_TIER, f"{entry['name']}/{tier} shape"
            assert 0 <= counts["proved"] <= counts["total"], (
                f"{entry['name']}/{tier}: proved {counts['proved']} out of range"
            )
        assert set(entry["tiers"]) <= ALLOWED_TIER_NAMES, (
            f"{entry['name']} has unknown tier names"
        )


def test_names_unique(board) -> None:
    names = [entry["name"] for entry in board]
    assert len(names) == len(set(names)), "duplicate leaderboard entries"


def test_max_steps_is_positive(board) -> None:
    for entry in board:
        assert isinstance(entry["max_steps"], int) and entry["max_steps"] >= 1


# ---- result versioning / comparability (agent.main bench + leaderboard) ----

def _entry(name="m", schema=1, solver="prove_best_of", steps=20,
           sha="abc123", toolchain="tc", hashes=None, score=5,
           result_version=1):
    return {
        "name": name, "schema_version": schema, "solver": solver,
        "max_steps": steps, "problems_sha256": sha, "toolchain": toolchain,
        "score": score, "total": 10, "date": "2026-01-01",
        "result_version": result_version,
        "statement_hashes": hashes or {},
        "tiers": {"hard": {"proved": score, "total": 10}},
    }


def _cfg(**over):
    cfg = {
        "max_steps": 20, "solver": "prove_best_of",
        "problems_sha256": "abc123", "toolchain": "tc",
        "statement_hashes": {},
    }
    cfg.update(over)
    return cfg


def test_upsert_supersedes_same_name_config() -> None:
    from agent import main as main_mod

    cfg = _cfg()
    old = _entry(name="m", score=3)
    new = _entry(name="m", score=9, result_version=2)
    out = main_mod._upsert_board([old], new, cfg)
    assert [e["name"] for e in out] == ["m"], "older run of same name+config replaced"
    assert out[0]["score"] == 9


def test_upsert_keeps_distinct_names() -> None:
    from agent import main as main_mod

    cfg = _cfg()
    a = _entry(name="a", score=9)
    b = _entry(name="b", score=5)
    out = main_mod._upsert_board([a], b, cfg)
    assert sorted(e["name"] for e in out) == ["a", "b"]
    assert out[0]["name"] == "a"  # ranked by score descending


def test_entry_reasons_flags_mismatched_config() -> None:
    from agent import main as main_mod

    assert main_mod._entry_reasons(_entry(), _cfg()) == []
    # wrong max_steps + wrong problems_sha256
    reasons = main_mod._entry_reasons(_entry(steps=6, sha="other"), _cfg())
    assert any("max_steps" in r for r in reasons)
    assert any("problems_sha256" in r for r in reasons)
    # wrong solver
    reasons = main_mod._entry_reasons(_entry(solver="prove"), _cfg())
    assert any("solver" in r for r in reasons)


def test_entry_reasons_flags_schema_version() -> None:
    from agent import main as main_mod

    reasons = main_mod._entry_reasons(_entry(schema=0), _cfg())
    assert any("schema_version" in r for r in reasons)


def test_entry_reasons_flags_statement_drift() -> None:
    from agent import main as main_mod

    cfg = _cfg(statement_hashes={"a": "newhash"})
    reasons = main_mod._entry_reasons(_entry(hashes={"a": "oldhash"}), cfg)
    assert any("drifted" in r for r in reasons)


def test_leaderboard_run_records_versioned_entry(monkeypatch, tmp_path) -> None:
    import argparse
    import json

    from agent import main as main_mod

    p = tmp_path / "problems.json"
    p.write_text(json.dumps(
        [{"id": "t1", "difficulty": "easy",
          "statement": "theorem t1 : True := by\n  sorry"}]))
    calls = []

    def fake_prove_one(pd, ms, idx, total, *a, **k):
        calls.append((pd["id"], ms))
        return {"id": pd["id"], "difficulty": pd["difficulty"],
                "statement": pd["statement"], "statement_hash": "h",
                "proved": True}, 0, 0.0

    monkeypatch.setattr(main_mod, "_prove_one", fake_prove_one)
    monkeypatch.setattr(main_mod, "load_board", list)
    boardf = tmp_path / "lb.json"
    monkeypatch.setattr(main_mod, "BOARD_FILE", str(boardf))
    args = argparse.Namespace(run=True, problems=str(p), max_steps=20,
                              name="m", set="nope")
    assert main_mod._leaderboard_run(args) == 0
    assert calls == [("t1", 20)]
    board = json.loads(boardf.read_text())
    assert len(board) == 1
    e = board[0]
    assert e["schema_version"] == main_mod.SCHEMA_VERSION
    assert e["solver"] == "prove_best_of"
    assert e["result_version"] == 1
    assert e["name"] == "m"


def test_cmd_bench_refuses_drifted_frozen_set(monkeypatch, tmp_path, capsys) -> None:
    import argparse
    import json

    from agent import main as main_mod

    p = tmp_path / "problems.json"
    p.write_text(json.dumps(
        [{"id": "t1", "difficulty": "hard", "statement": "s"}]))
    sets = tmp_path / "sets"
    sets.mkdir()
    (sets / "core.toml").write_text('problems = [{ id = "t1", revision = 2 }]\n')
    args = argparse.Namespace(problems=str(p), set="core", max_steps=20, start=1,
                              parallel=1, no_goal_feedback=False, no_record=False,
                              no_hammers=False, n_attempts=1, full_file=False,
                              adaptive=False, validate=False, report=None)
    assert main_mod.cmd_bench(args) == 1
    assert "refusing to run" in capsys.readouterr().err


def test_prove_one_uses_prove_best_of_and_stamps_hash(monkeypatch) -> None:
    from types import SimpleNamespace

    from agent import main as main_mod
    from agent.problems import statement_hash

    called = {"best": 0, "plain": 0}

    def fake_best(*a, **k):
        called["best"] += 1
        return SimpleNamespace(proved=True, steps=3, seconds=1.0,
                               total_tokens=10, total_prompt_tokens=5,
                               total_completion_tokens=5,
                               estimated_cost_usd=0.001,
                               session_path=None, trace=[], attempts=[[]])

    def fake_plain(*a, **k):
        called["plain"] += 1
        return fake_best(*a, **k)

    monkeypatch.setattr(main_mod, "prove_best_of", fake_best)
    monkeypatch.setattr(main_mod, "prove", fake_plain)
    res, _tok, _cost = main_mod._prove_one(
        {"id": "t1", "difficulty": "easy", "statement": "theorem t : True := by\n  sorry"},
        20, 1, 1)
    assert called["best"] == 1 and called["plain"] == 0, "bench/leaderboard worker uses prove_best_of"
    assert res["statement_hash"] == statement_hash("theorem t : True := by\n  sorry")
