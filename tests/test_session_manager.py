"""Session manager tests — ported from huggingface/tau tests/test_session_manager.py.

Tau builds a per-project index.jsonl, lists newest-first, upserts by id.
Same contract here, plus the history-reconstruction round-trip that backs
resume/branch.
"""

from __future__ import annotations

from pathlib import Path

from agent.session import read_session
from agent.session_manager import (
    SessionManager,
    SessionRecord,
    history_from_records,
    last_body,
)


def _record(tmp_path: Path, sid: str, updated: float = 0.0, status: str = "failed") -> SessionRecord:
    return SessionRecord(
        id=sid, path=str(tmp_path / f"{sid}.jsonl"), problem_id=sid,
        model="Qwen/Qwen3.8-27B", status=status, created_at=updated, updated_at=updated,
    )


def test_session_manager_creates_and_lists_sessions(tmp_path: Path) -> None:
    manager = SessionManager(sessions_dir=tmp_path)

    manager.upsert(_record(tmp_path, "s-one", 100.0))
    assert manager.index_path == tmp_path / "index.jsonl"
    assert manager.index_path.exists()

    assert manager.get("s-one") == manager.list_sessions()[0]


def test_upsert_replaces_same_id_record(tmp_path: Path) -> None:
    manager = SessionManager(sessions_dir=tmp_path)
    manager.upsert(_record(tmp_path, "s-one", 100.0, status="running"))
    manager.upsert(_record(tmp_path, "s-one", 101.0, status="proved"))
    manager.upsert(_record(tmp_path, "s-two", 102.0, status="failed"))

    records = manager.list_sessions()
    assert [r.id for r in records] == ["s-two", "s-one"]  # newest updated first
    assert manager.get("s-one").status == "proved"


def test_list_sessions_newest_first(tmp_path: Path) -> None:
    manager = SessionManager(sessions_dir=tmp_path)
    manager.upsert(_record(tmp_path, "old", 100.0))
    manager.upsert(_record(tmp_path, "new", 200.0))
    assert [r.id for r in manager.list_sessions()] == ["new", "old"]


def test_touch_updates_fields_and_bumps_time(tmp_path: Path) -> None:
    manager = SessionManager(sessions_dir=tmp_path)
    manager.upsert(_record(tmp_path, "s-one", 100.0, status="running"))

    updated = manager.touch("s-one", status="failed", proved=False, steps=20)
    assert updated is not None
    assert updated.status == "failed"
    assert updated.proved is False
    assert updated.steps == 20
    assert updated.updated_at >= 100.0
    assert manager.get("s-one").status == "failed"


def test_recover_interrupted_marks_running_without_result_as_failed(tmp_path: Path) -> None:
    import json

    manager = SessionManager(sessions_dir=tmp_path)
    # Interrupted: running record whose transcript has start but no result.
    interrupted = _record(tmp_path, "interrupted", 100.0, status="running")
    manager.upsert(interrupted)
    Path(interrupted.path).write_text(
        json.dumps({"event": "start", "statement": "s"}) + "\n"
    )
    # Completed: running record but the transcript does have a result event.
    done = _record(tmp_path, "done", 101.0, status="running")
    manager.upsert(done)
    Path(done.path).write_text(
        json.dumps({"event": "start", "statement": "s"}) + "\n"
        + json.dumps({"event": "result", "proved": True}) + "\n"
    )

    assert manager.recover_interrupted() == 1
    assert manager.get("interrupted").status == "failed"
    assert manager.get("done").status == "running"


def test_touch_unknown_returns_none(tmp_path: Path) -> None:
    manager = SessionManager(sessions_dir=tmp_path)
    assert manager.touch("nope", status="proved") is None


def test_index_survives_corrupt_lines(tmp_path: Path) -> None:
    manager = SessionManager(sessions_dir=tmp_path)
    manager.upsert(_record(tmp_path, "s-one", 100.0))
    # Append garbage — the reader must skip it, not crash.
    with manager.index_path.open("a") as fh:
        fh.write("{not valid json\n")
    assert manager.get("s-one") is not None
    assert len(manager.list_sessions()) == 1


def test_session_record_json_round_trip(tmp_path: Path) -> None:
    rec = SessionRecord(
        id="abc", path=str(tmp_path / "abc.jsonl"), problem_id="p",
        model="m", status="proved", proved=True, steps=3,
        created_at=1.0, updated_at=2.0,
    )
    assert SessionRecord.from_json(rec.to_json()) == rec


# ---------------------------------------------------------------- history rebuild


def _write_session(path: Path, body_lines: list[str], signature: str) -> Path:
    """Write a minimal session mirroring what prove() records for N LLM steps."""
    recs = [{"t": 0, "event": "start", "problem_id": path.stem,
             "statement": signature, "max_steps": 20, "model": "m"}]
    recs.append({"t": 0, "event": "llm_start"})
    for i, body in enumerate(body_lines, 1):
        recs.append({"t": i, "event": "build", "step": i, "ok": False,
                     "diagnostics": 1, "summary": "goals remain",
                     "report": f"goal report {i}"})
        recs.append({"t": i, "event": "goals", "step": i, "goals": f"open goals {i}"})
        recs.append({"t": i, "event": "llm_request", "step": i})
        recs.append({"t": i, "event": "llm_response", "step": i,
                     "prompt_tokens": 1, "completion_tokens": 1, "tokens": 2,
                     "body": body})
    path.write_text("\n".join(__import__("json").dumps(r) for r in recs) + "\n")
    return path


def test_history_from_records_round_trips_proof_loop(tmp_path: Path) -> None:
    signature = "theorem prover_foo (n : ℕ) : n + 0 = n :="
    sp = _write_session(tmp_path / "foo.jsonl", ["  induction n", "  omega"], signature)
    history = history_from_records(read_session(sp))

    # One user/assistant pair per LLM step.
    assert len(history) == 4
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    # User turns carry the same shape prove() sends.
    assert "Theorem signature:" in history[0]["content"]
    assert "Compiler diagnostics:" in history[0]["content"]
    assert "Open goals" in history[0]["content"]
    assert "goal report 1" in history[0]["content"]
    assert "open goals 1" in history[0]["content"]
    # Assistant turns carry the raw body.
    assert history[1]["content"] == "  induction n"


def test_history_skips_empty_bodies(tmp_path: Path) -> None:
    sp = _write_session(tmp_path / "foo.jsonl", ["", "  omega"],
                        "theorem t : 1 = 1 :=")
    history = history_from_records(read_session(sp))
    # empty first body contributes no assistant turn, but its build report
    # still pairs with the second reply → assistant count stays 1
    assistants = [m for m in history if m["role"] == "assistant"]
    assert len(assistants) == 1
    assert assistants[0]["content"] == "  omega"


def test_last_body_returns_last_nonempty(tmp_path: Path) -> None:
    sp = _write_session(tmp_path / "foo.jsonl", ["  a", "  b"], "theorem t : 1 = 1 :=")
    assert last_body(read_session(sp)) == "  b"


def test_last_body_none_for_hammer_only_sessions(tmp_path: Path) -> None:
    """A hammer-only session (no llm_response) has no replayable body."""
    recs = [{"t": 0, "event": "start", "statement": "theorem t : 1 = 1 :="},
            {"t": 1, "event": "hammer", "i": 1, "total": 10, "tactic": "ring",
             "ok": True, "output": ""}]
    sp = tmp_path / "hammer.jsonl"
    sp.write_text("\n".join(__import__("json").dumps(r) for r in recs) + "\n")
    assert last_body(read_session(sp)) is None


def test_branch_truncates_history_at_turn(tmp_path: Path) -> None:
    """branch_at keeps the first N turns and discards the rest (tau: repoint
    the LeafEntry at an earlier entry)."""
    sp = _write_session(tmp_path / "foo.jsonl", ["  a", "  b", "  c"],
                        "theorem t : 1 = 1 :=")
    full = history_from_records(read_session(sp))
    assert len(full) == 6

    branched = full[: max(0, 1) * 2]
    assert len(branched) == 2
    assert branched[1]["content"] == "  a"  # only the first turn survives


def test_session_manager_rename_updates_title(tmp_path: Path) -> None:
    from agent.session_manager import SessionRecord

    manager = SessionManager(sessions_dir=tmp_path)
    manager.upsert(SessionRecord(
        id="20260101-000000-sq_nonneg", path=str(tmp_path / "x.jsonl"),
        problem_id="sq_nonneg", model="m", created_at=1.0, updated_at=2.0,
    ))
    updated = manager.rename("20260101-000000-sq_nonneg", "Clean proof of sq_nonneg")
    assert updated is not None
    assert updated.title == "Clean proof of sq_nonneg"
    assert manager.get("20260101-000000-sq_nonneg").title == "Clean proof of sq_nonneg"


def test_session_manager_rename_creates_record_for_unindexed(tmp_path: Path) -> None:
    sp = tmp_path / "20260101-111111-prime_dvd.jsonl"
    sp.write_text('{"t": 1.0, "event": "start", "problem_id": "prime_dvd", '
                  '"statement": "theorem x := by sorry"}\n')
    manager = SessionManager(sessions_dir=tmp_path)
    updated = manager.rename("20260101-111111-prime_dvd", "Named after rename")
    assert updated is not None
    assert updated.title == "Named after rename"
    assert manager.get("20260101-111111-prime_dvd").title == "Named after rename"
