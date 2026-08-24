"""End-to-end resume/branch tests for the proof loop.

Mocks the LLM (tau's FakeProvider analogue) and drives loop.prove() against
a hand-written recorded session, asserting the resume contract:
- a `resume` event is emitted with seed metadata
- the hammer pre-pass is skipped on resume
- the LLM sees the prior turn's diagnostics + reply as seeded history
- branch_at=0 discards the seed entirely (fresh start)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import llm, loop
from agent.session_manager import SessionManager, SessionRecord

SEED_SESSION_ID = "20260101-000000-seed"
STATEMENT = "theorem prover_seed (n : ℕ) : n + 0 = n := by\n  sorry"


@pytest.fixture(autouse=True)
def _real_lean_dir(monkeypatch):
    """These are true end-to-end tests: the real lake/Mathlib workspace must
    be used, so undo the conftest-wide PROVER_LEAN_DIR isolation."""
    monkeypatch.delenv("PROVER_LEAN_DIR", raising=False)


def _seed_session(sessions_dir: Path) -> None:
    """Write a one-step failed session + index record to seed a resume."""
    records = [
        {"t": 0, "event": "start", "problem_id": "seed", "statement": STATEMENT,
         "max_steps": 20, "model": "fake"},
        {"t": 0, "event": "llm_start"},
        {"t": 1, "event": "build", "step": 1, "ok": False, "diagnostics": 1,
         "summary": "goals remain", "report": "fake report"},
        {"t": 1, "event": "goals", "step": 1, "goals": "⊢ n + 0 = n"},
        {"t": 1, "event": "llm_request", "step": 1},
        {"t": 1, "event": "llm_response", "step": 1, "tokens": 10,
         "body": "  induction n with\n  | zero => rfl\n  | succ n ih =>"},
        {"t": 2, "event": "result", "proved": False, "steps": 20},
    ]
    sp = sessions_dir / f"{SEED_SESSION_ID}.jsonl"
    sessions_dir.mkdir(exist_ok=True)
    sp.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    SessionManager(sessions_dir).upsert(SessionRecord(
        id=SEED_SESSION_ID, path=str(sp), problem_id="seed",
        model="fake", status="failed", created_at=1, updated_at=1,
    ))


@pytest.fixture
def fake_llm(monkeypatch):
    """LLM that always answers with a ring body; counts calls."""
    calls = {"n": 0}

    def fake_chat(system, messages, temperature=0.2, retries=4, model_name=None):
        calls["n"] += 1
        return llm.LLMResponse(
            content="```lean\n  ring\n```",
            prompt_tokens=5, completion_tokens=2, total_tokens=7,
        )

    monkeypatch.setattr(llm, "chat", fake_chat)
    return calls


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("PROVER_SESSIONS_DIR", str(tmp_path / "sessions"))
    _seed_session(tmp_path / "sessions")
    return tmp_path / "sessions"


@pytest.mark.anyio
async def test_resume_seeds_history_and_skips_hammers(fake_llm, sessions):
    r = loop.prove(STATEMENT, max_steps=3, verbose=False, problem_id="resume-e2e",
                   resume_from=SEED_SESSION_ID, goal_feedback=False)
    events = [e["event"] for e in r.trace]

    assert "resume" in events
    assert "hammer" not in events  # hammer pass skipped on resume
    # history seeded with the prior turn's diagnostics + reply
    assert "fake report" in r.history[0]["content"]
    assert "induction n" in r.history[1]["content"]
    assert fake_llm["n"] == 1

    rev = next(e for e in r.trace if e["event"] == "resume")
    assert rev["seed_turns"] == 1
    assert rev["branch_at"] is None


@pytest.mark.anyio
async def test_resume_finishes_the_proof(fake_llm, sessions):
    """The fake LLM's ring answer closes the goal on the first build check."""
    r = loop.prove(STATEMENT, max_steps=3, verbose=False, problem_id="resume-fin",
                   resume_from=SEED_SESSION_ID, goal_feedback=False)
    assert r.proved is True


@pytest.mark.anyio
async def test_branch_at_zero_discards_seed(fake_llm, sessions):
    r = loop.prove(STATEMENT, max_steps=3, verbose=False, problem_id="branch-e2e",
                   resume_from=SEED_SESSION_ID, branch_at=0, goal_feedback=False)
    events = [e["event"] for e in r.trace]
    assert "resume" not in events  # nothing was seeded
    assert "hammer" in events  # full hammer pass ran instead


@pytest.mark.anyio
async def test_branch_at_keeps_only_first_turn(fake_llm, sessions):
    r = loop.prove(STATEMENT, max_steps=3, verbose=False, problem_id="branch1",
                   resume_from=SEED_SESSION_ID, branch_at=1, goal_feedback=False)
    rev = next(e for e in r.trace if e["event"] == "resume")
    assert rev["branch_at"] == 1
    assert rev["seed_turns"] == 1


@pytest.mark.anyio
async def test_branch_summary_seeded_before_history(fake_llm, sessions):
    """A model branch summary is prepended as its own user turn — the seeded
    history must still be visible after it (tau's preamble seeding)."""
    from agent.branch_summary import BRANCH_SUMMARY_PREAMBLE

    r = loop.prove(STATEMENT, max_steps=3, verbose=False, problem_id="branch2",
                   resume_from=SEED_SESSION_ID, branch_at=1, goal_feedback=False,
                   branch_summary="## Goal\nfinish the induction")
    assert r.history[0]["role"] == "user"
    assert r.history[0]["content"].startswith(BRANCH_SUMMARY_PREAMBLE)
    assert "finish the induction" in r.history[0]["content"]
    assert "fake report" in r.history[1]["content"]  # seeded turn still present
    assert r.proved is True


@pytest.mark.anyio
async def test_unknown_session_id_falls_through(fake_llm, sessions):
    """A missing session is ignored; the run proceeds from scratch."""
    r = loop.prove(STATEMENT, max_steps=3, verbose=False, problem_id="noid",
                   resume_from="does-not-exist", goal_feedback=False)
    assert "resume" not in [e["event"] for e in r.trace]
    assert "hammer" in [e["event"] for e in r.trace]
