"""Direct unit tests for the proof repair loop (agent/loop.py).

Mocks the Lean toolchain (`lean.check_file`) and the LLM (`llm.chat`)
so the loop's control flow is exercised without Lake/Mathlib or a network
call — the gap the e2e resume tests only brush indirectly.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from agent import llm, loop

STATEMENT = "theorem prover_loop (n : ℕ) : n + 0 = n := by sorry"

# 1 native hammer check + 1 failed step-1 check + 1 passing step-2 check
PASS_ON_STEP2 = 3


@pytest.fixture
def hermetic(tmp_path: Path, monkeypatch):
    """Point all ~/.prover writes and the Lean dir at tmp_path."""
    monkeypatch.setenv("PROVER_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("PROVER_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(loop, "LEAN_DIR", tmp_path)
    # Far above any realistic history so auto-compaction never fires via tokens.
    monkeypatch.setattr(llm, "context_window_tokens", lambda: 1_000_000)
    return tmp_path


def install_check(monkeypatch, ok_on: set[int] | None = None) -> dict:
    """Fake `lean.check_file`: fails unless the call number is in ok_on."""
    state = {"calls": 0}

    def fake(_file: Path, _dir: Path) -> tuple[bool, str]:
        state["calls"] += 1
        if ok_on is not None and state["calls"] in ok_on:
            return True, ""
        return False, "fake.lean:1:1: error: goals remain\n"

    monkeypatch.setattr(loop.lean, "check_file", fake)
    return state


def install_chat(monkeypatch, responses: list[str]) -> dict:
    """Fake `llm.chat` returning responses[i] in order (last repeats)."""
    state = {"calls": 0}

    def fake(system: str, messages: list[dict], temperature: float = 0.2,
             retries: int = 4, model_name: str | None = None) -> llm.LLMResponse:
        i = min(state["calls"], len(responses) - 1)
        state["calls"] += 1
        return llm.LLMResponse(content=responses[i], prompt_tokens=10,
                               completion_tokens=5, total_tokens=15)

    monkeypatch.setattr(loop.llm, "chat", fake)
    return state


def _event_names(r: loop.Result) -> list[str]:
    return [e["event"] for e in r.trace]


# ---------------------------------------------------------------- hammer pass

def test_native_hammer_solves_without_llm(hermetic, monkeypatch) -> None:
    """`prover_finish` (one Lean invocation) that closes the goal never
    spends LLM tokens."""
    install_check(monkeypatch, ok_on={1})
    chat_state = install_chat(monkeypatch, ["```lean\n  ring\n```"])
    r = loop.prove(STATEMENT, max_steps=5, verbose=False,
                   problem_id="hammer-solve", goal_feedback=False,
                   record_session=False)
    assert r.proved
    assert r.steps == 1
    assert chat_state["calls"] == 0
    names = _event_names(r)
    assert "hammer" in names
    assert "llm_start" not in names
    assert "prover_finish" in r.proof
    file_text = (hermetic / "lean" / "tmp" / "Prover_hammer-solve.lean").read_text()
    assert "prover_finish" in file_text


def test_fallback_hammer_loop_when_module_missing(hermetic, monkeypatch) -> None:
    """Without the ProverSupport olean the pre-pass falls back to running
    each hammer separately (old behavior)."""
    state = {"calls": 0}

    def check(_f, _d):
        state["calls"] += 1
        if state["calls"] == 1:
            return False, "fake.lean:1:1: error: unknown module prefix 'ProverSupport'\n"
        return (True, "") if state["calls"] == 4 else (False, "fake.lean:1:1: error: goals remain\n")

    monkeypatch.setattr(loop.lean, "check_file", check)
    chat_state = install_chat(monkeypatch, ["```lean\n  simp\n```"])
    r = loop.prove(STATEMENT, max_steps=5, verbose=False,
                   problem_id="hammer-late", goal_feedback=False,
                   record_session=False)
    assert r.proved
    assert r.steps == 3  # 3rd fallback hammer (linarith) closes it
    assert chat_state["calls"] == 0
    hammers = [e for e in r.trace if e["event"] == "hammer"]
    assert [h["tactic"] for h in hammers] == ["prover_finish", "ring", "omega", "linarith"]
    assert hammers[-1]["ok"] is True


def test_native_hammer_failure_goes_to_llm(hermetic, monkeypatch) -> None:
    """When `prover_finish` runs but fails, the loop proceeds to the LLM."""
    install_check(monkeypatch, ok_on=None)
    chat_state = install_chat(monkeypatch, ["```lean\n  ring\n```"])
    r = loop.prove(STATEMENT, max_steps=2, verbose=False,
                   problem_id="hammer-fail", goal_feedback=False,
                   record_session=False)
    assert chat_state["calls"] == 2
    hammers = [e for e in r.trace if e["event"] == "hammer"]
    assert len(hammers) == 1
    assert hammers[0]["tactic"] == "prover_finish"
    assert hammers[0]["ok"] is False


def test_prover_search_prepass_closes_without_llm(hermetic, monkeypatch) -> None:
    """PROVER_SEARCH=1 upgrades the pre-pass to `prover_search`; when it
    closes the goal, no LLM tokens are spent and the file carries the
    search heartbeat option."""
    install_check(monkeypatch, ok_on={1})
    chat_state = install_chat(monkeypatch, ["```lean\n  ring\n```"])
    monkeypatch.setenv("PROVER_SEARCH", "1")
    r = loop.prove(STATEMENT, max_steps=5, verbose=False,
                   problem_id="search-solve", goal_feedback=False,
                   record_session=False)
    assert r.proved
    assert r.steps == 1
    assert chat_state["calls"] == 0
    hammers = [e for e in r.trace if e["event"] == "hammer"]
    assert hammers[0]["tactic"] == "prover_search 3"
    assert hammers[0]["ok"] is True
    assert "prover_search" in r.proof
    file_text = (hermetic / "lean" / "tmp" / "Prover_search-solve.lean").read_text()
    assert "set_option maxHeartbeats 0" in file_text
    assert "prover_search" in file_text


def test_prover_search_prepass_failure_goes_to_llm(hermetic, monkeypatch) -> None:
    """When `prover_search` fails, the loop proceeds to the LLM exactly as
    with `prover_finish`."""
    install_check(monkeypatch, ok_on=None)
    chat_state = install_chat(monkeypatch, ["```lean\n  ring\n```"])
    monkeypatch.setenv("PROVER_SEARCH", "1")
    r = loop.prove(STATEMENT, max_steps=2, verbose=False,
                   problem_id="search-fail", goal_feedback=False,
                   record_session=False)
    assert chat_state["calls"] == 2
    hammers = [e for e in r.trace if e["event"] == "hammer"]
    assert len(hammers) == 1
    assert hammers[0]["tactic"] == "prover_search"
    assert hammers[0]["ok"] is False


# --------------------------------------------------------------- LLM repairs

def test_llm_repairs_until_build_passes(hermetic, monkeypatch) -> None:
    """Hammers fail → LLM drafts → next build check passes."""
    install_check(monkeypatch, ok_on={PASS_ON_STEP2})
    chat_state = install_chat(monkeypatch, ["```lean\n  ring\n```"])
    r = loop.prove(STATEMENT, max_steps=5, verbose=False,
                   problem_id="llm-repair", goal_feedback=False,
                   record_session=False)
    assert r.proved
    assert r.steps == 2
    assert chat_state["calls"] == 1
    names = _event_names(r)
    assert "llm_start" in names
    assert "llm_request" in names
    assert "llm_response" in names
    build = [e for e in r.trace if e["event"] == "build"]
    assert build[-1]["ok"] is True


def test_llm_error_is_emitted_and_logged(hermetic, monkeypatch,
                                         tmp_path: Path) -> None:
    """An [LLM error …] reply surfaces as an llm_error event and lands in the
    structured failure log; the loop keeps going until the build passes."""
    install_check(monkeypatch, ok_on={PASS_ON_STEP2 + 1})  # one retry turn
    install_chat(monkeypatch,
                 ["[LLM error: 429 rate limited]", "```lean\n  ring\n```"])
    r = loop.prove(STATEMENT, max_steps=5, verbose=False,
                   problem_id="llm-error", goal_feedback=False,
                   record_session=False)
    assert r.proved
    errors = [e for e in r.trace if e["event"] == "llm_error"]
    assert len(errors) == 1
    assert "429" in errors[0]["error"]
    log = tmp_path / "logs" / "agent-calls.jsonl"
    assert log.exists()
    assert '"kind": "llm_error"' in log.read_text()
    assert "429" in log.read_text()


def test_failed_extraction_keeps_assistant_turn(hermetic, monkeypatch) -> None:
    """A reply whose body extraction fails must still append the assistant
    turn: dropping it puts two user messages back-to-back (strict
    OpenAI-compatible backends reject that) and breaks compaction pairing."""
    install_check(monkeypatch, ok_on=None)
    histories: list[list[dict]] = []

    def chat(system, messages, temperature=0.2, retries=4, model_name=None):
        histories.append([dict(m) for m in messages])
        # Only a restated theorem inside the block → _extract_body() == ""
        return llm.LLMResponse(
            content="```lean\ntheorem t : True := by trivial\n```",
            prompt_tokens=1, completion_tokens=1, total_tokens=2)

    monkeypatch.setattr(loop.llm, "chat", chat)
    r = loop.prove(STATEMENT, max_steps=2, verbose=False,
                   problem_id="alternation", goal_feedback=False,
                   record_session=False, skip_hammers=True)
    assert not r.proved
    assert len(histories) == 2
    for msgs in histories:
        roles = [m["role"] for m in msgs]
        for a, b in itertools.pairwise(roles):
            assert not (a == "user" and b == "user")
    # Step 2 saw the model's own (rejected) reply before its new user turn.
    assert histories[1][-2]["role"] == "assistant"
    assert histories[1][-1]["role"] == "user"


def test_exhausts_max_steps(hermetic, monkeypatch) -> None:
    """A never-solved problem burns the whole budget and reports failure."""
    check_state = install_check(monkeypatch, ok_on=None)
    chat_state = install_chat(monkeypatch, ["```lean\n  simp\n```"])
    r = loop.prove(STATEMENT, max_steps=3, verbose=False,
                   problem_id="exhaust", goal_feedback=False,
                   record_session=False)
    assert not r.proved
    assert r.steps == 3
    assert not r.stopped
    assert chat_state["calls"] == 3
    assert r.total_tokens == 3 * 15
    assert check_state["calls"] == 1 + 3  # 1 native hammer check + 3 step checks
    result = [e for e in r.trace if e["event"] == "result"][-1]
    assert result["proved"] is False


# --------------------------------------------------------------- control flow

def test_stop_requested_aborts(hermetic, monkeypatch) -> None:
    """should_stop is honored: the run reports stopped and stops early."""
    install_check(monkeypatch, ok_on=None)
    install_chat(monkeypatch, ["```lean\n  simp\n```"])
    r = loop.prove(STATEMENT, max_steps=10, verbose=False,
                   problem_id="stop", goal_feedback=False,
                   record_session=False, should_stop=lambda: True)
    assert not r.proved
    assert r.stopped
    assert r.steps == 0
    result = [e for e in r.trace if e["event"] == "result"][-1]
    assert result["stopped"] is True


def test_skip_hammers_goes_straight_to_llm(hermetic, monkeypatch) -> None:
    """skip_hammers=True: no hammer events, first check is the LLM's sorry."""
    install_check(monkeypatch, ok_on={2})  # step-1 fail + step-2 pass
    chat_state = install_chat(monkeypatch, ["```lean\n  ring\n```"])
    r = loop.prove(STATEMENT, max_steps=5, verbose=False,
                   problem_id="no-hammers", goal_feedback=False,
                   record_session=False, skip_hammers=True)
    assert r.proved
    assert "hammer" not in _event_names(r)
    assert chat_state["calls"] == 1


def test_goal_feedback_emits_goals(hermetic, monkeypatch) -> None:
    """With goal_feedback on, the loop asks the LSP and emits a goals event."""
    install_check(monkeypatch, ok_on={PASS_ON_STEP2})
    install_chat(monkeypatch, ["```lean\n  ring\n```"])

    class FakeLSP:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def update(self, text: str) -> None:
            pass

        def goal_at_end(self, text: str) -> str:
            return "⊢ n + 0 = n"

        def close(self) -> None:
            pass

    monkeypatch.setattr(loop.lsp, "LeanLSP", FakeLSP)
    r = loop.prove(STATEMENT, max_steps=5, verbose=False,
                   problem_id="goals", goal_feedback=True,
                   record_session=False)
    goals = [e for e in r.trace if e["event"] == "goals"]
    assert len(goals) == 1
    assert "⊢ n + 0 = n" in goals[0]["goals"]
    assert r.proved


def test_compaction_event_after_many_failed_steps(hermetic, monkeypatch) -> None:
    """Long runs fold old dead-end turns into a compaction summary (turn-based)."""
    install_check(monkeypatch, ok_on=None)
    chat_state = install_chat(monkeypatch, ["```lean\n  simp\n```"])
    r = loop.prove(STATEMENT, max_steps=22, verbose=False,
                   problem_id="compact", goal_feedback=False,
                   record_session=False)
    assert not r.proved
    assert chat_state["calls"] == 22
    compactions = [e for e in r.trace if e["event"] == "compaction"]
    assert len(compactions) == 1
    assert compactions[0]["dropped"] == 12
    assert r.total_tokens == 22 * 15


# ------------------------------------------------------------------- recording

def test_record_session_writes_jsonl_and_index(hermetic, tmp_path: Path,
                                               monkeypatch) -> None:
    """record_session=True persists the event stream + an index record."""
    install_check(monkeypatch, ok_on={PASS_ON_STEP2})
    install_chat(monkeypatch, ["```lean\n  ring\n```"])
    r = loop.prove(STATEMENT, max_steps=5, verbose=False,
                   problem_id="recorded", goal_feedback=False)
    assert r.session_path is not None
    session_file = Path(r.session_path)
    assert session_file.exists()
    text = session_file.read_text()
    assert '"event": "start"' in text
    assert '"event": "result"' in text
    index = tmp_path / "sessions" / "index.jsonl"
    assert index.exists()
    assert '"status": "proved"' in index.read_text()


# ------------------------------------------------------------- best-of-N search

def _fake_prove(monkeypatch, outcomes: list[tuple[bool, int]], captures: dict) -> None:
    """Fake `loop.prove` returning canned (proved, steps) outcomes in order."""
    i = {"n": 0}

    def fake(*args, **kwargs):
        idx = min(i["n"], len(outcomes) - 1)
        i["n"] += 1
        proved, steps = outcomes[idx]
        stmt = kwargs.get("statement") or (args[0] if args else "")
        captures["calls"].append(kwargs)
        return loop.Result(statement=stmt, proved=proved, steps=steps,
                           seconds=0.1, total_tokens=1, estimated_cost_usd=0.0)

    monkeypatch.setattr(loop, "prove", fake)
    return i


def test_best_of_single_attempt_delegates(hermetic, monkeypatch) -> None:
    captures = {"calls": []}
    _fake_prove(monkeypatch, [(True, 3)], captures)
    r = loop.prove_best_of(STATEMENT, n_attempts=1, verbose=False)
    assert len(captures["calls"]) == 1
    assert r.proved
    assert len(r.attempts) == 1


def test_best_of_stops_on_first_proof(hermetic, monkeypatch) -> None:
    captures = {"calls": []}
    _fake_prove(monkeypatch, [(True, 2), (False, 5)], captures)
    r = loop.prove_best_of(STATEMENT, n_attempts=3, verbose=False)
    assert len(captures["calls"]) == 1
    assert r.proved and r.steps == 2


def test_best_of_all_fail_returns_furthest(hermetic, monkeypatch) -> None:
    captures = {"calls": []}
    _fake_prove(monkeypatch, [(False, 2), (False, 7), (False, 4)], captures)
    r = loop.prove_best_of(STATEMENT, n_attempts=3, verbose=False)
    assert len(captures["calls"]) == 3
    assert not r.proved
    assert r.steps == 7
    assert len(r.attempts) == 3
    assert [a.steps for a in r.attempts] == [2, 7, 4]


def test_best_of_ramps_temperature_and_skips_hammers(hermetic, monkeypatch) -> None:
    captures = {"calls": []}
    _fake_prove(monkeypatch, [(False, 1), (False, 1)], captures)
    loop.prove_best_of(STATEMENT, n_attempts=2, verbose=False, temperature=0.2)
    temps = [c["temperature"] for c in captures["calls"]]
    assert temps == pytest.approx([0.2, 0.6])
    assert captures["calls"][0]["skip_hammers"] is False
    assert captures["calls"][1]["skip_hammers"] is True


def test_best_of_clamps_n_attempts(hermetic, monkeypatch) -> None:
    captures = {"calls": []}
    _fake_prove(monkeypatch, [(False, 1)], captures)
    loop.prove_best_of(STATEMENT, n_attempts=0, verbose=False)
    assert len(captures["calls"]) == 1


def test_prove_threads_temperature_to_chat(hermetic, monkeypatch) -> None:
    """The temperature passed to prove() reaches the LLM call."""
    install_check(monkeypatch, ok_on=None)  # every build fails
    temps: list[float] = []

    def chat(system, messages, temperature=0.2, retries=4, model_name=None):
        temps.append(temperature)
        return llm.LLMResponse(content="```lean\n  simp\n```", prompt_tokens=10,
                               completion_tokens=5, total_tokens=15)

    monkeypatch.setattr(loop.llm, "chat", chat)
    loop.prove(STATEMENT, max_steps=2, verbose=False,
               problem_id="temp", goal_feedback=False,
               record_session=False, temperature=0.7)
    assert temps and all(t == 0.7 for t in temps)


# ------------------------------------------------------------------ retrieval

def _retrieval_chat(monkeypatch, captured: list) -> None:
    def chat(system, messages, temperature=0.2, retries=4, model_name=None):
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        captured.append(user_msgs[-1] if user_msgs else "")
        return llm.LLMResponse(content="```lean\n  simp\n```", prompt_tokens=10,
                               completion_tokens=5, total_tokens=15)

    monkeypatch.setattr(loop.llm, "chat", chat)


def test_retrieval_disabled_by_default(hermetic, monkeypatch) -> None:
    install_check(monkeypatch, ok_on=None)
    captured: list = []
    _retrieval_chat(monkeypatch, captured)
    r = loop.prove(STATEMENT, max_steps=1, verbose=False,
                   problem_id="no-retrieve", goal_feedback=False,
                   record_session=False)
    assert "retrieve" not in [e["event"] for e in r.trace]
    assert "Relevant Mathlib lemmas" not in captured[0]


def test_retrieval_hints_injected_when_enabled(hermetic, monkeypatch) -> None:
    install_check(monkeypatch, ok_on=None)
    monkeypatch.setenv("PROVER_RETRIEVE", "1")
    monkeypatch.setattr("agent.retrieval.search_lemmas",
                        lambda *a, **k: [{"name": "nat_add_comm", "signature": "theorem nat_add_comm (a b : ℕ) : a + b = b + a"}])
    captured: list = []
    _retrieval_chat(monkeypatch, captured)
    r = loop.prove(STATEMENT, max_steps=1, verbose=False,
                   problem_id="retrieve", goal_feedback=False,
                   record_session=False)
    events = [e["event"] for e in r.trace]
    assert "retrieve" in events
    retrieve = next(e for e in r.trace if e["event"] == "retrieve")
    assert retrieve["lemmas"] == ["nat_add_comm"]
    assert "nat_add_comm : theorem nat_add_comm" in captured[0]


def test_retrieval_failure_does_not_break_loop(hermetic, monkeypatch) -> None:
    """A broken index must never abort the proof loop."""
    install_check(monkeypatch, ok_on=None)
    monkeypatch.setenv("PROVER_RETRIEVE", "1")
    def boom(*a, **k):
        raise RuntimeError("index corrupt")
    monkeypatch.setattr("agent.retrieval.search_lemmas", boom)
    captured: list = []
    _retrieval_chat(monkeypatch, captured)
    r = loop.prove(STATEMENT, max_steps=1, verbose=False,
                   problem_id="retrieve-boom", goal_feedback=False,
                   record_session=False)
    assert "retrieve" not in [e["event"] for e in r.trace]
    assert captured[0].startswith("Theorem signature:")


def test_retrieval_corpus_hit_shows_proof(hermetic, monkeypatch) -> None:
    """A corpus entry (Lean-proved statement) is surfaced with its tactic so
    the model knows the goal is reachable by the native chain."""
    install_check(monkeypatch, ok_on=None)
    monkeypatch.setenv("PROVER_RETRIEVE", "1")
    monkeypatch.setattr(
        "agent.retrieval.search_lemmas",
        lambda *a, **k: [{
            "name": "tpl_add_zero_nat",
            "signature": "theorem prover_tpl_add_zero_nat (a : ℕ) : a + 0 = a",
            "file": "corpus",
            "proof": "prover_finish",
        }])
    captured: list = []
    _retrieval_chat(monkeypatch, captured)
    r = loop.prove(STATEMENT, max_steps=1, verbose=False,
                   problem_id="retrieve-corpus", goal_feedback=False,
                     record_session=False)
    assert "proven by prover_finish" in captured[0]
    retrieve = next(e for e in r.trace if e["event"] == "retrieve")
    assert retrieve["corpus"] == 1

def test_proof_body_extract_and_cap() -> None:
    assert loop._proof_body("prover_search") == ""
    assert loop._proof_body("") == ""
    body = loop._proof_body("theorem t : True := by trivial")
    assert body == "  trivial"
    body = loop._proof_body("by\n  refine ⟨?_, ?_⟩\n  · simp")
    assert "refine ⟨?_, ?_⟩" in body and all(ln.startswith("  ") for ln in body.splitlines())
    long_body = "by\n" + "\n".join(f"  have h{i} : True := trivial" for i in range(40))
    assert loop._proof_body(long_body) == ""  # exceeds 300-char cap


def test_split_signature_cuts_at_last_by() -> None:
    # Scaffolded benchmarks (Putnam `_solution` abbrevs, FormalQualBench defs)
    # carry several `:= by`s ahead of the target theorem: cut at the LAST one.
    stmt = (
        "namespace Foo\n\ndef helper (n : ℕ) : ℕ := by\n  exact n\n\n"
        "theorem prover_putnam_1962_a2 (P : Prop) : P := by\n  sorry"
    )
    sig = loop._split_signature(stmt)
    assert sig.endswith("theorem prover_putnam_1962_a2 (P : Prop) : P := by")
    assert sig.count(":= by") == 2  # helper's `:= by` kept intact
    assert loop._split_signature("theorem t : True") == "theorem t : True := by"


def test_retrieval_worked_example_injected(hermetic, monkeypatch) -> None:
    """LLM-verified corpus entry → worked example with real tactics in prompt."""
    install_check(monkeypatch, ok_on=None)
    monkeypatch.setenv("PROVER_RETRIEVE", "1")
    monkeypatch.setattr(
        "agent.retrieval.search_lemmas",
        lambda *a, **k: [{
            "name": "auto_parity_abclx",
            "signature": "theorem parity (n : ℕ) : 2 ∣ n * (n + 1)",
            "file": "corpus",
            "proof": "by\n  have h : n % 2 = 0 ∨ n % 2 = 1 := by omega\n  rcases h with (h | h)\n  · refine ⟨n / 2 * (n + 1), ?_⟩\n    omega",
        }])
    captured: list = []
    _retrieval_chat(monkeypatch, captured)
    r = loop.prove(STATEMENT, max_steps=1, verbose=False,
                   problem_id="retrieve-example", goal_feedback=False,
                   record_session=False)
    assert "Worked examples" in captured[0]
    assert "rcases h with (h | h)" in captured[0]
    retrieve = next(e for e in r.trace if e["event"] == "retrieve")
    assert retrieve["corpus"] == 1


# -------------------------------------------------------------------- routing

def test_router_selects_model_for_difficulty(hermetic, monkeypatch) -> None:
    """PROVER_MODEL_<TIER> routes the model actually used for the LLM call."""
    install_check(monkeypatch, ok_on=None)
    monkeypatch.setenv("PROVER_MODEL_EASY", "routed-model")
    monkeypatch.setenv("PROVER_TEMP_EASY", "0.5")
    monkeypatch.setenv("PROVER_STEPS_EASY", "3")
    seen: dict = {}

    def chat(system, messages, temperature=0.2, retries=4, model_name=None):
        seen["model"] = model_name
        seen["temperature"] = temperature
        return llm.LLMResponse(content="```lean\n  simp\n```", prompt_tokens=10,
                               completion_tokens=5, total_tokens=15)

    monkeypatch.setattr(loop.llm, "chat", chat)
    r = loop.prove(STATEMENT, max_steps=2, verbose=False,
                   problem_id="routed", goal_feedback=False,
                   record_session=False, difficulty="easy")
    assert seen["model"] == "routed-model"
    assert seen["temperature"] == 0.5
    # routed max_steps caps the loop
    assert not r.proved
    assert r.steps == 3


def test_router_unset_difficulty_uses_default_model(hermetic, monkeypatch) -> None:
    install_check(monkeypatch, ok_on=None)
    monkeypatch.delenv("PROVER_MODEL", raising=False)
    seen: dict = {}

    def chat(system, messages, temperature=0.2, retries=4, model_name=None):
        seen["model"] = model_name
        return llm.LLMResponse(content="```lean\n  simp\n```", prompt_tokens=10,
                               completion_tokens=5, total_tokens=15)

    monkeypatch.setattr(loop.llm, "chat", chat)
    loop.prove(STATEMENT, max_steps=1, verbose=False,
               problem_id="default-model", goal_feedback=False,
               record_session=False)
    assert seen["model"] == "gpt-4o"


# ------------------------------------------------------------------ lemma plan

def test_lemma_plan_prepends_proven_helpers(hermetic, monkeypatch) -> None:
    """PROVER_LEMMA_PLAN=1 proves helpers first and prepends them to the file."""
    install_check(monkeypatch, ok_on=None)
    monkeypatch.setenv("PROVER_LEMMA_PLAN", "1")
    lemma_decl = "theorem prover_plan_1 (a b : ℕ) : a + b = b + a := by\n  omega"
    monkeypatch.setattr("agent.plan.propose_lemmas",
                        lambda *a, **k: ["theorem prover_plan_1 (a b : ℕ) : a + b = b + a := by\n  sorry"])
    monkeypatch.setattr("agent.plan.prove_lemmas", lambda *a, **k: [lemma_decl])
    captured: list = []
    _retrieval_chat(monkeypatch, captured)
    r = loop.prove(STATEMENT, max_steps=1, verbose=False,
                   problem_id="planned", goal_feedback=False,
                   record_session=False)
    events = [e["event"] for e in r.trace]
    assert "plan" in events
    assert "plan_lemmas" in events
    assert r.trace[events.index("plan_lemmas")]["proven"] == [lemma_decl[:80]]
    # the helper lemma is in the file above the main theorem
    f = hermetic / "lean" / "tmp" / "Prover_planned.lean"
    text = f.read_text()
    assert "theorem prover_plan_1 (a b : ℕ) : a + b = b + a := by" in text
    assert text.index("prover_plan_1") < text.index("prover_loop")
    # the model was told about the helpers
    assert "prover_plan_1" in captured[0]


def test_lemma_plan_off_by_default(hermetic, monkeypatch) -> None:
    install_check(monkeypatch, ok_on=None)
    monkeypatch.delenv("PROVER_LEMMA_PLAN", raising=False)
    captured: list = []
    _retrieval_chat(monkeypatch, captured)
    r = loop.prove(STATEMENT, max_steps=1, verbose=False,
                   problem_id="unplanned", goal_feedback=False,
                   record_session=False)
    assert "plan" not in [e["event"] for e in r.trace]
    assert "prover_plan_" not in captured[0]


# --------------------------------------------------------------- full-file mode

SIG = "theorem prover_full (a b : ℕ) : a + b = b + a := by"


def test_extract_full_file_keeps_helpers_and_enforces_signature() -> None:
    reply = """```lean
import Mathlib

lemma helper (a : ℕ) : a + 0 = a := by
  omega

theorem prover_full (a b : ℕ) : a + b = b + a := by
  rw [add_comm]
```"""
    out = loop._extract_full_file(reply, SIG)
    assert out is not None
    assert "import Mathlib" not in out  # header owns the import
    assert "lemma helper (a : ℕ) : a + 0 = a := by" in out
    assert "theorem prover_full (a b : ℕ) : a + b = b + a := by" in out
    assert "  rw [add_comm]" in out
    # the canonical signature replaced the model's declaration (same here),
    # and the model's own decl text is not duplicated
    assert out.count("theorem prover_full") == 1


def test_extract_full_file_returns_none_when_theorem_missing() -> None:
    reply = """```lean
import Mathlib

lemma helper (a : ℕ) : a = a := by
  rfl
```"""
    assert loop._extract_full_file(reply, SIG) is None
    assert loop._extract_full_file("no code block here", SIG) is None


def test_extract_full_file_replaces_changed_statement() -> None:
    """The model's altered statement must be overwritten by ours."""
    reply = """```lean
theorem prover_full (a b : ℕ) : a + b = b := by
  omega
```"""
    out = loop._extract_full_file(reply, SIG)
    assert out is not None
    assert "theorem prover_full (a b : ℕ) : a + b = b + a := by" in out
    assert "a + b = b := by" not in out
    assert "  omega" in out


def _chat_with(monkeypatch, responses: list[str]) -> None:
    state = {"n": 0}

    def chat(system, messages, temperature=0.2, retries=4, model_name=None):
        i = min(state["n"], len(responses) - 1)
        state["n"] += 1
        return llm.LLMResponse(content=responses[i], prompt_tokens=10,
                               completion_tokens=5, total_tokens=15)

    monkeypatch.setattr(loop.llm, "chat", chat)


def test_full_file_mode_proves_with_helpers(hermetic, monkeypatch) -> None:
    """Full-file mode: model's whole file is used and the loop proves."""
    state = {"calls": 0}

    def check(_f, _d):
        state["calls"] += 1
        return (True, "") if state["calls"] >= 2 else (False, "fake.lean:1:1: error: goals remain\n")

    monkeypatch.setattr(loop.lean, "check_file", check)
    full_reply = """```lean
lemma helper (a : ℕ) : a + 0 = a := by
  omega

theorem prover_full (a b : ℕ) : a + b = b + a := by
  rw [add_comm]
```"""
    _chat_with(monkeypatch, [full_reply])
    stmt = "theorem prover_full (a b : ℕ) : a + b = b + a := by sorry"
    r = loop.prove(stmt, max_steps=5, verbose=False, problem_id="full",
                   goal_feedback=False, record_session=False,
                   skip_hammers=True, full_file=True)
    assert r.proved
    assert r.steps == 2
    text = (hermetic / "lean" / "tmp" / "Prover_full.lean").read_text()
    assert "lemma helper (a : ℕ) : a + 0 = a := by" in text
    assert "theorem prover_full (a b : ℕ) : a + b = b + a := by" in text
    assert "  rw [add_comm]" in text


def test_full_file_reject_fed_back_then_proves(hermetic, monkeypatch) -> None:
    """A reply without the target theorem is rejected and the note is fed back."""
    state = {"calls": 0}

    def check(_f, _d):
        state["calls"] += 1
        return (True, "") if state["calls"] >= 3 else (False, "fake.lean:1:1: error: goals remain\n")

    monkeypatch.setattr(loop.lean, "check_file", check)
    user_msgs: list[str] = []

    def chat(system, messages, temperature=0.2, retries=4, model_name=None):
        user_msgs.append(messages[-1]["content"])
        idx = len(user_msgs) - 1
        if idx == 0:
            return llm.LLMResponse(content="```lean\nlemma helper (a : ℕ) : a = a := by rfl\n```",
                                   prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return llm.LLMResponse(content="```lean\ntheorem prover_full (a b : ℕ) : a + b = b + a := by\n  rw [add_comm]\n```",
                               prompt_tokens=1, completion_tokens=1, total_tokens=2)

    monkeypatch.setattr(loop.llm, "chat", chat)
    stmt = "theorem prover_full (a b : ℕ) : a + b = b + a := by sorry"
    r = loop.prove(stmt, max_steps=5, verbose=False, problem_id="full2",
                   goal_feedback=False, record_session=False,
                   skip_hammers=True, full_file=True)
    assert r.proved
    assert "was not accepted" in user_msgs[1]
    assert "`theorem prover_full`" in user_msgs[1]


# ------------------------------------------------------------------ adaptive

def test_adaptive_steps_extends_on_progress(hermetic, monkeypatch) -> None:
    """Progress (fewer diagnostics) near the budget extends it."""
    state = {"n": 0}

    def check(_f, _d):
        state["n"] += 1
        return (False, "fake.lean:1:1: error: goals remain\n" * max(1, 5 - state["n"]))

    monkeypatch.setattr(loop.lean, "check_file", check)
    _chat_with(monkeypatch, ["```lean\n  simp\n```"] * 30)
    r = loop.prove(STATEMENT, max_steps=4, verbose=False,
                   problem_id="adaptive", goal_feedback=False,
                   record_session=False, skip_hammers=True, adaptive_steps=True)
    assert not r.proved
    events = [e["event"] for e in r.trace]
    assert "extend" in events
    ext = next(e for e in r.trace if e["event"] == "extend")
    assert ext["from_steps"] == 4 and ext["to_steps"] == 6
    assert r.steps >= 6


def test_adaptive_steps_no_extend_without_progress(hermetic, monkeypatch) -> None:
    """No progress → budget stays put."""
    def check(_f, _d):
        return (False, "fake.lean:1:1: error: goals remain\n")

    monkeypatch.setattr(loop.lean, "check_file", check)
    _chat_with(monkeypatch, ["```lean\n  simp\n```"] * 20)
    r = loop.prove(STATEMENT, max_steps=4, verbose=False,
                   problem_id="adaptive2", goal_feedback=False,
                   record_session=False, skip_hammers=True, adaptive_steps=True)
    assert not r.proved
    assert "extend" not in [e["event"] for e in r.trace]
    assert r.steps == 4
