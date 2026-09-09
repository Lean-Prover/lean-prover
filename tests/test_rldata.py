"""Unit tests for GRPO rollout generation (agent/rldata.py).

The rollout *logic* (trace → completions/rewards extraction + format) is
tested offline by faking ``prove_best_of``; no LLM, no Lean, no network.
"""

from __future__ import annotations

from pathlib import Path

from agent import rldata
from agent.prover_loop import Result


def _result(proved: bool, bodies: list[str]) -> Result:
    return Result(
        statement="s", proved=proved, steps=1, seconds=0.1,
        trace=[{"type": "llm_response", "accepted": True, "body": b} for b in bodies],
    )


def _group(monkeypatch, problems, attempts, n_attempts=2):
    def fake(statement, n_attempts=1, **kwargs):
        return Result(
            statement=statement, proved=any(a.proved for a in attempts),
            steps=1, seconds=0.1, attempts=attempts)

    import agent.prover_loop as pl

    monkeypatch.setattr(pl, "prove_best_of", fake)
    return rldata.generate(problems, n_attempts=n_attempts)


def test_generate_emits_prompt_completions_rewards(monkeypatch) -> None:
    problems = [{"id": "p1", "statement": "theorem p1 : True := by sorry"}]
    groups = _group(monkeypatch, problems,
                    attempts=[_result(False, ["omega"]), _result(True, ["simp"])])
    assert len(groups) == 1
    g = groups[0]
    assert g["id"] == "p1"
    assert "theorem p1 : True" in g["prompt"]
    assert g["completions"] == ["omega", "simp"]
    assert g["rewards"] == [0.0, 1.0]


def test_generate_uses_final_body_per_attempt(monkeypatch) -> None:
    problems = [{"id": "p1", "statement": "theorem p1 : True := by sorry"}]
    # attempt 1 tries two bodies, fails; attempt 2 succeeds on first try
    groups = _group(monkeypatch, problems,
                    attempts=[_result(False, ["simp", "omega"]), _result(True, ["exact rfl"])])
    g = groups[0]
    assert g["completions"] == ["omega", "exact rfl"]
    assert g["rewards"] == [0.0, 1.0]


def test_generate_skips_attempts_without_llm_body(monkeypatch) -> None:
    problems = [{"id": "p1", "statement": "theorem p1 : True := by sorry"}]
    # hammer-only attempt (no llm_response bodies) contributes nothing
    groups = _group(monkeypatch, problems,
                    attempts=[Result(statement="s", proved=True, steps=1,
                                     seconds=0.1, trace=[]),
                              _result(False, ["omega"])])
    g = groups[0]
    assert g["completions"] == ["omega"]
    assert g["rewards"] == [0.0]


def test_write_jsonl_roundtrip(tmp_path: Path) -> None:
    groups = [{"id": "p", "prompt": "sys", "completions": ["a"], "rewards": [1.0]}]
    out = tmp_path / "grpo.jsonl"
    n = rldata.write_jsonl(groups, out)
    assert n == 1
    assert out.exists()
    import json
    assert json.loads(out.read_text().strip()) == groups[0]
