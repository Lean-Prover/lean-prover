"""`prover usage` CLI + proof-loop diagnostic-log wiring tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import main


def _write_usage_session(dir_path: Path, name: str, steps: int = 2) -> Path:
    recs = [{"t": 0, "event": "start", "problem_id": name, "statement": "t",
             "max_steps": 20, "model": "gpt-4o"}]
    for i in range(1, steps + 1):
        recs.append({"t": i, "event": "llm_request", "step": i})
        recs.append({"t": i, "event": "llm_response", "step": i,
                     "prompt_tokens": 100, "completion_tokens": 50,
                     "tokens": 150, "body": f"  body {i}"})
    sp = dir_path / f"{name}.jsonl"
    sp.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return sp


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


def test_no_args_launches_tui(monkeypatch) -> None:
    calls: list[object] = []

    def fake_tui(args: argparse.Namespace) -> int:
        calls.append(args)
        return 0

    monkeypatch.setattr(main, "cmd_tui", fake_tui)
    assert main.cli([]) == 0
    assert len(calls) == 1
    assert calls[0].parallel == 1


def test_tui_parallel_flag_reaches_cmd_tui(monkeypatch) -> None:
    calls: list[object] = []

    def fake_tui(args: argparse.Namespace) -> int:
        calls.append(args)
        return 0

    monkeypatch.setattr(main, "cmd_tui", fake_tui)
    with pytest.raises(SystemExit) as exc:
        main.cli(["tui", "-p", "3"])
    assert exc.value.code == 0
    assert calls[0].parallel == 3


def test_usage_cli_single_session(monkeypatch, tmp_path, capsys) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    _write_usage_session(sessions, "sess-a")
    monkeypatch.setenv("PROVER_SESSIONS_DIR", str(sessions))
    rc = main.cmd_usage(_ns(id="sess-a"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Usage for sess-a" in out
    assert "Total tokens" in out


def test_usage_cli_all_sessions(monkeypatch, tmp_path, capsys) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    _write_usage_session(sessions, "sess-a")
    _write_usage_session(sessions, "sess-b")
    monkeypatch.setenv("PROVER_SESSIONS_DIR", str(sessions))
    rc = main.cmd_usage(_ns(id=None))
    out = capsys.readouterr().out
    assert rc == 0
    assert "across 2 session" in out


def test_usage_cli_missing_session(monkeypatch, tmp_path, capsys) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    _write_usage_session(sessions, "sess-a")
    monkeypatch.setenv("PROVER_SESSIONS_DIR", str(sessions))
    rc = main.cmd_usage(_ns(id="nope"))
    assert rc == 1
    assert "session not found" in capsys.readouterr().out


# --------------------------------------------------------------------------- prover ask


def _fake_result(proved: bool = True):
    return SimpleNamespace(
        proved=proved,
        steps=3,
        seconds=2.5,
        total_tokens=150,
        total_prompt_tokens=100,
        total_completion_tokens=50,
        estimated_cost_usd=0.000123,
        session_path="/x/s.jsonl",
        proof="by norm_num" if proved else None,
    )


def test_ask_writes_json_to_stdout(monkeypatch, capsys) -> None:
    calls: list[object] = []

    def fake_prove(*a, **kw):
        calls.append((a, kw))
        return _fake_result(proved=True)

    monkeypatch.setattr(main, "prove", fake_prove)
    rc = main.cmd_ask(_ns(statement="theorem t : 1+1=2 := by norm_num",
                          max_steps=20, no_goal_feedback=False, no_record=False,
                          full_file=False, adaptive_steps=False))
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["proved"] is True
    assert payload["steps"] == 3
    assert payload["proof"] == "by norm_num"
    assert '"proved": true' in out


def test_ask_human_prose_to_stderr(monkeypatch, capsys) -> None:
    def fake_prove(*a, **kw):
        return _fake_result(proved=False)

    monkeypatch.setattr(main, "prove", fake_prove)
    rc = main.cmd_ask(_ns(statement="t", max_steps=20, no_goal_feedback=False,
                          no_record=False, full_file=False, adaptive_steps=False))
    captured = capsys.readouterr()
    # stdout is a single clean JSON line
    assert "\n".join(captured.out.splitlines()[1:]) == ""
    json.loads(captured.out)
    # prose goes to stderr
    assert "proved=False" in captured.err
    assert rc == 1


# --------------------------------------------------------------------------- project-defaults wiring


def test_project_max_steps_becomes_cli_default(monkeypatch, tmp_path) -> None:
    """A committed .prover.json max_steps sets the prove subcommand default."""
    import json as _json

    (tmp_path / ".prover.json").write_text(_json.dumps({"max_steps": 7}))
    monkeypatch.chdir(tmp_path)
    ns = main._build_parser().parse_args(["prove", "theorem t : True := by trivial"])
    assert ns.max_steps == 7


def test_project_workers_becomes_bench_parallel_default(monkeypatch, tmp_path) -> None:
    import json as _json

    (tmp_path / ".prover.json").write_text(_json.dumps({"workers": 4}))
    monkeypatch.chdir(tmp_path)
    ns = main._build_parser().parse_args(["bench"])
    assert ns.parallel == 4
    assert ns.max_steps == 20  # built-in floor when repo sets only workers


def test_quiet_suppresses_bench_progress(monkeypatch, tmp_path, capsys) -> None:
    """quiet=true in .prover.json silences per-problem progress in bench."""
    import json as _json

    (tmp_path / ".prover.json").write_text(_json.dumps({"quiet": True}))
    monkeypatch.chdir(tmp_path)

    calls: list[dict] = []

    def fake_prove_best_of(*a, **kw):
        calls.append(kw)
        r = _fake_result(proved=True)
        # prove_best_of returns an object with .attempts
        r2 = SimpleNamespace(**{**vars(r), "attempts": [], "trace": []})
        return r2

    monkeypatch.setattr(main, "prove_best_of", fake_prove_best_of)

    problems = tmp_path / "problems.json"
    problems.write_text(_json.dumps([
        {"id": "p1", "statement": "theorem p1 : 1+1=2 := by norm_num", "difficulty": "trivial"},
    ]))
    rc = main.cmd_bench(_ns(problems=str(problems), max_steps=2, start=1, parallel=1,
                            no_goal_feedback=False, no_record=False, no_hammers=False,
                            n_attempts=1, full_file=False, adaptive=False,
                            report=None, validate=False))
    out = capsys.readouterr().out
    assert rc == 0
    # no per-progress "[1/1] p1" line, but the score summary still prints
    assert "[1/1] p1" not in out
    assert "Score: 1/1" in out
