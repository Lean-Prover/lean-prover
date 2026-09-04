"""Beam/tree search over the Lean repair loop (``agent/search.py``).

``prover search --beam N`` keeps a frontier of the N most promising proof
states and expands each with the LLM+Lean repair step, pruning dead ends by
a progress heuristic (fewer compiler diagnostics / open goals). This
concentrates the step budget on promising paths instead of running N
independent trajectories (best-of-N) with equal budget — the AlphaProof-class
idea, at beam granularity.

The generic ``beam_search`` core is deliberately provider-agnostic and is
unit-tested with a deterministic stub ``step``; ``_lean_step`` is the concrete
LLM+Lean expansion used by the CLI.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import lean, llm
from .loop import LEAN_DIR
from .prover_loop import SYSTEM_FULL, _extract_body, _split_signature

DEFAULT_BEAM = 3


@dataclass
class State:
    """One node in the proof-search frontier."""

    statement: str = ""
    body: str = ""
    history: list = field(default_factory=list)
    diag_count: int = 0
    goal_count: int = 0
    proved: bool = False
    proof: str = ""
    last_output: str = ""  # compiler diagnostics from the most recent build


def _score(s: State) -> tuple[int, int]:
    """Progress heuristic: fewer diagnostics, then fewer open goals, wins."""
    return (s.diag_count, s.goal_count)


def beam_search(
    initial: State,
    step,
    is_goal,
    beam_width: int = DEFAULT_BEAM,
    max_iters: int = 20,
) -> State:
    """Generic best-first beam search.

    ``step(state) -> list[State]`` expands a node; ``is_goal(state) -> bool``
    recognizes a solved node. Returns the first goal state, else the best
    (lowest-scoring) state after ``max_iters`` rounds.
    """
    frontier = [initial]
    best = initial
    for _ in range(max_iters):
        next_frontier: list[State] = []
        for s in frontier:
            for child in step(s):
                if is_goal(child):
                    return child
                next_frontier.append(child)
        if not next_frontier:
            break
        next_frontier.sort(key=_score)
        frontier = next_frontier[:beam_width]
        best = frontier[0]
    return best


def _new_state_from_build(state: State, body: str, lean_file: Path, lean_dir: Path,
                          timeout: int) -> State:
    """Write ``body``, compile it, and score the resulting state."""
    lean_file.write_text(HEADER + _split_signature(state.statement) + "\n" + body + "\n",
                         encoding="utf-8")
    rc, out = lean.compile_only(lean_file, lean_dir, timeout=timeout)
    diags = lean.parse_diagnostics(out)
    n_diag = sum(1 for d in diags if d.severity == "error")
    n_goal = sum(1 for d in diags if "unsolved goals" in d.message.lower())
    return State(
        statement=state.statement,
        body=body,
        history=state.history,
        diag_count=n_diag,
        goal_count=n_goal,
        proved=(rc == 0),
        proof=body if rc == 0 else "",
        last_output=out[-2000:],
    )


def _lean_step(
    state: State,
    *,
    signature: str,
    lean_dir: Path,
    lean_file: Path,
    system_prompt: str,
    temperature: float,
    model_name: str,
    timeout: int,
) -> list[State]:
    """Expand one node: LLM proposes a fix, Lean compiles it, return children."""
    diag_text = _build_diagnostics_text(state)
    user_msg = (f"Theorem signature:\n{signature}\n\n"
                f"Compiler diagnostics:\n{diag_text}\n\n"
                "Write ONLY the tactic proof body.")
    history = list(state.history) + [{"role": "user", "content": user_msg}]
    resp = llm.chat(system_prompt, history, temperature=temperature, model_name=model_name)
    if resp.content.startswith("[LLM error"):
        return []
    body = _extract_body(resp.content)
    if not body:
        return []
    history.append({"role": "assistant", "content": resp.content})
    child = _new_state_from_build(State(statement=state.statement, history=history),
                                  body, lean_file, lean_dir, timeout)
    return [child]


def search_problem(
    statement: str,
    *,
    beam_width: int = DEFAULT_BEAM,
    max_steps: int = 20,
    temperature: float = 0.2,
    model_name: str | None = None,
    lean_dir: Path | None = None,
    timeout: int = 60,
) -> State:
    """Beam-search a single theorem statement."""
    from .llm import model as active_model

    lean_dir = lean_dir or LEAN_DIR
    model_name = model_name or active_model()
    signature = _split_signature(statement)
    lean_file = lean_dir / "tmp" / "beam_search.lean"
    initial = _new_state_from_build(
        State(statement=statement, history=[]), "", lean_file, lean_dir, timeout)
    if initial.proved:
        return initial
    return beam_search(
        initial,
        lambda s: _lean_step(
            s, signature=signature, lean_dir=lean_dir, lean_file=lean_file,
            system_prompt=SYSTEM_FULL, temperature=temperature,
            model_name=model_name, timeout=timeout),
        lambda s: s.proved,
        beam_width=beam_width,
        max_iters=max_steps,
    )


def _build_diagnostics_text(state: State) -> str:
    if not state.last_output:
        return "(no diagnostics yet — initial attempt)"
    return state.last_output


def enabled() -> bool:
    return os.getenv("PROVER_BEAM") == "1"


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .problems import load_problems

    ap = argparse.ArgumentParser(prog="prover search", description=__doc__)
    ap.add_argument("--problems", default="benchmark/problems.json")
    ap.add_argument("--beam", type=int, default=DEFAULT_BEAM)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="benchmark/search_report.json")
    args = ap.parse_args(argv)

    problems = load_problems(args.problems)
    if args.limit is not None:
        problems = problems[: args.limit]
    import json

    results = []
    for p in problems:
        st = search_problem(p.statement, beam_width=args.beam, max_steps=args.max_steps)
        results.append({"id": p.id, "proved": st.proved, "proof": st.proof,
                        "diag_count": st.diag_count, "goal_count": st.goal_count})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    solved = sum(1 for r in results if r["proved"])
    print(f"beam search: {solved}/{len(results)} proved (beam={args.beam})")
    print(f"report: {args.out}")
    return 0


HEADER = "import Mathlib\n\nopen BigOperators Nat Finset\n\n"


if __name__ == "__main__":
    sys.exit(main())
