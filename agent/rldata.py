"""GRPO-ready rollout generation from the Lean verifier (``prover rl-data``).

This is the *data + reward* half of verifier-reward RL. It runs the existing
prover over a problem set with multiple attempts per problem and emits, for
each problem, the prompt plus every attempt's completion (the model's tactic
body) paired with the Lean verifier reward — 1.0 if that attempt closed the
goal, else 0.0.

Output is one JSON object per line (TRL-style GRPO groups):

    {"id", "prompt", "completions": [t1, t2, ...], "rewards": [r1, r2, ...]}

``completions`` and ``rewards`` are aligned per attempt. The reward is the
honest binary verifier signal — the same ``lake env lean`` gate the proof
loop uses — never a model estimate.

Honest scope: the *actual* GRPO fine-tune needs a trainable (local/GPU)
policy model, which this box lacks (no GPU, hosted chat endpoint) — the same
infra wall documented in ``agent/finetune.py``. What this module delivers is
the verifier-reward rollout dataset a GRPO trainer consumes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .prover_loop import SYSTEM_FULL


def _attempt_completions(attempt) -> list[str]:
    """The model's tactic bodies from one attempt's trace (in order)."""
    bodies = [e.get("body") for e in attempt.trace
              if e.get("type") == "llm_response" and e.get("body")]
    return [b for b in bodies if b]


def generate(
    problems: list[dict],
    n_attempts: int = 3,
    max_steps: int = 20,
    goal_feedback: bool = True,
    temperature: float = 0.2,
    temperature_delta: float = 0.4,
    lean_dir: Path | None = None,
) -> list[dict]:
    """Run the prover over ``problems`` and emit GRPO rollout groups.

    Each problem yields one group with aligned ``completions``/``rewards``.
    Attempts that produced no LLM body are skipped (nothing to train on).
    """
    from .prover_loop import prove_best_of

    groups: list[dict] = []
    for p in problems:
        res = prove_best_of(
            p["statement"],
            n_attempts=n_attempts,
            max_steps=max_steps,
            verbose=False,
            problem_id=p.get("id"),
            goal_feedback=goal_feedback,
            record_session=False,
            temperature=temperature,
            temperature_delta=temperature_delta,
            difficulty=p.get("difficulty"),
        )
        attempts = res.attempts or [res]
        completions: list[str] = []
        rewards: list[float] = []
        for a in attempts:
            bodies = _attempt_completions(a)
            if not bodies:
                continue
            completions.append(bodies[-1])  # the final body of this attempt
            rewards.append(1.0 if a.proved else 0.0)
        if not completions:
            continue  # no LLM rollouts (e.g. hammer-only solve) — nothing to train on
        groups.append({
            "id": p.get("id", "?"),
            "prompt": f"{SYSTEM_FULL}\n\n{p['statement']}",
            "completions": completions,
            "rewards": rewards,
        })
    return groups


def write_jsonl(groups: list[dict], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for g in groups:
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")
    return len(groups)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .problems import load_problems

    ap = argparse.ArgumentParser(prog="prover rl-data", description=__doc__)
    ap.add_argument("--problems", default="benchmark/problems.json",
                    help="problem set JSON (default benchmark/problems.json)")
    ap.add_argument("--out", default="benchmark/train_grpo.jsonl",
                    help="output GRPO rollout JSONL")
    ap.add_argument("--n-attempts", type=int, default=3,
                    help="rollouts (attempts) per problem")
    ap.add_argument("--max-steps", type=int, default=20,
                    help="repair steps per attempt")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process first N problems")
    args = ap.parse_args(argv)

    problems = load_problems(args.problems)
    if args.limit is not None:
        problems = problems[: args.limit]
    groups = generate([p.as_dict() for p in problems],
                      n_attempts=args.n_attempts, max_steps=args.max_steps)
    n = write_jsonl(groups, Path(args.out))
    print(f"wrote {n} GRPO rollout groups to {args.out}")
    solved = sum(1 for g in groups if any(r > 0 for r in g["rewards"]))
    print(f"groups with ≥1 successful rollout: {solved}/{len(groups)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
