"""Core agent loop: draft → compile → parse errors → patch → repeat.

Architecture: the agent NEVER lets the model rewrite the whole file. We own
the theorem statement; the model only supplies the proof body (the tactics
after `:= by`). This makes "prove a different theorem" structurally
impossible — the statement is assembled by us, not the model.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import events, lean, llm, lsp
from .compaction import compact_history
from .context_window import (
    auto_compaction_threshold_for_context_window,
    estimate_context_tokens,
)
from .diagnostics import (
    ProofCallDiagnosticContext,
    ProofCallDiagnosticLogger,
    new_proof_call_run_id,
)
from .session import Session, read_session
from .session_manager import SessionManager, SessionRecord, history_from_records

LEAN_DIR = Path(__file__).resolve().parent.parent / "lean"


def _resolve_lean_dir() -> Path:
    """Workspace override for hermetic runs/tests (PROVER_LEAN_DIR)."""
    return Path(os.environ["PROVER_LEAN_DIR"]) if os.environ.get("PROVER_LEAN_DIR") else LEAN_DIR

# Prepended to every agent-written file so tactics/theorems are in scope.
HEADER = "import Mathlib\nimport ProverSupport\n\nopen BigOperators Nat Finset\n\n"

# One-shot hammers tried before spending any LLM tokens. They run as ONE Lean
# invocation via the native `prover_finish` tactic (Lean-side chain over
# ProverSupport.hammerNames); the list below is the fallback only when the
# ProverSupport olean is not built yet (fresh clone). Each costs one
# `lake env lean` (~3s) and solves a surprising fraction of problems outright.
HAMMERS = [
    "ring",
    "omega",
    "linarith",
    "nlinarith",
    "simp",
    "norm_num",
    "decide",
    "aesop",
    "tauto",
    "positivity",
    "bv_decide",
    "trivial",
]

SYSTEM = """You are an expert Lean 4 theorem prover.
You are given a theorem SIGNATURE (everything up to and including `:= by`)
and, after each attempt, the compiler diagnostics from `lake build`.
Respond with ONLY the tactic proof body — the lines that go after `:= by`,
indented two spaces, in a single ```lean code block. Rules:
- Do NOT restate, rename, or change the theorem. Only write the proof body.
- Do NOT include the theorem signature in your reply, only the tactics.
- Prefer hammers first: `ring`, `omega`, `linarith`, `nlinarith`, `simp`,
  `norm_num`, `positivity`, `aesop`, `bv_decide` (Bool/BitVec goals). Only
  write manual induction/case analysis if hammers cannot close the goal.
- Use only core Lean 4 / Mathlib tactics available in the project.
- If OPEN GOALS are shown, your proof must close every one.
- If several goals remain, close the hardest one first.
- Fix diagnostics in priority order: syntax errors first, then type errors,
  then unsolved goals. Ignore linter warnings unless they block the build.
- Never resubmit a proof you already tried: if a tactic failed once it will
  fail again unchanged. Change something — a different hammer (`simp`,
  `aesop`, `norm_num`), an explicit `induction`, or a helper `have`.
- If the goal is universally quantified over ℕ or ℤ and algebraic tactics
  keep failing, use `induction n` with explicit `simp [Nat.succ_mul]`-style
  arithmetic rewriting.
- Prefer compact proofs: combine chained rewrites (`rw [a, b]`, not one
  `rw` per lemma) and drop redundant steps.
- No `sorry`. The proof must fully type-check.
- If diagnostics are shown, fix exactly those errors."""

# Full-file mode (--full-file): the model owns the whole file — imports,
# helper lemmas, definitions — and must prove the given theorem. The repair
# loop still enforces the canonical statement (see _extract_full_file).
SYSTEM_FULL = """You are an expert Lean 4 / Mathlib developer.
You are given a theorem SIGNATURE that MUST be proven, and the compiler
diagnostics from `lake build` of your previous file.
Respond with ONE complete ```lean code block containing an entire self-
contained Lean file that proves the theorem. Rules:
- The file MUST declare `theorem prover_<id>` with EXACTLY the given
  signature and prove it (no `sorry`).
- You may add any helper lemmas, definitions, `section`, `open`, or
  `import` lines you need ABOVE the theorem to make the proof work.
- Do NOT change the theorem's statement or binders in any way.
- Use only core Lean 4 / Mathlib definitions.
- Fix diagnostics in priority order: syntax errors first, then type errors,
  then unsolved goals. Ignore linter warnings unless they block the build.
- Never resubmit a proof you already tried: change something — a different
  tactic, an `induction`, or a helper lemma — instead of repeating a failure.
- Prefer compact proofs: combine chained rewrites and drop redundant steps.
- If diagnostics are shown, fix exactly those errors in the file.
- Output the whole file every time (not a diff)."""


@dataclass
class Result:
    statement: str
    proved: bool
    steps: int
    seconds: float
    proof: str = ""
    history: list[str] = field(default_factory=list)
    # Token/cost tracking
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    # Proof trace (event records — see agent/events.py)
    trace: list[dict] = field(default_factory=list)
    # True when the run was aborted by should_stop (e.g. TUI Stop)
    stopped: bool = False
    # JSONL session log for this run (None if recording disabled)
    session_path: str | None = None
    # Per-attempt results when run via prove_best_of (attempt 1 = index 0)
    attempts: list[Result] = field(default_factory=list)


def _get_lean_file(problem_id: str | None = None, lean_dir: Path | None = None) -> Path:
    """Get a unique Lean file for this problem. Uses lean/tmp/ for isolation."""
    root = lean_dir if lean_dir is not None else _resolve_lean_dir()
    if problem_id:
        tmp_dir = root / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_dir / f"Prover_{problem_id}.lean"
    # Fallback to original single file (for `prove` command)
    return root / "src" / "Prover.lean"


def _split_signature(statement: str) -> str:
    """Return the theorem signature up to and including `:= by`.

    Accepts statements given with or without a trailing proof. Benchmarks with
    scaffolding definitions ahead of the theorem (Putnam `_solution` scaffolds,
    FormalQualBench namespaces) carry multiple `:= by`s — the target theorem is
    always the LAST declaration, so we cut at the last one.
    """
    s = statement.strip()
    last_end = None
    for m in re.finditer(r":=\s*by\b", s):
        last_end = m.end()
    if last_end is not None:
        return s[:last_end]
    # No `:= by` present — append it.
    return s + " := by"


def _extract_body(text: str) -> str:
    """Pull the proof body out of a model reply.

    Prefers a ```lean block; strips any accidental theorem signature lines.
    """
    code = llm.extract_lean_code(text)
    lines = []
    for ln in code.splitlines():
        stripped = ln.strip()
        # Skip anything that restates a theorem or re-imports.
        if re.match(r"^(theorem|lemma|example)\b", stripped):
            continue
        if re.match(r"^import\b", stripped):
            continue
        if re.match(r"^open\b", stripped):
            continue
        lines.append(ln)
    body = "\n".join(lines).strip("\n")
    # Normalize indentation to two spaces per tactic line.
    out = []
    for ln in body.splitlines():
        if not ln.strip():
            continue
        out.append("  " + ln.strip())
    return "\n".join(out)


def _proof_body(proof_text: str, cap: int = 300) -> str:
    """Bounded, indented proof body from a corpus entry's stored text.

    The stored field may be a tactic tag (`prover_search`), a full file, or
    a raw `by` block. Only real bodies are returned; tags and empty inputs
    yield ''. Used to render worked examples inside the prompt (P1-3), so the
    cap keeps the budget for prefill on slow endpoints under control.
    """
    t = proof_text.strip()
    if not t or t in ("prover_finish", "prover_search"):
        return ""
    if "prover_finish" in t or "prover_search" in t:
        return ""
    m = re.search(r":=\s*by\b", t)
    body = t[m.end():] if m else t
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if not lines:
        return ""
    out = "\n".join("  " + ln for ln in lines)
    return out if len(out) <= cap else ""


def _extract_full_file(text: str, signature: str) -> str | None:
    """Extract a complete Lean file, enforcing our canonical theorem statement.

    The model writes a whole file (helpers, definitions, proof). We splice our
    canonical `signature` in place of whatever the model wrote for the target
    theorem's declaration (keeping the model's tactic body), so the model can
    add any supporting code but can never silently change the statement.

    Returns None when the reply has no code block or the target theorem is
    missing/renamed — the loop then tells the model what to fix.

    `import` lines are stripped: our header already imports Mathlib, and Lean
    rejects an `import` once any other command (e.g. our `open`) has run.
    """
    code = llm.extract_lean_code(text)
    if not code.strip():
        return None
    code = "\n".join(
        ln for ln in code.splitlines() if not ln.lstrip().startswith("import ")
    )
    m = re.search(r"theorem\s+([A-Za-z0-9_'.]+)", signature)
    if not m:
        return code
    name = m.group(1)
    decl = re.compile(rf"(?ms)^theorem\s+{re.escape(name)}\b.*?:=\s*by\b")
    mm = decl.search(code)
    if not mm:
        return None
    # keep anything before the decl (helpers/opens) and after `:= by` (the
    # model's tactic body); the declaration itself is replaced by ours.
    return code[: mm.start()] + signature + code[mm.end():]


def prove(
    statement: str,
    max_steps: int = 20,
    verbose: bool = True,
    problem_id: str | None = None,
    goal_feedback: bool = True,
    on_event: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    record_session: bool = True,
    resume_from: str | None = None,
    branch_at: int | None = None,
    branch_summary: str | None = None,
    skip_hammers: bool = False,
    temperature: float = 0.2,
    difficulty: str | None = None,
    model_name: str | None = None,
    lemma_plan: bool | None = None,
    full_file: bool = False,
    adaptive_steps: bool = False,
) -> Result:
    t0 = time.time()
    lean_dir = _resolve_lean_dir()
    signature = _split_signature(statement)
    history: list[dict] = []
    trace: list[dict] = []
    total_prompt = 0
    total_completion = 0
    total_tokens = 0

    # Per-difficulty routing: env overrides (PROVER_MODEL_<TIER> etc.) win,
    # otherwise fall back to the caller's model/temperature/step defaults.
    if difficulty is not None:
        from .router import select as router_select

        cfg = router_select(difficulty, model=model_name, temperature=temperature,
                            max_steps=max_steps)
        model_name = cfg["model"]
        temperature = cfg.get("temperature", temperature)
        max_steps = cfg.get("max_steps", max_steps)
    active_model = model_name or llm.model()

    session = Session(problem_id=problem_id)
    session_open = session.open() if record_session else False
    session_path = str(session.path) if session_open else None
    manager = SessionManager()

    # Structured failure log (tau AgentCallDiagnosticLogger parity): machine-
    # readable JSONL of LLM failures under the paths logs_dir, alongside the
    # human-readable llm_error event stream.
    diag = ProofCallDiagnosticLogger.from_paths()
    diag_context = ProofCallDiagnosticContext(
        model=active_model,
        cwd=lean_dir,
        session_id=session.id,
        run_id=new_proof_call_run_id(),
        problem_id=problem_id,
    )

    def emit(event: str, **payload) -> None:
        """Single emit path: trace + session JSONL + callback + CLI print."""
        rec = events.record(event, **payload)
        trace.append(rec)
        session.write(rec)
        if on_event:
            on_event(rec)
        if verbose:
            line = events.format(rec)
            if line:
                print(line)

    def stop_requested() -> bool:
        return bool(should_stop and should_stop())

    target_file = _get_lean_file(problem_id, lean_dir)

    # Lemma-bank: proven helper lemmas prepended above the main theorem.
    lemma_bank = ""
    plan_enabled = lemma_plan if lemma_plan is not None else (
        os.getenv("PROVER_LEMMA_PLAN") == "1"
    )

    # Full-file mode: the model owns the whole file; `content` is the text
    # after HEADER+lemma_bank (starts with our canonical signature).
    content = signature + "\n  sorry"
    system_prompt = SYSTEM_FULL if full_file else SYSTEM
    extract_note = ""  # fed back to the model when a reply can't be used

    def write_file(b: str) -> None:
        nonlocal content
        content = b
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(HEADER + lemma_bank + b + "\n")

    def check_file() -> tuple[bool, str]:
        """Run `lean --check` on the target file for fast, isolated verification,
        then the proof-integrity audit (injection scan + type-aware statement
        check + kernel `#print axioms` audit). A lean pass is not enough on its
        own: sorryAx / native_decide (Lean.ofReduceBool) / a custom `axiom` all
        compile clean, so an audit failure is reported like a failed check —
        the loop feeds it back for retry and never records/commits a proof that
        is not audit-clean. The cheap regex scan always runs; the expensive
        kernel audit is skipped under PROVER_SKIP_AXIOM_AUDIT=1."""
        proved, output = lean.check_file(target_file, lean_dir)
        if not proved:
            return False, output
        from .validate import audit_file

        res = audit_file(target_file, lean_dir, expected_signature=statement)
        if not res.ok:
            return False, f"{res.reason}\n"
        return True, output

    # LSP session for goal-state feedback (lazily started, may stay None).
    lsp_client: lsp.LeanLSP | None = None

    def get_goals() -> str | None:
        """Open goal state via LSP. Returns formatted goals or None."""
        nonlocal lsp_client
        if not goal_feedback:
            return None
        if lsp_client is None:
            lsp_client = lsp.LeanLSP(lean_dir, target_file)
        try:
            current_text = HEADER + lemma_bank + content
            lsp_client.update(current_text)
            goals = lsp_client.goal_at_end(current_text)
            return goals
        except Exception:  # noqa: BLE001 — LSP feedback is optional; never break the loop
            return None
    def finish(
        proved: bool, steps: int, proof: str, cost: float, stopped: bool = False
    ) -> Result:
        secs = time.time() - t0
        if lsp_client:
            lsp_client.close()
        if (
            proved and not stopped
            and os.environ.get("PROVER_CORPUS_GROW", "1") == "1"
            and not resume_from
        ):
            try:
                from .retrieval import corpus_append

                corpus_append(lean_dir, statement, proof)
            except Exception:  # noqa: BLE001, S110 — corpus growth is best-effort
                pass
        emit(
            "result",
            proved=proved,
            steps=steps,
            seconds=round(secs, 2),
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_tokens,
            cost_usd=cost,
            stopped=stopped,
            session_id=session.id if session_open else None,
        )
        session.close()
        if record_session:
            manager.upsert(SessionRecord(
                id=session.id, path=str(session.path), problem_id=problem_id,
                model=active_model,
                status="stopped" if stopped else ("proved" if proved else "failed"),
                proved=proved, steps=steps,
                created_at=t0, updated_at=time.time(),
            ))
        return Result(
            statement, proved, steps, secs, proof, history,
            total_prompt, total_completion, total_tokens, cost, trace,
            stopped, session_path,
        )

    emit("start", statement=statement, problem_id=problem_id,
         max_steps=max_steps, model=active_model)
    if record_session:
        manager.upsert(SessionRecord(
            id=session.id, path=str(session.path), problem_id=problem_id,
            model=active_model, status="running",
            created_at=t0, updated_at=time.time(),
        ))

    # ---- Resume/branch: seed the repair history from a recorded session
    # (tau: replay the root→leaf path to rebuild harness messages).
    # Full-file mode is a structural rewrite; resume seeding is body-mode only.
    body = None  # initial proof body; seeded on resume
    if resume_from and not full_file:
        records = []
        for rec in manager.list_sessions():
            if rec.id == resume_from:
                records = read_session(Path(rec.path))
                break
        if records:
            seed = history_from_records(records)
            if branch_at is not None:
                # branch_at = keep only the first N user/assistant turns,
                # discarding the rest of the recorded trajectory
                # (tau: repoint the LeafEntry at an earlier entry).
                seed = seed[: max(0, branch_at) * 2]
            if seed:
                history = seed[:12]
                if branch_summary:
                    # Context for a continuation from an earlier point: what
                    # the rest of the previous run already did (tau's
                    # BRANCH_SUMMARY_PREAMBLE seeding).
                    from .branch_summary import BRANCH_SUMMARY_PREAMBLE

                    history.insert(0, {
                        "role": "user",
                        "content": f"{BRANCH_SUMMARY_PREAMBLE}\n{branch_summary}",
                    })
                last_asst = next(
                    (m["content"] for m in reversed(seed)
                     if m["role"] == "assistant" and m["content"].strip()
                     and m["content"] != "(no response)"),
                    None,
                )
                if last_asst:
                    body = _extract_body(last_asst)
                emit("resume", session_id=resume_from, seed_turns=len(seed) // 2,
                     branch_at=branch_at)

    # ---- Hammer pre-pass: try one-shot tactics before spending LLM tokens.
    # Runs as a SINGLE Lean invocation: `prover_finish` executes the whole
    # hammer chain natively (Lean-side), instead of spawning one `lake env
    # lean` per hammer. If the ProverSupport olean isn't built (fresh clone,
    # "unknown module prefix"), fall back to the per-hammer Python loop.
    # PROVER_SEARCH=1 upgrades the pre-pass to `prover_search` (bounded
    # native search, which includes the hammer chain) — opt-in, slower, but
    # solves more problems without any LLM tokens.
    # Skipped on resume and when the caller knows hammers already failed.
    if body is None and not skip_hammers:
        prepass = "prover_search" if os.environ.get("PROVER_SEARCH") else "prover_finish"
        prepass_opts = ""
        depths = ["3"]
        budget = "1000"
        if prepass == "prover_search":
            raw_depths = os.environ.get("PROVER_SEARCH_DEPTH", "3")
            depths = [d for d in raw_depths.split(",") if d.strip().isdigit()] or ["3"]
            b = os.environ.get("PROVER_SEARCH_BUDGET", "1000")
            budget = b if b.isdigit() else "1000"
            prepass_opts = "set_option maxHeartbeats 0\n"
            if budget != "1000":
                prepass_opts += f"set_option prover_search.budget {budget}\n"
        # Depth ramp (PROVER_SEARCH_DEPTH="3,4,5"): retry the whole search at
        # each depth until something closes. Budget budget is shared per depth.
        for d in depths:
            tactic = f"prover_search {d}" if prepass == "prover_search" else "prover_finish"
            write_file(prepass_opts + signature + "\n  " + tactic)
            ok, output = check_file()
            if ok:
                emit("hammer", i=1, total=1, tactic=tactic, ok=True, output="")
                return finish(True, 1, target_file.read_text(), 0.0)
            if prepass != "prover_search":
                break
            if d != depths[-1]:
                emit("hammer", i=int(d), total=len(depths), tactic=tactic,
                     ok=False, output=output[-500:])
        if "unknown module prefix 'ProverSupport'" not in output:
            # native chain ran and failed — every hammer failed
            emit("hammer", i=1, total=1, tactic=prepass, ok=False,
                 output=output[-500:])
        else:
            # ProverSupport not built: per-hammer fallback loop
            emit("hammer", i=1, total=len(HAMMERS) + 1, tactic=prepass,
                 ok=False, output=output[-500:])
            for i, hammer in enumerate(HAMMERS, 1):
                if stop_requested():
                    break
                write_file(signature + "\n  " + hammer)
                ok, output = check_file()
                emit("hammer", i=i + 1, total=len(HAMMERS) + 1, tactic=hammer,
                     ok=ok, output="" if ok else output[-500:])
                if ok:
                    return finish(True, i, target_file.read_text(), 0.0)
        emit("llm_start")
        body = "  sorry"  # placeholder so the first build reports sorry
    if body is not None:
        write_file(signature + "\n" + body)
    elif not full_file:
        body = "  sorry"
        write_file(signature + "\n  sorry")
    else:
        write_file(content)  # full-file, no resume seed: initial `sorry` file

    # ---- Lemma planning: prove helper lemmas first, prepend proven ones
    # (PROVER_LEMMA_PLAN=1). Skipped on resume (the prior run already had its
    # chance) and in full-file mode (the model writes helpers itself).
    # Only *proven* lemmas enter the file — never `sorry`.
    if plan_enabled and not resume_from and not full_file:
        try:
            from .plan import propose_lemmas, prove_lemmas

            proposed = propose_lemmas(signature, model_name=active_model)
            emit("plan", proposed=[p[:80] for p in proposed])
            proven = prove_lemmas(proposed, problem_id=problem_id, max_steps=8,
                                  model_name=active_model)
            emit("plan_lemmas", proven=[p[:80] for p in proven])
            if proven:
                lemma_bank = "\n\n".join(proven) + "\n\n"
        except Exception:  # noqa: BLE001, S110 — planning is best-effort
            pass

    # ---- Lemma retrieval: keyword hints from the local Mathlib index
    # (PROVER_RETRIEVE=1). Optional, offline, deterministic — never blocks
    # the loop on the network.
    retrieval_hints = ""
    if os.getenv("PROVER_RETRIEVE") == "1":
        try:
            from .retrieval import search_lemmas

            hits = search_lemmas(signature, k=5)
        except Exception:  # noqa: BLE001 — retrieval is best-effort
            hits = []
        if hits:
            # Corpus hits with a real proof body (LLM-verified tactics) are
            # worth showing as worked examples — that's the cheapest way to
            # teach the lemma/tactic style. Native-tag hits stay one-liners.
            examples: list[str] = []
            for h in hits:
                if h.get("file") != "corpus" or not h.get("proof"):
                    continue
                if h["proof"].strip() in ("prover_finish", "prover_search"):
                    continue
                body = _proof_body(h["proof"])
                if body:
                    examples.append(f"{h['signature']} := by\n{body}")
                if len(examples) >= 2:
                    break
            ex_block = ""
            if examples:
                ex_block = ("Worked examples of proofs this project has already "
                            "verified (Lean-checked — imitate the style/lemmas):\n"
                            + "\n\n".join(examples) + "\n\n")
            retrieval_hints = (
                "Relevant lemmas found by local keyword search "
                "(name : signature):\n"
                + "\n".join(
                    f"- {h['name']} : {h['signature']}"
                    + (f" — proven by {h['proof'].splitlines()[0].strip()}"
                       if h.get("proof") and h["proof"].strip() in
                       ("prover_finish", "prover_search") else "")
                    for h in hits
                )
                + "\n\n" + ex_block
            )
            emit("retrieve", k=len(hits),
                 lemmas=[h["name"] for h in hits],
                 corpus=sum(1 for h in hits if h.get("file") == "corpus"))

    base_max_steps = max_steps
    extensions = 0
    prev_progress = (10**9, 10**9)  # (ndiag, ngoals) of the previous step
    tried_bodies: set[str] = set()  # exact proof bodies already submitted
    step = 0
    try:
        while step < max_steps:
            step += 1
            if stop_requested():
                return finish(False, step - 1, "", llm.estimate_cost(total_prompt, total_completion), stopped=True)
            ok, output = check_file()
            diags = lean.parse_diagnostics(output)
            ndiag = len(diags)
            if ok:
                emit("build", step=step, ok=True, diagnostics=0, summary="all goals solved")
                return finish(True, step, target_file.read_text(),
                              llm.estimate_cost(total_prompt, total_completion))

            report = lean.error_report(lean_dir, output)
            emit("build", step=step, ok=False, diagnostics=ndiag,
                 summary=(diags[0].message if diags else "sorry / not proved"),
                 report=report[:4000])
            goals = get_goals()
            ngoals = goals.count("⊢") if goals else 0
            if goals:
                emit("goals", step=step, goals=goals)

            # ---- Adaptive budget: if we just ran out of steps while making real
            # progress, extend the budget (bounded: ≤2 extensions, ≤4× original).
            # Placed BEFORE the LLM call (not at the loop bottom) so the extension
            # still happens when this step's LLM call fails and `continue`s — a
            # bottom placement is skipped on that path and prematurely exits.
            progress = (ndiag, ngoals)
            improving = progress < prev_progress
            prev_progress = progress
            if (
                adaptive_steps
                and step == max_steps
                and extensions < 2
                and max_steps < base_max_steps * 4
                and improving
            ):
                old = max_steps
                max_steps = int(max_steps * 1.5)
                extensions += 1
                emit("extend", from_steps=old, to_steps=max_steps,
                     reason=f"progress on step {step} (diags {ndiag}, goals {ngoals})")

            lemma_hints = ""
            if lemma_bank:
                names = [ln.split("(")[0].split()[1] for ln in lemma_bank.strip().splitlines()
                         if ln.startswith(("theorem ", "lemma "))]
                lemma_hints = (
                    "Helper lemmas already proven in this file (use them via "
                    f"rw/exact/simpa): {', '.join(names)}\n\n"
                )

            write_what = (
                "Write ONLY the tactic proof body."
                if not full_file else
                "Write the COMPLETE Lean file (helpers + imports allowed) proving "
                "the theorem with exactly the given signature."
            )
            user_msg = (
                f"Theorem signature:\n{signature}\n\n"
                + lemma_hints
                + retrieval_hints
                + f"Compiler diagnostics:\n{report}\n\n"
                + (f"Open goals at the end of your last proof attempt:\n{goals}\n\n" if goals else "")
                + (extract_note + "\n\n" if extract_note else "")
                + write_what
            )
            extract_note = ""  # consumed
            history.append({"role": "user", "content": user_msg})
            emit("llm_request", step=step)
            resp = llm.chat(system_prompt, history, temperature=temperature, model_name=active_model)
            if resp.content.startswith("[LLM error"):
                emit("llm_error", step=step, error=resp.content)
                diag.log_llm_error(
                    context=diag_context, phase=f"step-{step}", error=resp.content
                )
                history.append({"role": "assistant", "content": "(no response)"})
                continue
            rejected_dup = False
            if full_file:
                new_content = _extract_full_file(resp.content, signature)
                if new_content is None:
                    name = re.search(r"theorem\s+([A-Za-z0-9_'.]+)", signature).group(1)
                    extract_note = (
                        "Your reply was not accepted: it must contain a single "
                        f"```lean block that declares `theorem {name}` with the "
                        "given signature (write the whole file, do not rename it)."
                    )
                elif new_content in tried_bodies:
                    # Duplicate of an already-tried file → same build result.
                    rejected_dup = True
                    extract_note = (
                        "Your reply duplicates a file you already submitted; "
                        "it will produce the same diagnostics. Change your "
                        "approach: different tactic, `induction`, helper lemma."
                    )
                else:
                    tried_bodies.add(new_content)
                    write_file(new_content)
                history.append({"role": "assistant", "content": resp.content})
            else:
                new_body = _extract_body(resp.content)
                if new_body:
                    if new_body in tried_bodies:
                        # Same body → same diagnostics. Don't waste a build;
                        # bounce it back with an explicit variation demand.
                        rejected_dup = True
                        extract_note = (
                            "Your reply duplicates a proof you already "
                            "submitted; it produced exactly the diagnostics "
                            "shown above and will fail identically. Change "
                            "your approach: different tactic, `induction`, "
                            "a helper `have`, or a hammer (`simp`, `aesop`)."
                        )
                    else:
                        tried_bodies.add(new_body)
                        body = new_body
                        write_file(signature + "\n" + body)
                # Always keep the turn, even when extraction failed: dropping
                # it would put two user messages back-to-back in the history
                # (strict backends reject that) and break compaction pairing.
                history.append({"role": "assistant", "content": resp.content})
            total_prompt += resp.prompt_tokens
            total_completion += resp.completion_tokens
            total_tokens += resp.total_tokens
            payload_body = "" if rejected_dup else (new_content if full_file else new_body)
            emit("llm_response", step=step,
                 accepted=bool(payload_body),
                 prompt_tokens=resp.prompt_tokens,
                 completion_tokens=resp.completion_tokens,
                 tokens=resp.total_tokens,
                 body=payload_body)
            # Compaction beats truncation: fold old dead-end turns into a
            # summary so the model stops re-trying them (tau's memory model).
            n_msgs = len(history)
            history, summary = compact_history(history)
            # tau parity: if the estimated context crosses the auto-compaction
            # threshold (70% of the model's context window) without the turn
            # count having triggered yet, compact eagerly.
            threshold = auto_compaction_threshold_for_context_window(llm.context_window_tokens())
            if summary is None and threshold and estimate_context_tokens(system_prompt, history) >= threshold:
                history, summary = compact_history(history, keep_turns=8, compact_at_turns=14)
            if summary:
                emit("compaction", dropped=n_msgs - len(history))

    finally:
        # Release the lean --server subprocess and the session JSONL
        # handle even if an unexpected exception escapes the loop.
        if lsp_client:
            lsp_client.close()
        session.close()

    return finish(False, max_steps, "", llm.estimate_cost(total_prompt, total_completion))


def prove_best_of(
    statement: str,
    n_attempts: int = 1,
    max_steps: int = 20,
    verbose: bool = True,
    problem_id: str | None = None,
    goal_feedback: bool = True,
    on_event: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    record_session: bool = True,
    skip_hammers: bool = False,
    temperature: float = 0.2,
    temperature_delta: float = 0.4,
    difficulty: str | None = None,
    model_name: str | None = None,
    lemma_plan: bool | None = None,
    full_file: bool = False,
    adaptive_steps: bool = False,
) -> Result:
    """Best-of-N search: run up to N independent repair trajectories.

    Diversity comes from a temperature ramp (attempt 1 uses `temperature`,
    each later attempt adds `temperature_delta`). Hammers are deterministic,
    so later attempts skip them. Returns the first proved attempt; if none
    proves, returns the attempt that got furthest (highest `steps` reached).
    Each attempt gets its own session record; all results are kept on
    Result.attempts (index 0 = attempt 1).
    """
    n = max(1, n_attempts)
    results: list[Result] = []
    for i in range(1, n + 1):
        temp = temperature + temperature_delta * (i - 1)
        r = prove(
            statement,
            max_steps=max_steps,
            verbose=verbose,
            problem_id=problem_id,
            goal_feedback=goal_feedback,
            on_event=on_event,
            should_stop=should_stop,
            record_session=record_session,
            skip_hammers=skip_hammers or i > 1,
            temperature=temp,
            difficulty=difficulty,
            model_name=model_name,
            lemma_plan=lemma_plan,
            full_file=full_file,
            adaptive_steps=adaptive_steps,
        )
        results.append(r)
        if r.proved:
            break

    best = next((r for r in results if r.proved), None) or max(
        results, key=lambda r: r.steps
    )
    best.attempts = results
    return best