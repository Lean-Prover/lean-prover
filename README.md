# lean-prover

An agent that writes Lean 4 proofs. You give it a theorem; it iterates
(LLM draft → `lake` compile → parse diagnostics → fix) until the proof
type-checks. Ends with ∎

## Status

Working agent, 100-problem benchmark, interactive TUI, MCP server,
session resume/branching, history compaction, theming, best-of-N search,
local Mathlib retrieval, autoformalization, synthetic data tooling.
Current best: **Qwen3.8-27B on Mathlib v4.33.0 scores 68/100**
(trivial 20/20, easy 23/30, medium 21/30, hard 4/20), $0 cost (free HF endpoint).
Live leaderboard + org site: **http://100.96.13.61:8899** (self-hosted on Tailscale, repo `lean-prover-web`).

## Setup

```bash
# 1. Lean toolchain + Mathlib (once; Mathlib fetch takes a while)
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
source ~/.bashrc
cd lean && lake update && lake exe cache get && lake build && cd ..

# 2. Python deps
pip install -e .
pip install textual pyperclip   # for the TUI + clipboard
pip install pytest              # to run the test suite

# 3. LLM access (any OpenAI-compatible endpoint)
export OPENAI_API_KEY=...          # or
export OPENAI_BASE_URL=http://localhost:11434/v1  # e.g. Ollama
export PROVER_MODEL=gpt-4o

# Free option: OpenCode Zen (https://opencode.ai/docs/zen) — key from
# https://opencode.ai/auth, free model ids include big-pickle,
# x-preview-f-free, mimo-v2.5-free, hy3-free, nemotron-3-ultra-free,
# nemotron-3.5-lightning-free, deepseek-v4-flash-free, laguna-s-2.1-free.
export OPENAI_BASE_URL=https://opencode.ai/zen/v1
export OPENAI_API_KEY=<zen-key>
export PROVER_MODEL=big-pickle
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | — | Any OpenAI-compatible provider. |
| `PROVER_MODEL` | `gpt-4o` | Model name for `/status` + cost lookup. Wins over the stored active profile. |
| `PROVER_MODEL_<TIER>` | `PROVER_MODEL` | Per-difficulty model override (`<TIER>` = `TRIVIAL`/`EASY`/`MEDIUM`/`HARD`). |
| `PROVER_TEMP_<TIER>` | caller default | Per-difficulty sampling temperature (float). |
| `PROVER_STEPS_<TIER>` | caller default | Per-difficulty max repair steps (int). |
| model profiles (`~/.prover/models.json`) | — | Named profiles (model + endpoint + key + context window + cost) managed from the TUI with `/models`. The stored active profile is used when `PROVER_MODEL` is unset. |

## Usage

```bash
# Prove a single theorem interactively (edits lean/src/Prover.lean)
prover prove "theorem pythagoras (a b c : ℕ) : a ^ 2 + b ^ 2 = c ^ 2 ↔ a = 0" --max-steps 20
# Best-of-N search + beam/tree search over the repair loop
prover prove "..." --n-attempts 3      # best-of-N (temperature ramp per attempt)
prover search --problems benchmark/problems.json --beam 3 --limit 5   # beam search
prover prove "..." --full-file         # model writes the whole Lean file (helpers/imports);
                                       # the theorem statement is still enforced
prover prove "..." --adaptive          # extend the step budget when making progress

# Autoformalize a natural-language statement to a compilable Lean theorem
prover formalize "For all integers a and b, a + b = b + a."

# Generate a synthetic proof corpus (statement, proof, ok) as JSONL
prover synth-data --count 20 --out synth.jsonl   # train file: synth_train.jsonl (proven only)

# Import a standard benchmark (MiniF2F Lean4 port) + type-check it
python benchmark/import_standard.py minif2f --src <miniF2F-lean4 checkout> --split test --verify

# Physics benchmarks (32 physics-motivated theorems across two tiers:
# physics.json = kinematics/dynamics/energy/circuits/waves/gravitation;
# physics_hard.json = projectile/rotation/fluids/thermo/optics/magnetism —
# all type-check on Mathlib v4.33.0). Hammer-only floor: 31/32 solved by
# `prover_finish` with no LLM; only `circuit_parallel` (division) needs a model.
prover bench --problems benchmark/physics.json --max-steps 20
prover bench --problems benchmark/physics_hard.json --max-steps 20
prover bench --problems benchmark/minif2f_test.json

# Interactive TUI: browse problems, watch live repairs, slash commands
prover                # no args = TUI (the default entry point)
prover tui            # explicit; or: prover tui -p 4 (parallel workers)

# Slash commands inside the TUI prompt bar (Tab-less: ctrl+space completes):
#   /help /prove /run /stop /workers <n> /resume <id> /branch <id> [turn]
#   /export <path> /theme [name] /status /model /models /system /hotkeys /clear /quit
#   /usage [session-id|all] /reload

# Keyboard shortcuts (problem list focused):
#   j/k       navigate up/down
#   space     queue selected problem for proof
#   0/G       jump to top/bottom
#   /         focus search bar
#   m/Ctrl+O  open model profiles manager

# Run the benchmark (100 theorems, JSON in benchmark/problems.json)
prover bench --max-steps 20 --report report.json
prover bench --parallel 4            # isolation makes parallelism safe
prover bench --no-goal-feedback      # errors only, no LSP goal state
prover bench --no-record             # skip JSONL session logs
prover bench --n-attempts 2          # best-of-N per problem
prover bench --full-file             # model writes whole files per problem
prover bench --adaptive              # extend step budget on progress

# Inspect recorded proof sessions (event stream per run)
prover sessions                      # list recent sessions
prover sessions 20260817-015954-proof       # replay one
prover sessions 20260817-015954-proof --raw # with raw JSON records

# Token/cost dashboard (per session or across all)
prover usage                         # all sessions
prover usage 20260817-015954-proof   # one session

# Machine-readable proof output
prover prove "..." --output json         # one JSON event per line
prover prove "..." --output transcript   # colored step transcript

# Local leaderboard: run a subset and record the score
prover leaderboard --run --problems benchmark/trivial.json --name my-model
prover leaderboard --show
# Live board: http://100.96.13.61:8899 (self-hosted; leaderboard.json is
# served by that site — keep it updated via `prover leaderboard --run`)

# Use lean-prover from any MCP client (Claude, opencode, Cursor, …)
prover mcp    # JSON-RPC tools: prove_theorem, benchmark_score, problems
# …or from ACP editors (Zed, Devin Desktop, CodeCompanion, …)
prover acp    # Agent Client Protocol stdio server (session-based prove)

# No-LLM baseline: how many problems Lean itself solves (one `lake env lean` per problem)
prover lean-baseline --tactic prover_finish --out benchmark/lean_baseline.json
prover lean-baseline --tactic prover_search --out benchmark/lean_baseline_search.json

# Lean-proved corpus (JSONL) from a baseline report
prover synth-lean --report benchmark/lean_baseline.json --out corpus/lean_proved.jsonl

# SFT/RL training JSONL from Lean-verified corpus + all committed baseline reports
prover datagen --out benchmark/train_sft.jsonl
# Writes {"system", "instruction": <theorem>, "output": "```lean\n  <tactic>\n```"} entries.
# Run this as the loop proof more problems — the corpus auto-grows during proving.

# GRPO-ready verifier-reward rollouts (prompt + per-attempt completions + binary Lean reward)
prover rl-data --problems benchmark/problems.json --n-attempts 3 --out benchmark/train_grpo.jsonl
# Writes TRL-style groups: {"id", "prompt", "completions": [...], "rewards": [0/1...]}.
# Reward = the Lean verifier gate (same `lake env lean` the loop uses). Actual
# GRPO fine-tune needs a trainable/GPU policy model — see agent/finetune.py.

# Tests
pytest tests/        # 720 tests (loop, compaction, session, TUI, commands, baselines)
```

## How it works

```
        ┌────────────┐
        │ statement  │
        └─────┬──────┘
              ▼
   ┌─────────────────────┐
   │ hammer pre-pass     │  prover_finish: ring / omega / linarith / simp /
   │ (before any LLM)    │  aesop / … in ONE `lake env lean`, no model tokens
   └─────────┬───────────┘
             ▼ (no hammer worked)
   ┌─────────────────────┐
   │ LLM drafts/patches  │◄──────────────┐
   │ the proof body      │               │
   └─────────┬───────────┘               │
             ▼                           │
        ┌─────────────┐   diagnostics    │
        │ lake env    │   + goal state   │
        │ lean --check│──────────────────┘
        └─────┬───────┘   (LSP RPC, source context)
              ▼  type-checks
           [PROVED ∎]
```

The hammer pre-pass is now Lean-native: `prover_finish` runs the whole chain
(`grind` + `simp`/`ring`/`omega`/… in one Lean compile, tactics parsed at
run time via `runParserCategory`), replacing the old 10×`lake env lean`
spawns. `grind` (Lean 4.33+) is tried first as the strongest hammer. There is
also a bounded native search tactic, `prover_search` (case split / induction /
subst / use / simp_all with backtracking, depth- and budget-capped), and an
honest no-LLM baseline command to measure what Lean alone solves.

The model only ever supplies the proof body — the theorem statement is
assembled by us, so "prove a different theorem" is structurally
impossible. History is compacted (old attempts folded into a failed-attempts
summary) rather than truncated, so weak models stop re-trying dead ends.

Optional search/assistance layers (all env-gated, all default off, all
best-effort — a failure never breaks the loop):
- **Best-of-N** (`--n-attempts`): independent repair trajectories with a
  temperature ramp; hammers run once (they are deterministic); returns the
  first proof, else the attempt that got furthest.
- **Beam/tree search** (`prover search --beam N`): keeps a frontier of the N
  most promising proof states, expanding each with the LLM+Lean repair step
  and pruning dead ends by a progress heuristic (fewer diagnostics / open
  goals) — concentrates budget on promising paths instead of equal budget
  per trajectory.
- **Lemma retrieval** (`PROVER_RETRIEVE=1`): keyword index over ~150k Mathlib
  lemma signatures; the top-5 for the target statement are fed to the model
  as hints (no embeddings, no network). Neural variant (`PROVER_EMBED=1`,
  `PROVER_EMBED_MODEL=<embedding-model>`): embedding cosine-similarity over a
  cached index, falling back to keyword when the endpoint serves no
  `/embeddings`.
- **Lemma planning** (`PROVER_LEMMA_PLAN=1`): the model proposes ≤3 helper
  lemmas, each is statement-checked and proven by a bounded sub-loop, and
  only *proven* helpers are prepended above the main theorem (never `sorry`).
- **Per-difficulty routing** (`PROVER_MODEL_<TIER>` etc.): pick a cheaper
  model for trivial/easy and a stronger one for hard.
- **Full-file mode** (`--full-file`): the model writes a complete Lean file —
  helper lemmas, definitions, opens — instead of a single tactic body. The
  canonical theorem statement is still enforced by splicing our signature
  over whatever the model wrote for the theorem's declaration, so it can
  restructure the code but never change the statement.
- **Adaptive budget** (`--adaptive`): when a repair step reduces the number
  of compiler diagnostics or open goals, the loop extends its step budget
  (1.5×, bounded) instead of giving up at the hard cap.

## Layout

```
lean/               Lean 4 package (lake), Mathlib pinned to v4.33.0
  src/Prover.lean   target file the agent edits
  src/Physics/      dimensional-analysis layer (Dimension + Quantity, type-enforced units)
  tmp/              per-problem files (benchmark isolation)
agent/              the agent (Python)
  loop.py           repair loop + hammer pre-pass + resume/branch
  events.py         event protocol — one stream to trace/session/TUI
  session.py        JSONL sessions (~/.prover/sessions/)
  session_manager.py index.jsonl upsert/list + history rebuild
  session_stats.py  per-session token + cost aggregation
  compaction.py     failed-attempts summary (tau's memory model)
  lean.py           lake invocation + diagnostic parsing
  llm.py            LLM provider (OpenAI-compatible) + cost tracking
  lsp.py            Lean language server client (goal-state feedback, run_tactic RPC)
  retrieval.py      local Mathlib lemma-signature keyword index + search
  router.py         per-difficulty model/temperature/step routing (env table)
  formalize.py      autoformalization (NL → compilable Lean theorem)
  plan.py           lemma-bank planning (prove helpers before the main theorem)
  synth.py          synthetic proof corpus generation (JSONL + train split)
  mcp.py            MCP server (expose prove_theorem to any agent)
  acp.py            ACP server (session-based prove for editors)
  search.py         beam/tree search over the repair loop (progress-pruned frontier)
  commands.py       slash-command registry (tau pattern, 22 built-ins)
  paths.py          canonical user/project dir config (env overrides)
  context_window.py token-based context estimation + auto-compaction threshold
  session_usage.py  /usage token + cost dashboard
  thinking.py       reasoning levels → provider kwargs (tau thinking port)
  diagnostics.py    structured JSONL failure log (~/.prover/logs/, tau port)
  rendering.py      --output text/json/transcript renderers (tau rendering port)
  autocomplete.py   slash-command completions (tau pattern)
  themes.py         TUI themes as JSON data (tau pattern)
  terminal_title.py OSC terminal title + braille spinner (tau pattern)
  tui/              Textual TUI package (app.py, screens.py, handlers.py, *.tcss)
  main.py           CLI (prove / bench / tui / mcp / sessions / usage / leaderboard)
  lean_baseline.py  no-LLM baseline runner (prover lean-baseline)
  synth_lean.py     Lean-proved corpus JSONL writer (prover synth-lean)
benchmark/          fixed theorem set + runner + import_standard.py + merge_reports.py
tests/              pytest suite (720 tests)
leaderboard.json    local score history (prover leaderboard; served by lean-prover-web)
```

## Roadmap

- [x] 100-problem graded benchmark (benchmark/problems.json)
- [x] Better error-context extraction (surrounding source, not just line:col)
- [x] Per-problem Lean file isolation (parallel runs)
- [x] Proof trace logging + cost tracking
- [x] Goal-state feedback via Lean LSP (`getInteractiveGoals`)
- [x] MCP server wrapper (`prover mcp`)
- [x] Leaderboard (local: `prover leaderboard`; live board at http://100.96.13.61:8899)
- [x] TUI: custom prove, session replay, parallel workers, Errors panel
- [x] Clipboard (tau port: pyperclip + OSC-52 fallback, selection-aware)
- [x] Slash commands + completions (tau pattern, 21 built-ins) + ctrl+k palette
- [x] Session resume + branching (`/branch <session> [turn]`, model branch summaries)
- [x] File drops into the prompt bar (paths quoted/URI-decoded, tau port)
- [x] Session export (`/export`, JSONL + self-contained HTML transcript)
- [x] Project trust gating (`.prover` protected resources, modal + env policy)
- [x] Prompt templates (`/prompts` picker, slash expansion, project override)
- [x] History compaction (failed-attempts summary) + token-based auto-compaction
- [x] Themes + terminal-title chrome + OSC 9/99 completion notification (tau pattern)
- [x] Queued prompts while running (ctrl+e to edit), /new /compact /name (tau pattern)
- [x] Session token + cost dashboard (`/usage`, per-session and across all)
- [x] `/reload` resource change summary (problems/themes/prompts before→after)
- [x] Thinking levels, structured failure log, `--output` renderers, `prover usage` CLI (tau port)
- [x] Results post + public leaderboard (see leaderboard.json)
- [x] Live leaderboard site (self-hosted `lean-prover-web`, refreshed from `leaderboard.json`)
- [x] Best-of-N search (`--n-attempts`, temperature ramp, first-proof-wins)
- [x] Local Mathlib lemma retrieval (`PROVER_RETRIEVE=1`, ~150k-signature keyword index)
- [x] Per-difficulty model/temperature/step routing (`PROVER_MODEL_<TIER>` table)
- [x] Autoformalization (`prover formalize`, NL → compilable Lean, retry on diagnostics)
- [x] Lemma-bank planning (`PROVER_LEMMA_PLAN=1`, proven-helpers-first)
- [x] Synthetic data + expert-iteration corpus (`prover synth-data`)
- [x] MiniF2F benchmark import + type-check (`benchmark/import_standard.py`)
- [x] LSP `runTactic` primitive (RPC present on Lean ≥ v4.22; available on our pinned v4.33.0)
- [x] Full-file dynamic mode (`--full-file`: model writes whole files; statement enforced) + adaptive step budget (`--adaptive`)

## Not yet verified / deferred (honest scope)

- Live-model verification of best-of-N, retrieval, planning, routing and
  formalize is **unblocked** as of 2026-09-04: the active OpenCode Zen endpoint
  (`https://opencode.ai/zen/v1`) answers `/chat/completions` and the full
  repair loop has been verified end-to-end (a theorem proved through the LLM
  loop, $0 cost). The earlier "HF backup endpoint hangs" blocker no longer
  applies to the configured endpoint.
- The `runTactic` RPC exists on the pinned Lean v4.33.0 toolchain (added in
  Lean v4.22); the primitive is unit-tested but needs live re-verification.
- Full MiniF2F test run (244 problems) **completed 2026-09-04**: **83/244**
  (34%), all proved proofs validator-checked, $0 cost, 4916 tokens, parallel-4.
  Report: `benchmark/minif2f_report.json`. (Statements all type-check on
  Mathlib v4.33.0.)
- Mathlib v4.33.0 landed (`grind` tactic available and used as the strongest hammer). The verifier-reward RL *data* pipeline (`prover rl-data`) is in; actual GRPO fine-tuning is deferred — it needs a trainable/GPU policy model this box lacks (see `agent/finetune.py`). Upstream contributions are deferred.

## Model profiles (`/models` in the TUI)

The TUI manages named model profiles via `/models` (Enter = select active,
`a` add, `e` edit, `d` delete), persisted to `~/.prover/models.json`. Each
profile binds a model name to an optional endpoint and overrides:

```json
{
  "active": "Qwen/Qwen3.8-27B",
  "profiles": [
    {
      "name": "Qwen/Qwen3.8-27B",
      "label": "Qwen 27B",
      "base_url": "http://localhost:8000/v1",
      "api_key": "",
      "context_window": 262144,
      "cost_in": null,
      "cost_out": null
    }
  ]
}
```

- Empty `base_url`/`api_key` fall back to `OPENAI_BASE_URL`/`OPENAI_API_KEY`;
  empty cost fields fall back to the pricing table in `agent/llm.py`.
- `context_window`/`cost_in`/`cost_out` override the auto-compaction budget
  and cost dashboard for that model.
- Resolution: `PROVER_MODEL` env > stored `active` > `gpt-4o`. Env always
  wins, so headless/CI runs are unaffected by TUI-configured profiles.

## TUI keyboard shortcuts

| Key | Action |
|---|---|
| `j` / `k` | Navigate problem list down/up |
| `space` | Queue selected problem for proof |
| `0` / `G` | Jump to top/bottom of list |
| `/` | Focus search bar |
| `Escape` | Blur search, return to prompt |
| `m` / `Ctrl+O` | Open model profiles manager |
| `p` | Prove selected |
| `c` | Prove custom theorem |
| `r` | Run remaining problems |
| `s` | Stop current run |
| `v` | Browse sessions |
| `l` | Leaderboard |
| `w` | Set workers |
| `q` | Quit |

## Resizable panes

Drag the bar between the problem list and the side panels with the mouse to
resize the problem pane (left drag); double-click the bar to reset it to the
default width. The width persists in `~/.prover/tui.json` across runs.

## Live cost meter

The status bar shows real-time token usage and estimated cost during proofs:
`proved X · failed Y · cost≈$0.0123`. Reset with `/new`.
