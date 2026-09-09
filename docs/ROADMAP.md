# Roadmap — lean-prover × lean-prover-web

A living, grounded plan across the two repos in this workspace:

> **Status (2026-09-09): all phases implemented and verified.** Phase 1
> (`prover bench --results` + `/benchmarks` page), Phase 2 (`/docs` page) and
> Phase 3 (docs refresh) are complete. See the verification notes at the end.



| Repo | What it is |
|---|---|
| `lean-prover` | The Lean 4 theorem-proving agent (Python). CLI/TUI/MCP/ACP surfaces over one proof engine. |
| `lean-prover-web` | Astro static org site + live leaderboard (inlines `leaderboard.json` at build time). |

## State audit (2026-09-09)

Before planning, the code was audited against `docs/ARCHITECTURE.md`'s proposed
target. The architecture steps are **largely already done in code**; the docs
had not caught up.

| Step | Claimed in docs | Actual code state | Verdict |
|---|---|---|---|
| A — unify loops | done | `loop.py` is a thin facade over the single `prover_loop.py` engine | done |
| B — TUI package | done | `agent/tui/` package (`app.py`, `screens.py`, `handlers.py`) | done |
| C — ACL gate | done | `permissions.acl_before_tool_call()` gates local + MCP tools | done |
| D — stable core API | "next" | `loop.py` facade routes surfaces; a few modules still import engine constants/helpers (`LEAN_DIR`, `_resolve_lean_dir`, `SYSTEM_FULL`, `_extract_body`, `_split_signature`) | **mostly done** — only helper/constant imports remain, not engine calls |
| E — ports & adapters | "next" | `agent/ports.py` defines `LLMClient`/`LeanRunner`/`LSPClient`/`Retriever` protocols; `prove()` takes `llm_client`/`lean_runner`/`lsp_factory`/`retriever` injectable params with `Default*` adapters | **done** |
| F — robustness | "next" | `provider_retry.py` (exp backoff + full jitter), `CancellationToken`, JSONL append-only session storage, single `events.py` emit path | **done** |
| G — embeddability | optional | `agent/acp.py` ACP server (`initialize`/`session/*`) | **done** |
| H — CLI registry | done | `agent/cli/` package, `main.py` thin facade | done |

**Conclusion:** the architecture roadmap is complete in code. Remaining work is
(1) refreshing the stale docs, and (2) the genuinely-open gap: the web site only
shows an aggregate leaderboard, while the agent already records per-problem
results. Closing that gap is the cross-repo feature below.

## Phase 1 — Cross-repo: per-problem benchmark detail

The `bench` pipeline (`agent/cli/bench.py`) already produces per-problem results
(`_run_problems` → list of `{id, difficulty, statement, proved, steps, seconds,
tokens, ...}`), written today only via the heavy `--report` (includes `trace`
and `attempts`). The web site consumes only the aggregate `leaderboard.json`.

**Goal:** publish a lean, web-friendly per-problem results file from the agent
and render it as a benchmark detail page on the site.

### lean-prover
- Add `prover bench --results <path>`: writes a compact JSON document
  `{schema_version, model, solver, max_steps, date, provenance, results:[...]}`
  with one entry per problem (id, difficulty/tier, statement, proved, steps,
  seconds, tokens). No `trace`/`attempts` bloat.
- Reuse the same write path in `prover leaderboard --run` so recorded runs also
  emit the detail file.
- Document the file contract in `docs/GUIDE.md`.

### lean-prover-web
- `src/data/benchmarkResults.ts`: reads `benchmark-results.json` at build time
  (symlink → `../lean-prover/benchmark-results.json`, same pattern as the
  leaderboard), falling back to a seed file so the build never ships empty.
- `src/pages/benchmarks.astro`: renders a per-model, per-problem table
  (proved/failed, tier, steps, time, tokens), grouped by tier, with a summary
  header.
- Wire into nav + a link from the leaderboard section of `index.astro`.
- `scripts/check-leaderboard.mjs`: warn (not fail) when the results file is
  absent, mirroring the leaderboard fallback.

## Phase 2 — Web: docs page

The site has one landing page; the agent's real usage lives in
`lean-prover/docs/GUIDE.md`. Publish the essentials as a static page.

### lean-prover-web
- `src/pages/docs.astro`: install (Lean toolchain + Mathlib, Python deps, LLM
  config) and core CLI usage (`prover prove` / `ask` / `bench` / `leaderboard`),
  condensed from `GUIDE.md`.
- Link from nav.

## Phase 3 — Docs refresh (lean-prover)

- `docs/ARCHITECTURE.md`: update the "Roadmap status" section to mark steps
  D/E/F/G done with code evidence; note the remaining D cleanup (internal
  helper imports) as optional, non-blocking.
- `docs/GUIDE.md`: refresh the test count and document the new `--results` flag.

## Non-goals

- No re-refactor of the already-done architecture steps (ports, retry, ACP,
  CLI registry) — the code is ahead of the docs; churn would be wasted.
- No forcing every knob through `settings.py`.
- No rewrite of the Lean toolchain / Mathlib wrapper.
- No framework churn on the web site (stays Astro static, zero JS runtime).

## Verification

- lean-prover: `ruff check agent/ tests/`, fast `pytest` subset, real `prover`
  CLI smoke on a happy path.
- lean-prover-web: `npm run check` (astro check) + `npm run build`.
