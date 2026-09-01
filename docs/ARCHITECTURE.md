# Architecture — Current & Proposed

A living map of lean-prover: what exists today, and a future-ready target that
keeps the working prover intact while removing the sharp edges (the two loop
files, the god-class TUI, the fat CLI dispatcher).

---

## 1. Current architecture (as of `38549ad`)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            CLI — agent/main.py                              │
│   _build_parser() → cli() → dispatch                                        │
│   prove · ask · bench · mcp · formalize · synth-data · lean-baseline        │
│   synth-lean · datagen · finetune · tui · login · update · version          │
│   rpc · chat · export · sessions · usage · leaderboard                      │
└──────────────┬──────────────────────┬─────────────────┬────────────────────┘
               │                      │                 │
        ┌──────▼─────┐          ┌─────▼─────┐     ┌─────▼──────┐
        │  TUI       │          │  MCP      │     │ ask /bench │
        │ tui.py ────│          │ mcp.py    │     │ /renderers │
        │ (2 280 LOC)│          │ (298 LOC) │     │            │
        └──────┬─────┘          └─────┬─────┘     └────────────┘
               │      slash/CommandResult   ACL gate (permissions.py)
               │              │             │
               ▼              ▼             ▼
        ┌───────────────────────────────────────────────┐
        │            PROOF ENGINE                        │
        │   prover_loop.py (739 LOC)  ·  loop.py (408)   │  ◄── TWO loops (hazard)
        │   prove() · prove_best_of() · hammer · adaptive │
        └──────┬──────────┬───────────────┬─────────┬────┘
               ▼          ▼               ▼         ▼
        ┌──────────┐ ┌─────────┐   ┌───────────┐ ┌──────────┐
        │ llm.py   │ │ lean.py │   │ lsp.py    │ │ tools.py │
        │ providers│ │ lake    │   │ goal state│ │ hammers  │
        │  + cost  │ │ compile │   │ runTactic │ │ baselines│
        └────┬─────┘ └─────────┘   └──────┬────┘ └──────────┘
             └───────────────┬────────────┘
                             ▼
              OpenAI-compatible /v1 · Lean toolchain (lake/Mathlib)

   Already-clean packages: session/  providers/  rendering/  extensions/
   fx-adopted layer (new): settings.py · project_defaults.py · permissions.py
```

**Today's sharp edges** (what the proposal targets):

| # | Problem | Location | Status |
|---|---|---|---|
| 1 | `loop.py` was a fragile import-shim (try/except swallow + LEAN_DIR re-patch) | `loop.py` | **done** (facade, `3f12704`) |
| 2 | God-class TUI | `tui.py` 2 280 LOC | **done** (`tui/` package, below) |
| 3 | Fat CLI dispatcher | `main.py` 725 LOC | **done** (`cli/` registry, below) |
| 4 | Engine reached ad hoc by 7 modules | `coding_session`, `coding_tools`, `loop`, `main`, `rpc`, `system_prompt`, `tui` | open (facade thin end) |
| 5 | ACL gate only on MCP | `mcp.py` | **done** (local loop gate, next commit) |

---

## 2. Proposed future-ready architecture

Principles: **one engine, many thin surfaces; boundaries over batteries; a
stable core API; ports-and-adapters for robustness.**

```
┌───────────────────────────  SURFACES (thin, swappable)  ────────────────────────┐
│  CLI       TUI (Textual)    MCP stdio    ACP (editors)   RPC    Batch / CI      │
│ cli/*.py   tui/app.py      mcp.py        acp/*.py        rpc.py  bench/*.py     │
│           + screens/handlers                                                      │
└──────┬───────────┬───────────────┬──────────────┬────────────┬───────────────────┘
       │           │               │              │            │
       ▼           ▼               ▼              ▼            ▼
┌────────────────────────────  CORE API (stable contract)  ───────────────────────┐
│   ProofSession · ProofResult · attempt · EventsStream                            │
│   prove()  prove_best_of()  hammer()  formalize()  plan()  bench()               │
└───────────────┬────────────────────────────┬─────────────────────────────────────┘
                │                            │
   ┌────────────▼─────────────┐   ┌──────────▼───────────────────────────────┐
   │   PROOF ENGINE (pure)    │   │   CONTROL LAYER (fx-adopted)              │
   │   engine/ — the single   │   │   settings   project_defaults             │
   │   repair loop + budgeting │   │   permissions (ACL gate ALL tools)        │
   │   NO I/O in the core     │   │   sessions/compaction/events              │
   └────────────┬─────────────┘   └────────────────────────────────────────────┘
                │            (depends on ports, not concrete adapters)
   ┌────────────▼───────────────  INFRA / ADAPTERS (behind ports)  ──────────┐
   │  LLMClient   LeanRunner   LSPClient   Retriever   Tools   CostEstimator │
   │  providers/  lean.py      lsp.py      retrieval   tools.py llm.cost     │
   │  context_window · models · session/* · skills                           │
   └──────────────┬──────────────────────────────────────────────────────────┘
                  ▼
        OpenAI-compatible · Lean toolchain (lake/Mathlib) · providers fleet

```

---

## 3. What changes — with priority

| Step | Change | Priority | Risk |
|---|---|---|---|
| A | **Unify loops**: one `engine/` `ProofEngine`; delete/alias `loop.py`; single `attempts`/`Result` shape | High | Medium |
| B | **Extract TUI** into `tui/` package (`app.py`, `screens/`, `handlers/`); thin core-API calls | High | Medium-High |
| C | **Extend ACL gate** beyond MCP to local/TUI sensitive tools | High | Low |
| D | **Core API as the only entry**: surfaces call `engine.api.*`, never internals | High | Medium |
| E | **Ports & adapters**: `LLMClient`/`LeanRunner`/`LSPClient` interfaces (enables fakes/stubs for tests, provider swap) | Medium | Medium |
| F | **Robustness**: retry+backoff with jitter, per-attempt cancellation, transactional session recovery (WAL), structured telemetry sink behind a port | Medium | Medium |
| G | **Embeddability**: add ACP server; optional wasm via thin adapter (reuse today's `providers/` + pure engine) | Low (optional) | Medium |
| H | **CLI→`cli/` registry**: thin command modules, remove god-dispatcher | Low | Low-Medium | **done** |

Priority order is **A → D → C → E → F → B → H**; G only if embeddability is a
stated goal. A/D give the biggest maintainability win with the least churn;
C is a quick security win; B/H are cosmetic-but-worthwhile; G is truly optional.

---

## 4. Robustness details (what "robust" concretely adds)

- **Pure core**: the engine holds no I/O — all effects go through injected ports,
  so tests can drive the whole loop with fakes (no endpoint/LSP needed).
- **Retry & backoff**: exponential backoff + jitter on provider/transport errors
  (extends today's thin `provider_retry.py`).
- **Cancellation**: per-attempt cancel tokens; surfaces can stop cleanly.
- **Transactional sessions**: write-ahead append (already JSONL) + startup
  recovery for interrupted runs (extends `session_manager.py`/`session/storage.py`).
- **Observability**: everything already flows through the single `events.py`
  emit path — add a telemetry sink behind a port so TUI/JSON/`/usage` stay in sync
  without touching the engine.

## 5. Non-goals (kept out on purpose)

- No rewrite of the Lean toolchain / Mathlib — we keep wrapping `lake`/Mathlib.
- No forcing every knob through `settings.py` — domain-specific resolution
  (model profiles, family context tables) stays where it owns the semantics.
- No Zig / no framework churn.

## 6. Roadmap status

**Step A — DONE (`3f12704`).** Investigation corrected the assumption: there is
no duplicate engine. `prover_loop.py` is the single canonical proof engine;
`loop.py` was a fragile `try/except ImportError` import-shim (plus the async
coding loop). Step A removed the silent-swallow guard and the LEAN_DIR
re-patching, leaving `loop.py` a thin facade (`prove` syncs a redirected
`LEAN_DIR`; `prove_best_of` loops through `loop.prove` to keep the monkeypatch
contract). Verified: 38 hermetic loop tests + full fast suite + real CLI.

No `engine/` package was created — relocating working files would be churn with
no functional gain, since the facade already routes every surface through the
one canonical engine. Creating `engine/` is deferred and only worth it if we
later want to bound `prover_loop.py`'s size (currently 739 LOC).

**Step C — DONE.** The per-tool ACL now gates **local** tool invocation too, not
just MCP. `permissions.acl_before_tool_call()` builds a `before_tool_call` hook
(deny rules block; everything else open by default) wired as the default in the
coding-harness construction (`coding_session`), so the `loop.run_agent_loop`
path enforces the same ACL as the MCP server. Verified: 36 loop/permissions +
441 fast tests, ruff clean.

**Next (by value):** Step **D** (surfaces call a stable core API, not engine
internals — mostly done via the `loop.py` facade). Step **E/F** (ports &
adapters, robustness) after that.

**Step B — DONE (`tui/` package).** The earlier assessment that the TUI was
"not a clean split" was resolved by making `agent.tui` a **package** instead of
a module:

- `agent/tui/app.py` — `ProverApp` (composition, bindings, command dispatch)
- `agent/tui/screens.py` — the 12 `ModalScreen` subclasses + `build_session_label`
- `agent/tui/handlers.py` — `ProverRunMixin` (proof-run + live event rendering)
- `agent/tui/app.tcss` — app chrome styling; `agent/tui/screens.tcss` — screen
  styling, loaded via each screen's `CSS_PATH` (per-screen CSS)
- `agent/tui/__init__.py` — the public API + the `REPO` / `PROBLEMS_FILE` /
  `MAX_WORKERS` module constants.

The reason the earlier extraction failed was that the screens referenced
`render_event`, `REPO`/`PROBLEMS_FILE`, and `sess`/`SessionManager`. That is
handled by (a) keeping `render_event`/widgets in the existing `tui_render.py`
leaf module, and (b) defining `REPO`/`PROBLEMS_FILE`/`MAX_WORKERS` in the
package `__init__` *before* the submodule imports and having the app/screens
read them through the `agent.tui` namespace at call time — so the test
monkeypatch contract (`monkeypatch.setattr("agent.tui.REPO", ...)`) survives
the split.

The TUI now uses Textual idioms throughout: the parallel proof runner spawns
**one Textual thread-worker per problem** (`@work(thread=True, group="prove")`)
bounded by a semaphore (replacing the inner `ThreadPoolExecutor`), the run
coordinator and background jobs (branch/login/update) use `@work` decorators,
worker count is a Textual `Reactive`, and all widget access goes through
`call_from_thread`. Verified: 655 tests pass (twice), TUI worker smoke tests,
real `prover` CLI, ruff clean.

**Step H — DONE (`cli/` registry).** The fat CLI dispatcher in `agent/main.py`
(725 LOC) is now a thin re-export facade over the command registry in
`agent/cli/`:

- `agent/cli/parser.py` — the argparse dispatcher (`cli` / `_build_parser`)
- `agent/cli/prove.py` — `prover prove` / `prover ask`
- `agent/cli/bench.py` — `prover bench` (scoring core: `_prove_one`/`_run_problems`)
- `agent/cli/leaderboard.py` — `prover leaderboard` + provenance/comparability
- `agent/cli/sessions.py` — `prover sessions` / `prover usage`
- `agent/cli/misc.py` — thin delegators (mcp/formalize/synth/finetune/tui/…)
- `agent/cli/common.py` — shared helpers
- `agent/cli/__init__.py` — the registry; `agent/main.py` re-exports it.

Two things made this non-trivial and were handled deliberately:
1. The existing `agent/cli.py` was the separate *coding facade* (prog
   `prover-coding`, Tau port) — it collided with the `cli/` package name, so it
   moved to `agent/cli_coding.py` (main.py delegation + `test_phase16` updated).
2. The test suite monkeypatches `agent.main` module attributes (`prove`,
   `prove_best_of`, `_prove_one`, `load_board`, `BOARD_FILE`, `cmd_tui`). The
   command modules therefore read those names through the `agent.main` facade
   at call time (`from agent import main as _m; _m.prove(...)`), the same
   pattern as the TUI's `REPO`, so the monkeypatch contract survives unchanged.

Verified: 658 tests pass (twice), real `prover` CLI (help/version/sessions/
leaderboard/prove), ruff clean.
