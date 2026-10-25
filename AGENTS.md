# AGENTS.md

Instructions for AI coding agents working in this repository.

## Code style (fx-inspired)

Keep the codebase uncluttered and consistent:

- **CLI flags** use kebab-case (`--max-steps`, `--no-record`). Never camelCase
  for flags.
- **Python identifiers** use `snake_case`; classes use `PascalCase`.
- **No emojis** in code, output, or documentation. Unicode symbols such as `∎`
  (the proof terminator) are acceptable and desirable in theorem output.
- **Keep the public surface minimal.** Only expose names that are used outside
  their module.
- **Lint** with ruff before finishing: `ruff check agent/ tests/`.

## Process gate (fx-inspired)

Do not report work as "done" just because the unit tests pass. Before declaring
a change ready:

1. `ruff check agent/ tests/` is clean.
2. `pytest tests/` passes (or the fast subset if the slow LLM/Lean/LSP tests
   need an unavailable endpoint — say so explicitly).
3. Run the real CLI (`prover ...`) on at least one happy path and confirm it
   does not abort and stderr is clean. Tests do not construct the full runtime
   (TUI threads, subprocesses, LSP), so they can miss startup or wiring bugs.

## Invoking the CLI

Use the installed package, e.g. `prover prove "..."` / `prover bench` /
`prover ask "..."`. The `ask` subcommand prints parseable JSON to stdout and
routes human prose to stderr (fx-style single-shot, script-friendly).

## Config precedence (fx-inspired)

User-facing knobs resolve through one documented chain (highest wins):

1. `PROVER_<KEY>` environment variable
2. `<workspace>/.prover/settings.json` (project settings)
3. `~/.prover/settings.json` (user settings)
4. Built-in default

See `agent/settings.py` for the key table. A repository may commit
`.prover.json` with only *repo-safe* defaults (`max_steps`, `context_window`,
`workers`, `quiet`); model/credentials/permission keys are rejected — see
`agent/project_defaults.py`.

## Permissions (fx-inspired)

Per-tool exact allow/deny rules live in `agent/permissions.py`
(`~/.prover/permissions.json`), managed via the `/permissions` slash command
(`/permissions remember allow|deny <tool> [pattern]`, `revoke <id>`,
`mode <ask|auto|yolo>`, `list`). This is orthogonal to the existing per-project
trust in `agent/project_trust.py`.

## Cloudflare

This repo has no Cloudflare deployment. If one is added it must follow the Vaultly-only policy (vault `cf-leanprover`) used by lean-prover-web.
