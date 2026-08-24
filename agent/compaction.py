"""History compaction for the proof repair loop
(ports tau's ContextCompactor to the deterministic, provider-free case).

The loop keeps the LLM history bounded, but naive truncation throws away
everything the model already tried — so weak models ping-pong between the
same dead ends for 20 steps. Instead (tau's rule: summarize what was
already covered, keep recent context verbatim), compact_history() folds
older user/assistant turns into one summary turn listing the distinct
failed attempts and their first error, then keeps the most recent turns
intact.
"""

from __future__ import annotations

import re

MAX_HISTORY_TURNS = 12  # keep this many recent turns verbatim
COMPACT_AT_TURNS = 18   # compact once history grows past this many turns
MAX_LISTED_ATTEMPTS = 10

_ATTEMPT_RE = re.compile(r"```\w*\n(.*?)```", re.DOTALL)

# Errors that are not worth listing (the sorry placeholder step).
# Lean quotes the marker differently across versions ('sorry' vs `sorry`).
_NOISE_FIRST_ERRORS = (
    "declaration uses 'sorry'",
    "declaration uses `sorry`",
    "goals remain",
)


def compact_history(
    messages: list[dict],
    keep_turns: int = MAX_HISTORY_TURNS,
    compact_at_turns: int = COMPACT_AT_TURNS,
) -> tuple[list[dict], str | None]:
    """Compress old turns into a failed-attempts summary.

    Returns (new_messages, summary). summary is None when nothing was
    compacted. Recent `keep_turns` user/assistant pairs stay verbatim;
    older pairs are folded into one leading summary turn.
    """
    if len(messages) <= compact_at_turns * 2:
        return messages, None

    # Align on a user message so the compacted/kept split is a clean pair boundary.
    old = messages[: -keep_turns * 2]
    recent = messages[-keep_turns * 2:]
    while recent and recent[0]["role"] != "user":
        old = old[:-1]
        recent = recent[1:]
    if not old:
        return messages, None

    attempts = _attempt_summaries(old)
    if not attempts:
        return messages, None

    summary = (
        "Earlier attempts in this proof session (do NOT repeat them):\n"
        + "\n".join(f"  {i}. attempt:\n{body}\n     failed: {err}"
                    for i, (body, err) in enumerate(attempts, 1))
    )
    return [
        {"role": "user", "content": summary},
        {"role": "assistant", "content": "(noted — trying a different approach)"},
        *recent,
    ], summary


def _attempt_summaries(
    messages: list[dict],
) -> list[tuple[str, str]]:
    """Distinct (body, first-error) pairs from a run of old turns."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    users = [m for m in messages if m["role"] == "user"]
    assistants = [m for m in messages if m["role"] == "assistant"]
    for i, assistant in enumerate(assistants):
        content = str(assistant.get("content", ""))
        body = _first_code_lines(content)
        # Skip non-attempt turns (compaction placeholders, "(no response)").
        if not _ATTEMPT_RE.search(content) or body == "(empty)":
            continue
        if body in seen:
            continue
        seen.add(body)
        err = _first_error(users[i]) if i < len(users) else "?"
        out.append((body, err))
    return out[-MAX_LISTED_ATTEMPTS:]


def _first_code_lines(text: str, lines: int = 3) -> str:
    """The opening lines of the assistant's proof body, for the summary."""
    m = _ATTEMPT_RE.search(text)
    code = (m.group(1) if m else text).strip()
    code_lines = [ln for ln in code.splitlines() if ln.strip()]
    return "\n".join("     | " + ln.strip() for ln in code_lines[:lines]) or "(empty)"


def _first_error(user: dict) -> str:
    """First diagnostic line from a user turn's compiler report."""
    content = str(user.get("content", ""))
    for line in content.splitlines():
        if ".lean:" in line and ("error" in line or "warning" in line):
            return line.strip()[:120]
    for noise in _NOISE_FIRST_ERRORS:
        if noise in content:
            return noise
    return "(see diagnostics)"
