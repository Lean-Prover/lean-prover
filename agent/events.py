"""Event protocol for the proof loop.

Every state change in `loop.prove()` flows through a single `emit()` path as
a self-describing record:

    {"t": 1730000000.123, "event": "build", "step": 3, "diagnostics": 2, ...}

Three consumers, one stream:
- `trace`   — kept in memory, embedded in Result and report JSON
- session   — JSONL under ~/.prover/sessions/ (agent/session.py)
- `on_event` — live callback (TUI), see agent/tui.py

The CLI renders records with `format()`; the TUI renders the same records in
its own way. The loop itself prints nothing.
"""

from __future__ import annotations

import time


def record(event: str, **payload) -> dict:
    """Build a self-describing event record."""
    return {"t": time.time(), "event": event, **payload}


def result_summary(rec: dict) -> str:
    """Plain-text summary line for a ``result`` record.

    Single source of truth for the PROVED/FAILED/stopped wording, shared by
    the CLI ``format`` path and any other consumer (avoids drift).
    """
    secs = rec.get("seconds", 0.0)
    steps = rec.get("steps", 0)
    if rec.get("stopped"):
        return f"stopped by user ({secs:.1f}s)"
    if rec.get("proved"):
        return f"PROVED ∎ ({steps} steps, {secs:.1f}s)"
    return f"FAILED after {steps} steps ({secs:.1f}s)"


def format(rec: dict) -> str | None:
    """Human-readable line for a record, or None if it shouldn't be printed."""
    ev = rec.get("event")
    if ev == "start":
        stmt = rec.get("statement", "")
        return f"start: {rec.get('problem_id', '?')} (max_steps={rec.get('max_steps')}) {str(stmt)[:70]}"
    if ev == "hammer":
        mark = "✓" if rec.get("ok") else "✗"
        return f"  [hammer {rec['i']}/{rec['total']}] `{rec['tactic']}` {mark}"
    if ev == "resume":
        branch = f" at turn {rec['branch_at']}" if rec.get("branch_at") is not None else ""
        return (f"  resumed session {rec.get('session_id')} "
                f"({rec.get('seed_turns')} turns seeded){branch}")
    if ev == "llm_start":
        return "  no hammer worked, starting LLM loop"
    if ev == "build":
        if rec.get("ok"):
            return None  # the proved record carries the good news
        summary = str(rec.get("summary", ""))[:70]
        return f"  [step {rec['step']}] {rec['diagnostics']} diagnostics — {summary}"
    if ev == "goals":
        first = str(rec.get("goals", "")).strip().splitlines()
        return f"  [step {rec['step']}] goal: {first[0] if first else ''}"
    if ev == "llm_request":
        return None
    if ev == "compaction":
        return "  [compaction] folded old turns into a failed-attempts summary"
    if ev == "llm_response":
        return f"  [step {rec['step']}] LLM replied ({rec.get('tokens', '?')} tokens)"
    if ev == "llm_error":
        return f"  [step {rec['step']}] {str(rec.get('error', ''))[:100]}"
    if ev == "result":
        return f"  {result_summary(rec)}"
    return None


try:
    from typing import Annotated, Literal, TypeAlias

    from pydantic import Field

    from .messages import AgentMessage, ToolResultMessage, WireModel
    from .provider_events import AssistantMessageEvent
    from .tools import AgentToolResult
    from .types import JSONValue

    class AgentStartEvent(WireModel):
        type: Literal["agent_start"] = "agent_start"

    class AgentEndEvent(WireModel):
        type: Literal["agent_end"] = "agent_end"
        messages: list[AgentMessage] = Field(default_factory=list)  # type: ignore

    class TurnStartEvent(WireModel):
        type: Literal["turn_start"] = "turn_start"

    class TurnEndEvent(WireModel):
        type: Literal["turn_end"] = "turn_end"
        message: AgentMessage  # type: ignore
        tool_results: list[ToolResultMessage] = Field(default_factory=list)  # type: ignore

    class MessageStartEvent(WireModel):
        type: Literal["message_start"] = "message_start"
        message: AgentMessage  # type: ignore

    class MessageUpdateEvent(WireModel):
        type: Literal["message_update"] = "message_update"
        message: AgentMessage  # type: ignore
        assistant_message_event: AssistantMessageEvent = Field(serialization_alias="assistantMessageEvent")  # type: ignore

    class MessageEndEvent(WireModel):
        type: Literal["message_end"] = "message_end"
        message: AgentMessage  # type: ignore

    class ToolExecutionStartEvent(WireModel):
        type: Literal["tool_execution_start"] = "tool_execution_start"
        tool_call_id: str
        tool_name: str
        args: dict[str, JSONValue] = Field(default_factory=dict)  # type: ignore

    class ToolExecutionUpdateEvent(WireModel):
        type: Literal["tool_execution_update"] = "tool_execution_update"
        tool_call_id: str
        tool_name: str
        args: dict[str, JSONValue] = Field(default_factory=dict)  # type: ignore
        partial_result: AgentToolResult  # type: ignore

    class ToolExecutionEndEvent(WireModel):
        type: Literal["tool_execution_end"] = "tool_execution_end"
        tool_call_id: str
        tool_name: str
        result: AgentToolResult  # type: ignore
        is_error: bool

    AgentEvent: TypeAlias = Annotated[  # type: ignore
        AgentStartEvent
        | AgentEndEvent
        | TurnStartEvent
        | TurnEndEvent
        | MessageStartEvent
        | MessageUpdateEvent
        | MessageEndEvent
        | ToolExecutionStartEvent
        | ToolExecutionUpdateEvent
        | ToolExecutionEndEvent,
        Field(discriminator="type"),
    ]
except Exception as exc:  # noqa: BLE001 — typed-event layer is optional
    # The flat dict event stream still works everywhere; only the typed
    # pydantic AgentEvent union is unavailable. Surface it instead of silently
    # degrading the type layer.
    import warnings

    warnings.warn(
        f"agent.events typed-event layer unavailable ({exc!r}); "
        "flat event records still work as plain dicts",
        RuntimeWarning,
        stacklevel=1,
    )
