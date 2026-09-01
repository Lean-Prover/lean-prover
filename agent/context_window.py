"""Approximate context-size estimation — Tau context_window.py extension.

Adds typed-message estimation (``estimate_context_usage`` for ``AgentMessage``
tuples) alongside the existing dict-form estimator.  Also adds
``auto_compaction_threshold_for_context_window`` with a configurable reserve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4
TOOL_CALL_OVERHEAD_TOKENS = 16


@dataclass(frozen=True, slots=True)
class ContextUsageEstimate:
    total_tokens: int
    system_tokens: int
    message_tokens: int
    tool_tokens: int
    message_count: int


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def _message_text(message: dict | Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, (list, tuple)):
            parts = [str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content]
            return "\n".join(p for p in parts if p)
        return str(content) if content else ""
    if hasattr(message, "content") and message.content:
        parts = getattr(message, "content", ())
        if isinstance(parts, (list, tuple)):
            text_parts = [str(getattr(p, "text", p)) for p in parts if hasattr(p, "text") or isinstance(p, str)]
            return "\n".join(text_parts)
        return str(parts)
    return ""


def estimate_message_tokens(message: dict | Any) -> int:
    tokens = MESSAGE_OVERHEAD_TOKENS + estimate_text_tokens(_message_text(message))
    if isinstance(message, dict):
        if message.get("role") == "assistant":
            calls = message.get("tool_calls") or []
            tokens += sum(
                TOOL_CALL_OVERHEAD_TOKENS + estimate_text_tokens(str(c.get("name", "")) + str(c.get("arguments", "")))
                for c in calls
            )
        elif message.get("role") == "tool":
            tokens += estimate_text_tokens(str(message.get("name", "")))
    else:
        if hasattr(message, "tool_calls") and message.tool_calls:
            for call in getattr(message, "tool_calls", []):
                tokens += TOOL_CALL_OVERHEAD_TOKENS + estimate_text_tokens(
                    str(getattr(call, "name", "")) + str(getattr(call, "arguments", ""))
                )
        if hasattr(message, "tool_name") and message.tool_name:
            tokens += estimate_text_tokens(str(getattr(message, "tool_name", "")))
    return tokens


def estimate_context_tokens(system: str, messages: list[dict]) -> int:
    return estimate_context_usage(system=system, messages=messages).total_tokens


def auto_compaction_threshold_for_context_window(context_window_tokens: int) -> int | None:
    from .llm import DEFAULT_COMPACTION_RESERVE_TOKENS

    if context_window_tokens <= 0:
        return None
    return max(1, context_window_tokens - DEFAULT_COMPACTION_RESERVE_TOKENS)


def estimate_context_usage(system: str, messages: list[dict | Any]) -> ContextUsageEstimate:
    system_tokens = estimate_text_tokens(system)
    message_tokens = sum(estimate_message_tokens(m) for m in messages)
    tool_tokens = sum(
        estimate_text_tokens(
            str(m.get("name", "")) if isinstance(m, dict) else str(getattr(m, "tool_name", ""))
        )
        for m in messages
        if (m.get("role") == "tool" if isinstance(m, dict) else hasattr(m, "tool_name"))
    )
    total = system_tokens + message_tokens
    return ContextUsageEstimate(
        total_tokens=total,
        system_tokens=system_tokens,
        message_tokens=message_tokens,
        tool_tokens=tool_tokens,
        message_count=len(messages),
    )