"""Per-request token usage and a compact ASCII dashboard for prover sessions
(ports tau's session_usage analytics surface to the provider-free record stream).

Prover records are the JSONL event log: `llm_request`, `llm_response`,
`compaction`, `result`. We collect one `RequestUsage` per assistant response,
notable `UsageEvent`s (compactions), and tool-call counts, then render a small
inline dashboard.  Costing reuses llm.estimate_cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from . import llm

__all__ = [
    "USAGE_STYLES",
    "RequestUsage",
    "SessionUsage",
    "UsageEvent",
    "aggregate_usage_by_model",
    "collect_session_usage",
    "estimated_request_cost",
    "render_usage_by_model",
    "render_usage_dashboard",
]


@dataclass(frozen=True, slots=True)
class RequestUsage:
    """Token usage for one LLM request/response cycle."""

    number: int
    timestamp: str
    model: str
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float | None

    @property
    def prompt(self) -> int:
        return self.prompt_tokens

    @property
    def output(self) -> int:
        return self.output_tokens


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """A notable session event positioned against the next model request."""

    request_number: int
    timestamp: str
    kind: str
    label: str


@dataclass(frozen=True, slots=True)
class SessionUsage:
    """Aggregated usage for a session's exported records."""

    requests: tuple[RequestUsage, ...] = ()
    tool_calls: tuple[tuple[str, int], ...] = ()
    compactions: int = 0
    events: tuple[UsageEvent, ...] = ()

    @property
    def total_prompt(self) -> int:
        return sum(item.prompt_tokens for item in self.requests)

    @property
    def total_output(self) -> int:
        return sum(item.output_tokens for item in self.requests)

    @property
    def total_tokens(self) -> int:
        return sum(item.total_tokens for item in self.requests)

    @property
    def total_cost(self) -> float | None:
        costs = [r.estimated_cost for r in self.requests if r.estimated_cost is not None]
        return sum(costs) if costs else None


def estimated_request_cost(
    prompt_tokens: int, output_tokens: int, model: str | None = None
) -> float | None:
    """Estimate a request cost in USD (None when uncosted)."""
    cost = llm.estimate_cost(prompt_tokens, output_tokens, model)
    return cost


def collect_session_usage(records: list[dict], model: str | None = None) -> SessionUsage:
    """Collect per-request token usage, tool-call counts, and notable events."""
    requests: list[RequestUsage] = []
    tools: dict[str, int] = {}
    compactions = 0
    pending_events: list[tuple[str, str, str]] = []
    events: list[UsageEvent] = []
    request_number = 0
    for rec in records:
        ev = rec.get("event")
        ts = datetime.fromtimestamp(
            float(rec.get("timestamp", 0) or 0), tz=UTC
        ).strftime("%H:%M:%S")
        if ev == "compaction":
            compactions += 1
            pending_events.append((ts, "compaction", "Compaction"))
            continue
        if ev == "llm_request":
            pending_events.append((ts, "request", "LLM request"))
            continue
        if ev == "llm_response":
            request_number += 1
            prompt = int(rec.get("prompt_tokens") or 0)
            output = int(rec.get("completion_tokens") or 0)
            total = int(rec.get("tokens") or prompt + output)
            cost = estimated_request_cost(prompt, output, model)
            requests.append(
                RequestUsage(
                    number=request_number,
                    timestamp=ts,
                    model=model or rec.get("model") or "unknown",
                    prompt_tokens=prompt,
                    output_tokens=output,
                    total_tokens=total,
                    estimated_cost=cost,
                )
            )
            tool_calls = rec.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    name = call.get("name") if isinstance(call, dict) else str(call)
                    if name:
                        tools[name] = tools.get(name, 0) + 1
            events.extend(
                UsageEvent(
                    request_number=request_number,
                    timestamp=timestamp,
                    kind=kind,
                    label=label,
                )
                for timestamp, kind, label in pending_events
            )
            pending_events.clear()
    if pending_events:
        events.extend(
            UsageEvent(
                request_number=max(request_number, len(events)) if events else request_number,
                timestamp=timestamp,
                kind=kind,
                label=label,
            )
            for timestamp, kind, label in pending_events
        )
    ordered_tools = tuple(sorted(tools.items(), key=lambda item: (-item[1], item[0])))
    return SessionUsage(
        requests=tuple(requests),
        tool_calls=ordered_tools,
        compactions=compactions,
        events=tuple(events),
    )


_USAGE_CARD_COLORS = {
    "prompt": ("error", "error"),
    "output": ("success", "success"),
    "cached": ("accent", "accent"),
}
USAGE_STYLES = " ".join(f"usage-{k}={c[0]}" for k, c in _USAGE_CARD_COLORS.items())


def _compact_number(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{int(value)}"


def _format_cost(value: float | None) -> str:
    return "N/A" if value is None else f"${value:.2f}"


def _bar(value: float, maximum: float, width: int = 24) -> str:
    if maximum <= 0:
        return " " * width
    filled = min(width, max(0, round((min(value, maximum) / maximum) * width)))
    return "█" * filled + " " * (width - filled)


def aggregate_usage_by_model(usages) -> dict[str, SessionUsage]:
    """Aggregate multiple SessionUsage objects into per-model summaries."""
    by_model: dict[str, list[RequestUsage]] = {}
    for usage in usages:
        for r in usage.requests:
            by_model.setdefault(r.model, []).append(r)
    return {
        model: SessionUsage(requests=tuple(reqs))
        for model, reqs in sorted(by_model.items())
    }


def render_usage_by_model(by_model: dict[str, SessionUsage]) -> str:
    """Render a per-model token/cost summary (sorted by cost descending)."""
    lines = ["", "Usage — by model", "=" * 48]
    if not by_model:
        lines.append("  (no token usage recorded)")
        return "\n".join(lines)
    ordered = sorted(by_model.items(),
                     key=lambda kv: (kv[1].total_cost or 0), reverse=True)
    for model, usage in ordered:
        lines.append("")
        lines.append(f"### {model}")
        lines.append(f"  Requests:   {len(usage.requests):,}")
        lines.append(f"  Tokens:     {_compact_number(usage.total_tokens):>8}")
        lines.append(f"  Cost:       {_format_cost(usage.total_cost):>8}")
    return "\n".join(lines)


def render_usage_dashboard(usage: SessionUsage) -> str:
    """Render a compact ASCII usage dashboard (TUI-friendly)."""
    requests = usage.requests
    if not requests:
        return "No assistant responses with token usage were found in this session."
    total = max((r.total_tokens or 1) for r in requests)
    lines = ["", "Usage", "=" * 48]
    lines.append(f"  Requests:            {len(requests):,}")
    lines.append(f"  Total prompt input:  {_compact_number(usage.total_prompt):>8} tokens")
    lines.append(f"  Total output:        {_compact_number(usage.total_output):>8} tokens")
    lines.append(f"  Total tokens:        {_compact_number(usage.total_tokens):>8}")
    lines.append(f"  Estimated cost:      {_format_cost(usage.total_cost):>8}")
    lines.append(f"  Compactions:         {usage.compactions:>8}")
    if usage.tool_calls:
        top_tool = usage.tool_calls[0]
        lines.append(
            f"  Top tool:            {top_tool[0]} ({top_tool[1]})"
        )
    lines.append("")
    for r in requests[-12:]:
        bar = _bar(r.total_tokens, total)
        lines.append(
            f"  R{r.number:<3} {r.timestamp}  {bar} "
            f"{_compact_number(r.prompt_tokens):>5}→{_compact_number(r.output_tokens):>5} "
            f"{_format_cost(r.estimated_cost):>7}"
        )
    if usage.events:
        lines.append("")
        for ev in usage.events[-6:]:
            lines.append(f"  · {ev.timestamp}  {ev.label} (req {ev.request_number})")
    lines.append("")
    return "\n".join(lines)
