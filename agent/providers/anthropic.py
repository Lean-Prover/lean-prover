"""Anthropic provider — native Messages API streaming with thinking blocks.

Drives Anthropic's ``/v1/messages`` SSE stream from a worker thread and
bridges parsed events to the async generator, translating Anthropic's
``system`` + ``messages`` body shape and ``text``/``thinking``/``tool_use``
content blocks into the Pi-style provider contract (matching
OpenAICompatibleProvider's event ordering).
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from collections.abc import AsyncIterator

from ..messages import (
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
)
from ..provider import CancellationToken
from ..provider_events import (
    AssistantDoneEvent,
    AssistantErrorEvent,
    AssistantStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)

try:
    import httpx
except ImportError:  # pragma: no cover — httpx ships with openai
    httpx = None  # type: ignore[assignment]

_ANTHROPIC_VERSION = "2023-06-01"
_MAX_TOKENS = 4096

_STOP_REASONS = {"end_turn": "stop", "max_tokens": "length",
                 "stop_sequence": "stop", "tool_use": "toolUse"}


def _parse_sse_data(line: str) -> dict | None:
    """Parse one Anthropic SSE ``data:`` line into its event dict (or None)."""
    if not line or not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _convert_message(m: object) -> dict | None:
    """Map a transcript message to the Anthropic Messages wire shape."""
    if isinstance(m, dict):
        return m
    role = getattr(m, "role", None)
    if role == "assistant":
        blocks: list[dict] = []
        for block in getattr(m, "content", []) or []:
            if isinstance(block, TextContent):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ThinkingContent):
                blocks.append({
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": block.thinking_signature,
                })
            elif isinstance(block, ToolCall):
                blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.arguments,
                })
        if not blocks:
            blocks = [{"type": "text", "text": ""}]
        return {"role": "assistant", "content": blocks}
    if role == "toolResult" and isinstance(m, ToolResultMessage):
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": m.text or "",
            }],
        }
    text = getattr(m, "text", None)
    if text is None:
        text = str(getattr(m, "content", ""))
    return {"role": "user", "content": text}


def _convert_tools(tools: list) -> list[dict]:
    """Map AgentTool definitions to Anthropic tool schemas."""
    out = []
    for t in tools:
        name = getattr(t, "name", None)
        if not name:
            continue
        out.append({
            "name": name,
            "description": getattr(t, "description", ""),
            "input_schema": getattr(t, "parameters", {"type": "object"}),
        })
    return out


class AnthropicProvider:
    def __init__(self, base_url: str | None = None,
                 api_key: str | None = None,
                 *, headers: dict | None = None,
                 provider_name: str = "anthropic") -> None:
        self.base_url = (base_url or os.getenv("ANTHROPIC_BASE_URL")
                         or "https://api.anthropic.com")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._headers = dict(headers or {})
        self._provider_name = provider_name

    def _api_key_for(self, model: str) -> str | None:
        """Explicit key > env key > OAuth token (best-effort)."""
        if self.api_key:
            return self.api_key
        try:
            from ..oauth.access import api_key_for_model

            return api_key_for_model(model)
        except Exception:  # noqa: BLE001 — auth fallback must never break calls
            return None

    def _request_body(self, model: str, system: str, messages: list,
                      tools: list) -> dict:
        from ..thinking import (
            anthropic_thinking_budget_for_level,
            thinking_level_from_env,
        )

        body: dict = {
            "model": model,
            "max_tokens": _MAX_TOKENS,
            "system": system,
            "messages": messages,
        }
        converted_tools = _convert_tools(tools)
        if converted_tools:
            body["tools"] = converted_tools
        budget = anthropic_thinking_budget_for_level(thinking_level_from_env())
        if budget is not None:
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
        return body

    def _headers_for(self, model: str) -> dict:
        key = self._api_key_for(model)
        headers = {
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
            **self._headers,
        }
        if key:
            headers["x-api-key"] = key
        return headers

    def _emit_events(self, model: str, body: dict, chunks: queue.Queue) -> None:
        """Stream Anthropic SSE and enqueue parsed (kind, payload) tuples."""
        if httpx is None:
            chunks.put(("error", RuntimeError("httpx is required for Anthropic")))
            return
        url = self.base_url.rstrip("/") + "/v1/messages"
        try:
            with (
                httpx.Client(timeout=httpx.Timeout(connect=15.0, read=900.0,
                                                   write=15.0, pool=15.0)) as client,
                client.stream("POST", url, json=body,
                              headers=self._headers_for(model)) as resp,
            ):
                    if resp.status_code >= 400:
                        chunks.put(("error", RuntimeError(
                            f"Anthropic HTTP {resp.status_code}: {resp.text[:500]}")))
                        return
                    for line in resp.iter_lines():
                        event = _parse_sse_data(line)
                        if event is not None:
                            chunks.put(("event", event))
            chunks.put(("end", None))
        except Exception as e:  # noqa: BLE001 — surfaced as error event
            chunks.put(("error", e))

    async def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list,
        tools: list,
        signal: CancellationToken | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator:
        if signal and signal.is_cancelled():
            yield AssistantErrorEvent(reason="aborted", error=AssistantMessage(
                content=[], model=model, stop_reason="aborted",
                error_message="cancelled before request"))
            return

        ov_messages: list[dict] = []
        for m in messages:
            converted = _convert_message(m)
            if converted is not None:
                ov_messages.append(converted)
        body = self._request_body(model, system, ov_messages, tools)

        chunks: queue.Queue[tuple[str, object]] = queue.Queue()
        cancel_flag = threading.Event()
        threading.Thread(
            target=self._emit_events, args=(model, body, chunks),
            daemon=True, name="anthropic-stream").start()

        partial = AssistantMessage(content=[], model=model, api="anthropic")
        started = False
        errored = False
        stop_reason: str | None = None
        # Per content-block index: dict with type/state accumulated across
        # start/delta/stop SSE events.
        blocks: dict[int, dict] = {}
        poll = 0.2

        def _error(reason: str, message: str) -> AssistantErrorEvent:
            nonlocal errored
            errored = True
            stop = "aborted" if reason == "aborted" else "error"
            return AssistantErrorEvent(reason=reason, error=AssistantMessage(
                content=list(partial.content), model=model, stop_reason=stop,
                error_message=message))

        while True:
            try:
                kind, payload = await asyncio.to_thread(chunks.get, True, poll)
            except queue.Empty:
                if signal and signal.is_cancelled():
                    cancel_flag.set()
                    yield _error("aborted", "cancelled during stream")
                    return
                continue
            except Exception:  # noqa: BLE001 — executor shutdown mid-stream
                return

            if kind == "error":
                cancel_flag.set()
                yield _error("error", str(payload))
                return
            if kind == "end":
                break

            event = payload
            etype = event.get("type")
            if etype == "error":
                msg = (event.get("error") or {}).get("message", "unknown error")
                yield _error("error", msg)
                return
            if etype == "message_delta":
                stop_reason = (event.get("delta") or {}).get("stop_reason")
                continue
            if etype not in ("content_block_start", "content_block_delta",
                             "content_block_stop"):
                continue

            index = event.get("index", 0)
            block = blocks.setdefault(index, {"type": None, "state": {}})
            if etype == "content_block_start":
                cb = event.get("content_block") or {}
                btype = cb.get("type")
                block["type"] = btype
                if not started:
                    started = True
                    yield AssistantStartEvent(partial=partial)
                if btype == "text":
                    partial.content.append(TextContent(text=""))
                    block["index"] = len(partial.content) - 1
                    yield TextStartEvent(content_index=block["index"],
                                         partial=partial)
                elif btype == "thinking":
                    partial.content.append(ThinkingContent(thinking=""))
                    block["index"] = len(partial.content) - 1
                    yield ThinkingStartEvent(content_index=block["index"],
                                             partial=partial)
                elif btype == "tool_use":
                    tool = ToolCall(id=cb.get("id", ""), name=cb.get("name", ""),
                                    arguments={})
                    partial.content.append(tool)
                    block["tool"] = tool
                    block["index"] = len(partial.content) - 1
                    yield ToolCallStartEvent(content_index=block["index"],
                                             partial=partial)
            elif etype == "content_block_delta":
                delta = event.get("delta") or {}
                dtype = delta.get("type")
                ci = block.get("index")
                if dtype == "text_delta":
                    assert isinstance(partial.content[ci], TextContent)
                    partial.content[ci].text += delta.get("text", "")
                    yield TextDeltaEvent(content_index=ci,
                                         delta=delta.get("text", ""),
                                         partial=partial)
                elif dtype == "thinking_delta":
                    assert isinstance(partial.content[ci], ThinkingContent)
                    partial.content[ci].thinking += delta.get("thinking", "")
                    yield ThinkingDeltaEvent(content_index=ci,
                                             delta=delta.get("thinking", ""),
                                             partial=partial)
                elif dtype == "signature_delta":
                    assert isinstance(partial.content[ci], ThinkingContent)
                    partial.content[ci].thinking_signature = delta.get("signature")
                elif dtype == "input_json_delta":
                    tool = block.get("tool")
                    if tool is not None:
                        block["json"] = block.get("json", "") + delta.get("partial_json", "")
                        try:
                            tool.arguments = json.loads(block["json"])
                        except json.JSONDecodeError:
                            pass
                        yield ToolCallDeltaEvent(delta=delta.get("partial_json", ""),
                                                 content_index=ci, partial=partial)
            elif etype == "content_block_stop":
                ci = block.get("index")
                if block["type"] == "text":
                    yield TextEndEvent(content_index=ci,
                                       content=partial.content[ci].text,
                                       partial=partial)
                elif block["type"] == "thinking":
                    yield ThinkingEndEvent(content_index=ci,
                                           content=partial.content[ci].thinking,
                                           partial=partial)
                elif block["type"] == "tool_use":
                    tool = block.get("tool")
                    if tool is not None:
                        yield ToolCallEndEvent(content_index=ci,
                                               tool_call=tool, partial=partial)

        if errored:
            return
        if not started:
            yield AssistantStartEvent(partial=partial)
        reason = _STOP_REASONS.get(stop_reason or "", "stop")
        partial.stop_reason = reason  # type: ignore[assignment]
        yield AssistantDoneEvent(reason=reason, message=partial)  # type: ignore[arg-type]
