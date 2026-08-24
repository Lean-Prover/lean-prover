"""OpenAI-compatible provider — real streaming, tool forwarding, cancellation.

Drives the sync SDK's streaming API from a worker thread and bridges chunks
to the async generator via a queue, so consumers get incremental text and
tool-call events (Pi-style provider contract) instead of one blocking blob.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from collections.abc import AsyncIterator

from openai import OpenAI, Timeout

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
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)

_TIMEOUT = Timeout(connect=15.0, read=900.0, write=15.0, pool=15.0)


def _convert_message(m: object) -> dict | None:
    """Map a transcript message to its OpenAI chat wire shape."""
    if isinstance(m, dict):
        return m
    role = getattr(m, "role", None)
    if role == "assistant":
        text_parts: list[str] = []
        tool_calls = []
        for block in getattr(m, "content", []) or []:
            if isinstance(block, TextContent):
                text_parts.append(block.text)
            elif isinstance(block, ThinkingContent):
                continue  # not replayable on OpenAI-compatible endpoints
            elif isinstance(block, ToolCall):
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.arguments),
                    },
                })
        msg: dict = {"role": "assistant", "content": "".join(text_parts)}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg
    if role == "toolResult" and isinstance(m, ToolResultMessage):
        return {
            "role": "tool",
            "tool_call_id": m.tool_call_id,
            "content": m.text or "",
        }
    # user and anything else with .text/.content
    text = getattr(m, "text", None)
    if text is None:
        text = str(getattr(m, "content", ""))
    return {"role": "user", "content": text}


def _convert_tools(tools: list) -> list[dict]:
    """Map AgentTool definitions to OpenAI function-tool schemas."""
    out = []
    for t in tools:
        name = getattr(t, "name", None)
        if not name:
            continue
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": getattr(t, "description", ""),
                "parameters": getattr(t, "parameters", {"type": "object"}),
            },
        })
    return out


_FINISH_REASONS = {"tool_calls": "toolUse", "length": "length", "stop": "stop"}


class _PendingToolCall:
    __slots__ = ("arguments", "block", "content_index", "id", "name")

    def __init__(self) -> None:
        self.id = ""
        self.name = ""
        self.arguments = ""
        self.block: ToolCall | None = None
        self.content_index = -1

    def _parsed_arguments(self) -> dict:
        try:
            args = json.loads(self.arguments) if self.arguments else {}
        except json.JSONDecodeError:
            return {}
        return args if isinstance(args, dict) else {}


class OpenAICompatibleProvider:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 *, compat: object | None = None,
                 headers: dict | None = None,
                 provider_name: str = "openai-compatible") -> None:
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "unused")
        self._explicit_key = self.api_key != "unused"
        self._compat = compat
        self._headers = dict(headers or {})
        self._provider_name = provider_name
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=_TIMEOUT,
            max_retries=2,
            default_headers=self._headers or None,
        )
        self._oauth_clients: dict[tuple[str | None, str], OpenAI] = {}

    def _client_for(self, model: str) -> OpenAI:
        """Client honoring a stored OAuth credential when no API key is set.

        Uses the *exchanged* inference key (``api_key_for_model``): GitHub
        Copilot must send its short-lived ``copilot_internal/v2`` key, not the
        raw GitHub OAuth token.  Non-Copilot providers fall through to the
        OAuth bearer token as-is.
        """
        if self._explicit_key:
            return self._client
        try:
            from ..oauth.access import api_key_for_model

            token = api_key_for_model(model)
        except Exception:  # noqa: BLE001 — auth fallback must never break calls
            token = None
        if not token:
            return self._client
        key = (self.base_url, token)
        cached = self._oauth_clients.get(key)
        if cached is None:
            cached = OpenAI(api_key=token, base_url=self.base_url,
                            timeout=_TIMEOUT, max_retries=2,
                            default_headers=self._headers or None)
            self._oauth_clients[key] = cached
        return cached

    def _request_kwargs(self, model: str, ov_messages: list, ov_tools: list) -> dict:
        """Build the chat-completions request, applying compat flags.

        ``thinking_format``/``reasoning_effort`` translate the resolved
        thinking level (agent/thinking.py) per backend dialect; the remaining
        flags toggle endpoint-specific request extras.
        """
        kwargs: dict = {
            "model": model,
            "messages": ov_messages,
            "tools": ov_tools or None,
            "temperature": 0.2,
            "max_tokens": 4096,
            "stream": True,
        }
        compat = self._compat
        if compat is None:
            return kwargs
        from ..thinking import reasoning_effort_for_level, thinking_level_from_env

        level = thinking_level_from_env()
        if compat.thinking_format == "qwen":
            # vLLM/HF Qwen3 chat templates need an explicit enable_thinking
            # switch; otherwise output lands in a `reasoning` field.
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": level != "off"}}
        elif compat.reasoning_effort and level != "off":
            kwargs["reasoning_effort"] = reasoning_effort_for_level(level)
        if compat.zai_tool_stream:
            # z.ai-class endpoints emit tool-call deltas only when usage is
            # requested in the stream options.
            kwargs["stream_options"] = {"include_usage": True}
        if compat.content_response_format:
            kwargs["response_format"] = {"type": "text"}
        if compat.response_provider_header or compat.provider_key:
            provider = compat.provider_key or self._provider_name
            kwargs["extra_headers"] = {"x-provider": provider}
        return kwargs

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

        ov_messages: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            converted = _convert_message(m)
            if converted is not None:
                ov_messages.append(converted)
        ov_tools = _convert_tools(tools)

        chunks: queue.Queue[tuple[str, object]] = queue.Queue()
        cancel_flag = threading.Event()

        def _run() -> None:
            try:
                stream = self._client_for(model).chat.completions.create(
                    **self._request_kwargs(model, ov_messages, ov_tools))
                for chunk in stream:
                    if cancel_flag.is_set():
                        break
                    chunks.put(("chunk", chunk))
                chunks.put(("end", None))
            except Exception as e:  # noqa: BLE001 — surfaced as AssistantErrorEvent
                chunks.put(("error", e))

        threading.Thread(target=_run, daemon=True, name="llm-stream").start()

        partial = AssistantMessage(content=[], model=model,
                                   api="openai-compatible")
        started = False
        errored = False
        finish_reason: str | None = None
        pending: dict[int, _PendingToolCall] = {}
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

            chunk = payload
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            choice = choices[0]
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            content_delta = getattr(delta, "content", None)
            if content_delta:
                if not started:
                    started = True
                    yield AssistantStartEvent(partial=partial)
                    partial.content.append(TextContent(text=""))
                    yield TextStartEvent(content_index=len(partial.content) - 1,
                                         partial=partial)
                assert isinstance(partial.content[-1], TextContent)
                partial.content[-1].text += content_delta
                yield TextDeltaEvent(content_index=len(partial.content) - 1,
                                     delta=content_delta, partial=partial)

            for tcd in getattr(delta, "tool_calls", None) or []:
                idx = getattr(tcd, "index", 0) or 0
                slot = pending.setdefault(idx, _PendingToolCall())
                if getattr(tcd, "id", None):
                    slot.id = tcd.id
                fn = getattr(tcd, "function", None)
                if fn is None:
                    continue
                if getattr(fn, "name", None) and not slot.name:
                    slot.name = fn.name
                if slot.name and slot.block is None and (
                        slot.arguments or getattr(fn, "arguments", None)):
                    if not started:
                        started = True
                        yield AssistantStartEvent(partial=partial)
                    slot.block = ToolCall(id=slot.id or f"call_{slot.name}",
                                          name=slot.name, arguments={})
                    partial.content.append(slot.block)
                    slot.content_index = len(partial.content) - 1
                    yield ToolCallStartEvent(content_index=slot.content_index,
                                             partial=partial)
                arg_delta = getattr(fn, "arguments", None)
                if arg_delta:
                    slot.arguments += arg_delta
                    if slot.block is not None:
                        slot.block.arguments = slot._parsed_arguments()
                        yield ToolCallDeltaEvent(delta=arg_delta,
                                                 content_index=slot.content_index,
                                                 partial=partial)

        if errored:
            return

        # Finalize: close the text block, then emit tool-call end events.
        if partial.content and isinstance(partial.content[0], TextContent):
            yield TextEndEvent(content_index=0,
                               content=partial.content[0].text, partial=partial)
        for idx in sorted(pending):
            slot = pending[idx]
            if slot.block is None:
                continue
            slot.block.arguments = slot._parsed_arguments()
            yield ToolCallEndEvent(content_index=slot.content_index,
                                   tool_call=slot.block, partial=partial)

        if not started:
            yield AssistantStartEvent(partial=partial)
        reason = _FINISH_REASONS.get(finish_reason or "", "stop")
        partial.stop_reason = reason
        yield AssistantDoneEvent(reason=reason, message=partial)  # type: ignore[arg-type]
