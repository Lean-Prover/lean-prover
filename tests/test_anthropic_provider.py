"""Unit tests for the native Anthropic provider (agent/providers/anthropic.py).

Covers Anthropic's system + messages wire shape, thinking-budget translation,
SSE parsing, and the Pi-style event sequence emitted from content blocks.
No network: the httpx client is faked.
"""

from __future__ import annotations

import json
import queue
from typing import ClassVar

import pytest

from agent.messages import (
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
)
from agent.provider_events import AssistantDoneEvent
from agent.providers.anthropic import (
    AnthropicProvider,
    _convert_message,
    _convert_tools,
    _parse_sse_data,
)


def test_convert_assistant_round_trips_blocks() -> None:
    m = AssistantMessage(content=[
        ThinkingContent(thinking="think here", thinking_signature="sig1"),
        TextContent(text="answer"),
        ToolCall(id="c1", name="lean_check", arguments={"file": "t.lean"}),
    ], model="m")
    out = _convert_message(m)
    assert out["role"] == "assistant"
    assert out["content"][0] == {
        "type": "thinking", "thinking": "think here", "signature": "sig1"}
    assert out["content"][1] == {"type": "text", "text": "answer"}
    assert out["content"][2]["type"] == "tool_use"
    assert out["content"][2]["name"] == "lean_check"
    assert out["content"][2]["input"] == {"file": "t.lean"}


def test_convert_tool_result_uses_anthropic_shape() -> None:
    m = ToolResultMessage(tool_call_id="c1", tool_name="lean_check",
                          content=[TextContent(text="ok")])
    assert _convert_message(m) == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "ok"}],
    }


def test_convert_tools() -> None:
    class _T:
        name = "lean_check"
        description = "d"
        parameters: ClassVar[dict] = {"type": "object"}
    tools = _convert_tools([_T()])
    assert tools[0]["name"] == "lean_check"
    assert tools[0]["input_schema"] == {"type": "object"}


def test_request_body_thinking_budget(monkeypatch) -> None:
    monkeypatch.setenv("PROVER_THINKING", "high")
    p = AnthropicProvider(base_url="https://api.anthropic.com", api_key="k")
    body = p._request_body("claude", "sys", [{"role": "user", "content": "hi"}], [])
    assert body["system"] == "sys"
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert "tools" not in body
    monkeypatch.setenv("PROVER_THINKING", "off")
    off_body = p._request_body("claude", "sys", [], [])
    assert "thinking" not in off_body


def test_parse_sse_data() -> None:
    assert _parse_sse_data('data: {"type":"ping"}') == {"type": "ping"}
    assert _parse_sse_data("data: [DONE]") is None
    assert _parse_sse_data("event: ping") is None
    assert _parse_sse_data("not data") is None


# ------------------------------------------------------------- streaming


class _FakeResp:
    status_code = 200

    def __init__(self, lines) -> None:
        self.lines = lines

    def iter_lines(self):
        yield from self.lines


class _FakeStreamCtx:
    def __init__(self, resp) -> None:
        self._resp = resp

    def __enter__(self):
        return self._resp

    def __exit__(self, *a) -> None:
        pass


class _FakeHttpx:
    lines: ClassVar[list[str]] = []

    class Timeout:
        def __init__(self, **kwargs) -> None:
            pass

    class Client:
        def __init__(self, **kwargs) -> None:
            self._lines = _FakeHttpx.lines
            self.request = None

        def __enter__(self):
            return self

        def __exit__(self, *a) -> None:
            pass

        def stream(self, method, url, json=None, headers=None):
            self.request = {"method": method, "url": url, "json": json,
                            "headers": headers}
            return _FakeStreamCtx(_FakeResp(self._lines))


def _sse(*events) -> list[str]:
    return ["data: " + json.dumps(e) for e in events]


def test_emit_events_parses_blocks(monkeypatch) -> None:
    monkeypatch.setattr("agent.providers.anthropic.httpx", _FakeHttpx)
    p = AnthropicProvider(base_url="https://api.anthropic.com", api_key="k")
    body = {"model": "claude", "max_tokens": 4096, "system": "sys",
            "messages": []}
    chunks: queue.Queue = queue.Queue()
    _FakeHttpx.lines = _sse(
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "hel"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
    )
    p._emit_events("claude", body, chunks)
    kinds = []
    while True:
        kind, payload = chunks.get_nowait()
        if kind == "end":
            break
        kinds.append(payload["type"])
    assert kinds == [
        "content_block_start", "content_block_delta", "content_block_stop",
        "message_delta",
    ]


@pytest.mark.anyio
async def test_stream_response_emits_pi_event_sequence(monkeypatch) -> None:
    monkeypatch.setattr("agent.providers.anthropic.httpx", _FakeHttpx)
    p = AnthropicProvider(base_url="https://api.anthropic.com", api_key="k")
    _FakeHttpx.lines = _sse(
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "thinking"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "thinking_delta", "thinking": "think"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "text_delta", "text": "answer"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2,
         "content_block": {"type": "tool_use", "id": "c1", "name": "lean_check"}},
        {"type": "content_block_delta", "index": 2,
         "delta": {"type": "input_json_delta", "partial_json": '{"x": 1}'}},
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
    )
    evs = [e async for e in p.stream_response(
        model="claude", system="sys", messages=[], tools=[])]
    kinds = [type(e).__name__ for e in evs]
    assert kinds == [
        "AssistantStartEvent",
        "ThinkingStartEvent", "ThinkingDeltaEvent", "ThinkingEndEvent",
        "TextStartEvent", "TextDeltaEvent", "TextEndEvent",
        "ToolCallStartEvent", "ToolCallDeltaEvent", "ToolCallEndEvent",
        "AssistantDoneEvent",
    ]
    done = evs[-1]
    assert isinstance(done, AssistantDoneEvent)
    assert done.reason == "toolUse"
    assert done.message.stop_reason == "toolUse"
    blocks = {type(b).__name__ for b in done.message.content}
    assert blocks == {"ThinkingContent", "TextContent", "ToolCall"}
    call = next(b for b in done.message.content if isinstance(b, ToolCall))
    assert call.arguments == {"x": 1}
    assert done.message.thinking_text == "think"
