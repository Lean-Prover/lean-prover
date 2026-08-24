"""Unit tests for the OpenAI-compatible provider (agent/providers/openai_compatible.py).

The network client is stubbed with a fake streaming `create()`; the tests
drive the async generator and assert the Pi-style event sequence, tool-call
reassembly, message/tool wire conversion, and cancellation.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import ClassVar

import pytest

from agent.messages import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from agent.provider_events import (
    AssistantDoneEvent,
    AssistantErrorEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from agent.providers.openai_compatible import (
    OpenAICompatibleProvider,
    _convert_message,
    _convert_tools,
)


def _chunk(delta=None, finish_reason=None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)])


def _text_delta(text: str) -> SimpleNamespace:
    return _chunk(SimpleNamespace(content=text, tool_calls=None))


def _tool_delta(index: int, tid=None, name=None, arguments=None) -> SimpleNamespace:
    fn = SimpleNamespace(name=name, arguments=arguments)
    return _chunk(SimpleNamespace(content=None, tool_calls=[
        SimpleNamespace(index=index, id=tid, function=fn)]))


class FakeStreamClient:
    """Stands in for the cached OpenAI client; records kwargs, replays chunks."""

    def __init__(self, chunks: list) -> None:
        self.chunks = chunks
        self.kwargs: dict | None = None

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return iter(self.chunks)


def _provider(chunks: list) -> tuple[OpenAICompatibleProvider, FakeStreamClient]:
    p = OpenAICompatibleProvider(base_url="http://localhost/v1", api_key="test")
    client = FakeStreamClient(chunks)
    p._client = client
    return p, client


async def _collect(agen) -> list:
    return [ev async for ev in agen]


# ---------------------------------------------------------------- conversion


def test_convert_assistant_message_round_trips_tool_calls() -> None:
    m = AssistantMessage(
        content=[TextContent(text="let me check"),
                 ToolCall(id="c1", name="lean_check", arguments={"file": "t.lean"})],
        model="m")
    out = _convert_message(m)
    assert out["role"] == "assistant"
    assert out["content"] == "let me check"
    assert out["tool_calls"][0]["function"]["name"] == "lean_check"
    assert json.loads(out["tool_calls"][0]["function"]["arguments"]) == {"file": "t.lean"}


def test_convert_tool_result_uses_wire_role() -> None:
    m = ToolResultMessage(tool_call_id="c1", tool_name="lean_check",
                          content=[TextContent(text="ok")])
    out = _convert_message(m)
    assert out == {"role": "tool", "tool_call_id": "c1", "content": "ok"}


def test_convert_user_and_tools() -> None:
    assert _convert_message(UserMessage(content="hi")) == {
        "role": "user", "content": "hi"}
    class _T:
        name = "lean_check"
        description = "d"
        parameters: ClassVar[dict] = {"type": "object"}
    tools = _convert_tools([_T()])
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "lean_check"


# ------------------------------------------------------------------ streaming


@pytest.mark.anyio
async def test_stream_text_events_in_order() -> None:
    p, client = _provider([
        _text_delta("hel"), _text_delta("lo"),
        _chunk(finish_reason="stop"),
    ])
    evs = await _collect(p.stream_response(model="m", system="s", messages=[],
                                           tools=[]))
    kinds = [type(e).__name__ for e in evs]
    assert kinds == ["AssistantStartEvent", "TextStartEvent", "TextDeltaEvent",
                     "TextDeltaEvent", "TextEndEvent", "AssistantDoneEvent"]
    done = evs[-1]
    assert done.message.text == "hello"
    assert done.reason == "stop"
    # streaming request actually requested
    assert client.kwargs["stream"] is True


@pytest.mark.anyio
async def test_stream_reassembles_tool_call_fragments() -> None:
    p, _ = _provider([
        _text_delta("checking"),
        _tool_delta(0, tid="call_1", name="lean_check", arguments='{"fi'),
        _tool_delta(0, arguments='le": "a.lean"}'),
        _chunk(finish_reason="tool_calls"),
    ])
    evs = await _collect(p.stream_response(model="m", system="s", messages=[],
                                           tools=[]))
    ends = [e for e in evs if isinstance(e, ToolCallEndEvent)]
    assert len(ends) == 1
    call = ends[0].tool_call
    assert call.id == "call_1"
    assert call.name == "lean_check"
    assert call.arguments == {"file": "a.lean"}
    starts = [e for e in evs if isinstance(e, ToolCallStartEvent)]
    deltas = [e for e in evs if isinstance(e, ToolCallDeltaEvent)]
    assert len(starts) == 1 and len(deltas) == 2
    done = evs[-1]
    assert isinstance(done, AssistantDoneEvent)
    assert done.reason == "toolUse"
    assert done.message.stop_reason == "toolUse"
    # the final message carries both the text block and the parsed call
    blocks = {type(b).__name__ for b in done.message.content}
    assert blocks == {"TextContent", "ToolCall"}


@pytest.mark.anyio
async def test_stream_forwards_tools_to_api() -> None:
    class _T:
        name = "lsp_goals"
        description = "goal state"
        parameters: ClassVar[dict] = {"type": "object"}

    p, client = _provider([_chunk(finish_reason="stop")])
    await _collect(p.stream_response(model="m", system="s", messages=[],
                                     tools=[_T()]))
    assert client.kwargs["tools"][0]["function"]["name"] == "lsp_goals"


@pytest.mark.anyio
async def test_stream_error_before_start_yields_error_event() -> None:
    class BoomClient:
        @property
        def chat(self):
            raise RuntimeError("connection reset")

    p = OpenAICompatibleProvider(base_url="http://x/v1", api_key="k")
    p._client = BoomClient()
    evs = await _collect(p.stream_response(model="m", system="s", messages=[],
                                           tools=[]))
    assert len(evs) == 1
    assert isinstance(evs[0], AssistantErrorEvent)
    assert evs[0].reason == "error"


@pytest.mark.anyio
async def test_cancelled_signal_aborts_immediately() -> None:
    signal = SimpleNamespace(is_cancelled=lambda: True)
    p, client = _provider([_text_delta("never")])
    evs = await _collect(p.stream_response(model="m", system="s", messages=[],
                                           tools=[], signal=signal))
    assert len(evs) == 1
    assert isinstance(evs[0], AssistantErrorEvent)
    assert evs[0].reason == "aborted"
    assert client.kwargs is None  # no request was made
