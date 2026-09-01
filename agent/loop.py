"""Pure Pi-compatible provider/tool agent loop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence

from .events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from .messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from .provider import CancellationToken, ModelProvider
from .provider_events import (
    AssistantDoneEvent,
    AssistantErrorEvent,
    AssistantMessageEvent,
    AssistantStartEvent,
)
from .tool_history import repair_tool_history
from .tools import AgentTool, AgentToolResult

BeforeToolCall = Callable[[ToolCall], Awaitable[tuple[bool, str | None]]]
AfterToolCall = Callable[
    [ToolCall, AgentToolResult, bool],
    Awaitable[tuple[AgentToolResult, bool]],
]


async def run_agent_loop(
    *,
    provider: ModelProvider,
    model: str,
    system: str,
    messages: list[AgentMessage],
    tools: list[AgentTool],
    prompts: Sequence[AgentMessage] = (),
    prelude_messages: Sequence[AgentMessage] = (),
    max_turns: int | None = None,
    signal: CancellationToken | None = None,
    session_id: str | None = None,
    get_steering_messages: Callable[[], Sequence[AgentMessage]] | None = None,
    get_follow_up_messages: Callable[[], Sequence[AgentMessage]] | None = None,
    before_tool_call: BeforeToolCall | None = None,
    after_tool_call: AfterToolCall | None = None,
) -> AsyncIterator[AgentEvent]:
    """Run the provider/tool loop and emit Pi-compatible agent events."""
    new_messages = list(prompts)
    if prompts:
        messages.extend(prompts)

    yield AgentStartEvent()
    yield TurnStartEvent()
    for message in prelude_messages:
        yield MessageStartEvent(message=message)
        yield MessageEndEvent(message=message)
    for prompt in prompts:
        yield MessageStartEvent(message=prompt)
        yield MessageEndEvent(message=prompt)

    if max_turns is not None and max_turns < 1:
        error = _error_message(model, "max_turns must be at least 1")
        messages.append(error)
        new_messages.append(error)
        yield MessageStartEvent(message=error)
        yield MessageEndEvent(message=error)
        yield TurnEndEvent(message=error)
        yield AgentEndEvent(messages=new_messages)
        return

    tool_by_name = {tool.name: tool for tool in tools}
    turn = 1
    first_turn = True
    pending = tuple(get_steering_messages() if get_steering_messages else ())

    while True:
        has_more_tools = True
        while has_more_tools or pending:
            if not first_turn:
                yield TurnStartEvent()
            first_turn = False

            for message in pending:
                messages.append(message)
                new_messages.append(message)
                yield MessageStartEvent(message=message)
                yield MessageEndEvent(message=message)
            pending = ()

            if max_turns is not None and turn > max_turns:
                error = _error_message(model, f"Agent stopped after max_turns={max_turns}")
                messages.append(error)
                new_messages.append(error)
                yield MessageStartEvent(message=error)
                yield MessageEndEvent(message=error)
                yield TurnEndEvent(message=error)
                yield AgentEndEvent(messages=new_messages)
                return

            # Python async generators cannot pass a yielding callback through a
            # normal await cleanly, so consume the assistant sub-generator and
            # retain its final message through the terminal event.
            assistant = None
            async for event in _assistant_events(
                provider=provider,
                model=model,
                system=system,
                messages=_provider_context(messages),
                tools=tools,
                signal=signal,
                session_id=session_id,
            ):
                yield event
                if isinstance(event, MessageEndEvent) and isinstance(
                    event.message, AssistantMessage
                ):
                    assistant = event.message

            if assistant is None:  # defensive: _assistant_events always terminates
                assistant = _error_message(model, "Provider produced no assistant message")
                yield MessageStartEvent(message=assistant)
                yield MessageEndEvent(message=assistant)

            messages.append(assistant)
            new_messages.append(assistant)
            if assistant.stop_reason in {"error", "aborted"}:
                yield TurnEndEvent(message=assistant)
                yield AgentEndEvent(messages=new_messages)
                return

            tool_results: list[ToolResultMessage] = []
            calls = list(assistant.tool_calls)
            has_more_tools = bool(calls)
            for call in calls:
                async for event in _execute_tool_call(
                    call,
                    tool_by_name,
                    signal,
                    before_tool_call,
                    after_tool_call,
                ):
                    yield event
                    if isinstance(event, MessageEndEvent) and isinstance(
                        event.message, ToolResultMessage
                    ):
                        tool_results.append(event.message)
                        messages.append(event.message)
                        new_messages.append(event.message)

            yield TurnEndEvent(message=assistant, tool_results=tool_results)
            turn += 1
            pending = tuple(get_steering_messages() if get_steering_messages else ())

        follow_ups = tuple(get_follow_up_messages() if get_follow_up_messages else ())
        if follow_ups:
            pending = follow_ups
            continue
        break

    yield AgentEndEvent(messages=new_messages)


def _provider_context(messages: list[AgentMessage]) -> list[AgentMessage]:
    """Return replayable messages while retaining failures in durable history.

    Providers cannot consistently accept an assistant turn with no content. Tau
    persists terminal failures for diagnostics, but an empty failed or aborted
    turn is not model context and must not poison the next request.
    """
    replayable = tuple(
        message
        for message in messages
        if not (
            isinstance(message, AssistantMessage)
            and message.stop_reason in {"error", "aborted"}
            and not message.content
        )
    )
    return list(repair_tool_history(replayable).messages)


async def _assistant_events(
    *,
    provider: ModelProvider,
    model: str,
    system: str,
    messages: list[AgentMessage],
    tools: list[AgentTool],
    signal: CancellationToken | None,
    session_id: str | None,
) -> AsyncIterator[AgentEvent]:
    source: AsyncIterator[AssistantMessageEvent] = provider.stream_response(
        model=model,
        system=system,
        messages=messages,
        tools=tools,
        signal=signal,
        session_id=session_id,
    )
    started = False
    async for event in source:
        if isinstance(event, AssistantStartEvent):
            started = True
            yield MessageStartEvent(message=event.partial)
        elif isinstance(event, AssistantDoneEvent):
            if not started:
                yield MessageStartEvent(message=event.message)
            yield MessageEndEvent(message=event.message)
        elif isinstance(event, AssistantErrorEvent):
            if not started:
                yield MessageStartEvent(message=event.error)
            yield MessageEndEvent(message=event.error)
        else:
            yield MessageUpdateEvent(
                message=event.partial,
                assistant_message_event=event,
            )


async def _execute_tool_call(
    call: ToolCall,
    tools: Mapping[str, AgentTool],
    signal: CancellationToken | None,
    before_tool_call: BeforeToolCall | None,
    after_tool_call: AfterToolCall | None,
) -> AsyncIterator[AgentEvent]:
    yield ToolExecutionStartEvent(
        tool_call_id=call.id,
        tool_name=call.name,
        args=call.arguments,
    )

    blocked = False
    block_reason: str | None = None
    if before_tool_call is not None:
        blocked, block_reason = await before_tool_call(call)

    if blocked:
        result = _error_result(block_reason or "Tool execution was blocked")
        is_error = True
    elif signal is not None and signal.is_cancelled():
        result = _error_result("Operation aborted")
        is_error = True
    else:
        tool = tools.get(call.name)
        if tool is None:
            result = _error_result(f"Tool {call.name} not found")
            is_error = True
        else:
            result, is_error, updates = await _run_tool(tool, call, signal)
            for update in updates:
                yield ToolExecutionUpdateEvent(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    args=call.arguments,
                    partial_result=update,
                )

    if after_tool_call is not None:
        result, is_error = await after_tool_call(call, result, is_error)

    yield ToolExecutionEndEvent(
        tool_call_id=call.id,
        tool_name=call.name,
        result=result,
        is_error=is_error,
    )
    message = ToolResultMessage(
        tool_call_id=call.id,
        tool_name=call.name,
        content=result.content,
        details=result.details,
        added_tool_names=result.added_tool_names,
        is_error=is_error,
    )
    yield MessageStartEvent(message=message)
    yield MessageEndEvent(message=message)


async def _run_tool(
    tool: AgentTool,
    call: ToolCall,
    signal: CancellationToken | None,
) -> tuple[AgentToolResult, bool, list[AgentToolResult]]:
    updates: list[AgentToolResult] = []
    accepting = True

    def on_update(partial: AgentToolResult) -> None:
        if accepting:
            updates.append(partial.model_copy(deep=True))

    try:
        result = await tool.execute(call.id, call.arguments, signal, on_update)
        return result, False, updates
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - tools are an isolation boundary
        return _error_result(str(exc)), True, updates
    finally:
        accepting = False


def _error_result(message: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=message)], details={})


def _error_message(model: str, message: str) -> AssistantMessage:
    return AssistantMessage(
        model=model,
        content=[],
        stop_reason="error",
        error_message=message,
    )


# ---------------------------------------------------------------------------
# Lean compat facade over the canonical engine (agent/prover_loop.py).
#
# Two concerns live here: the async coding loop above, and this thin facade
# re-exporting the proof engine so existing `from .loop import ...` call sites
# keep working. The submodules are reachable as `loop.lean/llm/lsp` so tests
# can monkeypatch them. `prove` only syncs a redirected `loop.LEAN_DIR` (bench
# isolation + hermetic tests) before delegating; `prove_best_of` loops through
# this module's `prove` so a ``loop.prove`` monkeypatch stays honoured.
from .prover_loop import (  # noqa: F401
    HAMMERS,
    HEADER,
    LEAN_DIR,
    SYSTEM,
    SYSTEM_FULL,
    Result,
    _extract_body,
    _extract_full_file,
    _proof_body,
    _split_signature,
    lean,
    llm,
    lsp,
)


def prove(*args, **kwargs):
    """Thin facade over :func:`prover_loop.prove`.

    The engine resolves its lean dir per-call from ``PROVER_LEAN_DIR``
    (``_resolve_lean_dir``), so no module-global sync is needed here — a
    mutated ``LEAN_DIR`` would only introduce a data race under --parallel.
    """
    import agent.prover_loop as _pl

    return _pl.prove(*args, **kwargs)


def prove_best_of(
    statement: str,
    n_attempts: int = 1,
    max_steps: int = 20,
    verbose: bool = True,
    problem_id: str | None = None,
    goal_feedback: bool = True,
    on_event=None,
    should_stop=None,
    record_session: bool = True,
    skip_hammers: bool = False,
    temperature: float = 0.2,
    temperature_delta: float = 0.4,
    difficulty: str | None = None,
    model_name: str | None = None,
    lemma_plan: bool | None = None,
    full_file: bool = False,
    adaptive_steps: bool = False,
):
    """Best-of-N facade over :func:`prove`, mirroring prover_loop.prove_best_of.

    Delegates to the shared :func:`prover_loop._best_of_attempts` driver with
    this module's ``prove``, so a ``loop.prove`` monkeypatch (hermetic tests)
    is honoured without duplicating the temperature-ramp body.
    """
    from .prover_loop import _best_of_attempts

    return _best_of_attempts(
        prove,
        statement,
        n_attempts=n_attempts,
        max_steps=max_steps,
        verbose=verbose,
        problem_id=problem_id,
        goal_feedback=goal_feedback,
        on_event=on_event,
        should_stop=should_stop,
        record_session=record_session,
        skip_hammers=skip_hammers,
        temperature=temperature,
        temperature_delta=temperature_delta,
        difficulty=difficulty,
        model_name=model_name,
        lemma_plan=lemma_plan,
        full_file=full_file,
        adaptive_steps=adaptive_steps,
    )
