"""Phase 14 tests — coding session substrate (Tau session/tools/config port).

Covers CodingSession.load ordering, provider/toola round-trips through
run_agent_loop, coding-tool edit/truncation semantics, the session entry
JSONL tree, and provider-config persistence + legacy models.json migration.
"""

from __future__ import annotations

import json

import pytest

from agent import coding_tools
from agent.coding_session import CodingSession, CodingSessionConfig
from agent.messages import AssistantMessage, ToolCall
from agent.provider_events import AssistantDoneEvent, AssistantStartEvent
from agent.providers.fake import FakeProvider
from agent.session.entries import SessionEntry
from agent.session.jsonl import entries_from_json_lines, entry_to_json_line

# ------------------------------------------------------------ CodingSession


class ScriptedProvider:
    """Provider that emits a scripted assistant turn followed by a stop turn."""

    def __init__(self, tool_call: ToolCall | None, final_text: str) -> None:
        self.tool_call = tool_call
        self.final_text = final_text
        self.tool_responses_seen: list[str] = []

    async def stream_response(self, *, model, system, messages, tools, signal=None, session_id=None):
        for message in messages:
            if getattr(message, "role", "") == "toolResult":
                self.tool_responses_seen.append(message.text)
        if self.tool_call is not None:
            call = self.tool_call
            self.tool_call = None
            msg = AssistantMessage(model=model, content=[call], stop_reason="toolUse")
            yield AssistantStartEvent(partial=msg)
            yield AssistantDoneEvent(reason="toolUse", message=msg)
            return
        msg = AssistantMessage(model=model, content=[], stop_reason="stop")
        yield AssistantStartEvent(partial=msg)
        yield AssistantDoneEvent(reason="stop", message=msg)


@pytest.fixture
def hermetic(tmp_path, monkeypatch):
    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("PROVER_NO_SESSIONS", "1")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _load(tmp_path, provider):
    cfg = CodingSessionConfig(provider=provider, model="fake", cwd=tmp_path)
    return CodingSession.load(cfg)


@pytest.mark.anyio
async def test_load_composes_tools_and_system_prompt(hermetic) -> None:
    session = await _load(hermetic, FakeProvider("hi"))
    names = [d["name"] for d in (session.config.resolved_tools or [])]
    assert {"read", "edit", "write", "bash"} <= set(names)
    assert session.config.system  # built, not empty
    assert session.session_id


@pytest.mark.anyio
async def test_load_binds_lean_tools_when_lean_dir_exists(hermetic) -> None:
    (hermetic / "lean").mkdir()
    session = await _load(hermetic, FakeProvider("hi"))
    names = [d["name"] for d in (session.config.resolved_tools or [])]
    assert {"lean_check", "lsp_goals", "retrieval_search"} <= set(names)


@pytest.mark.anyio
async def test_prompt_round_trip_with_tool_call(hermetic) -> None:
    (hermetic / "note.txt").write_text("lean says hi\n")
    call = ToolCall(id="c1", name="read", arguments={"path": "note.txt"})
    provider = ScriptedProvider(tool_call=call, final_text="done")
    session = await _load(hermetic, provider)
    events = [e async for e in session.prompt("read the note")]
    names = [type(e).__name__ for e in events]
    assert names[0] == "AgentStartEvent"
    assert names[-1] == "AgentEndEvent"
    assert "ToolExecutionEndEvent" in names
    assert provider.tool_responses_seen and "lean says hi" in provider.tool_responses_seen[0]
    last = session.harness.messages[-1]
    assert isinstance(last, AssistantMessage)


@pytest.mark.anyio
async def test_tool_error_propagates_is_error(hermetic) -> None:
    call = ToolCall(id="c1", name="read", arguments={"path": "missing.txt"})
    provider = ScriptedProvider(tool_call=call, final_text="done")
    session = await _load(hermetic, provider)
    [e async for e in session.prompt("read missing")]
    tool_msg = next(
        m for m in session.harness.messages if getattr(m, "role", "") == "toolResult"
    )
    assert tool_msg.is_error
    assert "read failed" in tool_msg.text


@pytest.mark.anyio
async def test_set_model_and_custom_entries(hermetic) -> None:
    session = await _load(hermetic, FakeProvider("hi"))
    session.set_model("other-model", provider="openrouter")
    assert session.model_choice.to_dict() == {"model": "other-model", "provider": "openrouter"}
    session.append_custom_entry("label", {"name": "checkpoint"})
    assert session.drain_pending_entries() == [{"type": "custom", "kind": "label", "payload": {"name": "checkpoint"}}]
    assert session.drain_pending_entries() == []


# ------------------------------------------------------------ coding tools


def test_detect_line_ending() -> None:
    assert coding_tools.detect_line_ending("a\r\nb\r\nc") == "\r\n"
    assert coding_tools.detect_line_ending("a\rb\rc") == "\r"
    assert coding_tools.detect_line_ending("a\nb\n") == "\n"
    assert coding_tools.detect_line_ending("") == "\n"


def test_truncate_for_read_head_and_tail() -> None:
    text = "\n".join(f"line {i}" for i in range(3000)) + "\n"
    truncated, flag = coding_tools.truncate_for_read(text)
    assert flag
    assert "line 0" in truncated
    assert "line 2999" in truncated
    assert "more lines truncated" in truncated


def test_apply_edits_not_found() -> None:
    _, _, error = coding_tools.apply_edits_to_normalized_content(
        "abc\n", [{"old_string": "xyz", "new_string": "w"}]
    )
    assert error and "not found" in error


def test_apply_edits_duplicate_exact_match() -> None:
    _, _, error = coding_tools.apply_edits_to_normalized_content(
        "aaa\n", [{"old_string": "a", "new_string": "b"}]
    )
    assert error and "multiple times" in error


def test_apply_edits_overlapping_ranges() -> None:
    _, _, error = coding_tools.apply_edits_to_normalized_content(
        "abcdef\n",
        [
            {"old_string": "abc", "new_string": "x"},
            {"old_string": "cde", "new_string": "y"},
        ],
    )
    assert error and "overlaps" in error


def test_apply_edits_batch_and_patch() -> None:
    new_content, patch, error = coding_tools.apply_edits_to_normalized_content(
        "alpha\nbeta\n",
        [
            {"old_string": "alpha", "new_string": "ALPHA"},
            {"old_string": "beta", "new_string": "BETA"},
        ],
        path="t.txt",
    )
    assert error is None
    assert new_content == "ALPHA\nBETA\n"
    assert "-alpha" in patch and "+ALPHA" in patch


def test_apply_edits_crlf_round_trip(tmp_path) -> None:
    new_content, _, error = coding_tools.apply_edits_to_normalized_content(
        "one\r\ntwo\r\n",
        [{"old_string": "one", "new_string": "ONE"}],
        line_ending="\r\n",
    )
    assert error is None
    assert new_content == "ONE\r\ntwo\r\n"


def test_format_size() -> None:
    assert coding_tools.format_size(10) == "10 B"
    assert coding_tools.format_size(2048).endswith("KB")
    assert coding_tools.format_size(5 * 1024 * 1024).endswith("MB")


@pytest.mark.anyio
async def test_create_coding_tools_write_edit_round_trip(tmp_path) -> None:
    tools = {t["name"]: t["execute"] for t in coding_tools.create_coding_tools(tmp_path)}
    result = await tools["write"]({"path": "a/b.txt", "content": "first\n"})
    assert tmp_path.joinpath("a/b.txt").read_text() == "first\n"
    result = await tools["edit"](
        {"path": "a/b.txt", "old_string": "first", "new_string": "second"}
    )
    assert tmp_path.joinpath("a/b.txt").read_text() == "second\n"
    assert result.get("content")
    read = await tools["read"]({"path": "a/b.txt"})
    assert read.get("content") == "second\n"


# ------------------------------------------------------------ session entries


def test_entry_jsonl_round_trip() -> None:
    entries = [
        SessionEntry(entry_id="1", type="message", role="user", content="hi"),
        SessionEntry(entry_id="2", type="message", role="assistant", content="yo"),
        SessionEntry(entry_id="3", type="custom", kind="checkpoint", payload={"k": "v"}),
        SessionEntry(entry_id="4", type="leaf", message_entry_id="2"),
    ]
    lines = [entry_to_json_line(e) for e in entries]
    parsed = entries_from_json_lines(lines)
    assert len(parsed) == 4
    assert parsed[0].content == "hi"
    assert parsed[2].payload == {"k": "v"}
    assert parsed[3].message_entry_id == "2"


def test_entries_from_json_lines_skips_corrupt() -> None:
    parsed = entries_from_json_lines(["{}", "not json", '{"id": "1", "type": "label", "label": "x"}'])
    assert [e.entry_id for e in parsed] == ["", "1"]


# ---------------------------------------------------------- provider config


def test_provider_config_save_load_and_backup(tmp_path, monkeypatch) -> None:
    from agent import provider_config as pc

    monkeypatch.setenv("PROVER_PROVIDERS_PATH", str(tmp_path / "providers.json"))
    configs = [pc.ProviderConfig(name="local", base_url="http://x/v1", env_key="K")]
    assert pc.save_provider_settings(configs) is not None
    # Tau backs up the pre-existing file on overwrite (second write).
    assert pc.save_provider_settings(configs) is not None
    loaded = pc.load_provider_settings()
    assert [c.name for c in loaded] == ["local"]
    assert (tmp_path / "providers.json.bak").exists()


def test_provider_config_backup_is_owner_only(tmp_path, monkeypatch) -> None:
    """The .bak holds api_key material, so copyfile's default perms (0644)
    must be tightened to 0600."""
    import os

    from agent import provider_config as pc

    monkeypatch.setenv("PROVER_PROVIDERS_PATH", str(tmp_path / "providers.json"))
    configs = [pc.ProviderConfig(name="local", base_url="http://x/v1",
                                 env_key="K", api_key="sk-secret")]
    assert pc.save_provider_settings(configs) is not None
    assert pc.save_provider_settings(configs) is not None
    mode = os.stat(tmp_path / "providers.json.bak").st_mode & 0o777
    assert mode == 0o600, f"backup is world-readable: {oct(mode)}"


def test_provider_config_legacy_models_migration(tmp_path, monkeypatch) -> None:
    from agent import provider_config as pc

    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("PROVER_PROVIDERS_PATH", raising=False)
    (tmp_path / "models.json").write_text(json.dumps({
        "active": "m1",
        "profiles": [
            {"name": "m1", "base_url": "http://y/v1", "api_key": "sk", "context_window": 9},
            {"name": "m2", "base_url": ""},
        ],
    }))
    configs = pc.load_provider_settings()
    assert [c.name for c in configs] == ["m1", "m2"]
    assert configs[0].api_key == "sk"
    assert configs[0].max_context_window == 9


# ------------------------------------------------------- provider kind routing


def test_provider_from_config_routes_by_kind(monkeypatch) -> None:
    """ProviderConfig.kind must select a real per-backend adapter, not always
    OpenAICompatibleProvider."""
    from agent import coding_session
    from agent.provider_config import ProviderConfig
    from agent.providers.anthropic import AnthropicProvider
    from agent.providers.openai_compatible import OpenAICompatibleProvider

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    anthropic = coding_session._provider_from_config(
        ProviderConfig(name="claude", kind="anthropic",
                       base_url="https://api.anthropic.com"))
    assert isinstance(anthropic, AnthropicProvider)
    assert anthropic.api_key is None  # resolved lazily at request time

    oai = coding_session._provider_from_config(
        ProviderConfig(name="local", kind="openai_compatible",
                       base_url="http://localhost/v1"))
    assert isinstance(oai, OpenAICompatibleProvider)

    codex = coding_session._provider_from_config(
        ProviderConfig(name="codex", kind="openai_codex"))
    assert isinstance(codex, OpenAICompatibleProvider)
    assert codex.base_url == "https://api.openai.com/v1"

    none_provider = coding_session._provider_from_config(None)
    assert isinstance(none_provider, OpenAICompatibleProvider)


def test_provider_from_config_forwards_compat_and_headers() -> None:
    from agent import coding_session
    from agent.provider_config import OpenAICompatibleConfig, ProviderConfig
    from agent.providers.openai_compatible import OpenAICompatibleProvider

    p = coding_session._provider_from_config(ProviderConfig(
        name="qwen-local",
        kind="openai_compatible",
        base_url="http://localhost/v1",
        headers={"X-Org": "acme"},
        compat=OpenAICompatibleConfig(thinking_format="qwen",
                                      reasoning_effort=True),
    ))
    assert isinstance(p, OpenAICompatibleProvider)
    assert p._compat.thinking_format == "qwen"
    assert p._headers == {"X-Org": "acme"}


# ------------------------------------------------------------- harness bits


@pytest.mark.anyio
async def test_harness_queues_via_run(hermetic) -> None:
    session = await _load(hermetic, FakeProvider("hi"))
    pending = session.harness.steer("steer-1")
    assert pending.count == 1
    [e async for e in session.prompt("go")]
    assert session.harness.is_running is False
    assert session.harness.pending_message_count == 0
