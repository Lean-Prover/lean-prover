"""llm._call thinking-level → provider kwargs tests (tau thinking wiring port)."""

from __future__ import annotations

from typing import ClassVar

import pytest

from agent import llm


class _FakeMessage:
    content = "ok"
    reasoning = None


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices: ClassVar = [_FakeChoice()]
    usage = None


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeResponse()


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


@pytest.fixture
def fake_client(monkeypatch):
    client = _FakeClient()

    def _client(model_name=None):
        return client

    monkeypatch.setattr(llm, "client", _client)
    return client


def test_thinking_off_with_base_url_sends_disable_switch(monkeypatch, fake_client) -> None:
    monkeypatch.delenv("PROVER_THINKING", raising=False)
    monkeypatch.setenv("PROVER_DISABLE_THINKING", "1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1/v1")
    llm._call("sys", [], 0.2)
    kwargs = fake_client.chat.completions.kwargs
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "reasoning_effort" not in kwargs


def test_thinking_off_without_base_url_sends_nothing(monkeypatch, fake_client) -> None:
    monkeypatch.delenv("PROVER_THINKING", raising=False)
    monkeypatch.setenv("PROVER_DISABLE_THINKING", "1")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    llm._call("sys", [], 0.2)
    kwargs = fake_client.chat.completions.kwargs
    assert "extra_body" not in kwargs
    assert "reasoning_effort" not in kwargs


def test_explicit_thinking_level_sends_reasoning_effort(monkeypatch, fake_client) -> None:
    monkeypatch.setenv("PROVER_THINKING", "high")
    monkeypatch.delenv("PROVER_DISABLE_THINKING", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1/v1")
    llm._call("sys", [], 0.2)
    kwargs = fake_client.chat.completions.kwargs
    assert kwargs["reasoning_effort"] == "high"
    assert "extra_body" not in kwargs


class _FakeModel:
    def __init__(self, mid: str) -> None:
        self.id = mid


class _FakeModelsResponse:
    def __init__(self, ids: list[str]) -> None:
        self.data = [_FakeModel(i) for i in ids]


class _FakeModelsRoute:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids
        self.timeout = None

    def list(self, timeout: float = 15.0):
        self.timeout = timeout
        return _FakeModelsResponse(self._ids)


class _FakeModelClient:
    def __init__(self, ids: list[str]) -> None:
        self.models = _FakeModelsRoute(ids)


def test_available_models_lists_served_ids(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1/v1")
    client = _FakeModelClient(["Qwen/Qwen3.8-27B"])
    monkeypatch.setattr(llm, "client", lambda model_name=None: client)
    assert llm.available_models() == ["Qwen/Qwen3.8-27B"]


def test_available_models_empty_when_endpoint_unreachable(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1/v1")

    def _boom():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(llm, "client", _boom)
    assert llm.available_models() == []


def test_validate_model_flags_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("PROVER_MODEL", "gpt-4o")
    monkeypatch.setattr(llm, "client", lambda: _FakeModelClient(["Qwen/Qwen3.8-27B"]))
    hint = llm.validate_model()
    assert hint is not None
    assert "gpt-4o" in hint and "Qwen/Qwen3.8-27B" in hint


def test_validate_model_accepts_served_name(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("PROVER_MODEL", "Qwen/Qwen3.8-27B")
    monkeypatch.setattr(llm, "client", lambda: _FakeModelClient(["Qwen/Qwen3.8-27B"]))
    assert llm.validate_model() is None


def test_validate_model_silent_without_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert llm.validate_model() is None


def test_qwen_context_window() -> None:
    assert llm.context_window_tokens("Qwen/Qwen3.8-27B") == 262_144
