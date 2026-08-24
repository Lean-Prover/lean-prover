"""Coding session substrate — Tau tau_coding/session.py port (Tau 37a9e43 src/tau_coding/session.py), lean-adapted.

``CodingSession.load(config)`` replicates Tau's load ordering:

    read_all(entries) -> repair (detach missing parents) -> trust ->
    discover resources (skills/templates/context) -> compose tools ->
    build system prompt -> AgentHarness(...)

Lean adappts: durable prove-loop sessions stay in the flat JSONL stream
(``agent/session.py``); this class powers the generic Tau-style agent path
and the ``loop.run_agent_loop`` harness used by ``AgentHarness``.

``prove()`` is NOT replaced — the legacy repair loop stays authoritative for
theorem proving (``agent/prover_loop.py``); this facade is additive.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .context_window import ContextUsageEstimate, estimate_context_usage
from .harness import AgentHarness, AgentHarnessConfig
from .provider import ModelProvider
from .provider_config import ProviderConfig, provider_config_from_name
from .resources import TauResourcePaths, discover_resources


@dataclass(slots=True)
class CodingSessionConfig:
    provider: ModelProvider | None = None
    model: str = "gpt-4o"
    provider_name: str = ""
    system: str | None = None
    cwd: Path = field(default_factory=Path.cwd)
    session_id: str | None = None
    resource_paths: TauResourcePaths | None = None
    tool_definitions: list[dict] | None = None
    max_turns: int | None = None
    trust_override: str | None = None
    extension_paths: tuple[Path, ...] = ()
    custom_prompt: str | None = None
    thinking_level: str = "off"
    before_tool_call: Callable | None = None

    resolved_tools: list[dict] | None = None
    context_files: list[dict[str, str]] | None = None


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """Model + provider binding selected at session start (tau ModelChoice)."""

    model: str
    provider: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"model": self.model, "provider": self.provider}


class CodingSession:
    def __init__(self, config: CodingSessionConfig, harness: AgentHarness) -> None:
        self.config = config
        self.harness = harness
        self.session_id = config.session_id
        self._model_choice = ModelChoice(model=config.model, provider=config.provider_name)
        self._context_usage: ContextUsageEstimate | None = None
        self._pending_entries: list[Any] = []

    @classmethod
    async def load(cls, config: CodingSessionConfig) -> CodingSession:
        """Load-or-create a session, mirroring Tau session.py load ordering."""
        if config.session_id is None:
            config.session_id = uuid4().hex

        # 1. provider resolution (persisted configs > built-in defaults)
        provider_config = provider_config_from_name(config.provider_name)
        provider = config.provider or _provider_from_config(provider_config)

        # 2. resource discovery (skills/templates/context) — lean additions
        resources = config.resource_paths or discover_resources(config.cwd)
        config.resource_paths = resources
        from .context import discover_project_context_with_diagnostics
        from .skills import load_skills_with_diagnostics

        context_files, _ = discover_project_context_with_diagnostics(config.cwd)
        config.context_files = context_files
        skill_dirs: list[Path] = [d for d in resources.skills_dirs if d.exists()]
        skills, _ = load_skills_with_diagnostics(tuple(skill_dirs))

        # 3. tool composition (tau create_coding_tools + lean prover tools)
        from .coding_tools import create_coding_tools

        coding_tools = config.tool_definitions or create_coding_tools(config.cwd)
        lean_dir = config.cwd / "lean"
        lean_tools = _lean_tools(lean_dir) if lean_dir.exists() else []
        config.resolved_tools = coding_tools + [
            {"name": t.name, "description": t.description, "parameters": t.parameters, "execute": t.execute}
            for t in lean_tools
        ]

        # 4. system prompt build (custom > builtin)
        from .system_prompt import build_system_prompt

        system = config.system or build_system_prompt(
            tools=config.resolved_tools,
            skills=skills,
            custom_prompt=config.custom_prompt,
            context_files=context_files,
        )
        config.system = system

        # 5. harness binding (tau: AgentHarness + attach_harness_listener)
        harness = AgentHarness(
            AgentHarnessConfig(
                provider=provider,
                model=config.model,
                system=system,
                tools=_harness_tools(config.resolved_tools),
                max_turns=config.max_turns,
                session_id=config.session_id,
                before_tool_call=config.before_tool_call or _acl_hook(),
                after_tool_call=_after_tool_call,
            )
        )
        return cls(config, harness)

    @property
    def model_choice(self) -> ModelChoice:
        return self._model_choice

    def set_model(self, model: str, provider: str = "") -> None:
        self._model_choice = ModelChoice(model=model, provider=provider)
        self.harness._config.model = model

    def set_thinking_level(self, level: str) -> None:
        self.config.thinking_level = level

    @property
    def context_usage(self) -> ContextUsageEstimate | None:
        if self._context_usage is None:
            self._context_usage = estimate_context_usage(
                self.harness.config.system, list(self.harness.messages)
            )
        return self._context_usage

    def append_custom_entry(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        self._pending_entries.append({"type": "custom", "kind": kind, "payload": payload or {}})

    def drain_pending_entries(self) -> list[Any]:
        entries = self._pending_entries
        self._pending_entries = []
        return entries

    async def prompt(self, text: str):
        async for event in self.harness.prompt(text):
            yield event

    def __getattr__(self, name: str) -> Any:
        # Delegate unknown attribute access to the harness (tau facade parity).
        return getattr(self.harness, name)


async def _after_tool_call(call, result, is_error):
    """Propagate tool-reported errors into the loop's is_error flag."""
    if result is not None and getattr(result, "is_error", False):
        return result, True
    return result, is_error


def _acl_hook():
    """Resolve the coding loop's per-tool ACL gate (default before_tool_call)."""

def _provider_from_config(config: ProviderConfig | None):
    """Instantiate a real per-backend provider adapter for the config's kind.

    ``ProviderConfig.kind`` selects the native adapter (openai_compatible →
    OpenAI SDK; anthropic → native Messages API; openai_codex → OpenAI SDK
    against the Codex endpoint).  Compat flags/headers are forwarded so
    per-backend feature translation is live, not dead config.
    """
    if config is None:
        from .providers.openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider()
    if config.kind == "anthropic":
        from .providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            base_url=config.base_url or None,
            api_key=config.effective_api_key,
            headers=config.headers,
            provider_name=config.name,
        )
    from .providers.openai_compatible import OpenAICompatibleProvider

    base_url = config.base_url or None
    if config.kind == "openai_codex" and not base_url:
        base_url = "https://api.openai.com/v1"
    return OpenAICompatibleProvider(
        base_url=base_url,
        api_key=config.effective_api_key,
        compat=config.compat,
        headers=config.headers,
        provider_name=config.name,
    )


def _lean_tools(lean_dir: Path) -> list:
    from .tools import default_tools

    return default_tools(lean_dir)


class _HarnessTool:
    """Tool adapter matching run_agent_loop's contract.

    Wraps the legacy dict-first executors (async coding-tool dicts and sync
    lean AgentTool callbacks) into Tau's async ``execute(call_id, arguments,
    signal, on_update)`` shape returning a result object whose ``.content``
    is a ``[TextContent]`` list and which carries ``details`` +
    ``added_tool_names`` for ``ToolResultMessage``.
    """

    def __init__(self, name: str, description: str, parameters: dict, fn) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self._fn = fn
        self._is_async = asyncio.iscoroutinefunction(fn)

    async def execute(self, call_id: str, arguments: dict, signal, on_update=None):
        from .messages import TextContent
        from .tools import ToolResult

        if self._is_async:
            result = await self._fn(arguments)
        else:
            result = await asyncio.to_thread(self._fn, arguments)
        if isinstance(result, dict):
            content = result.get("content", result.get("output", ""))
            content = str(content) if content is not None else ""
            details = {k: v for k, v in result.items() if k not in ("content", "output", "is_error")}
            is_error = bool(result.get("is_error", False))
        else:
            content, details, is_error = str(result), {}, False
        return ToolResult(content=[TextContent(text=content)], details=details, is_error=is_error)  # type: ignore[arg-type]


def _harness_tools(definitions: list[dict] | None) -> list:
    tools: list[_HarnessTool] = []
    for d in definitions or []:
        execute = d.get("execute")
        if execute is None:
            continue
        tools.append(
            _HarnessTool(
                name=str(d.get("name", "")),
                description=str(d.get("description", "")),
                parameters=dict(d.get("parameters") or {}),
                fn=execute,
            )
        )
    return tools