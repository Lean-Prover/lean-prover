"""Durable provider configuration — Tau provider_config.py port (Tau 37a9e43 src/tau_coding/provider_config.py), lean-adapted.

A ``ProviderConfig`` describes how to reach one inference provider (endpoint,
auth, compat flags).  Configs persist to ``providers.json`` under
``ProverPaths.config_dir`` with atomic write + ``.bak`` backup.  Legacy
``~/.prover/models.json`` (agent/models.py) remains readable for backward
compat: its profiles are folded in as an ``openai-compatible`` provider.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .paths import ProverPaths

ProviderKind = Literal["openai_compatible", "anthropic", "openai_codex"]

DEFAULT_PROVIDERS_PATH = "providers.json"
_LEGACY_MODELS_FILE = "models.json"


@dataclass(frozen=True, slots=True)
class OpenAICompatibleConfig:
    """Compat flags for OpenAI-style chat endpoints (tau OpenAICompatibleConfig)."""

    supports_images: bool = False
    supports_prompt_cache_key: bool = False
    reasoning_effort: bool = False
    thinking_format: str | None = None  # "openai" | "qwen" | "deepseek" | None
    response_provider_header: bool = False
    zai_tool_stream: bool = False
    content_response_format: bool = False
    provider_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenAICompatibleConfig:
        d = data or {}
        return cls(
            supports_images=bool(d.get("supports_images", False)),
            supports_prompt_cache_key=bool(d.get("supports_prompt_cache_key", False)),
            reasoning_effort=bool(d.get("reasoning_effort", False)),
            thinking_format=d.get("thinking_format"),
            response_provider_header=bool(d.get("response_provider_header", False)),
            zai_tool_stream=bool(d.get("zai_tool_stream", False)),
            content_response_format=bool(d.get("content_response_format", False)),
            provider_key=d.get("provider_key"),
        )


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """One named provider endpoint (tau ProviderConfig)."""

    name: str
    kind: ProviderKind = "openai_compatible"
    base_url: str = ""
    api_key: str | None = None
    env_key: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    compat: OpenAICompatibleConfig = field(default_factory=OpenAICompatibleConfig)
    thinking_default: str | None = None
    max_context_window: int | None = None

    @property
    def effective_api_key(self) -> str | None:
        """Resolve the API key: explicit > env var > env_key-configured var."""
        if self.api_key:
            return self.api_key
        if self.env_key and os.environ.get(self.env_key):
            return os.environ.get(self.env_key)
        if not self.env_key and self.base_url:
            # OpenAI-compatible defaults: OPENAI_API_KEY for direct OpenAI.
            for candidate in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
                if candidate == "OPENAI_API_KEY" and os.environ.get(candidate):
                    return os.environ.get(candidate)
                if candidate == "OPENAI_BASE_URL" and self.base_url and "openai" in self.base_url:
                    return os.environ.get("OPENAI_API_KEY")
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "env_key": self.env_key,
            "headers": self.headers,
            "compat": self.compat.to_dict(),
            "thinking_default": self.thinking_default,
            "max_context_window": self.max_context_window,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProviderConfig:
        d = data or {}
        compat = d.get("compat") or {}
        return cls(
            name=str(d.get("name") or ""),
            kind=str(d.get("kind") or "openai_compatible"),  # type: ignore[assignment]
            base_url=str(d.get("base_url") or ""),
            api_key=d.get("api_key"),
            env_key=str(d.get("env_key") or ""),
            headers=dict(d.get("headers") or {}),
            compat=OpenAICompatibleConfig.from_dict(compat if isinstance(compat, dict) else {}),
            thinking_default=d.get("thinking_default"),
            max_context_window=d.get("max_context_window"),
        )


def provider_settings_path() -> Path:
    """Where provider configs persist (``~/.prover/providers.json``)."""
    override = os.environ.get("PROVER_PROVIDERS_PATH")
    if override:
        return Path(override)
    return ProverPaths().config_dir / DEFAULT_PROVIDERS_PATH


def provider_settings_from_json(data: dict[str, Any]) -> list[ProviderConfig]:
    """Parse a provider-settings file body (tolerates legacy ``models.json``)."""
    if isinstance(data, dict) and isinstance(data.get("providers"), list):
        providers = []
        for entry in data["providers"]:
            if isinstance(entry, dict):
                try:
                    providers.append(ProviderConfig.from_dict(entry))
                except (TypeError, ValueError):
                    continue
        return providers
    # Legacy models.json shape: {"active": ..., "profiles": [...]}
    profiles = data.get("profiles") if isinstance(data, dict) else []
    providers: list[ProviderConfig] = []
    if isinstance(profiles, list):
        for profile in profiles:
            if not isinstance(profile, dict) or not profile.get("name"):
                continue
            providers.append(
                ProviderConfig(
                    name=str(profile["name"]),
                    kind="openai_compatible",
                    base_url=str(profile.get("base_url") or ""),
                    api_key=profile.get("api_key") or None,
                    max_context_window=profile.get("context_window"),
                )
            )
    return providers


def load_provider_settings() -> list[ProviderConfig]:
    """Load persisted provider configs (falls back to legacy models.json)."""
    path = provider_settings_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return provider_settings_from_json(data)
        except (json.JSONDecodeError, OSError):
            pass
    legacy = ProverPaths().config_dir / _LEGACY_MODELS_FILE
    if legacy.exists():
        try:
            return provider_settings_from_json(json.loads(legacy.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    return []


def save_provider_settings(providers: list[ProviderConfig]) -> Path | None:
    """Persist provider configs atomically with a ``.bak`` backup (tau parity)."""
    path = provider_settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    body = json.dumps({"providers": [p.to_dict() for p in providers]}, indent=2)
    try:
        if path.exists():
            bak = path.with_suffix(".json.bak")
            shutil.copyfile(path, bak)
            # copyfile does not carry permissions; the backup holds api_key.
            try:
                os.chmod(bak, 0o600)
            except OSError:
                bak.unlink(missing_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".providers-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                file.write(body)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
    except OSError:
        return None
    return path


def _default_provider_configs() -> list[ProviderConfig]:
    """Built-in provider defaults (lean-adapted: qwen-local first)."""
    return [
        ProviderConfig(
            name="qwen-local",
            kind="openai_compatible",
            base_url=os.environ.get("OPENAI_BASE_URL", ""),
            env_key="OPENAI_API_KEY",
            compat=OpenAICompatibleConfig(
                thinking_format="qwen", reasoning_effort=True
            ),
            thinking_default="off",
        ),
        ProviderConfig(
            name="openai",
            kind="openai_compatible",
            base_url="https://api.openai.com/v1",
            env_key="OPENAI_API_KEY",
            compat=OpenAICompatibleConfig(
                supports_images=True,
                supports_prompt_cache_key=True,
                reasoning_effort=True,
                thinking_format="openai",
                response_provider_header=True,
            ),
            thinking_default="off",
        ),
        ProviderConfig(
            name="anthropic",
            kind="anthropic",
            base_url="https://api.anthropic.com",
            env_key="ANTHROPIC_API_KEY",
            thinking_default="low",
        ),
        ProviderConfig(
            name="openrouter",
            kind="openai_compatible",
            base_url="https://openrouter.ai/api/v1",
            env_key="OPENROUTER_API_KEY",
            compat=OpenAICompatibleConfig(
                supports_images=True, reasoning_effort=True
            ),
            thinking_default="off",
        ),
    ]


def effective_provider_configs() -> list[ProviderConfig]:
    """Persisted configs, falling back to built-in defaults when empty."""
    persisted = load_provider_settings()
    if persisted:
        return persisted
    return _default_provider_configs()


def provider_config_from_name(name: str) -> ProviderConfig | None:
    """Look up a provider by name across persisted + built-in configs."""
    for provider in effective_provider_configs():
        if provider.name == name:
            return provider
    return None
