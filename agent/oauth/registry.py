"""OAuth provider registry — Tau oauth_registry.py port (Tau 37a9e43 src/tau_coding/oauth_registry.py), lean-adapted.

Maintains a global registry of named ``OAuthProvider`` implementations so
commands like ``/login <provider>`` can dispatch dynamically. The three
built-ins register at import time.
"""

from __future__ import annotations

from ..credentials import OAuthCredential
from .types import OAuthProvider

_REGISTRY: dict[str, OAuthProvider] = {}


def get_oauth_provider(name: str) -> OAuthProvider | None:
    return _REGISTRY.get(name)


class _FunctionProvider:
    """Adapts module-level login functions to the OAuthProvider protocol."""

    def __init__(self, name: str, login) -> None:
        self.name = name
        self._login = login

    async def login(self, prompt_cb) -> OAuthCredential | None:
        return await self._login(prompt_cb)

    async def refresh(self, credential: OAuthCredential) -> OAuthCredential | None:
        from .access import refresh_credential

        return refresh_credential(credential)


def _register_builtins() -> None:
    from .anthropic import login_anthropic
    from .codex import login_openai_codex
    from .github_copilot import login_github_copilot

    _REGISTRY.update({
        "anthropic": _FunctionProvider("anthropic", login_anthropic),
        "github-copilot": _FunctionProvider("github-copilot", login_github_copilot),
        "openai-codex": _FunctionProvider("openai-codex", login_openai_codex),
    })


_register_builtins()
