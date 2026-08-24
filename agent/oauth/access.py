"""OAuth consumption layer — turns stored credentials into usable tokens.

This is the seam between ``/login`` (which persists credentials) and the
model clients (``llm.client``, ``providers.openai_compatible``), which call
:func:`token_for_model` / :func:`api_key_for_model` when no explicit API key
is configured.

Model → OAuth-provider resolution follows the same family hints as
``models._MODEL_PROVIDER_HINTS``:

- ``claude*``  → ``anthropic``
- ``gpt*``/``o*``/``codex*``/``chatgpt*`` → ``openai-codex``
- ``*copilot*`` → ``github-copilot``

Precedence everywhere: an explicitly configured API key (profile or env)
always wins over an OAuth token — OAuth is a fallback, never a hijack.
"""

from __future__ import annotations

import time

from ..credentials import FileCredentialStore, OAuthCredential
from .anthropic import ANTHROPIC_CLIENT_ID, refresh_anthropic
from .codex import OPENAI_CODEX_CLIENT_ID, refresh_codex
from .github_copilot import (
    GITHUB_CLIENT_ID,
    _copilot_exchange,
    refresh_github_copilot,
)

# Refresh margin: re-up before expiry so in-flight calls don't race the clock.
_EXPIRY_MARGIN_S = 300.0

_MODEL_OAUTH_HINTS: tuple[tuple[str, str], ...] = (
    ("claude", "anthropic"),
    ("copilot", "github-copilot"),
    ("gpt", "openai-codex"),
    ("codex", "openai-codex"),
    ("chatgpt", "openai-codex"),
    ("o1", "openai-codex"),
    ("o3", "openai-codex"),
    ("o4", "openai-codex"),
)

_REFRESHERS = {
    "anthropic": (refresh_anthropic, ANTHROPIC_CLIENT_ID),
    "github-copilot": (refresh_github_copilot, GITHUB_CLIENT_ID),
    "openai-codex": (refresh_codex, OPENAI_CODEX_CLIENT_ID),
}


def oauth_provider_for_model(model_name: str) -> str | None:
    """The OAuth provider whose token family matches this model, or None."""
    lowered = (model_name or "").lower()
    for prefix, provider in _MODEL_OAUTH_HINTS:
        if lowered.startswith(prefix):
            return provider
    return None


def _is_expired(credential: OAuthCredential, now: float) -> bool:
    return (credential.expires_at is not None
            and credential.expires_at - _EXPIRY_MARGIN_S <= now)


def _persist_refreshed(store: FileCredentialStore,
                       old: OAuthCredential, token: dict) -> OAuthCredential:
    fresh = OAuthCredential(
        provider=old.provider,
        access_token=str(token["access_token"]),
        refresh_token=token.get("refresh_token") or old.refresh_token,
        expires_at=(time.time() + float(token["expires_in"]))
        if token.get("expires_in") else old.expires_at,
        issued_at=time.time(),
        extras=old.extras,
    )
    store.set(fresh)
    return fresh


def refresh_credential(credential: OAuthCredential,
                       store: FileCredentialStore | None = None,
                       ) -> OAuthCredential | None:
    """Refresh a stored credential via its provider's refresh endpoint.

    Returns the persisted refreshed credential, or None when the provider
    has no refresher, no refresh token exists, or the endpoint refuses.
    """
    entry = _REFRESHERS.get(credential.provider)
    if entry is None or not credential.refresh_token:
        return None
    refresher, client_id = entry
    try:
        token = refresher(credential.refresh_token, client_id=client_id)  # type: ignore[call-arg]
    except TypeError:
        token = refresher(credential.refresh_token)  # type: ignore[call-arg]
    if not token or not token.get("access_token"):
        return None
    return _persist_refreshed(store or FileCredentialStore(), credential, token)


def get_valid_credential(provider: str,
                         store: FileCredentialStore | None = None,
                         ) -> OAuthCredential | None:
    """Stored credential for `provider`, refreshed if within expiry margin."""
    if not provider:
        return None
    st = store or FileCredentialStore()
    cred = st.get(provider)
    if cred is None or not cred.access_token:
        return None
    if _is_expired(cred, time.time()):
        cred = refresh_credential(cred, st) or cred
    return cred


def token_for_model(model_name: str,
                    store: FileCredentialStore | None = None,
                    ) -> str | None:
    """Bearer token for `model_name`'s OAuth provider, or None.

    Refreshes expired credentials transparently. Never raises: auth is
    best-effort and callers fall back to API keys.
    """
    provider = oauth_provider_for_model(model_name)
    if provider is None:
        return None
    try:
        cred = get_valid_credential(provider, store)
    except Exception:  # noqa: BLE001 — network/IO failures must not break auth
        return None
    return cred.access_token if cred else None


def api_key_for_model(model_name: str,
                      store: FileCredentialStore | None = None,
                      ) -> str | None:
    """Inference key for `model_name`, exchanging where the API requires it.

    GitHub Copilot needs its short-lived ``copilot_internal/v2`` API key
    rather than the GitHub OAuth token; other providers use the token as-is.
    """
    provider = oauth_provider_for_model(model_name)
    if provider != "github-copilot":
        return token_for_model(model_name, store)
    try:
        st = store or FileCredentialStore()
        cred = get_valid_credential(provider, st)
        if cred is None:
            return None
        payload = _copilot_exchange(cred.access_token)
        if payload is None and cred.refresh_token:
            fresh = refresh_credential(cred, st)
            if fresh is not None:
                payload = _copilot_exchange(fresh.access_token)
        if payload and payload.get("token"):
            return str(payload["token"])
    except Exception:  # noqa: BLE001
        return None
    return None
