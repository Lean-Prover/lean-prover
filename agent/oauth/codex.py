"""OpenAI Codex PKCE OAuth — Tau oauth.py port (Tau 37a9e43 src/tau_coding/oauth.py), lean-adapted.

Full AuthorizationCode + PKCE flow for ChatGPT Codex subscriptions:

1. generate code_verifier/code_challenge (S256)
2. open the authorization URL in the browser via a callback prompt
3. spin up a local redirect server capturing the ``?code=`` response
4. exchange code + verifier at the token endpoint
5. persist the credential to ``FileCredentialStore``

Portable: stdlib-only (http.server, urllib) so it works on headless CI when
a prompt callback is provided; the OpenAI SDK handles the token exchange
when its client is available.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import secrets
import time
import urllib.parse
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..credentials import FileCredentialStore, OAuthCredential
from .types import OAuthPrompt

OPENAI_CODEX_AUTH_BASE_URL = "https://chatgpt.com/backend-api/codex"
OPENAI_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_CODEX_REDIRECT_URI = "http://localhost:1455/auth/callback"
OPENAI_CODEX_SCOPE = "email openid"
LOCAL_PORT = 1455


@dataclass(frozen=True, slots=True)
class AuthorizationCode:
    """Result of the local redirect-server capture."""

    code: str
    state: str


def create_pkce_pair() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) for S256 PKCE."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def _local_platform_server(expected_state: str, timeout: float = 300.0) -> AuthorizationCode | None:
    """Spin up a 127.0.0.1 listener capturing the OAuth redirect (stdlib).

    Binds the fixed LOCAL_PORT because that is the redirect URI registered with
    the provider (changing it would break the token exchange); the handler is
    hardened to only accept the exact callback path with the expected state.
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer

    result: dict[str, str] = {}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            # Only the exact OAuth callback path is accepted; anything else is
            # not part of this flow and gets a 404 (not a captured code).
            if parsed.path != "/auth/callback":
                self.send_response(404)
                self.end_headers()
                return
            params = dict(urllib.parse.parse_qsl(parsed.query))
            # Validate state BEFORE recording any code, so a request without
            # the expected state can never satisfy the flow.
            if expected_state and params.get("state") != expected_state:
                self.send_response(400)
                self.end_headers()
                return
            result["code"] = params.get("code", "")
            result["state"] = params.get("state", "")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body>Sign-in complete. You may close this tab.</body></html>")

        def log_message(self, *args: Any) -> None:
            pass

    server = None
    try:
        server = HTTPServer(("127.0.0.1", LOCAL_PORT), _Handler)
    except OSError:
        # Port busy (another login, or the real Codex CLI): fail the flow
        # instead of crashing the caller.
        return None
    server.timeout = timeout
    deadline = time.time() + timeout
    while time.time() < deadline:
        server.handle_request()
        if result.get("code"):
            break
    server.server_close()
    if not result.get("code"):
        return None
    return AuthorizationCode(code=result["code"], state=result.get("state", ""))


def _exchange_code(code: str, verifier: str, client_id: str) -> dict[str, Any] | None:
    """POST the token exchange form (PKCE verifier attached)."""
    from .http import OAuthHTTPError, post_form

    try:
        return post_form(f"{OPENAI_CODEX_AUTH_BASE_URL}/oauth/token", {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": OPENAI_CODEX_REDIRECT_URI,
            "client_id": client_id,
            "code_verifier": verifier,
        })
    except OAuthHTTPError:
        return None


def refresh_codex(refresh_token: str,
                  client_id: str = OPENAI_CODEX_CLIENT_ID) -> dict | None:
    """Exchange a Codex refresh token for a fresh token payload."""
    from .http import OAuthHTTPError, post_form

    try:
        token = post_form(f"{OPENAI_CODEX_AUTH_BASE_URL}/oauth/token", {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": refresh_token,
        })
    except OAuthHTTPError:
        return None
    return token if token.get("access_token") else None


class AuthorizationFlow:
    """Drives one PKCE login round-trip (tau AuthorizationFlow)."""

    def __init__(self, callback: Callable[[OAuthPrompt], Any] | None = None) -> None:
        self.callback = callback
        self._loop: asyncio.AbstractEventLoop | None = None
        self._opened_browser = False

    def _prompt(self, prompt: OAuthPrompt) -> None:
        if self.callback is not None:
            import inspect

            result = self.callback(prompt)
            if inspect.isawaitable(result):
                if self._loop is not None and self._loop.is_running():
                    # Called from the loop's thread we'd deadlock on .result();
                    # this flow is sync-driven, so only cross-thread dispatch.
                    asyncio.run_coroutine_threadsafe(result, self._loop).result(timeout=30)
                else:
                    asyncio.run(result)  # type: ignore[arg-type]
        elif prompt.url and not self._opened_browser:
            webbrowser.open(prompt.url)
            self._opened_browser = True


def login_openai_codex(
    prompt_callback: Callable[[OAuthPrompt], Any] | None = None,
    *,
    credential_store: FileCredentialStore | None = None,
    client_id: str = OPENAI_CODEX_CLIENT_ID,
    timeout: float = 300.0,
) -> OAuthCredential | None:
    """Run the full Codex PKCE login and persist the credential.

    ``prompt_callback`` receives an ``OAuthPrompt`` for each interactive step
    (URL open, waiting). Returns the stored credential on success.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    flow = AuthorizationFlow(callback=prompt_callback)
    flow._loop = loop

    verifier, challenge = create_pkce_pair()
    state = secrets.token_urlsafe(32)
    auth_url = (
        f"{OPENAI_CODEX_AUTH_BASE_URL}/oauth/authorize"
        f"?client_id={urllib.parse.quote(client_id)}"
        f"&response_type=code"
        f"&redirect_uri={urllib.parse.quote(OPENAI_CODEX_REDIRECT_URI)}"
        f"&scope={urllib.parse.quote(OPENAI_CODEX_SCOPE)}"
        f"&state={state}"
        f"&code_challenge={urllib.parse.quote(challenge)}"
        f"&code_challenge_method=S256"
        f"&prompt=login"
    )
    flow._prompt(OAuthPrompt(kind="url", text="Open this URL to sign in", url=auth_url))
    flow._prompt(OAuthPrompt(kind="wait", text="Waiting for sign-in to complete..."))

    auth = _local_platform_server(state, timeout=timeout)
    if auth is None:
        return None
    token = _exchange_code(auth.code, verifier, client_id)
    if token is None or not token.get("access_token"):
        return None
    credential = OAuthCredential(
        provider="openai-codex",
        access_token=str(token["access_token"]),
        refresh_token=token.get("refresh_token"),
        expires_at=(time.time() + float(token["expires_in"])) if token.get("expires_in") else None,
        extras={k: v for k, v in token.items() if k not in ("access_token", "refresh_token", "expires_in", "token_type")},
    )
    store = credential_store or FileCredentialStore()
    store.set(credential)
    return credential
