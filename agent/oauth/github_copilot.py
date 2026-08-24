"""GitHub Copilot device-code OAuth — Tau oauth_github_copilot.py port (Tau 37a9e43 src/tau_coding/oauth_github_copilot.py), lean-adapted.

Copilot's OAuth flow is device-code based with a ``copilot_internal/v2/token``
exchange. This port requests a device code from GitHub, displays the user code
prompt, and polls via ``poll_oauth_device_code``.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ..credentials import FileCredentialStore, OAuthCredential
from .device import DeviceFlowError, parse_device_code_response, poll_oauth_device_code
from .http import OAuthHTTPError, get_json, post_form
from .types import OAuthPrompt

GITHUB_CLIENT_ID = "2b01893e4b92f7498773"
GITHUB_SCOPE = "read:user user:email"
GITHUB_DEVICE_URL = "https://github.com/login/device/code"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"

# Identity + Copilot entitlement only: full `repo` write access is not needed
# for sign-in-for-inference and would turn a leaked token into repo access.
_COPILOT_HEADERS = {
    "Editor-Version": "lean-prover/0.2",
    "Editor-Plugin-Version": "lean-prover/0.2",
}


def _post_json(url: str, fields: dict[str, str]) -> dict | None:
    try:
        return post_form(url, fields)
    except OAuthHTTPError:
        return None


def refresh_github_copilot(refresh_token: str,
                           client_id: str = GITHUB_CLIENT_ID) -> dict | None:
    """Exchange a GitHub refresh token for a fresh token payload."""
    try:
        token = post_form(GITHUB_TOKEN_URL, {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": refresh_token,
        })
    except OAuthHTTPError:
        return None
    return token if token.get("access_token") else None


def copilot_api_key(github_token: str,
                    should_refresh: Callable[[], dict | None] | None = None,
                    ) -> dict | None:
    """Exchange a GitHub OAuth token for a short-lived Copilot API key.

    Returns the raw exchange payload (``token``/``expires_at`` keys) or None.
    On 401 the caller-supplied ``should_refresh`` hook is invoked once to
    obtain a fresh GitHub token payload before retrying.
    """
    payload = _copilot_exchange(github_token)
    if payload is not None:
        return payload
    if should_refresh is None:
        return None
    fresh = should_refresh()
    if not fresh or not fresh.get("access_token"):
        return None
    return _copilot_exchange(str(fresh["access_token"]))


def _copilot_exchange(github_token: str) -> dict | None:
    try:
        return get_json(COPILOT_TOKEN_URL, headers={
            "Authorization": f"Bearer {github_token}", **_COPILOT_HEADERS})
    except OAuthHTTPError as e:
        if "401" in str(e):
            return None  # signal refresh-and-retry to the caller
        return None


async def login_github_copilot(
    prompt_callback: Callable[[OAuthPrompt], object] | None = None,
    *,
    credential_store: FileCredentialStore | None = None,
    client_id: str = GITHUB_CLIENT_ID,
    should_stop: Callable[[], bool] | None = None,
) -> OAuthCredential | None:
    """Run the GitHub Copilot device-code login; persist on success."""
    auth = _post_json(GITHUB_DEVICE_URL, {
        "client_id": client_id,
        "scope": GITHUB_SCOPE,
    })
    if auth is None or not auth.get("device_code"):
        return None
    device = parse_device_code_response(auth)
    if prompt_callback is not None:
        prompt_callback(OAuthPrompt(
            kind="url",
            text=f"Open {device.verification_uri} and enter {device.user_code}",
            url=device.verification_uri,
        ))

    async def fetch_token(device_code: str) -> dict | None:
        return _post_json(GITHUB_TOKEN_URL, {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": device_code,
            "client_id": client_id,
        })

    try:
        token = await poll_oauth_device_code(device, fetch_token, should_stop=should_stop)
    except DeviceFlowError:
        return None
    if token is None or not token.get("access_token"):
        return None
    credential = OAuthCredential(
        provider="github-copilot",
        access_token=str(token["access_token"]),
        refresh_token=token.get("refresh_token"),
        expires_at=(time.time() + float(token["expires_in"])) if token.get("expires_in") else None,
    )
    store = credential_store or FileCredentialStore()
    store.set(credential)
    return credential
