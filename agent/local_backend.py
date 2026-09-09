"""Local inference backends — llama.cpp (Tau /local parity, lean-adapted).

A ``LocalBackend`` describes a local model server (llama.cpp) reachable over an
OpenAI-compatible HTTP API. lean-prover does not install, start, stop, or scan
for llama.cpp; it probes a single configured endpoint, discovers model ids via
``/v1/models``, and records a safe integration snapshot (endpoint + selected
model + timestamp) so ``prove`` / the TUI can target the server.

Endpoint precedence: explicit > saved snapshot > ``LLAMA_BASE_URL`` > default.
API key: stored credential (provider ``llama.cpp``) > ``LLAMA_API_KEY``.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .paths import ProverPaths

BACKEND_ID = "llama.cpp"
DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
_SNAPSHOT_VERSION = 1
_DEFAULT_TIMEOUT = 10.0


class LocalBackend(Protocol):
    """A local inference server (provider-neutral, Tau local-backend parity)."""

    id: str

    def probe(self) -> list[str]:
        """Discovered model ids (exact ids from the server, never invented)."""
        ...

    def discover_models(self) -> list[dict]:
        """Discovered models as a list of ``{id}`` dicts."""
        ...

    def doctor(self) -> dict:
        """Reachability + discovery diagnostics."""
        ...


@dataclass(frozen=True, slots=True)
class LlamaCppBackend:
    """llama.cpp server over its OpenAI-compatible HTTP API.

    ``endpoint`` / ``api_key`` are explicit overrides; when unset they resolve
    from the saved snapshot, ``LLAMA_BASE_URL`` / ``LLAMA_API_KEY``, then the
    default endpoint. ``config_dir`` is injectable for tests.
    """

    endpoint: str | None = None
    api_key: str | None = None
    config_dir: Path | None = None

    @property
    def id(self) -> str:
        return BACKEND_ID

    def _dir(self) -> Path:
        return self.config_dir or ProverPaths().config_dir

    @property
    def snapshot_path(self) -> Path:
        return self._dir() / "state" / f"{BACKEND_ID}.json"

    # ------------------------------------------------------------- resolution

    def effective_endpoint(self) -> str:
        if self.endpoint:
            return self.endpoint.rstrip("/")
        snap = self._load_snapshot()
        if snap and snap.get("endpoint"):
            return str(snap["endpoint"]).rstrip("/")
        env = os.environ.get("LLAMA_BASE_URL")
        if env:
            return env.rstrip("/")
        return DEFAULT_ENDPOINT

    def effective_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        from .credentials import FileCredentialStore

        cred = FileCredentialStore(self._dir() / "credentials").get(BACKEND_ID)
        if cred and cred.access_token:
            return cred.access_token
        return os.environ.get("LLAMA_API_KEY")

    # ---------------------------------------------------------------- probing

    def _http_json(self, path: str, timeout: float = _DEFAULT_TIMEOUT) -> dict:
        url = f"{self.effective_endpoint()}{path}"
        req = urllib.request.Request(url)
        key = self.effective_api_key()
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def probe(self) -> list[str]:
        """Discover model ids from ``/v1/models`` (exact ids, never invented)."""
        data = self._http_json("/v1/models")
        models = data.get("data") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return []
        ids: list[str] = []
        for model in models:
            if isinstance(model, dict) and model.get("id"):
                ids.append(str(model["id"]))
        return ids

    def discover_models(self) -> list[dict]:
        return [{"id": mid} for mid in self.probe()]

    def doctor(self) -> dict:
        """Endpoint reachability + model discovery + streaming check.

        A server that streams but whose GGUF chat template cannot emit tool
        calls is a compatibility warning, not a connectivity failure; we only
        verify reachability and discovery here.
        """
        result = {
            "endpoint": self.effective_endpoint(),
            "reachable": False,
            "models": [],
            "streaming": False,
            "warning": None,
        }
        try:
            ids = self.probe()
            result["models"] = ids
            result["reachable"] = True
            if ids:
                result["streaming"] = True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError) as exc:
            result["warning"] = str(exc)
        return result

    # ------------------------------------------------------------- persistence

    def _load_snapshot(self) -> dict:
        path = self.snapshot_path
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def snapshot(self) -> dict:
        """The safe integration snapshot (endpoint, model, timestamp)."""
        snap = self._load_snapshot()
        return {
            "version": _SNAPSHOT_VERSION,
            "endpoint": snap.get("endpoint"),
            "model": snap.get("model"),
            "timestamp": snap.get("timestamp"),
        }

    def configure(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> dict:
        """Persist endpoint + optional model (safe snapshot) and optional key.

        ``api_key`` is stored in the credential store (never the snapshot).
        """
        effective = (
            endpoint or self.endpoint or os.environ.get("LLAMA_BASE_URL") or DEFAULT_ENDPOINT
        ).rstrip("/")
        if api_key:
            from .credentials import FileCredentialStore, OAuthCredential

            FileCredentialStore(self._dir() / "credentials").set(
                OAuthCredential(provider=BACKEND_ID, access_token=api_key)
            )
        body = {
            "version": _SNAPSHOT_VERSION,
            "endpoint": effective,
            "model": model,
            "timestamp": time.time(),
        }
        path = self.snapshot_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".llama-")
        try:
            os.chmod(tmp, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(body, file, indent=2)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return body

    def reset(self) -> None:
        """Remove only lean-prover's integration settings + snapshot.

        Never stops the external server or deletes model files.
        """
        try:
            self.snapshot_path.unlink(missing_ok=True)
        except OSError:
            pass
        from .credentials import FileCredentialStore

        FileCredentialStore(self._dir() / "credentials").delete(BACKEND_ID)

    # ------------------------------------------------------------ provider

    def provider_config(self):
        """An ``openai_compatible`` provider config targeting this endpoint.

        Lets a resolved model route to the local server without touching the
        engine (``prover local use --model <id>`` registers a model profile
        backed by this config).
        """
        from .provider_config import OpenAICompatibleConfig, ProviderConfig

        return ProviderConfig(
            name=BACKEND_ID,
            kind="openai_compatible",
            base_url=f"{self.effective_endpoint()}/v1",
            env_key="LLAMA_API_KEY",
            compat=OpenAICompatibleConfig(thinking_format="qwen"),
        )
