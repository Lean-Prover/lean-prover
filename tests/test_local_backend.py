"""llama.cpp local backend — discovery, doctor, snapshot, CLI, /local command."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest

from agent.local_backend import DEFAULT_ENDPOINT, LlamaCppBackend


class _FakeLlama(BaseHTTPRequestHandler):
    """Minimal llama.cpp /v1/models responder for hermetic tests."""

    models: ClassVar[list[str]] = ["qwen3-8b", "llama-3.2-3b"]

    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": m} for m in self.models]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture()
def llama_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeLlama)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_endpoint_resolution_default(tmp_path):
    backend = LlamaCppBackend(config_dir=tmp_path)
    assert backend.effective_endpoint() == DEFAULT_ENDPOINT


def test_endpoint_resolution_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_BASE_URL", "http://10.0.0.5:9999")
    assert LlamaCppBackend(config_dir=tmp_path).effective_endpoint() == "http://10.0.0.5:9999"


def test_endpoint_resolution_explicit_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_BASE_URL", "http://env:1")
    backend = LlamaCppBackend(endpoint="http://explicit:2", config_dir=tmp_path)
    assert backend.effective_endpoint() == "http://explicit:2"


def test_endpoint_resolution_snapshot(tmp_path):
    backend = LlamaCppBackend(config_dir=tmp_path)
    backend.configure(endpoint="http://saved:3", model="m1")
    assert LlamaCppBackend(config_dir=tmp_path).effective_endpoint() == "http://saved:3"


def test_api_key_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_API_KEY", "env-key")
    assert LlamaCppBackend(config_dir=tmp_path).effective_api_key() == "env-key"


def test_api_key_credential_wins_over_env(tmp_path, monkeypatch):
    from agent.credentials import FileCredentialStore, OAuthCredential

    monkeypatch.setenv("LLAMA_API_KEY", "env-key")
    FileCredentialStore(tmp_path / "credentials").set(
        OAuthCredential(provider="llama.cpp", access_token="stored-key")
    )
    assert LlamaCppBackend(config_dir=tmp_path).effective_api_key() == "stored-key"


def test_probe_discovers_models(llama_server, tmp_path):
    backend = LlamaCppBackend(endpoint=llama_server, config_dir=tmp_path)
    assert backend.probe() == ["qwen3-8b", "llama-3.2-3b"]
    assert backend.discover_models() == [{"id": "qwen3-8b"}, {"id": "llama-3.2-3b"}]


def test_doctor_ok(llama_server, tmp_path):
    backend = LlamaCppBackend(endpoint=llama_server, config_dir=tmp_path)
    report = backend.doctor()
    assert report["reachable"] is True
    assert report["models"] == ["qwen3-8b", "llama-3.2-3b"]
    assert report["streaming"] is True


def test_doctor_unreachable(tmp_path):
    backend = LlamaCppBackend(endpoint="http://127.0.0.1:1", config_dir=tmp_path)
    report = backend.doctor()
    assert report["reachable"] is False
    assert report["warning"]


def test_configure_snapshot_and_reset(tmp_path):
    backend = LlamaCppBackend(config_dir=tmp_path)
    body = backend.configure(endpoint="http://127.0.0.1:8080", model="qwen3-8b",
                             api_key="secret")
    snap_path = tmp_path / "state" / "llama.cpp.json"
    assert snap_path.exists()
    assert body["endpoint"] == "http://127.0.0.1:8080"
    assert body["model"] == "qwen3-8b"
    snap = backend.snapshot()
    assert snap["endpoint"] == "http://127.0.0.1:8080"
    assert snap["model"] == "qwen3-8b"
    # api_key never lands in the snapshot
    assert "api_key" not in json.loads(snap_path.read_text())
    backend.reset()
    assert not snap_path.exists()


def test_provider_config_targets_local(llama_server, tmp_path):
    backend = LlamaCppBackend(endpoint=llama_server, config_dir=tmp_path)
    pc = backend.provider_config()
    assert pc.name == "llama.cpp"
    assert pc.kind == "openai_compatible"
    assert pc.base_url == f"{llama_server}/v1"


# ------------------------------------------------------------- CLI + command


def test_cli_local_configure_and_status(llama_server, tmp_path, monkeypatch):
    from agent import main as m
    from agent.local_backend import LlamaCppBackend

    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path))
    # configure against the live fake server (writes into tmp_path config dir)
    rc = m.cmd_local(m._build_parser().parse_args(
        ["local", "configure", "--url", llama_server, "--model", "qwen3-8b"]))
    assert rc == 0
    backend = LlamaCppBackend(config_dir=tmp_path)
    assert backend.effective_endpoint() == llama_server
    assert backend.probe() == ["qwen3-8b", "llama-3.2-3b"]
    assert backend.snapshot()["model"] == "qwen3-8b"


def test_local_slash_command_registered():
    from agent.commands import create_default_command_registry

    registry = create_default_command_registry()
    cmd = registry.get("local")
    assert cmd is not None
    assert "llama.cpp" in cmd.description
    # alias
    assert registry.get("llama") is cmd


@pytest.mark.anyio
async def test_local_slash_opens_screen(monkeypatch, tmp_path) -> None:
    from agent.tui.app import ProverApp

    monkeypatch.setenv("PROVER_CONFIG_DIR", str(tmp_path / "config"))
    app = ProverApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt")
        prompt.focus()
        prompt.value = "/local"
        await pilot.press("enter")
        await pilot.pause()
        assert any(type(s).__name__ == "LocalScreen" for s in app.screen_stack)
        await pilot.press("q")
        await pilot.pause()
