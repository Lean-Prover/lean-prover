"""agent.settings — layered settings.json + precedence tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import settings


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    settings.clear_settings_cache()
    for key in settings.DEFAULTS:
        monkeypatch.delenv(settings.env_name(key), raising=False)
    yield
    settings.clear_settings_cache()


def _write_user_settings(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / ".prover" / "settings.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    return p


def test_builtin_defaults(monkeypatch, tmp_path) -> None:
    # No env, no files -> defaults. Point HOME at a fresh dir so nothing exists.
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("PROVER_CONFIG_DIR", raising=False)
    assert settings.get("max_steps") == 20
    assert settings.get("permission_mode") == "ask"
    assert settings.get("quiet") is False


def test_user_file_applies(monkeypatch, tmp_path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("PROVER_CONFIG_DIR", raising=False)
    _write_user_settings(fake_home, {"max_steps": 7, "quiet": True})
    assert settings.get("max_steps") == 7
    assert settings.get("quiet") is True


def test_env_beats_file(monkeypatch, tmp_path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("PROVER_CONFIG_DIR", raising=False)
    _write_user_settings(fake_home, {"max_steps": 7})
    monkeypatch.setenv("PROVER_MAX_STEPS", "11")
    assert settings.get("max_steps") == 11


def test_unknown_key_raises() -> None:
    with pytest.raises(KeyError):
        settings.get("bogus")


def test_canonical_key_normalises() -> None:
    assert settings.canonical_key("Max-Steps") == "max_steps"


def test_bad_file_ignored(monkeypatch, tmp_path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("PROVER_CONFIG_DIR", raising=False)
    p = _write_user_settings(fake_home, "{not json")
    assert p.exists()
    # Malformed file must not crash; falls back to default.
    assert settings.get("max_steps") == 20


def test_wrong_typed_value_falls_back(monkeypatch, tmp_path) -> None:
    """A valid-JSON-but-wrong-typed value must not crash the agent."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("PROVER_CONFIG_DIR", raising=False)
    p = _write_user_settings(fake_home, json.dumps({"max_steps": "twenty"}))
    assert p.exists()
    # Wrong-typed value is ignored; falls back to the built-in default.
    assert settings.get("max_steps") == 20


def test_explicit_path_override(monkeypatch, tmp_path) -> None:
    p = tmp_path / "alt.json"
    p.write_text(json.dumps({"workers": 5}))
    assert settings.get("workers") == 1
    assert settings.get("workers", path=p) == 5


def test_all_settings() -> None:
    res = settings.all_settings()
    assert set(res) == set(settings.DEFAULTS)


def test_prover_json_repo_default_layer(monkeypatch, tmp_path) -> None:
    """settings.get falls back to .prover.json repo-safe defaults."""
    import json as _json

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("PROVER_CONFIG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".prover.json").write_text(_json.dumps({"max_steps": 9}))
    assert settings.get("max_steps") == 9
    assert settings.get("permission_mode") == "ask"  # built-in floor


def test_env_wins_over_prover_json(monkeypatch, tmp_path) -> None:
    import json as _json

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("PROVER_CONFIG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".prover.json").write_text(_json.dumps({"max_steps": 9}))
    monkeypatch.setenv("PROVER_MAX_STEPS", "12")
    assert settings.get("max_steps") == 12


def test_project_settings_cannot_weaken_acl(monkeypatch, tmp_path) -> None:
    """A committed workspace .prover/settings.json must not set permission_mode."""
    import json as _json

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("PROVER_CONFIG_DIR", raising=False)
    proj = tmp_path / ".prover"
    proj.mkdir()
    (proj / "settings.json").write_text(
        _json.dumps({"permission_mode": "yolo", "max_steps": 7}))
    monkeypatch.chdir(tmp_path)
    # safe knob still applies from the project layer
    assert settings.get("max_steps") == 7
    # policy key is ignored from the project layer -> built-in floor
    assert settings.get("permission_mode") == "ask"


def test_project_layer_reloads_on_cwd_change(monkeypatch, tmp_path) -> None:
    """The project (.prover/settings.json) layer must reload when cwd changes."""
    import json as _json

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("PROVER_CONFIG_DIR", raising=False)
    a = tmp_path / "projA"
    b = tmp_path / "projB"
    for d, v in ((a, 7), (b, 9)):
        d.mkdir()
        (d / ".prover").mkdir()
        (d / ".prover" / "settings.json").write_text(_json.dumps({"max_steps": v}))
    monkeypatch.chdir(a)
    assert settings.get("max_steps") == 7
    monkeypatch.chdir(b)
    # Cached value must not leak from the previous working directory.
    assert settings.get("max_steps") == 9
