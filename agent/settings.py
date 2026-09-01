"""Layered user settings — adapted from fx's settings/config-precedence model.

fx resolves configuration through a single documented precedence chain:
environment > user settings (workspace > global) > project defaults > built-in.

lean-prover adopts the same idea as one small, typed loader: every user-facing
knob is read from ``~/.prover/settings.json`` with a single precedence rule
(highest wins), so there is one place to look and one rule to remember.

Precedence (highest wins):

1. Environment variable (``PROVER_<KEY>``), when set
2. ``<workspace>/.prover/settings.json`` (project settings) — optional
3. ``~/.prover/settings.json`` (user/global settings)
4. Built-in default for the key

Keys are the lower-cased, underscore form of each setting name (see
:py:data:`DEFAULTS`). Environment overrides use the ``PROVER_`` prefix and
uppercase the key, e.g. setting ``max_steps`` -> ``PROVER_MAX_STEPS``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .paths import ProverPaths

# name -> (default, json_type, doc). json_type is the expected JSON value type
# used to reject wrong-typed values in a config file.
DEFAULTS: dict[str, tuple[Any, str, str]] = {
    "max_steps": (20, "int", "default max repair steps for a proof"),
    "context_window": (0, "int", "0 = auto; token budget for history compaction"),
    "workers": (1, "int", "default parallel proof workers"),
    "theme": ("prover-dark", "str", "TUI theme name"),
    "thinking": ("off", "str", "default thinking level"),
    "permission_mode": ("ask", "str", "ask|auto|yolo baseline for tool approval"),
    "quiet": (False, "bool", "suppress non-essential output (best for scripts)"),
}

# Keys a *project* (committed workspace `<cwd>/.prover/settings.json`) file
# must never set. A repo is untrusted: letting it set `permission_mode` would
# weaken the tool-policy ACL. (model/base_url/api_key are not settings at all
# — they live in the provider/models stores, not DEFAULTS — so only the real
# ACL knob needs stripping here.)
_PROJECT_FORBIDDEN = frozenset({
    "permission_mode",
})


def canonical_key(name: str) -> str:
    """Normalise a setting name to its canonical (lower, underscore) form."""
    return name.strip().lower().replace("-", "_")


def env_name(key: str) -> str:
    """Return the env var that overrides ``key`` (``PROVER_<KEY>``)."""
    return "PROVER_" + key.upper()


def _coerce(key: str, raw: Any) -> Any:
    _, json_type, _ = DEFAULTS[key]
    if json_type == "int":
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"setting {key}: expected an int, got {raw!r}")
        return raw
    if json_type == "bool":
        if not isinstance(raw, bool):
            raise ValueError(f"setting {key}: expected a bool, got {raw!r}")
        return raw
    # str
    if not isinstance(raw, str):
        raise TypeError(f"setting {key}: expected a string, got {raw!r}")
    return raw


def _load_layers(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the user and project settings files; later files shadow earlier ones.

    Return value maps layer name -> parsed JSON dict (malformed/missing files
    are ignored so a bad settings file never breaks the agent).
    """
    paths = _resolve_files(path)
    layers: dict[str, dict[str, Any]] = {}
    for name, p in paths:
        if p is None or not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        layers[name] = data
    # The workspace project layer is untrusted (a committed repo file): strip
    # policy/API keys so a repo can pin harmless knobs but never weaken the
    # ACL or steer credential use. User-layer control is untouched.
    proj = layers.get("project")
    if proj:
        for key in _PROJECT_FORBIDDEN:
            proj.pop(key, None)
    return layers


def _resolve_files(path: Path | None) -> list[tuple[str, Path | None]]:
    """Return [(layer_name, path_or_None), ...] in low->high precedence order."""
    pp = ProverPaths()
    user = path if path is not None else pp.config_dir / "settings.json"
    project = Path.cwd() / ".prover" / "settings.json"
    # project layer is higher precedence than user layer
    return [("user", user), ("project", project)]


def get(key: str, *, path: Path | None = None) -> Any:
    """Resolve a single setting through the precedence chain (highest wins):

    1. ``PROVER_<KEY>`` environment variable
    2. ``<workspace>/.prover/settings.json`` (project settings)
    3. ``~/.prover/settings.json`` (user/global settings)
    4. ``<workspace>/.prover.json`` repo-safe defaults (project_defaults)
    5. Built-in default
    """
    ckey = canonical_key(key)
    if ckey not in DEFAULTS:
        raise KeyError(f"unknown setting: {key}")
    default, json_type, _ = DEFAULTS[ckey]

    env = os.environ.get(env_name(ckey))
    if env is not None:
        return _from_env(ckey, json_type, env)

    for layer in ("project", "user"):
        data = _layers(path).get(layer, {})
        if ckey in data:
            try:
                return _coerce(ckey, data[ckey])
            except (TypeError, ValueError):
                # A wrong-typed value in one layer must never crash the agent;
                # fall through to the next layer / default (bad settings files
                # are tolerated by design).
                continue

    # Repo-safe committed defaults (.prover.json) form the next-lowest layer.
    from .project_defaults import effective_defaults

    proj = effective_defaults()
    if ckey in proj:
        try:
            return _coerce(ckey, proj[ckey])
        except (TypeError, ValueError):
            pass

    return default


def all_settings(*, path: Path | None = None) -> dict[str, Any]:
    """Return every known setting resolved to its effective value."""
    return {key: get(key, path=path) for key in DEFAULTS}


def _layers(path: Path | None = None) -> dict[str, dict[str, Any]]:
    # Cache keyed by BOTH resolved file paths AND their mtimes: the value
    # depends on the user settings file and the cwd-relative project settings
    # file, and edits to either made while a long-lived process runs must be
    # picked up (a bare path key would go stale).
    from .paths import ProverPaths

    user = path if path is not None else ProverPaths().config_dir / "settings.json"
    project = Path.cwd() / ".prover" / "settings.json"
    stamp = "|".join(
        f"{p}:{(p.stat().st_mtime_ns if p.exists() else '')}" for p in (user, project)
    )
    key = f"{user}::{project}::{stamp}"
    cache = getattr(_layers, "_cache", None)  # type: ignore[attr-defined]
    if cache is None:
        cache = {}
        _layers._cache = cache  # type: ignore[attr-defined]
    if key not in cache:
        cache[key] = _load_layers(path)
    return cache[key]


def _from_env(key: str, json_type: str, env: str) -> Any:
    if json_type == "int":
        try:
            return int(env)
        except ValueError:
            default, _, _ = DEFAULTS[key]
            return default
    if json_type == "bool":
        return env.strip().lower() in ("1", "true", "yes", "on")
    return env


def clear_settings_cache() -> None:
    """Drop the in-process settings cache (used by tests)."""
    _layers._cache = {}
