"""Extension loader — Tau extensions/loader.py port (Tau 37a9e43 src/tau_coding/extensions/loader.py), lean-adapted.

Scans the configured extension dirs plus any extra paths and imports each
``*.py`` (or ``*/__init__.py`` for package extensions) as an isolated module.
A loaded module may expose a ``setup(tau)`` callback (Tau's extension
protocol); failures during setup are captured, not fatal.

Lean path: user extensions live under ``ProverPaths().config_dir /
"extensions"``; example extensions ship under ``data/examples/extensions``.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_EXTENSION_MODULE_PREFIX = "lean_prover_ext_"
_warned_trusted: set[bool] = set()


@dataclass
class LoadedExtension:
    """One successfully imported extension module (tau LoadedExtension)."""

    name: str
    path: Path
    module: Any = None
    setup_error: str | None = None
    registered: list[str] = field(default_factory=list)


def load_extensions(
    extensions_dirs: list[Path] | None = None,
    extra_paths: list[Path] | None = None,
) -> tuple[list[LoadedExtension], list[tuple[Path, str]]]:
    """Import extensions from dirs + extra single-file paths.

    Returns (loaded, failures) where failures is a list of (path, error)
    pairs — failures never abort the loader.

    Lean adaptation: package extensions (directories with ``__init__.py``)
    are supported alongside single-file ``*.py``.
    """
    pairs: list[tuple[str, Path]] = []
    seen = set()
    for d in extensions_dirs or []:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.py")):
            if p.stem.startswith("_"):
                continue
            pairs.append((p.stem, p))
        for init in sorted(d.glob("*/__init__.py")):
            name = init.parent.name
            if name.startswith("_"):
                continue
            pairs.append((name, init.parent))

    for p in extra_paths or []:
        p = Path(p)
        if p.is_file():
            pairs.append((p.stem, p))
        elif (p / "__init__.py").exists():
            pairs.append((p.name, p))

    if pairs and not _warned_trusted:
        _warned_trusted.add(True)
        import sys

        print("[extensions warn] loading extensions — these are fully trusted "
              "arbitrary Python executed with your privileges. Only load "
              "extensions you wrote or audited.", file=sys.stderr)

    loaded: list[LoadedExtension] = []
    failures: list[tuple[Path, str]] = []
    for name, path in pairs:
        if name in seen:
            continue
        seen.add(name)
        result = _import_extension(name, path)
        if result is None:
            failures.append((path, f"failed to import extension {name}"))
        else:
            loaded.append(result)
    return loaded, failures


def _import_extension(name: str, path: Path) -> LoadedExtension | None:
    module_name = f"{_EXTENSION_MODULE_PREFIX}{name}"
    try:
        module_path = path / "__init__.py" if path.is_dir() else path
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return LoadedExtension(name=name, path=path, module=module)
    except ImportError as exc:
        sys.modules.pop(module_name, None)
        return LoadedExtension(
            name=name, path=path, setup_error=f"import failed: {exc}"
        )
    except Exception:  # noqa: BLE001 — extension isolation boundary
        tb = traceback.format_exc(limit=5)
        sys.modules.pop(module_name, None)
        return LoadedExtension(name=name, path=path, setup_error=tb[-300:])


def _call_setup(extension: LoadedExtension, tau: Any) -> None:
    """Invoke an extension's ``setup(tau)`` hook, capturing failures."""
    module = extension.module
    if module is None:
        return
    setup = getattr(module, "setup", None)
    if setup is None or not callable(setup):
        return
    try:
        setup(tau)
        extension.registered.append("setup")
    except Exception:  # noqa: BLE001
        extension.setup_error = traceback.format_exc(limit=5)[-300:]


def unload_extension_modules() -> int:
    """Purge extension modules from sys.modules (tau unload_extension_modules)."""
    to_remove = [k for k in sys.modules if k.startswith(_EXTENSION_MODULE_PREFIX)]
    for k in to_remove:
        del sys.modules[k]
    return len(to_remove)