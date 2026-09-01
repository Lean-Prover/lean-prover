"""Root conftest — puts the repo root on ``sys.path`` for the test suite.

pytest inserts each test module's *basedir* (``tests/``) into ``sys.path``,
not the repo root. The ``benchmark`` (implicit namespace package) and ``data``
top-level packages live at the root, so without this they are not importable
from ``pytest tests/`` (``from benchmark import import_standard`` etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
