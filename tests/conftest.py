"""Pin the anyio backend for Textual's app.run_test() to asyncio."""

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _reset_allow_select():
    """ALLOW_SELECT is set on the class; reset around each test."""
    from agent.tui import ProverApp

    ProverApp.ALLOW_SELECT = True
    yield
    ProverApp.ALLOW_SELECT = True
    # The TUI's _apply_thinking_level sets a process-level override in
    # agent.thinking; never let it leak between tests.
    from agent import thinking

    thinking.clear_thinking_level()


@pytest.fixture(autouse=True)
def _no_corpus_growth(monkeypatch):
    """Keep test-driven proofs from writing into the real repo corpus."""
    monkeypatch.setenv("PROVER_CORPUS_GROW", "0")


@pytest.fixture(autouse=True)
def _hermetic_lean_dir(tmp_path, monkeypatch):
    """Keep test-driven proofs from touching the real lean/ workspace.

    prover_loop.LEAN_DIR is read once at import; tests that patch the module
    attribute directly still win, but everything else (prove() without a
    patched fixture) now writes into tmp instead of lean/src|tmp.
    """
    monkeypatch.setenv("PROVER_LEAN_DIR", str(tmp_path / "lean"))
