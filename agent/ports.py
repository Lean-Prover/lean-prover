"""Ports & adapters — the engine's external dependencies as protocols.

ARCHITECTURE.md step E: the proof engine depends on injected ports
(``LLMClient``, ``LeanRunner``, ``LSPClient``, ``Retriever``) rather than
concrete modules, so fakes can drive the whole loop in tests and providers
can be swapped without touching the engine.

All four ports are injectable parameters on :func:`prover_loop.prove`, with
``Default*`` adapters that delegate to the real ``agent.llm`` / ``agent.lean``
/ ``agent.lsp`` / ``agent.retrieval`` modules. The default adapters preserve
the exact call shapes the engine used before injection, so tests that
monkeypatch those modules keep working.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """The language-model dependency of the proof repair loop."""

    def chat(self, system: str, messages: list[dict], temperature: float = 0.2,
             model_name: str | None = None) -> object: ...
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int,
                      model_name: str | None = None) -> float: ...
    def context_window_tokens(self, model_name: str | None = None) -> int: ...
    def model(self) -> str: ...


@runtime_checkable
class LeanRunner(Protocol):
    """Compile/check Lean files and parse diagnostics."""

    def check_file(self, lean_file: object, lean_dir: object,
                   timeout: int = 60) -> tuple[bool, str]: ...
    def parse_diagnostics(self, output: str) -> list[object]: ...
    def error_report(self, lean_dir: object, output: str) -> str: ...


@runtime_checkable
class LSPClient(Protocol):
    """Lean language-server goal-state feedback."""

    def update(self, text: str) -> None: ...
    def goal_at_end(self, text: str) -> str | None: ...
    def close(self) -> None: ...


@runtime_checkable
class LeanLSPFactory(Protocol):
    """Creates an :class:`LSPClient` for a file (the LSP session is lazy)."""

    def __call__(self, lean_dir: object, target_file: object) -> LSPClient: ...


@runtime_checkable
class Retriever(Protocol):
    """Local lemma/statement retrieval + corpus growth."""

    def search_lemmas(self, statement: str, k: int = 5) -> list[dict]: ...
    def corpus_append(self, lean_dir: object, statement: str, proof: str) -> bool: ...


class DefaultLLMClient:
    """Default ``LLMClient`` delegating to the real ``agent.llm`` module.

    Kept a thin pass-through so tests that monkeypatch ``llm.chat`` /
    ``llm.estimate_cost`` / ``llm.context_window_tokens`` / ``llm.model``
    keep working through the injected port.
    """

    def chat(self, system, messages, temperature=0.2, model_name=None):
        from . import llm

        kwargs = {} if model_name is None else {"model_name": model_name}
        return llm.chat(system, messages, temperature=temperature, **kwargs)

    def estimate_cost(self, prompt_tokens, completion_tokens, model_name=None):
        from . import llm

        kwargs = {} if model_name is None else {"model_name": model_name}
        return llm.estimate_cost(prompt_tokens, completion_tokens, **kwargs)

    def context_window_tokens(self, model_name=None):
        from . import llm

        # Match the engine's original call shape: no args when unset (tests
        # monkeypatch this as a no-arg lambda).
        if model_name is None:
            return llm.context_window_tokens()
        return llm.context_window_tokens(model_name=model_name)

    def model(self):
        from . import llm

        return llm.model()


class DefaultLeanRunner:
    """Default ``LeanRunner`` delegating to the real ``agent.lean`` module.

    Preserves the engine's original call shapes (2-arg ``check_file``,
    etc.) so hermetic tests that monkeypatch ``lean.*`` keep working.
    """

    def check_file(self, lean_file, lean_dir, timeout=60):
        from . import lean

        return lean.check_file(lean_file, lean_dir)

    def parse_diagnostics(self, output):
        from . import lean

        return lean.parse_diagnostics(output)

    def error_report(self, lean_dir, output):
        from . import lean

        return lean.error_report(lean_dir, output)


class DefaultLeanLSPFactory:
    """Default ``LeanLSPFactory`` returning a real ``lsp.LeanLSP`` session."""

    def __call__(self, lean_dir, target_file):
        from . import lsp

        return lsp.LeanLSP(lean_dir, target_file)


class DefaultRetriever:
    """Default ``Retriever`` delegating to the real ``agent.retrieval`` module."""

    def search_lemmas(self, statement, k=5):
        from . import retrieval

        return retrieval.search_lemmas(statement, k=k)

    def corpus_append(self, lean_dir, statement, proof):
        from . import retrieval

        return retrieval.corpus_append(lean_dir, statement, proof)
