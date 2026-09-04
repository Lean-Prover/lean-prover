"""LLM provider — any OpenAI-compatible endpoint via env vars."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from openai import OpenAI, Timeout

from . import models

# Clients are cached per (base_url, api_key): model profiles may point at
# different endpoints, so a single global singleton is no longer enough.
_clients: dict[tuple[str | None, str], OpenAI] = {}

# Hard wall-clock cap per LLM call. httpx read-timeouts reset on every byte
# received, so slow-streaming reasoning endpoints can hang forever without
# this. The call runs in a daemon thread so a hung request can never block
# process exit (ThreadPoolExecutor's atexit join would otherwise do that).
HARD_TIMEOUT = float(os.environ.get("PROVER_LLM_TIMEOUT", "600"))


def client(model_name: str | None = None) -> OpenAI:
    """Client for the active model profile, or the env endpoint when the
    active model has no profile (or the profile specifies no base_url)."""
    name = model_name or models.resolved_model_name()
    profile = models.active_profile()
    base_url = profile.base_url if profile and profile.base_url else None
    if base_url is None:
        base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = profile.api_key if profile and profile.api_key else ""
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        # No explicit key: fall back to a stored OAuth credential for this
        # model family (claude* → anthropic, gpt*/codex* → openai-codex).
        from .oauth.access import token_for_model

        oauth_token = token_for_model(name)
        if oauth_token:
            api_key = oauth_token
    if not api_key:
        api_key = "unused"
    key = (base_url, api_key)
    cached = _clients.get(key)
    if cached is not None:
        return cached
    # No total-deadline: slow serverless endpoints stream at ~1 tok/s and
    # would trip a scalar total timeout. Per-phase timeouts keep failures
    # fast (connect/write/pool); read must be huge because the server can
    # sit silent for minutes on prefill before the first byte. The wall-
    # clock cap is enforced by HARD_TIMEOUT via the worker thread join.
    _clients[key] = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=Timeout(connect=15.0, read=900.0, write=15.0, pool=15.0),
        max_retries=2,
    )
    return _clients[key]


def model() -> str:
    return models.resolved_model_name()


def embed(texts: list[str], model_name: str | None = None) -> list[list[float]] | None:
    """Embed `texts` via an OpenAI-compatible ``/embeddings`` endpoint.

    Returns one vector per input (ordered by index), or None when the
    endpoint does not serve embeddings (many chat-only endpoints 404/HTML
    on ``/embeddings``). Callers MUST treat None as "neural retrieval
    unavailable — fall back to keyword". The embedding model defaults to
    ``PROVER_EMBED_MODEL`` (an embeddings model, e.g. ``text-embedding-3``)
    and falls back to the active chat model.
    """
    emb_model = model_name or os.environ.get("PROVER_EMBED_MODEL") or model()
    try:
        resp = client(emb_model).embeddings.create(model=emb_model, input=texts)
    except Exception:  # noqa: BLE001 — best-effort; None means "no embeddings here"
        return None
    data = getattr(resp, "data", None)
    if not data:
        return None
    data = sorted(data, key=lambda d: getattr(d, "index", 0))
    return [d.embedding for d in data]


def available_models(timeout: float = 15.0) -> list[str]:
    """Model names the configured endpoint actually serves (subject to its
    own request/timeout behavior). Empty when no endpoint is configured, the
    endpoint is unreachable, or the models route itself times out."""
    profile = models.active_profile()
    has_endpoint = bool((profile and profile.base_url) or os.environ.get("OPENAI_BASE_URL"))
    if not has_endpoint:
        return []
    try:
        raw = client().models.list(timeout=timeout)
    except Exception:  # noqa: BLE001 — best-effort
        return []
    return [m.id for m in getattr(raw, "data", [])]


def validate_model(model_name: str | None = None) -> str | None:
    """Return a hint string when the configured model is not served by the
    endpoint, or None when it is (or when no custom endpoint is configured).

    Turns a 180s hang into an immediate, actionable message (the endpoint's
    /models route is fast even when completions are stalled).
    """
    base = os.environ.get("OPENAI_BASE_URL")
    if not base:
        return None
    name = model_name or model()
    served = available_models()
    if not served:
        return f"endpoint {base} does not list any models (completions may still be stalling)"
    if name in served:
        return None
    return (f"model '{name}' is not served by {base}; available models: "
            + (", ".join(sorted(served)) or "none"))


# Rough context-window estimates per model family. Used by context_window.py to
# compute the automatic compaction budget (tau parity). Env override
# PROVER_CONTEXT_WINDOW short-circuits this lookup. Extended via tau_ai/model_limits.py.
_DEFAULT_CONTEXT_WINDOW_TOKENS = {
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_384,
    "qwen": 262_144,
    "claude": 200_000,
    "anthropic": 200_000,
    "gemini": 1_000_000,
    "gemma": 128_000,
    "mistral": 128_000,
    "deepseek": 128_000,
    "glm": 128_000,
}
DEFAULT_CONTEXT_WINDOW_TOKENS = 128_000
DEFAULT_COMPACTION_RESERVE_TOKENS = 16_384


def context_window_tokens(model_name: str | None = None) -> int:
    """Return the best-known context window for the active/default model."""
    override = os.environ.get("PROVER_CONTEXT_WINDOW")
    if override and override.isdigit():
        return int(override)
    name = model_name or model()
    profile = models.profile_for(name)
    if profile is not None and profile.context_window:
        return profile.context_window
    name = name.lower()
    for prefix, window in _DEFAULT_CONTEXT_WINDOW_TOKENS.items():
        if name == prefix or name.startswith(prefix):
            return window
    return DEFAULT_CONTEXT_WINDOW_TOKENS


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def _call(system: str, messages: list[dict], temperature: float, model_name: str | None = None) -> LLMResponse:
    from .thinking import reasoning_effort_for_level, thinking_level_from_env

    kwargs: dict = {
        "model": model_name or model(),
        "messages": [{"role": "system", "content": system}, *messages],
        "temperature": temperature,
        # Cap output: proofs are ~200-800 tokens; a generous ceiling keeps the
        # worst case (no EOS emitted) from burning hours on slow endpoints.
        "max_tokens": 4096,
    }
    level = thinking_level_from_env()
    if level == "off" and os.environ.get("OPENAI_BASE_URL"):
        # Thinking off (the prover default, agent/thinking.py): vLLM/HF
        # endpoints serving Qwen3+ put output in a `reasoning` field and
        # burn minutes when thinking is on without the chat-template
        # switch. Proofs are repaired by the compile loop, so fast
        # non-thinking mode wins.
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    elif level != "off":
        # Explicit thinking level (PROVER_THINKING): surface it as an
        # OpenAI-compatible reasoning effort (tau thinking.py → provider).
        kwargs["reasoning_effort"] = reasoning_effort_for_level(level)
    resp = client(model_name).chat.completions.create(**kwargs)
    usage = resp.usage
    content = resp.choices[0].message.content or ""
    if not content:
        content = getattr(resp.choices[0].message, "reasoning", None) or ""
    return LLMResponse(
        content=content,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
    )


class _CallResult:
    def __init__(self) -> None:
        self.response: LLMResponse | None = None
        self.error: str | None = None


def chat(
    system: str,
    messages: list[dict],
    temperature: float = 0.2,
    retries: int = 4,
    model_name: str | None = None,
) -> LLMResponse:
    """One LLM turn with a hard wall-clock cap and transient backoff (tau_ai/retry.py)."""
    from .provider_retry import is_transient_error, retry_delay_seconds

    def run(result: _CallResult) -> None:
        try:
            result.response = _call(system, messages, temperature, model_name)
        except BaseException as e:  # noqa: BLE001 — report any failure as an error response
            result.error = str(e)

    for attempt in range(retries + 1):
        result = _CallResult()
        worker = threading.Thread(target=run, args=(result,), daemon=True)
        worker.start()
        worker.join(timeout=HARD_TIMEOUT)
        if worker.is_alive():
            return LLMResponse(content=f"[LLM error: hard timeout after {HARD_TIMEOUT:.0f}s]")
        if result.error:
            msg = result.error
            if is_transient_error(msg) and attempt < retries:
                # Retry transient failures (429/5xx/connection/timeout) with
                # exponential backoff + jitter; deterministic errors fail fast.
                delay = retry_delay_seconds(attempt, max_delay_seconds=60.0)
                time.sleep(delay)
                continue
            return LLMResponse(content=f"[LLM error: {msg}]")
        return result.response  # type: ignore[return-value]
    return LLMResponse(content="[LLM error: retries exhausted]")


def extract_lean_code(text: str) -> str:
    """Pull the first ```lean block out of a model reply, or return as-is."""
    import re

    m = re.search(r"```(?:lean)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


# Rough cost estimates per 1M tokens (USD). Update as pricing changes.
# Covers common OpenAI-compatible endpoints. Free tiers = $0.
_COST_PER_1M = {
    "gpt-4o": (5.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "qwen/qwen3-8b": (0.0, 0.0),
    "qwen/qwen3-8b-max-free": (0.0, 0.0),
    "qwen/qwen3-235b": (0.0, 0.0),
    "qwen/qwen3-max-free": (0.0, 0.0),
    "Qwen/Qwen3-8B": (0.0, 0.0),
    "Qwen/Qwen3-27B": (0.0, 0.0),
    "Qwen/Qwen3-235B": (0.0, 0.0),
}

def estimate_cost(prompt_tokens: int, completion_tokens: int, model_name: str | None = None) -> float:
    """Estimate cost in USD for the given token counts."""
    model_name = model_name or model()
    # Profile cost overrides win when both halves are present.
    profile = models.profile_for(model_name)
    if profile is not None and profile.cost_in is not None and profile.cost_out is not None:
        return (prompt_tokens * profile.cost_in + completion_tokens * profile.cost_out) / 1_000_000
    # Exact match wins; else longest-prefix match so "gpt-4o-mini" is not
    # claimed by the shorter key "gpt-4o". Keys are sorted longest-first.
    for key, (in_cost, out_cost) in sorted(
        _COST_PER_1M.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        if model_name == key or model_name.startswith(key.rstrip('-*')):
            return (prompt_tokens * in_cost + completion_tokens * out_cost) / 1_000_000
    # Unknown model: assume free (conservative for budgeting)
    return 0.0
