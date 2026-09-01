from __future__ import annotations

import random

RETRY_BASE_DELAY_SECONDS = 0.25

# HTTP statuses / error markers that warrant a bounded retry (transient
# transport or provider-side failures), as opposed to a deterministic error
# that retrying would not fix (auth, 4xx validation, malformed request).
_TRANSIENT_STATUS = ("429", "500", "502", "503", "504")
_TRANSIENT_MARKERS = (
    "rate limit",
    "connection",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "server error",
    "overloaded",
    "try again",
)


def is_transient_error(message: str) -> bool:
    """Whether an error string describes a retryable transient failure.

    Matches 429/5xx status codes and common transport markers (connection,
    timeout, overload). Used to decide backoff+retry vs. an immediate
    deterministic failure.
    """
    msg = message.lower()
    if any(code in message for code in _TRANSIENT_STATUS):
        return True
    return any(marker in msg for marker in _TRANSIENT_MARKERS)


def retry_delay_seconds(attempt: int, *, max_delay_seconds: float) -> float:
    if max_delay_seconds <= 0:
        return 0.0
    base_delay = min(RETRY_BASE_DELAY_SECONDS, max_delay_seconds)
    delay = float(min(max_delay_seconds, base_delay * (2**attempt)))
    # Full jitter avoids thundering-herd alignment on shared endpoints while
    # keeping the worst case bounded by the exponential delay.
    return random.uniform(0.0, delay)
