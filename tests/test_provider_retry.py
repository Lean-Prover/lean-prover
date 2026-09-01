"""agent.provider_retry — transient-error classification + backoff tests."""

from __future__ import annotations

from agent.provider_retry import is_transient_error, retry_delay_seconds


def test_transient_statuses() -> None:
    assert is_transient_error("429 rate limit exceeded")
    assert is_transient_error("500 Internal Server Error")
    assert is_transient_error("502 Bad Gateway")
    assert is_transient_error("503 Service Unavailable")
    assert is_transient_error("504 Gateway Timeout")


def test_transient_markers() -> None:
    assert is_transient_error("Connection reset by peer")
    assert is_transient_error("Read timed out")
    assert is_transient_error("server is temporarily unavailable")


def test_deterministic_errors_are_not_transient() -> None:
    assert not is_transient_error("401 Unauthorized")
    assert not is_transient_error("400 Bad Request: malformed JSON")
    assert not is_transient_error("404 Not Found")
    assert not is_transient_error("invalid api key")


def test_backoff_is_bounded_and_jittered() -> None:
    delays = [retry_delay_seconds(i, max_delay_seconds=60.0) for i in range(6)]
    # All delays stay within [0, cap].
    assert all(0.0 <= d <= 60.0 for d in delays)
    # Backoff grows with attempt count (upper bound at least doubles).
    assert delays[5] >= delays[0]
    # Jitter: two samples of the same attempt differ (with high probability).
    a = retry_delay_seconds(3, max_delay_seconds=60.0)
    b = retry_delay_seconds(3, max_delay_seconds=60.0)
    assert a != b


def test_backoff_zero_cap() -> None:
    assert retry_delay_seconds(3, max_delay_seconds=0.0) == 0.0
