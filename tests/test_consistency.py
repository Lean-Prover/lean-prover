"""Cross-module consistency: shared HEADER + loogle offline fallback."""

from __future__ import annotations


def test_all_writers_share_canonical_header() -> None:
    from agent import formalize, lean_baseline, prover_loop, search

    assert formalize.HEADER == prover_loop.HEADER
    assert search.HEADER == prover_loop.HEADER
    assert lean_baseline.HEADER == prover_loop.HEADER
    assert "ProverSupport" in prover_loop.HEADER
    assert "import Mathlib" in prover_loop.HEADER


def test_loogle_empty_query_short_circuits() -> None:
    from agent import loogle

    assert loogle.search_loogle("") == []
    assert loogle.search_loogle("   ") == []


def test_loogle_network_failure_falls_back_or_empty(monkeypatch) -> None:
    import urllib.error

    from agent import loogle

    def boom(*_a, **_k):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(loogle.urllib.request, "urlopen", boom)
    # Retrieval disabled in test env (no PROVER_RETRIEVE) -> [] ; if an
    # index exists it returns keyword hits. Either way it must not raise.
    out = loogle.search_loogle("Nat.Prime", limit=3)
    assert isinstance(out, list)
    assert len(out) <= 3
