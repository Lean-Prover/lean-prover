"""Unit tests for the local Mathlib retrieval index (agent/retrieval.py).

The index is exercised against tiny synthetic "Mathlib" trees so tests stay
offline and fast; scoring/ranking and cache invalidation are the focus.
"""

from __future__ import annotations

from pathlib import Path

from agent import retrieval


def write_mathlib(lean_dir: Path, files: dict[str, str]) -> None:
    src = lean_dir / ".lake" / "packages" / "mathlib" / "Mathlib"
    for rel, text in files.items():
        f = src / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")


PRIME = """import Mathlib

theorem prime_mul_iff {a b : ℕ} : Nat.Prime (a * b) ↔ a.Prime ∧ b = 1 ∨ b.Prime ∧ a = 1 := by
  sorry

lemma prime_dvd_mul_iff {p a b : ℕ} : p.Prime → p ∣ a * b → p ∣ a ∨ p ∣ b := by
  sorry

theorem Nat.Prime.isPrimePow {p : ℕ} (hp : p.Prime) : IsPrimePow p := by
  sorry
"""

GCD = """import Mathlib

theorem gcd_mul_lcm (a b : ℕ) : Nat.gcd a b * Nat.lcm a b = a * b := by
  sorry

def SomeDef (x : ℕ) : ℕ where
  toFun := x
"""

FIB = """import Mathlib

theorem fib_add (m n : ℕ) : Nat.fib (m + n + 1) = Nat.fib m * Nat.fib n + Nat.fib (m + 1) * Nat.fib (n + 1) := by
  sorry
"""

FILES = {"Data/Nat/Prime/Basic.lean": PRIME, "Data/Nat/GCD.lean": GCD, "Data/Nat/Fib.lean": FIB}


def test_build_index_extracts_decls(tmp_path: Path) -> None:
    write_mathlib(tmp_path, FILES)
    idx = retrieval.build_index(tmp_path)
    names = {e["name"] for e in idx}
    assert {"prime_mul_iff", "prime_dvd_mul_iff", "gcd_mul_lcm", "fib_add", "Nat.Prime.isPrimePow"} <= names
    # `def ... where` bodies are not lemmas — skipped
    assert "SomeDef" not in names


def test_build_index_skips_import_lines_in_signature(tmp_path: Path) -> None:
    write_mathlib(tmp_path, FILES)
    idx = retrieval.build_index(tmp_path)
    fib = next(e for e in idx if e["name"] == "fib_add")
    assert ":= by" not in fib["signature"]
    assert "Nat.fib (m + n + 1) = Nat.fib m * Nat.fib n" in fib["signature"]


def test_search_ranks_relevant_lemma_first(tmp_path: Path) -> None:
    write_mathlib(tmp_path, FILES)
    idx = retrieval.build_index(tmp_path)
    stmt = "theorem prover_x (p a b : ℕ) : Nat.Prime p → p ∣ a * b → p ∣ a ∨ p ∣ b"
    hits = retrieval.search_lemmas(stmt, k=3, lean_dir=tmp_path, index=idx)
    assert hits, "expected at least one hit"
    assert hits[0]["name"] == "prime_dvd_mul_iff"


def test_search_respects_k(tmp_path: Path) -> None:
    write_mathlib(tmp_path, FILES)
    idx = retrieval.build_index(tmp_path)
    hits = retrieval.search_lemmas("Nat.fib m * Nat.fib n + Nat.fib (m + 1)", k=1, index=idx)
    assert len(hits) == 1
    assert hits[0]["name"] == "fib_add"


def test_search_returns_empty_for_unrelated_statement(tmp_path: Path) -> None:
    write_mathlib(tmp_path, FILES)
    idx = retrieval.build_index(tmp_path)
    hits = retrieval.search_lemmas("theorem xyz (q : Prop) : q → q", index=idx,
                                   lean_dir=tmp_path)
    assert hits == []


def test_cache_rebuilt_on_version_or_tree_change(tmp_path: Path, monkeypatch) -> None:
    write_mathlib(tmp_path, FILES)
    idx = retrieval.load_index(tmp_path)
    assert len(idx) == 5
    cache = tmp_path / "tmp" / "lemma_index.json"
    assert cache.exists()
    # tree change (new file) invalidates the cache
    write_mathlib(tmp_path, {"Data/Nat/Extra.lean": "theorem extra (n : ℕ) : n = n := by rfl\n"})
    idx2 = retrieval.load_index(tmp_path)
    assert {e["name"] for e in idx2} == {"prime_mul_iff", "prime_dvd_mul_iff", "gcd_mul_lcm", "fib_add", "Nat.Prime.isPrimePow", "extra"}
    # version bump forces a rebuild even without a tree change
    monkeypatch.setattr(retrieval, "CACHE_VERSION", 999)
    idx3 = retrieval.load_index(tmp_path)
    assert {e["name"] for e in idx3} == {e["name"] for e in idx2}


def test_enabled_flag(monkeypatch) -> None:
    assert retrieval.enabled() is False
    monkeypatch.setenv("PROVER_RETRIEVE", "1")
    assert retrieval.enabled() is True
    monkeypatch.delenv("PROVER_RETRIEVE")
    monkeypatch.setenv("PROVER_EMBED", "1")
    assert retrieval.enabled() is True


# ---- neural (embedding) retrieval ----

_VOCAB = ["prime", "gcd", "fib", "mul", "dvd", "nat", "add", "comm"]


def _stub_embed(texts: list[str]) -> list[list[float]] | None:
    """Deterministic bag-of-words vectors over _VOCAB (offline stub)."""
    return [[1.0 if w in t.lower() else 0.0 for w in _VOCAB] for t in texts]


def test_cosine_similarity() -> None:
    import pytest as _pytest

    a = [1.0, 0.0, 1.0]
    assert retrieval._cosine(a, a) == _pytest.approx(1.0)
    assert retrieval._cosine(a, [0.0, 1.0, 0.0]) == _pytest.approx(0.0)
    # orthogonal and same-length: similarity is normalized
    assert 0.0 < retrieval._cosine([1.0, 1.0], [1.0, 0.0]) < 1.0


def test_search_lemmas_embed_ranks_by_similarity(tmp_path: Path, monkeypatch) -> None:
    write_mathlib(tmp_path, FILES)
    monkeypatch.setenv("PROVER_EMBED", "1")
    # gcd_mul_lcm mentions gcd/mul/nat; the query shares those tokens.
    hits = retrieval.search_lemmas_embed(
        "Nat.gcd a b * Nat.lcm a b = a * b", k=3, lean_dir=tmp_path, embed=_stub_embed)
    assert hits and hits[0]["name"] == "gcd_mul_lcm"


def test_search_lemmas_embed_unavailable_falls_back_to_keyword(tmp_path, monkeypatch) -> None:
    write_mathlib(tmp_path, FILES)
    monkeypatch.setenv("PROVER_EMBED", "1")
    # Embedder returns None (chat-only endpoint): neural path yields [] and
    # search_lemmas must fall back to keyword ranking, not raise.
    hits = retrieval.search_lemmas(
        "theorem prime_mul_iff {a b : ℕ} : Nat.Prime (a * b) ↔ ...",
        k=3, lean_dir=tmp_path,
        index=retrieval.load_index(tmp_path))
    assert hits  # keyword path still returns relevant lemmas
    assert hits[0]["name"] == "prime_mul_iff"


def test_load_index_recovers_from_corrupt_cache(tmp_path: Path) -> None:
    write_mathlib(tmp_path, FILES)
    cache = tmp_path / "tmp" / "lemma_index.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("{{{ not json", encoding="utf-8")
    idx = retrieval.load_index(tmp_path)
    assert len(idx) == 5


def _write_corpus(tmp_path: Path, id_: str, statement: str, tactic: str) -> None:
    import json

    corpus = tmp_path.parent / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    f = corpus / "lean_proved.jsonl"
    entry = json.dumps({"id": id_, "difficulty": "trivial",
                        "statement": statement, "tactic": tactic})
    with f.open("a", encoding="utf-8") as fh:
        fh.write(entry + "\n")


def test_load_corpus_parses_entries(tmp_path: Path) -> None:
    write_mathlib(tmp_path, FILES)
    _write_corpus(tmp_path, "tpl_add_comm",
                  "theorem prover_tpl_add_comm (a b : ℕ) : a + b = b + a := by\n  sorry",
                  "prover_finish")
    entries = retrieval.load_corpus(tmp_path)
    assert len(entries) == 1
    e = entries[0]
    assert e["name"] == "tpl_add_comm"
    assert e["proof"] == "prover_finish"
    assert e["file"] == "corpus"
    assert "a + b = b + a" in e["signature"]
    assert ":=" not in e["signature"]


def test_search_includes_corpus_hints(tmp_path: Path) -> None:
    write_mathlib(tmp_path, FILES)
    _write_corpus(tmp_path, "tpl_add_comm",
                  "theorem prover_tpl_add_comm (a b : ℕ) : a + b = b + a := by\n  sorry",
                  "prover_finish")
    idx = retrieval.build_index(tmp_path)
    hits = retrieval.search_lemmas(
        "theorem prover_x (a b : ℕ) : a + b = b + a", k=3,
        lean_dir=tmp_path, index=idx)
    assert any(h["name"] == "tpl_add_comm" and h["proof"] == "prover_finish"
               for h in hits)


def test_load_corpus_missing_file_returns_empty(tmp_path: Path) -> None:
    bare = tmp_path / "no_corpus_here"
    bare.mkdir()
    assert retrieval.load_corpus(bare) == []


def test_corpus_append_writes_dedupe_and_loads(tmp_path: Path) -> None:
    # Nest so corpus_path (lean_dir.parent / "corpus") stays test-local.
    lean_dir = tmp_path / "proj" / "lean"
    write_mathlib(lean_dir, FILES)
    stmt = "theorem auto_proved (a b c : ℕ) (h : a ∣ b) : a ∣ b * c := by sorry"
    assert retrieval.corpus_append(lean_dir, stmt, "  prover_finish") is True
    assert retrieval.corpus_append(lean_dir, stmt, "  prover_finish") is False
    entries = retrieval.load_corpus(lean_dir)
    assert len(entries) == 1
    assert entries[0]["proof"] == "prover_finish"
    assert "a ∣ b" in entries[0]["signature"]


def test_corpus_append_stores_llm_tactic_text(tmp_path: Path) -> None:
    lean_dir = tmp_path / "proj2" / "lean"
    lean_dir.mkdir(parents=True)
    stmt = "theorem auto2 (n : ℕ) : n + 0 = n := by sorry"
    proof = "by\n  induction n\n  · rfl\n  · simp_all [Nat.succ_add]"
    assert retrieval.corpus_append(lean_dir, stmt, proof) is True
    entries = retrieval.load_corpus(lean_dir)
    assert "simp_all [Nat.succ_add]" in entries[0]["proof"]
