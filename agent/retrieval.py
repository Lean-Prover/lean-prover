"""Local Mathlib lemma retrieval for proof-time hints.

Builds a lightweight index of top-level ``theorem``/``lemma``/``def``
signatures from the pinned Mathlib checkout and scores them against a target
statement by keyword overlap. This is deliberately *not* an embedding model:
it is fast, deterministic, offline, and honest about being a keyword matcher
(surrogate for mathlib ``exact?``-style search without the server).

Usage::

    import agent.retrieval as r
    hints = r.search_lemmas("theorem prover_foo (p n : ℕ) : Nat.Prime p → p ∣ n → p ∣ n ^ 2", k=5)

The index is built lazily and cached at ``<lean_dir>/tmp/lemma_index.json``;
rebuilds automatically when the mathlib tree changes.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

DECL_RE = re.compile(
    r"^(?P<kw>theorem|lemma|def)\s+(?P<name>[A-Za-z0-9_'.]+)"
)
# Unicode-aware: captures ℕ/ℝ/ℤ etc. (word chars) plus the divisibility
# symbol, which distinguishes `p ∣ a`-style lemmas from generic ones.
TOKEN_RE = re.compile(r"[\w']+|∣")

# Declarative keywords appear in both statement and signature — they would
# match everything, so they are stripped from both sides.
STOPWORDS = {"theorem", "lemma", "def", "example", "instance", "prover", "by"}

# Identifiers/types that make a lemma likely relevant when they co-occur.
KEY_TOKENS = {
    "ℕ", "ℤ", "ℝ", "ℂ", "ℚ", "Prime", "Nat", "Int", "Real", "Complex",
    "Finset", "List", "Fib", "gcd", "sqrt", "choose", "factorial",
    "dvd", "Even", "Odd", "Sum", "∣",
}

DEFAULT_K = 5
SIG_CAP = 250  # chars of signature kept per declaration
INDEX_META_KEY = "_meta"
CACHE_VERSION = 2  # bump to force a rebuild when the extractor changes


def _signature_for(path: Path) -> list[dict]:
    """Yield {name, signature} for top-level declarations in one file."""
    out = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = DECL_RE.match(line)
        if not m or line.startswith((" ", "\t")):
            i += 1
            continue
        # Signature = this line plus continuation lines until `:=` (bounded).
        sig_lines = [line]
        j = i + 1
        while j < n and j - i <= 6 and ":=" not in "".join(sig_lines):
            sig_lines.append(lines[j])
            j += 1
        sig = " ".join(l.strip() for l in sig_lines if l.strip())[:SIG_CAP]
        if ":=" in sig:
            sig = sig[: sig.index(":=")]
        # Skip field-style definitions (`where`) — they are not lemmas.
        if " where" in sig:
            i = j
            continue
        out.append({"name": m.group("name"), "signature": sig.strip(), "file": str(path)})
        i = j if j > i else i + 1
    return out


def _index_stamp(lean_dir: Path) -> tuple[int, int]:
    """(file_count, total_bytes) of the Mathlib tree.

    Not memoized: it is the sentinel that invalidates the disk index cache, so
    it must reflect the current tree (a new/removed file changes count/bytes).
    """
    src = lean_dir / ".lake" / "packages" / "mathlib" / "Mathlib"
    total = 0
    count = 0
    if src.is_dir():
        for p in src.rglob("*.lean"):
            count += 1
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return count, total


def _cache_path(lean_dir: Path) -> Path:
    tmp = lean_dir / "tmp"
    tmp.mkdir(exist_ok=True)
    return tmp / "lemma_index.json"


def build_index(lean_dir: Path) -> list[dict]:
    """Scan Mathlib once; return the full lemma list (no caching)."""
    src = lean_dir / ".lake" / "packages" / "mathlib" / "Mathlib"
    if not src.is_dir():
        return []
    entries: list[dict] = []
    for p in sorted(src.rglob("*.lean")):
        try:
            entries.extend(_signature_for(p))
        except OSError:
            continue
    return entries


def load_index(lean_dir: Path) -> list[dict]:
    """Load the cached index, rebuilding when the Mathlib tree changed."""
    cache = _cache_path(lean_dir)
    stamp = _index_stamp(lean_dir)
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(data, list):
                meta = next((d for d in data if isinstance(d, dict) and d.get(INDEX_META_KEY)), None)
                if meta and meta[INDEX_META_KEY] == {"count": stamp[0], "bytes": stamp[1], "version": CACHE_VERSION}:
                    return [e for e in data if isinstance(e, dict) and e.get("name")]
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    entries = build_index(lean_dir)
    data = [dict(e) for e in entries]
    data.append({INDEX_META_KEY: {"count": stamp[0], "bytes": stamp[1], "version": CACHE_VERSION}})
    cache.write_text(json.dumps(data), encoding="utf-8")
    return entries


def _tokens(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall(text) if t not in STOPWORDS}


def _score(signature: str, target: set[str]) -> float:
    """Keyword-overlap score: shared tokens, weighted for key tokens.

    Long signatures are normalized so a single strong keyword match on a
    short lemma ranks higher than a dozen incidental matches on a giant one.
    """
    sig_tokens = _tokens(signature)
    if not sig_tokens:
        return 0.0
    shared = sig_tokens & target
    weighted = sum(2.0 if t in KEY_TOKENS else 1.0 for t in shared)
    # Prefer exact identifier matches over generic words.
    exact = sum(1.0 for t in shared if t in target and len(t) > 3)
    return (weighted + exact) / (len(sig_tokens) ** 0.5)


def corpus_path(lean_dir: Path) -> Path:
    """The Lean-proved corpus JSONL for a checkout (``<repo>/corpus/``)."""
    return lean_dir.parent / "corpus" / "lean_proved.jsonl"


_corpus_seen_cache: dict[str, set[str]] = {}


def corpus_append(lean_dir: Path, statement: str, proof: str) -> bool:
    """Record a freshly proved theorem in the corpus (idempotent by id).

    Called by the loop after every successful proof, so the corpus grows by
    use. The stored tactic column is the native tactic when the file was
    closed by a pre-pass, otherwise the verified tactic text. Best-effort:
    never raises into the proof loop.

    The seen-id set is cached in memory and updated on each append, so a
    long proving session does not re-read the whole corpus (O(n)) per solved
    proof.
    """
    import hashlib

    from .synth_lean import append_lines

    m = re.search(r"(?:theorem|lemma|def|example)\s+([A-Za-z0-9_'.]+)", statement)
    base = m.group(1) if m else "auto"
    name = base + "_" + hashlib.sha1(statement.encode("utf-8")).hexdigest()[:8]
    header = statement.split(":=")[0].strip()
    if "prover_finish" in proof or "prover_search" in proof:
        tag = "prover_finish" if "prover_finish" in proof else "prover_search"
    else:
        tag = proof.strip()
    path = corpus_path(lean_dir)
    key = str(path)
    seen = _corpus_seen_cache.get(key)
    if seen is None:
        seen = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    seen.add(json.loads(line).get("id", ""))
                except json.JSONDecodeError:
                    continue
        _corpus_seen_cache[key] = seen
    if name in seen:
        return False
    append_lines([
        {"id": name, "difficulty": "auto", "statement": header, "tactic": tag},
    ], path)
    seen.add(name)
    return True


def load_corpus(lean_dir: Path) -> list[dict]:
    """Lean-proved corpus entries as searchable index items.

    Reads ``<repo>/corpus/lean_proved.jsonl`` (written by ``prover
    synth-lean`` and grown by the loop after each solved proof) so the loop
    can surface statements Lean already proved as retrieval hints, tagged
    with the tactic that closed them. Missing or unparseable corpus files
    are simply skipped — retrieval stays best-effort.
    """
    f = corpus_path(lean_dir)
    if not f.exists():
        return []
    out: list[dict] = []
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        stmt = str(e.get("statement", ""))
        sig = stmt.split(":=")[0].strip()[:SIG_CAP]
        if not stmt:
            continue
        out.append({
            "name": str(e.get("id", "?")),
            "signature": sig,
            "file": "corpus",
            "proof": str(e.get("tactic", "")),
        })
    return out


def search_lemmas(
    statement: str,
    k: int = DEFAULT_K,
    lean_dir: Path | None = None,
    index: list[dict] | None = None,
) -> list[dict]:
    """Return the top-k most relevant Mathlib lemma signatures for a statement."""
    from .loop import LEAN_DIR

    lean_dir = lean_dir or LEAN_DIR
    idx = list(index) if index is not None else load_index(lean_dir)
    idx += load_corpus(lean_dir)
    target = _tokens(statement)
    scored = [(e, _score(e["signature"], target)) for e in idx]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    best = [e for e, s in scored[:k] if s > 0]
    return best


def enabled() -> bool:
    """True when proof-time retrieval is on (PROVER_RETRIEVE=1)."""
    return os.getenv("PROVER_RETRIEVE") == "1"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 1:
        print("usage: python -m agent.retrieval '<theorem statement>' [k]")
        return 2
    stmt = args[0]
    k = int(args[1]) if len(args) > 1 else DEFAULT_K
    for hit in search_lemmas(stmt, k=k):
        print(f"{hit['name']}  [{hit['file']}]")
        print(f"    {hit['signature']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
