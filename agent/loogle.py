from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request


def search_loogle(query: str, limit: int = 5, timeout: int = 10) -> list[dict]:
    query = query.strip()
    if not query:
        return []
    url = os.getenv("LOOGLE_API_URL", "https://loogle.lean-lang.org/json")
    params = urllib.parse.urlencode({"q": query})
    full = f"{url}?{params}"
    try:
        with urllib.request.urlopen(full, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        # Only network/parse failures degrade to the local fallback — never a
        # broad catch that hides a real bug behind a full index build.
        return _fallback_keyword(query, limit)
    hits: list[dict] = []
    if isinstance(data, dict) and "hits" in data:
        raw = data["hits"]
    elif isinstance(data, list):
        raw = data
    else:
        raw = []
    for h in raw[:limit]:
        if isinstance(h, dict):
            hits.append({"name": h.get("name", ""), "type": h.get("type", h.get("decl", "")), "doc": h.get("doc", "")[:300]})
        else:
            hits.append({"name": str(h), "type": ""})
    if not hits:
        return _fallback_keyword(query, limit)
    return hits


def _fallback_keyword(query: str, limit: int) -> list[dict]:
    try:
        from .retrieval import enabled, search_lemmas

        if not enabled():
            return []
        lemmas = search_lemmas(query, k=limit)
        return [{"name": name, "type": sig, "doc": ""} for name, sig in lemmas]
    except Exception:  # noqa: BLE001
        return []
