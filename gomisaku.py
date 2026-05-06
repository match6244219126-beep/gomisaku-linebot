"""Gomisaku data loader and search module.

Data source: https://www.gomisaku.jp/0372/ja/dictionary.js
             https://www.gomisaku.jp/0372/ja/type.js
"""

import json
import re
import unicodedata
from typing import Optional

import httpx

MUNICIPALITY_CODE = "0372"
_BASE = f"https://www.gomisaku.jp/{MUNICIPALITY_CODE}/ja"
_DICT_URL = f"{_BASE}/dictionary.js"
_TYPE_URL = f"{_BASE}/type.js"


async def _fetch_js(url: str) -> dict:
    """Fetch a gomisaku JS data file and return the embedded JSON."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()

    # Files look like: gomisakuGetData('type', { ... });
    match = re.search(r"gomisakuGetData\('[^']+',(.+)\);\s*$", resp.text, re.DOTALL)
    if not match:
        raise ValueError(f"Unexpected format in {url}")
    return json.loads(match.group(1))


def _extract_rows(data: dict) -> list[dict]:
    """Convert the plist-style array of key/string pairs into plain dicts."""
    rows = []
    for entry in data.get("array", {}).get("dict", []):
        keys = entry.get("key", [])
        values = entry.get("string", [])
        if not isinstance(values, list) or len(keys) != len(values):
            continue
        rows.append({k: (v if isinstance(v, str) else "") for k, v in zip(keys, values)})
    return rows


def _norm(text: str) -> str:
    """NFKC-normalise and lower-case for comparison."""
    return unicodedata.normalize("NFKC", text).lower().strip()


class GomisakuDB:
    def __init__(self) -> None:
        self.items: list[dict] = []
        self.types: dict[str, dict] = {}

    async def load(self) -> None:
        """Download and parse dictionary + type data."""
        dict_data = await _fetch_js(_DICT_URL)
        type_data = await _fetch_js(_TYPE_URL)

        self.items = _extract_rows(dict_data)
        for t in _extract_rows(type_data):
            if tid := t.get("typeID"):
                self.types[tid] = t

    def search(self, query: str) -> Optional[dict]:
        """Search for a garbage item by name.

        Returns an item dict enriched with 'type_info', or None if not found.
        Tries exact match first, then longest partial match (≥40 % overlap).
        """
        q = _norm(query)
        if not q:
            return None

        # 1. Exact match
        for item in self.items:
            if _norm(item.get("name", "")) == q or _norm(item.get("name_reduced", "")) == q:
                return self._enrich(item)

        # 2. Partial match — pick the candidate with the highest overlap ratio
        best_score = 0.0
        best_item: Optional[dict] = None

        for item in self.items:
            name = _norm(item.get("name", ""))
            name_r = _norm(item.get("name_reduced", ""))

            score = 0.0
            if name and q in name:
                score = max(score, len(q) / len(name))
            if name_r and q in name_r:
                score = max(score, len(q) / len(name_r))
            if name and name in q:
                score = max(score, len(name) / len(q))
            if name_r and name_r in q:
                score = max(score, len(name_r) / len(q))

            if score > best_score:
                best_score = score
                best_item = item

        if best_item is not None and best_score >= 0.4:
            return self._enrich(best_item)

        return None

    def _enrich(self, item: dict) -> dict:
        type_info = self.types.get(item.get("typeID", ""), {})
        return {**item, "type_info": type_info}
