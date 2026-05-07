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

        1. Exact match
        2. Bigram-precision partial match (≥ 0.7)
           バイグラム精度 = クエリのバイグラムのうち辞典品名に含まれる割合
           これにより「インスタントラーメン容器」→「インスタントラーメン袋・容器」が
           「インスタントラーメン（生ごみ）」より高スコアになる。
        """
        q = _norm(query)
        if not q:
            return None

        # 1. Exact match
        for item in self.items:
            if _norm(item.get("name", "")) == q or _norm(item.get("name_reduced", "")) == q:
                return self._enrich(item)

        # 2. Bigram-precision partial match
        result, score = self._best_bigram_match(q, threshold=0.7)
        return self._enrich(result) if result else None

    def get_suggestions(self, query: str, limit: int = 3) -> list[dict]:
        """検索ゼロヒット時に近い品目を返す（バイグラム精度 ≥ 0.4）。
        同じ分別種類の重複を避けて上位 limit 件を返す。"""
        q = _norm(query)
        if len(q) < 2:
            return []

        q_bigrams = frozenset(q[i:i + 2] for i in range(len(q) - 1))
        scored: list[tuple[float, dict]] = []

        for item in self.items:
            score = self._precision(q_bigrams, item)
            if score >= 0.4:
                scored.append((score, item))

        scored.sort(key=lambda x: -x[0])

        seen_types: set[str] = set()
        results: list[dict] = []
        for _, item in scored:
            tid = item.get("typeID", "")
            if tid not in seen_types:
                results.append(self._enrich(item))
                seen_types.add(tid)
            if len(results) >= limit:
                break

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _best_bigram_match(
        self, q: str, threshold: float
    ) -> tuple[Optional[dict], float]:
        """クエリのバイグラム精度で全品目をスコアリングし最高スコアの品目を返す。"""
        if len(q) < 2:
            return None, 0.0

        q_bigrams = frozenset(q[i:i + 2] for i in range(len(q) - 1))
        best_score = 0.0
        best_item: Optional[dict] = None

        for item in self.items:
            score = self._precision(q_bigrams, item)
            if score > best_score:
                best_score = score
                best_item = item

        if best_item is not None and best_score >= threshold:
            return best_item, best_score
        return None, best_score

    @staticmethod
    def _precision(q_bigrams: frozenset, item: dict) -> float:
        """クエリのバイグラムのうち品目名に含まれる割合（精度）を返す。"""
        if not q_bigrams:
            return 0.0
        best = 0.0
        for field in ("name", "name_reduced"):
            n = _norm(item.get(field, ""))
            if len(n) >= 2:
                t_bigrams = frozenset(n[i:i + 2] for i in range(len(n) - 1))
                precision = len(q_bigrams & t_bigrams) / len(q_bigrams)
                if precision > best:
                    best = precision
        return best

    def _enrich(self, item: dict) -> dict:
        type_info = self.types.get(item.get("typeID", ""), {})
        return {**item, "type_info": type_info}
