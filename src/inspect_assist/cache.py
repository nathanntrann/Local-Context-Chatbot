"""Semantic query cache — avoids redundant retrieval for similar queries."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class CacheEntry:
    query: str
    embedding: list[float]
    results: list[dict[str, Any]]
    timestamp: float = field(default_factory=time.monotonic)


class SemanticCache:
    """LRU cache keyed by query embedding similarity.

    If a new query's embedding has cosine similarity > threshold to a cached
    query, the cached results are returned instead of re-searching.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600,
        max_size: int = 200,
    ) -> None:
        self._threshold = similarity_threshold
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, query_embedding: list[float]) -> list[dict[str, Any]] | None:
        """Check cache for a sufficiently similar query. Returns results or None."""
        now = time.monotonic()
        expired_keys = []

        for key, entry in self._cache.items():
            # Check TTL
            if now - entry.timestamp > self._ttl:
                expired_keys.append(key)
                continue
            # Check similarity
            sim = _cosine_similarity(query_embedding, entry.embedding)
            if sim >= self._threshold:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                logger.debug("semantic_cache_hit", query=entry.query, similarity=sim)
                return entry.results

        # Prune expired entries
        for key in expired_keys:
            del self._cache[key]

        return None

    def put(
        self,
        query: str,
        query_embedding: list[float],
        results: list[dict[str, Any]],
    ) -> None:
        """Store query results in the cache."""
        # Evict oldest if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        key = f"{hash(query)}_{time.monotonic()}"
        self._cache[key] = CacheEntry(
            query=query,
            embedding=query_embedding,
            results=results,
        )
        logger.debug("semantic_cache_put", query=query, cache_size=len(self._cache))

    def invalidate(self) -> None:
        """Clear the entire cache (e.g., after knowledge re-indexing)."""
        self._cache.clear()
        logger.info("semantic_cache_invalidated")

    @property
    def size(self) -> int:
        return len(self._cache)
