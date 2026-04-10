"""Tests for the semantic query cache."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from inspect_assist.cache import SemanticCache, _cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_both_zero(self):
        assert _cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_high_similarity(self):
        a = [1.0, 0.1]
        b = [1.0, 0.0]
        sim = _cosine_similarity(a, b)
        assert sim > 0.99


class TestSemanticCache:
    def test_put_and_get_exact(self):
        cache = SemanticCache(similarity_threshold=0.95)
        emb = [1.0, 0.0, 0.0]
        results = [{"document": "test"}]
        cache.put("query", emb, results)
        assert cache.size == 1

        got = cache.get(emb)
        assert got is not None
        assert got == results

    def test_get_similar(self):
        cache = SemanticCache(similarity_threshold=0.95)
        cache.put("query", [1.0, 0.0, 0.0], [{"doc": "result"}])
        # Very similar embedding
        got = cache.get([0.999, 0.01, 0.0])
        assert got is not None

    def test_get_dissimilar_returns_none(self):
        cache = SemanticCache(similarity_threshold=0.95)
        cache.put("query", [1.0, 0.0, 0.0], [{"doc": "result"}])
        # Orthogonal embedding
        got = cache.get([0.0, 1.0, 0.0])
        assert got is None

    def test_get_empty_cache(self):
        cache = SemanticCache()
        assert cache.get([1.0, 0.0]) is None

    def test_invalidate_clears_cache(self):
        cache = SemanticCache()
        cache.put("q1", [1.0, 0.0], [{"a": 1}])
        cache.put("q2", [0.0, 1.0], [{"b": 2}])
        assert cache.size == 2
        cache.invalidate()
        assert cache.size == 0
        assert cache.get([1.0, 0.0]) is None

    def test_max_size_eviction(self):
        cache = SemanticCache(max_size=3, similarity_threshold=0.999)
        for i in range(5):
            # Use clearly distinct embeddings so they don't match each other
            emb = [0.0] * 10
            emb[i] = 1.0
            cache.put(f"q{i}", emb, [{"i": i}])
        assert cache.size == 3  # oldest 2 evicted

    def test_ttl_expiration(self):
        cache = SemanticCache(ttl_seconds=1, similarity_threshold=0.95)
        emb = [1.0, 0.0]
        cache.put("q", emb, [{"doc": "result"}])

        # Manually expire the entry by patching its timestamp
        key = list(cache._cache.keys())[0]
        cache._cache[key].timestamp = time.monotonic() - 2

        got = cache.get(emb)
        assert got is None
        assert cache.size == 0  # expired entry pruned

    def test_lru_order(self):
        cache = SemanticCache(max_size=2, similarity_threshold=0.999)
        cache.put("q1", [1.0, 0.0, 0.0], [{"a": 1}])
        cache.put("q2", [0.0, 1.0, 0.0], [{"b": 2}])
        # Access q1 to make it most recently used
        cache.get([1.0, 0.0, 0.0])
        # Add q3 — should evict q2 (least recently used)
        cache.put("q3", [0.0, 0.0, 1.0], [{"c": 3}])
        assert cache.size == 2
        # q1 should still be there
        assert cache.get([1.0, 0.0, 0.0]) is not None
        # q2 should be evicted
        assert cache.get([0.0, 1.0, 0.0]) is None
