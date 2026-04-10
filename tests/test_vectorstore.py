"""Tests for the ChromaDB vector store wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from inspect_assist.vectorstore import VectorStore


class TestVectorStore:
    def test_init_creates_collections(self, tmp_path: Path):
        store = VectorStore(tmp_path / "vs")
        assert store.chunk_count == 0
        assert store.parent_count == 0

    def test_upsert_and_query_chunks(self, tmp_path: Path):
        store = VectorStore(tmp_path / "vs")
        ids = ["c1", "c2", "c3"]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        documents = ["about thermal seals", "about calibration", "about thresholds"]
        metadatas = [
            {"article_slug": "seals", "category": "concepts"},
            {"article_slug": "calibration", "category": "procedures"},
            {"article_slug": "thresholds", "category": "parameters"},
        ]

        store.upsert_chunks(ids, embeddings, documents, metadatas)
        assert store.chunk_count == 3

        # Query with embedding close to c1
        results = store.query([0.9, 0.1, 0.0], n_results=2)
        assert len(results) == 2
        assert results[0]["id"] == "c1"
        assert results[0]["document"] == "about thermal seals"
        assert results[0]["metadata"]["article_slug"] == "seals"

    def test_query_with_where_filter(self, tmp_path: Path):
        store = VectorStore(tmp_path / "vs")
        store.upsert_chunks(
            ids=["c1", "c2"],
            embeddings=[[1.0, 0.0], [1.0, 0.0]],
            documents=["doc1", "doc2"],
            metadatas=[{"category": "concepts"}, {"category": "procedures"}],
        )
        results = store.query([1.0, 0.0], n_results=5, where={"category": "procedures"})
        assert len(results) == 1
        assert results[0]["id"] == "c2"

    def test_upsert_and_get_parents(self, tmp_path: Path):
        store = VectorStore(tmp_path / "vs")
        store.upsert_parents(
            ids=["p1", "p2"],
            documents=["parent content 1", "parent content 2"],
            metadatas=[{"article_slug": "a"}, {"article_slug": "b"}],
        )
        assert store.parent_count == 2

        results = store.get_parents(["p2", "p1"])
        assert len(results) == 2
        docs = {r["id"]: r["document"] for r in results}
        assert docs["p1"] == "parent content 1"
        assert docs["p2"] == "parent content 2"

    def test_get_parents_deduplicates(self, tmp_path: Path):
        store = VectorStore(tmp_path / "vs")
        store.upsert_parents(
            ids=["p1"],
            documents=["content"],
            metadatas=[{"slug": "a"}],
        )
        results = store.get_parents(["p1", "p1", "p1"])
        assert len(results) == 1

    def test_get_parents_empty_list(self, tmp_path: Path):
        store = VectorStore(tmp_path / "vs")
        assert store.get_parents([]) == []

    def test_get_parents_with_none_ids(self, tmp_path: Path):
        store = VectorStore(tmp_path / "vs")
        store.upsert_parents(ids=["p1"], documents=["c"], metadatas=[{"s": "a"}])
        results = store.get_parents([None, "p1", None])
        assert len(results) == 1

    def test_upsert_empty_does_nothing(self, tmp_path: Path):
        store = VectorStore(tmp_path / "vs")
        store.upsert_chunks([], [], [], [])
        store.upsert_parents([], [], [])
        assert store.chunk_count == 0
        assert store.parent_count == 0

    def test_delete_all(self, tmp_path: Path):
        store = VectorStore(tmp_path / "vs")
        store.upsert_chunks(["c1"], [[1.0]], ["doc"], [{"k": "v"}])
        store.upsert_parents(["p1"], ["parent"], [{"k": "v"}])
        assert store.chunk_count == 1
        assert store.parent_count == 1

        store.delete_all()
        assert store.chunk_count == 0
        assert store.parent_count == 0

    def test_upsert_overwrites_existing(self, tmp_path: Path):
        store = VectorStore(tmp_path / "vs")
        store.upsert_chunks(["c1"], [[1.0, 0.0]], ["original"], [{"v": "1"}])
        assert store.chunk_count == 1

        store.upsert_chunks(["c1"], [[0.0, 1.0]], ["updated"], [{"v": "2"}])
        assert store.chunk_count == 1

        results = store.query([0.0, 1.0], n_results=1)
        assert results[0]["document"] == "updated"

    def test_query_empty_store_returns_empty(self, tmp_path: Path):
        store = VectorStore(tmp_path / "vs")
        results = store.query([1.0, 0.0, 0.0], n_results=5)
        assert results == []
