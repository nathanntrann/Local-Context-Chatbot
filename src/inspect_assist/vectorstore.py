"""ChromaDB vector store wrapper for knowledge chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class VectorStore:
    """Persistent ChromaDB store with two collections: small chunks (search) and parent chunks (context)."""

    def __init__(self, persist_dir: Path | str) -> None:
        import chromadb

        self._persist_dir = str(persist_dir)
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._chunks_col = self._client.get_or_create_collection(
            name="knowledge_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        self._parents_col = self._client.get_or_create_collection(
            name="knowledge_parents",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("vectorstore_initialized", path=self._persist_dir)

    def upsert_chunks(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Upsert small (search) chunks with their embeddings."""
        if not ids:
            return
        # ChromaDB has a batch limit; chunk into groups of 5000
        for i in range(0, len(ids), 5000):
            self._chunks_col.upsert(
                ids=ids[i : i + 5000],
                embeddings=embeddings[i : i + 5000],
                documents=documents[i : i + 5000],
                metadatas=metadatas[i : i + 5000],
            )
        logger.info("chunks_upserted", count=len(ids))

    def upsert_parents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Upsert parent (context) chunks — stored without embeddings for lookup by ID."""
        if not ids:
            return
        # Parents are stored as documents for ID-based lookup, no embedding needed
        # Use dummy embeddings since ChromaDB requires them for upsert
        dummy = [[0.0] for _ in ids]
        for i in range(0, len(ids), 5000):
            self._parents_col.upsert(
                ids=ids[i : i + 5000],
                embeddings=dummy[i : i + 5000],
                documents=documents[i : i + 5000],
                metadatas=metadatas[i : i + 5000],
            )
        logger.info("parents_upserted", count=len(ids))

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query small chunks by embedding similarity.

        Returns list of dicts with keys: id, document, metadata, distance.
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self._chunks_col.count() or 1),
        }
        if where:
            kwargs["where"] = where
        if kwargs["n_results"] <= 0:
            return []

        results = self._chunks_col.query(**kwargs)
        out: list[dict[str, Any]] = []
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                out.append({
                    "id": chunk_id,
                    "document": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                })
        return out

    def get_parents(self, parent_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch parent chunks by their IDs."""
        if not parent_ids:
            return []
        # Deduplicate while preserving order
        seen = set()
        unique_ids = []
        for pid in parent_ids:
            if pid and pid not in seen:
                seen.add(pid)
                unique_ids.append(pid)
        if not unique_ids:
            return []

        results = self._parents_col.get(ids=unique_ids)
        out: list[dict[str, Any]] = []
        if results and results["ids"]:
            for i, pid in enumerate(results["ids"]):
                out.append({
                    "id": pid,
                    "document": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })
        return out

    def delete_all(self) -> None:
        """Delete both collections and recreate them."""
        self._client.delete_collection("knowledge_chunks")
        self._client.delete_collection("knowledge_parents")
        self._chunks_col = self._client.get_or_create_collection(
            name="knowledge_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        self._parents_col = self._client.get_or_create_collection(
            name="knowledge_parents",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("vectorstore_cleared")

    @property
    def chunk_count(self) -> int:
        return self._chunks_col.count()

    @property
    def parent_count(self) -> int:
        return self._parents_col.count()
