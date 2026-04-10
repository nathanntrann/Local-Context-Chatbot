"""Knowledge engine — loads Markdown articles, chunks them, and provides RAG search."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

from inspect_assist.cache import SemanticCache
from inspect_assist.chunking import (
    KnowledgeChunk,
    chunk_article,
    generate_context_prefix,
)
from inspect_assist.vectorstore import VectorStore

logger = structlog.get_logger()


@dataclass
class KnowledgeArticle:
    """A single knowledge base article."""

    slug: str
    title: str
    category: str
    tags: list[str]
    content: str
    metadata: dict = field(default_factory=dict)
    path: Path | None = None

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "title": self.title,
            "category": self.category,
            "tags": self.tags,
        }

    def embedding_text(self) -> str:
        """Text representation used for embedding."""
        parts = [self.title, self.category]
        if self.tags:
            parts.append(", ".join(self.tags))
        parts.append(self.content[:2000])
        return "\n".join(parts)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (pure Python)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _content_hash(articles: list[KnowledgeArticle]) -> str:
    """Hash article contents to detect changes (invalidates embedding cache)."""
    h = hashlib.sha256()
    for a in sorted(articles, key=lambda x: x.slug):
        h.update(a.content.encode("utf-8"))
    return h.hexdigest()[:16]


class KnowledgeEngine:
    """Loads Markdown knowledge articles, chunks them, and provides RAG search.

    Features:
    - Two-tier chunking (small for search precision, parent for response context)
    - Contextual retrieval (article summary prepended to chunks before embedding)
    - ChromaDB vector store with metadata filtering
    - Hybrid search (BM25 + semantic) with Reciprocal Rank Fusion
    - Cross-encoder reranking
    - HyDE (Hypothetical Document Embeddings)
    - Semantic query caching
    """

    def __init__(
        self,
        knowledge_path: Path,
        *,
        vectorstore_path: Path | None = None,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        parent_chunk_size: int = 1024,
        parent_chunk_overlap: int = 128,
        embed_model: str = "text-embedding-3-small",
        llm_model: str = "llama3.1:8b",
        contextual_retrieval: bool = True,
        hybrid_search: bool = True,
        rrf_k: int = 60,
        reranker_enabled: bool = True,
        reranker_type: str = "cross-encoder",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_top_n: int = 20,
        hyde_enabled: bool = False,
        cache_enabled: bool = True,
        cache_similarity_threshold: float = 0.95,
        cache_ttl_seconds: int = 3600,
        cache_max_size: int = 200,
        max_context_tokens: int = 4096,
    ) -> None:
        self._root = knowledge_path
        self._articles: list[KnowledgeArticle] = []
        self._loaded = False

        # Chunking config
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._parent_chunk_size = parent_chunk_size
        self._parent_chunk_overlap = parent_chunk_overlap
        self._contextual_retrieval = contextual_retrieval

        # Chunk storage
        self._small_chunks: list[KnowledgeChunk] = []
        self._parent_chunks: list[KnowledgeChunk] = []

        # Embedding / vector store
        self._embed_client = None
        self._embed_model = embed_model
        self._llm_model = llm_model
        self._vectorstore: VectorStore | None = None
        self._vs_path = vectorstore_path or (knowledge_path.parent / "vectorstore" if knowledge_path else None)
        self._embeddings_ready = False

        # BM25
        self._hybrid_search = hybrid_search
        self._rrf_k = rrf_k
        self._bm25 = None  # built after chunking
        self._bm25_corpus: list[str] = []
        self._bm25_chunk_ids: list[str] = []

        # Reranking
        self._reranker_enabled = reranker_enabled
        self._reranker_type = reranker_type
        self._reranker_model = reranker_model
        self._rerank_top_n = rerank_top_n

        # HyDE
        self._hyde_enabled = hyde_enabled

        # Semantic cache
        self._cache: SemanticCache | None = None
        if cache_enabled:
            self._cache = SemanticCache(
                similarity_threshold=cache_similarity_threshold,
                ttl_seconds=cache_ttl_seconds,
                max_size=cache_max_size,
            )

        self._max_context_tokens = max_context_tokens

    # --- Article loading ---

    def _load(self) -> None:
        if self._loaded:
            return
        self._articles = []
        if not self._root or not self._root.exists():
            self._loaded = True
            return

        for md_file in sorted(self._root.rglob("*.md")):
            article = self._parse_article(md_file)
            if article:
                self._articles.append(article)
        self._loaded = True

    @staticmethod
    def _parse_article(path: Path) -> KnowledgeArticle | None:
        text = path.read_text(encoding="utf-8")

        metadata: dict = {}
        content = text
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                try:
                    metadata = yaml.safe_load(parts[1]) or {}
                except yaml.YAMLError:
                    metadata = {}
                content = parts[2].strip()

        return KnowledgeArticle(
            slug=path.stem,
            title=metadata.get("title", path.stem.replace("-", " ").title()),
            category=metadata.get("category", path.parent.name),
            tags=[str(t) for t in metadata.get("tags", [])],
            content=content,
            metadata=metadata,
            path=path,
        )

    def reload(self) -> None:
        self._loaded = False
        self._load()

    def get_by_slug(self, slug: str) -> KnowledgeArticle | None:
        self._load()
        for article in self._articles:
            if article.slug == slug:
                return article
        return None

    def get_by_category(self, category: str) -> list[KnowledgeArticle]:
        self._load()
        return [a for a in self._articles if a.category.lower() == category.lower()]

    def list_all(self) -> list[dict]:
        self._load()
        return [a.to_dict() for a in self._articles]

    # --- Legacy keyword search (fallback) ---

    def search(self, query: str, limit: int = 5) -> list[KnowledgeArticle]:
        """Keyword search across titles, tags, and content."""
        self._load()
        query_lower = query.lower()
        terms = query_lower.split()

        scored: list[tuple[float, KnowledgeArticle]] = []
        for article in self._articles:
            score = 0.0
            text_lower = article.content.lower()
            title_lower = article.title.lower()
            tags_lower = [t.lower() for t in article.tags]

            for term in terms:
                if term in title_lower:
                    score += 3.0
                if any(term in tag for tag in tags_lower):
                    score += 2.0
                if term in article.category.lower():
                    score += 1.5
                count = text_lower.count(term)
                if count > 0:
                    score += min(count * 0.5, 3.0)

            if score > 0:
                scored.append((score, article))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [article for _, article in scored[:limit]]

    # --- Embedding & Indexing ---

    def set_embed_client(self, client, model: str | None = None) -> None:
        """Inject an AsyncOpenAI-compatible client for embeddings."""
        self._embed_client = client
        if model is not None:
            self._embed_model = model

    async def build_embeddings(self) -> None:
        """Chunk articles, embed, and index into ChromaDB + BM25. Safe to call repeatedly."""
        if self._embed_client is None:
            logger.warning("no_embed_client_set_skipping_rag_indexing")
            return
        self._load()
        if not self._articles:
            return

        content_hash = _content_hash(self._articles)

        # Check if vector store already has up-to-date data
        if self._vs_path:
            hash_file = Path(self._vs_path) / ".content_hash"
            if hash_file.exists():
                stored_hash = hash_file.read_text(encoding="utf-8").strip()
                if stored_hash == content_hash and self._vectorstore is not None:
                    logger.info("rag_index_up_to_date", hash=content_hash)
                    self._embeddings_ready = True
                    self._build_bm25()
                    return

        logger.info("rag_indexing_start", articles=len(self._articles))

        # --- Step 1: Chunk all articles ---
        self._small_chunks = []
        self._parent_chunks = []

        for article in self._articles:
            prefix = ""
            if self._contextual_retrieval:
                prefix = await generate_context_prefix(
                    article, self._embed_client, model=self._llm_model
                )

            small, parents = chunk_article(
                article,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                parent_chunk_size=self._parent_chunk_size,
                parent_chunk_overlap=self._parent_chunk_overlap,
                context_prefix=prefix,
            )
            self._small_chunks.extend(small)
            self._parent_chunks.extend(parents)

        logger.info(
            "chunks_created",
            small=len(self._small_chunks),
            parents=len(self._parent_chunks),
        )

        # --- Step 2: Embed small chunks ---
        if not self._small_chunks:
            return

        texts_to_embed = [c.embedding_text for c in self._small_chunks]
        embeddings: list[list[float]] = []
        # Batch embeddings (OpenAI limit ~2048 per call)
        batch_size = 2000
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i : i + batch_size]
            try:
                response = await self._embed_client.embeddings.create(
                    input=batch, model=self._embed_model
                )
                embeddings.extend([d.embedding for d in response.data])
            except Exception as e:
                logger.error("embedding_batch_failed", batch_start=i, error=str(e))
                self._embeddings_ready = False
                return

        # --- Step 3: Store in ChromaDB ---
        if self._vs_path:
            Path(self._vs_path).mkdir(parents=True, exist_ok=True)
            self._vectorstore = VectorStore(self._vs_path)
            self._vectorstore.delete_all()

            # Upsert small chunks with embeddings
            self._vectorstore.upsert_chunks(
                ids=[c.id for c in self._small_chunks],
                embeddings=embeddings,
                documents=[c.content for c in self._small_chunks],
                metadatas=[
                    {
                        "article_slug": c.article_slug,
                        "article_title": c.article_title,
                        "category": c.category,
                        "tags": ",".join(c.tags),
                        "chunk_index": c.chunk_index,
                        "parent_id": c.parent_id or "",
                    }
                    for c in self._small_chunks
                ],
            )

            # Upsert parent chunks (no embeddings needed, just for ID-based lookup)
            self._vectorstore.upsert_parents(
                ids=[c.id for c in self._parent_chunks],
                documents=[c.content for c in self._parent_chunks],
                metadatas=[
                    {
                        "article_slug": c.article_slug,
                        "article_title": c.article_title,
                        "category": c.category,
                        "chunk_index": c.chunk_index,
                    }
                    for c in self._parent_chunks
                ],
            )

            # Save content hash
            hash_file = Path(self._vs_path) / ".content_hash"
            hash_file.write_text(content_hash, encoding="utf-8")

        # --- Step 4: Build BM25 index ---
        self._build_bm25()

        # Invalidate semantic cache (knowledge changed)
        if self._cache:
            self._cache.invalidate()

        self._embeddings_ready = True
        logger.info(
            "rag_indexing_complete",
            chunks=len(self._small_chunks),
            parents=len(self._parent_chunks),
        )

    def _build_bm25(self) -> None:
        """Build BM25 index from small chunk texts."""
        if not self._small_chunks:
            # Rebuild chunk list from vectorstore if available
            if self._vectorstore and self._vectorstore.chunk_count > 0:
                # We need the in-memory chunks for BM25; reload articles and re-chunk
                # (without re-embedding since vectorstore is already populated)
                self._load()
                if not self._small_chunks:
                    self._rebuild_chunks_from_articles()

        if not self._small_chunks:
            return

        try:
            from rank_bm25 import BM25Okapi

            self._bm25_corpus = [c.content.lower() for c in self._small_chunks]
            self._bm25_chunk_ids = [c.id for c in self._small_chunks]
            tokenized = [doc.split() for doc in self._bm25_corpus]
            self._bm25 = BM25Okapi(tokenized)
            logger.info("bm25_index_built", chunks=len(self._bm25_corpus))
        except ImportError:
            logger.warning("rank_bm25_not_installed_hybrid_search_disabled")
            self._bm25 = None

    def _rebuild_chunks_from_articles(self) -> None:
        """Re-chunk articles in memory (without embedding) for BM25 index."""
        self._small_chunks = []
        self._parent_chunks = []
        for article in self._articles:
            tags = ", ".join(article.tags[:5])
            prefix = f"From '{article.title}' ({article.category}, tags: {tags}).\n\n"
            small, parents = chunk_article(
                article,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                parent_chunk_size=self._parent_chunk_size,
                parent_chunk_overlap=self._parent_chunk_overlap,
                context_prefix=prefix,
            )
            self._small_chunks.extend(small)
            self._parent_chunks.extend(parents)

    # --- Embedding helper ---

    async def _embed_query(self, text: str) -> list[float] | None:
        """Embed a single query string. Returns None on failure."""
        if not self._embed_client:
            return None
        try:
            response = await self._embed_client.embeddings.create(
                input=[text], model=self._embed_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning("query_embedding_failed", error=str(e))
            return None

    # --- HyDE ---

    async def _hyde_transform(self, query: str) -> str:
        """Generate a hypothetical document that answers the query (HyDE)."""
        if not self._embed_client or not self._hyde_enabled:
            return query
        try:
            response = await self._embed_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Write a short passage (3-5 sentences) that would answer the "
                            "following question about thermal seal inspection systems. "
                            "Write as if you are a technical knowledge base article."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                max_tokens=200,
                temperature=0.3,
            )
            hypothetical = response.choices[0].message.content.strip()
            logger.debug("hyde_generated", query=query, hypothetical=hypothetical[:100])
            return hypothetical
        except Exception as e:
            logger.warning("hyde_failed", error=str(e))
            return query

    # --- Core search methods ---

    async def semantic_search(
        self,
        query: str,
        limit: int = 5,
        category: str | None = None,
        return_parent_context: bool = True,
    ) -> list[dict[str, Any]]:
        """Full RAG search pipeline: cache → HyDE → hybrid search → rerank → parent lookup.

        Returns list of dicts with keys: document, metadata, article_title, article_slug,
        category, parent_content (if return_parent_context=True).
        Falls back to legacy keyword search if RAG index not available.
        """
        if not self._embeddings_ready or self._vectorstore is None:
            # Fallback to legacy search
            articles = self.search(query, limit=limit)
            return [
                {
                    "document": a.content[:2000],
                    "metadata": {"article_slug": a.slug, "article_title": a.title},
                    "article_title": a.title,
                    "article_slug": a.slug,
                    "category": a.category,
                }
                for a in articles
            ]

        # --- Step 1: Check semantic cache ---
        query_for_embed = query
        if self._hyde_enabled:
            query_for_embed = await self._hyde_transform(query)

        query_embedding = await self._embed_query(query_for_embed)
        if query_embedding is None:
            articles = self.search(query, limit=limit)
            return [
                {
                    "document": a.content[:2000],
                    "metadata": {"article_slug": a.slug, "article_title": a.title},
                    "article_title": a.title,
                    "article_slug": a.slug,
                    "category": a.category,
                }
                for a in articles
            ]

        if self._cache:
            cached = self._cache.get(query_embedding)
            if cached is not None:
                return cached[:limit]

        # --- Step 2: Hybrid search with RRF ---
        where_filter = None
        if category:
            where_filter = {"category": category}

        if self._hybrid_search and self._bm25 is not None:
            results = self._hybrid_search_rrf(
                query, query_embedding, n_results=self._rerank_top_n, where=where_filter
            )
        else:
            # Semantic-only search
            results = self._vectorstore.query(
                query_embedding, n_results=self._rerank_top_n, where=where_filter
            )

        if not results:
            return []

        # --- Step 3: Rerank ---
        if self._reranker_enabled and len(results) > 1:
            results = self._rerank(query, results)

        # Trim to requested limit
        results = results[:limit]

        # --- Step 4: Parent document retrieval ---
        if return_parent_context:
            results = self._attach_parent_context(results)

        # --- Step 5: Cache results ---
        if self._cache:
            self._cache.put(query, query_embedding, results)

        return results

    def _hybrid_search_rrf(
        self,
        query: str,
        query_embedding: list[float],
        n_results: int = 20,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Combine semantic + BM25 results using Reciprocal Rank Fusion."""
        k = self._rrf_k

        # Semantic results from ChromaDB
        semantic_results = self._vectorstore.query(
            query_embedding, n_results=n_results, where=where
        )
        semantic_ranks: dict[str, int] = {}
        semantic_docs: dict[str, dict] = {}
        for rank, r in enumerate(semantic_results):
            semantic_ranks[r["id"]] = rank + 1
            semantic_docs[r["id"]] = r

        # BM25 results
        bm25_ranks: dict[str, int] = {}
        if self._bm25 is not None:
            tokenized_query = query.lower().split()
            scores = self._bm25.get_scores(tokenized_query)
            # Get top N indices
            indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            for rank, (idx, score) in enumerate(indexed_scores[:n_results]):
                if score > 0 and idx < len(self._bm25_chunk_ids):
                    chunk_id = self._bm25_chunk_ids[idx]
                    bm25_ranks[chunk_id] = rank + 1

        # Merge via RRF
        all_ids = set(semantic_ranks.keys()) | set(bm25_ranks.keys())
        rrf_scores: list[tuple[float, str]] = []
        for chunk_id in all_ids:
            score = 0.0
            if chunk_id in semantic_ranks:
                score += 1.0 / (k + semantic_ranks[chunk_id])
            if chunk_id in bm25_ranks:
                score += 1.0 / (k + bm25_ranks[chunk_id])
            rrf_scores.append((score, chunk_id))

        rrf_scores.sort(key=lambda x: x[0], reverse=True)

        # Build result list, preferring semantic_docs for chunk content
        merged: list[dict[str, Any]] = []
        for score, chunk_id in rrf_scores[:n_results]:
            if chunk_id in semantic_docs:
                entry = dict(semantic_docs[chunk_id])
            else:
                # Chunk found by BM25 but not in semantic top-N; get from memory
                entry = self._chunk_dict_by_id(chunk_id)
                if entry is None:
                    continue
            entry["rrf_score"] = score
            merged.append(entry)

        return merged

    def _chunk_dict_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        """Look up a small chunk by ID from in-memory list."""
        for c in self._small_chunks:
            if c.id == chunk_id:
                return {
                    "id": c.id,
                    "document": c.content,
                    "metadata": {
                        "article_slug": c.article_slug,
                        "article_title": c.article_title,
                        "category": c.category,
                        "tags": ",".join(c.tags),
                        "chunk_index": c.chunk_index,
                        "parent_id": c.parent_id or "",
                    },
                    "distance": 0.0,
                }
        return None

    def _rerank(self, query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply cross-encoder reranking."""
        if self._reranker_type == "cross-encoder":
            from inspect_assist.reranker import rerank_cross_encoder

            return rerank_cross_encoder(
                query, results, model_name=self._reranker_model, top_n=self._rerank_top_n
            )
        # LLM reranking is async, handled separately in semantic_search
        return results

    def _attach_parent_context(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fetch parent chunks for the results and attach their content."""
        if not self._vectorstore:
            return results

        parent_ids = []
        for r in results:
            pid = r.get("metadata", {}).get("parent_id", "")
            if pid:
                parent_ids.append(pid)

        if not parent_ids:
            return results

        parents = self._vectorstore.get_parents(parent_ids)
        parent_map = {p["id"]: p for p in parents}

        for r in results:
            pid = r.get("metadata", {}).get("parent_id", "")
            if pid and pid in parent_map:
                parent = parent_map[pid]
                r["parent_content"] = parent["document"]
                r["article_title"] = parent["metadata"].get("article_title", "")
                r["article_slug"] = parent["metadata"].get("article_slug", "")
                r["category"] = parent["metadata"].get("category", "")
            else:
                r["article_title"] = r.get("metadata", {}).get("article_title", "")
                r["article_slug"] = r.get("metadata", {}).get("article_slug", "")
                r["category"] = r.get("metadata", {}).get("category", "")

        return results

    # --- Filtered search (for multi-hop RAG) ---

    async def search_filtered(
        self,
        query: str,
        limit: int = 5,
        category: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search with metadata filters — used by the LLM for targeted multi-hop retrieval."""
        return await self.semantic_search(
            query, limit=limit, category=category, return_parent_context=True
        )

    def get_article_section(self, article_slug: str, section_heading: str) -> str | None:
        """Get a specific section from an article by heading match."""
        article = self.get_by_slug(article_slug)
        if not article:
            return None

        lines = article.content.split("\n")
        section_lines: list[str] = []
        capturing = False
        heading_lower = section_heading.lower().strip()

        for line in lines:
            stripped = line.strip().lstrip("#").strip().lower()
            if stripped == heading_lower or heading_lower in stripped:
                capturing = True
                section_lines = [line]
                continue
            if capturing:
                # Stop at next heading of same or higher level
                if line.strip().startswith("#") and section_lines:
                    break
                section_lines.append(line)

        return "\n".join(section_lines).strip() if section_lines else None
