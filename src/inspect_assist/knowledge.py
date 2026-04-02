"""Knowledge engine — loads Markdown articles with YAML frontmatter."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import structlog
import yaml

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
        h.update(a.embedding_text().encode("utf-8"))
    return h.hexdigest()[:16]


class KnowledgeEngine:
    """Loads and searches Markdown knowledge articles with YAML frontmatter."""

    def __init__(self, knowledge_path: Path) -> None:
        self._root = knowledge_path
        self._articles: list[KnowledgeArticle] = []
        self._loaded = False
        # Vector search state
        self._embeddings: dict[str, list[float]] = {}  # slug → embedding
        self._embeddings_ready = False
        self._embed_client = None  # set via set_embed_client()
        self._embed_model = "text-embedding-3-small"
        self._cache_path = knowledge_path / ".embeddings_cache.json" if knowledge_path else None

    def _load(self) -> None:
        if self._loaded:
            return
        self._articles = []
        if not self._root.exists():
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

        # Parse YAML frontmatter between --- markers
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

    # --- Vector / semantic search ---

    def set_embed_client(self, client, model: str = "text-embedding-3-small") -> None:
        """Inject an AsyncOpenAI-compatible client for embeddings."""
        self._embed_client = client
        self._embed_model = model

    async def build_embeddings(self) -> None:
        """Compute and cache embeddings for all articles. Safe to call repeatedly."""
        if self._embed_client is None:
            return
        self._load()
        if not self._articles:
            return

        content_hash = _content_hash(self._articles)

        # Try loading from cache first
        cached = self._load_embedding_cache()
        if cached and cached.get("hash") == content_hash:
            self._embeddings = cached["embeddings"]
            self._embeddings_ready = True
            logger.info("embeddings_loaded_from_cache", count=len(self._embeddings))
            return

        # Build fresh embeddings
        texts = [a.embedding_text() for a in self._articles]
        try:
            response = await self._embed_client.embeddings.create(
                input=texts,
                model=self._embed_model,
            )
            self._embeddings = {}
            for article, embedding_data in zip(self._articles, response.data):
                self._embeddings[article.slug] = embedding_data.embedding
            self._embeddings_ready = True
            self._save_embedding_cache(content_hash)
            logger.info("embeddings_built", count=len(self._embeddings))
        except Exception as e:
            logger.warning("embeddings_failed", error=str(e))
            self._embeddings_ready = False

    async def semantic_search(self, query: str, limit: int = 5) -> list[KnowledgeArticle]:
        """Search articles by embedding similarity. Falls back to keyword search."""
        if not self._embeddings_ready or self._embed_client is None:
            return self.search(query, limit=limit)

        try:
            response = await self._embed_client.embeddings.create(
                input=[query],
                model=self._embed_model,
            )
            query_embedding = response.data[0].embedding
        except Exception:
            return self.search(query, limit=limit)

        scored: list[tuple[float, KnowledgeArticle]] = []
        for article in self._articles:
            if article.slug not in self._embeddings:
                continue
            sim = _cosine_similarity(query_embedding, self._embeddings[article.slug])
            scored.append((sim, article))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [article for _, article in scored[:limit]]

    def _load_embedding_cache(self) -> dict | None:
        if self._cache_path is None or not self._cache_path.exists():
            return None
        try:
            return json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _save_embedding_cache(self, content_hash: str) -> None:
        if self._cache_path is None:
            return
        cache = {
            "hash": content_hash,
            "model": self._embed_model,
            "embeddings": self._embeddings,
        }
        try:
            self._cache_path.write_text(json.dumps(cache), encoding="utf-8")
        except OSError:
            pass
