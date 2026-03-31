"""Knowledge engine — loads Markdown articles with YAML frontmatter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


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


class KnowledgeEngine:
    """Loads and searches Markdown knowledge articles with YAML frontmatter."""

    def __init__(self, knowledge_path: Path) -> None:
        self._root = knowledge_path
        self._articles: list[KnowledgeArticle] = []
        self._loaded = False

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
