"""Two-tier document chunking with contextual enrichment for RAG."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from inspect_assist.knowledge import KnowledgeArticle

logger = structlog.get_logger()

# Markdown-aware separators — split on headings first, then paragraphs, then lines
_MD_SEPARATORS = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]


@dataclass
class KnowledgeChunk:
    """A single chunk of a knowledge article."""

    id: str
    article_slug: str
    article_title: str
    category: str
    tags: list[str]
    chunk_index: int
    total_chunks: int
    content: str  # raw chunk text
    contextualized_content: str  # with article context prepended (used for embedding)
    parent_id: str | None = None  # ID of the parent chunk that covers this range
    metadata: dict = field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        return self.contextualized_content


def _chunk_id(slug: str, tier: str, index: int) -> str:
    """Deterministic chunk ID based on article slug + tier + index."""
    raw = f"{slug}:{tier}:{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _create_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_MD_SEPARATORS,
        length_function=len,  # character-based; token-based via tiktoken optional
        is_separator_regex=False,
    )


def chunk_article(
    article: KnowledgeArticle,
    chunk_size: int = 256,
    chunk_overlap: int = 32,
    parent_chunk_size: int = 1024,
    parent_chunk_overlap: int = 128,
    context_prefix: str = "",
) -> tuple[list[KnowledgeChunk], list[KnowledgeChunk]]:
    """Split an article into small search chunks and larger parent chunks.

    Returns (small_chunks, parent_chunks).
    Small chunks reference their parent via parent_id.
    """
    content = article.content
    if not content.strip():
        return [], []

    # --- Parent (large) chunks ---
    parent_splitter = _create_splitter(parent_chunk_size, parent_chunk_overlap)
    parent_texts = parent_splitter.split_text(content)

    parent_chunks: list[KnowledgeChunk] = []
    for i, text in enumerate(parent_texts):
        pid = _chunk_id(article.slug, "parent", i)
        parent_chunks.append(KnowledgeChunk(
            id=pid,
            article_slug=article.slug,
            article_title=article.title,
            category=article.category,
            tags=list(article.tags),
            chunk_index=i,
            total_chunks=len(parent_texts),
            content=text,
            contextualized_content=f"{context_prefix}{text}" if context_prefix else text,
            parent_id=None,
            metadata={"tier": "parent"},
        ))

    # --- Small (search) chunks ---
    small_splitter = _create_splitter(chunk_size, chunk_overlap)
    small_texts = small_splitter.split_text(content)

    small_chunks: list[KnowledgeChunk] = []
    for i, text in enumerate(small_texts):
        # Find which parent chunk this small chunk belongs to (by text overlap)
        parent_id = _find_parent(text, parent_chunks)
        cid = _chunk_id(article.slug, "small", i)
        ctx_text = f"{context_prefix}{text}" if context_prefix else text
        small_chunks.append(KnowledgeChunk(
            id=cid,
            article_slug=article.slug,
            article_title=article.title,
            category=article.category,
            tags=list(article.tags),
            chunk_index=i,
            total_chunks=len(small_texts),
            content=text,
            contextualized_content=ctx_text,
            parent_id=parent_id,
            metadata={"tier": "small"},
        ))

    logger.debug(
        "article_chunked",
        slug=article.slug,
        small_chunks=len(small_chunks),
        parent_chunks=len(parent_chunks),
    )
    return small_chunks, parent_chunks


def _find_parent(small_text: str, parent_chunks: list[KnowledgeChunk]) -> str | None:
    """Find the parent chunk that best contains this small chunk text."""
    best_overlap = 0
    best_id = parent_chunks[0].id if parent_chunks else None
    # Use first 80 chars of small text as probe
    probe = small_text[:80]
    for pc in parent_chunks:
        if probe in pc.content:
            return pc.id
    # Fallback: find parent with most word overlap
    small_words = set(small_text.lower().split())
    for pc in parent_chunks:
        parent_words = set(pc.content.lower().split())
        overlap = len(small_words & parent_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = pc.id
    return best_id


async def generate_context_prefix(
    article: KnowledgeArticle,
    llm_client,
    model: str = "llama3.1:8b",
) -> str:
    """Generate a one-sentence contextual summary for an article.

    This is prepended to each chunk before embedding (Contextual Retrieval).
    Uses a cheap/fast model since it only runs once per article at index time.
    """
    try:
        response = await llm_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the following knowledge article in one sentence. "
                    "Focus on what specific topic and subtopics it covers.",
                },
                {
                    "role": "user",
                    "content": f"Title: {article.title}\nCategory: {article.category}\n\n"
                    f"{article.content[:3000]}",
                },
            ],
            max_tokens=100,
            temperature=0,
        )
        summary = response.choices[0].message.content.strip()
        return f"From '{article.title}' ({article.category}): {summary}\n\n"
    except Exception as e:
        logger.warning("context_prefix_generation_failed", slug=article.slug, error=str(e))
        # Fallback to a static prefix
        tags = ", ".join(article.tags[:5])
        return f"From '{article.title}' ({article.category}, tags: {tags}).\n\n"
