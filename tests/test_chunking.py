"""Tests for the two-tier document chunking module."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from inspect_assist.chunking import KnowledgeChunk, chunk_article, _chunk_id, _find_parent


# Minimal stand-in for KnowledgeArticle to avoid importing the full engine
@dataclass
class _FakeArticle:
    slug: str = "test-article"
    title: str = "Test Article"
    category: str = "concepts"
    tags: tuple = ("test",)
    content: str = ""
    metadata: dict = None
    path: str = ""

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TestChunkId:
    def test_deterministic(self):
        a = _chunk_id("slug", "small", 0)
        b = _chunk_id("slug", "small", 0)
        assert a == b

    def test_different_for_different_tiers(self):
        a = _chunk_id("slug", "small", 0)
        b = _chunk_id("slug", "parent", 0)
        assert a != b

    def test_different_for_different_indices(self):
        a = _chunk_id("slug", "small", 0)
        b = _chunk_id("slug", "small", 1)
        assert a != b

    def test_length(self):
        cid = _chunk_id("some-article", "small", 5)
        assert len(cid) == 16


class TestChunkArticle:
    def test_empty_content_returns_empty(self):
        article = _FakeArticle(content="")
        small, parent = chunk_article(article)
        assert small == []
        assert parent == []

    def test_whitespace_only_returns_empty(self):
        article = _FakeArticle(content="   \n  \n  ")
        small, parent = chunk_article(article)
        assert small == []
        assert parent == []

    def test_short_content_single_chunk(self):
        article = _FakeArticle(content="A short paragraph about thermal seals.")
        small, parent = chunk_article(article, chunk_size=500, parent_chunk_size=2000)
        assert len(small) == 1
        assert len(parent) == 1
        assert small[0].article_slug == "test-article"
        assert small[0].category == "concepts"
        assert small[0].content == "A short paragraph about thermal seals."

    def test_produces_two_tiers(self):
        content = "\n\n".join(f"Paragraph {i}. " * 20 for i in range(10))
        article = _FakeArticle(content=content)
        small, parent = chunk_article(article, chunk_size=100, chunk_overlap=10, parent_chunk_size=400, parent_chunk_overlap=40)
        assert len(small) > len(parent)
        assert all(c.metadata["tier"] == "small" for c in small)
        assert all(c.metadata["tier"] == "parent" for c in parent)

    def test_small_chunks_have_parent_ids(self):
        content = "\n\n".join(f"Paragraph {i}. " * 20 for i in range(10))
        article = _FakeArticle(content=content)
        small, parent = chunk_article(article, chunk_size=100, parent_chunk_size=400)
        parent_ids = {p.id for p in parent}
        for sc in small:
            assert sc.parent_id is not None
            assert sc.parent_id in parent_ids

    def test_context_prefix_prepended(self):
        article = _FakeArticle(content="Content about inspection.")
        prefix = "This article covers thermal inspection. "
        small, parent = chunk_article(article, chunk_size=500, parent_chunk_size=2000, context_prefix=prefix)
        assert small[0].contextualized_content.startswith(prefix)
        assert small[0].content == "Content about inspection."

    def test_no_context_prefix(self):
        article = _FakeArticle(content="Raw content.")
        small, _ = chunk_article(article, chunk_size=500, parent_chunk_size=2000, context_prefix="")
        assert small[0].contextualized_content == small[0].content

    def test_chunk_metadata(self):
        article = _FakeArticle(content="Content " * 50, slug="my-slug", title="My Title", category="procedures", tags=("a", "b"))
        small, _ = chunk_article(article, chunk_size=100, parent_chunk_size=500)
        for c in small:
            assert c.article_slug == "my-slug"
            assert c.article_title == "My Title"
            assert c.category == "procedures"
            assert c.tags == ["a", "b"]
            assert c.total_chunks == len(small)


class TestFindParent:
    def test_finds_parent_by_text_overlap(self):
        parents = [
            KnowledgeChunk(id="p0", article_slug="s", article_title="T", category="c", tags=[], chunk_index=0, total_chunks=2, content="first parent chunk with unique text", contextualized_content="", parent_id=None),
            KnowledgeChunk(id="p1", article_slug="s", article_title="T", category="c", tags=[], chunk_index=1, total_chunks=2, content="second parent chunk about different topic", contextualized_content="", parent_id=None),
        ]
        # Small text whose first 80 chars are in parent 0
        result = _find_parent("first parent chunk with unique text here", parents)
        assert result == "p0"

    def test_falls_back_to_word_overlap(self):
        parents = [
            KnowledgeChunk(id="p0", article_slug="s", article_title="T", category="c", tags=[], chunk_index=0, total_chunks=2, content="alpha bravo charlie delta", contextualized_content="", parent_id=None),
            KnowledgeChunk(id="p1", article_slug="s", article_title="T", category="c", tags=[], chunk_index=1, total_chunks=2, content="echo foxtrot golf hotel", contextualized_content="", parent_id=None),
        ]
        result = _find_parent("foxtrot golf india juliet", parents)
        assert result == "p1"

    def test_empty_parents_returns_none(self):
        result = _find_parent("some text", [])
        assert result is None
