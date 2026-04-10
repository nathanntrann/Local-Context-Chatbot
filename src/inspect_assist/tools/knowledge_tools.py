"""Knowledge tools — search and explain using the RAG-augmented knowledge base."""

from __future__ import annotations

import json

from inspect_assist.tools import ToolParam, tool

_knowledge_engine = None


def set_knowledge_engine(engine) -> None:
    global _knowledge_engine
    _knowledge_engine = engine


def _get_engine():
    if _knowledge_engine is None:
        raise RuntimeError("Knowledge engine not initialized")
    return _knowledge_engine


def _assemble_context(results: list[dict], max_tokens: int = 4096) -> list[dict]:
    """Assemble context from search results, using parent content when available.

    Deduplicates overlapping parent chunks and caps total output size.
    """
    seen_parents: set[str] = set()
    assembled: list[dict] = []
    total_chars = 0
    # Rough char-to-token estimate: 1 token ≈ 4 chars
    char_limit = max_tokens * 4

    for r in results:
        parent_id = r.get("metadata", {}).get("parent_id", "")
        # Use parent content if available and not already included
        if parent_id and parent_id not in seen_parents and "parent_content" in r:
            content = r["parent_content"]
            seen_parents.add(parent_id)
        else:
            content = r.get("document", "")

        if not content:
            continue

        # Check token budget
        if total_chars + len(content) > char_limit:
            remaining = char_limit - total_chars
            if remaining > 100:
                content = content[:remaining]
            else:
                break

        total_chars += len(content)
        assembled.append({
            "title": r.get("article_title", r.get("metadata", {}).get("article_title", "")),
            "slug": r.get("article_slug", r.get("metadata", {}).get("article_slug", "")),
            "category": r.get("category", r.get("metadata", {}).get("category", "")),
            "section": r.get("metadata", {}).get("chunk_index", 0),
            "content": content,
        })

    return assembled


@tool(
    name="search_knowledge",
    description=(
        "Search the knowledge base for troubleshooting guides, known issues, procedures, "
        "and engineering notes. Returns matching content chunks ranked by relevance with "
        "source citations. Uses hybrid search (semantic + keyword) with reranking. "
        "Use this when users ask about problems, best practices, or system behavior."
    ),
    params=[
        ToolParam(
            name="query",
            type="string",
            description="Search query — keywords describing the topic or problem",
        ),
        ToolParam(
            name="limit",
            type="integer",
            description="Max result chunks to return (default 5)",
            required=False,
        ),
    ],
)
async def search_knowledge(query: str, limit: int = 5) -> str:
    engine = _get_engine()
    results = await engine.semantic_search(query, limit=limit)
    if not results:
        return json.dumps({"results": [], "message": "No matching articles found."})

    assembled = _assemble_context(results, max_tokens=engine._max_context_tokens)
    return json.dumps({"results": assembled, "count": len(assembled)}, indent=2)


@tool(
    name="search_knowledge_filtered",
    description=(
        "Search the knowledge base with metadata filters. Use this when you need to "
        "narrow results to a specific category (concepts, troubleshooting, procedures, "
        "known-issues, parameters) or when a previous search returned too many unrelated results. "
        "Good for multi-hop retrieval — first search broadly, then narrow by category."
    ),
    params=[
        ToolParam(
            name="query",
            type="string",
            description="Search query",
        ),
        ToolParam(
            name="category",
            type="string",
            description="Filter by category: concepts, troubleshooting, procedures, known-issues, parameters",
            required=False,
        ),
        ToolParam(
            name="limit",
            type="integer",
            description="Max results (default 5)",
            required=False,
        ),
    ],
)
async def search_knowledge_filtered(
    query: str,
    category: str | None = None,
    limit: int = 5,
) -> str:
    engine = _get_engine()
    results = await engine.search_filtered(query, limit=limit, category=category)
    if not results:
        return json.dumps({"results": [], "message": "No matching articles found."})

    assembled = _assemble_context(results, max_tokens=engine._max_context_tokens)
    return json.dumps({"results": assembled, "count": len(assembled)}, indent=2)


@tool(
    name="get_article_section",
    description=(
        "Retrieve a specific section from a knowledge article by its slug and heading. "
        "Use this when you've identified a relevant article from search results and want "
        "to read a specific section in full detail. The slug is the article filename without "
        "the .md extension (e.g. 'false-positives', 'thermal-seal-inspection')."
    ),
    params=[
        ToolParam(
            name="article_slug",
            type="string",
            description="Article slug (filename without .md, e.g. 'false-positives')",
        ),
        ToolParam(
            name="section_heading",
            type="string",
            description="Section heading to retrieve (e.g. 'Root Causes', 'Diagnosis Checklist')",
        ),
    ],
)
async def get_article_section(article_slug: str, section_heading: str) -> str:
    engine = _get_engine()

    section = engine.get_article_section(article_slug, section_heading)
    if section:
        return json.dumps({
            "article_slug": article_slug,
            "section_heading": section_heading,
            "content": section,
        }, indent=2)

    # Section not found — return available headings
    article = engine.get_by_slug(article_slug)
    if article:
        headings = [
            line.strip().lstrip("#").strip()
            for line in article.content.split("\n")
            if line.strip().startswith("#")
        ]
        return json.dumps({
            "article_slug": article_slug,
            "message": f"Section '{section_heading}' not found.",
            "available_headings": headings,
        }, indent=2)

    return json.dumps({"message": f"Article '{article_slug}' not found."})


@tool(
    name="explain_concept",
    description=(
        "Explain an inspection-related concept, parameter, or term using the knowledge base. "
        "If a specific article exists, returns it. Otherwise searches for related content. "
        "Use when users ask 'what is X?' or 'explain Y' or 'what does Z do?'."
    ),
    params=[
        ToolParam(
            name="concept",
            type="string",
            description="The concept, parameter, or term to explain",
        ),
    ],
)
async def explain_concept(concept: str) -> str:
    engine = _get_engine()

    # Try exact slug match first
    article = engine.get_by_slug(concept.lower().replace(" ", "-"))
    if article:
        return json.dumps(
            {
                "title": article.title,
                "category": article.category,
                "tags": article.tags,
                "content": article.content,
            },
            indent=2,
        )

    # Fall back to RAG search
    results = await engine.semantic_search(concept, limit=3)
    if results:
        assembled = _assemble_context(results, max_tokens=engine._max_context_tokens)
        return json.dumps(
            {
                "related_articles": assembled,
                "note": f"No exact article for '{concept}', but found related content.",
            },
            indent=2,
        )

    return json.dumps(
        {"message": f"No knowledge base content found for '{concept}'. Using general knowledge."}
    )
