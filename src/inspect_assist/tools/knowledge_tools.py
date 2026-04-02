"""Knowledge tools — search and explain using the knowledge base."""

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


@tool(
    name="search_knowledge",
    description=(
        "Search the knowledge base for troubleshooting guides, known issues, procedures, "
        "and engineering notes. Returns matching articles ranked by relevance. "
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
            description="Max results to return (default 5)",
            required=False,
        ),
    ],
)
async def search_knowledge(query: str, limit: int = 5) -> str:
    engine = _get_engine()
    results = await engine.semantic_search(query, limit=limit)
    if not results:
        return json.dumps({"results": [], "message": "No matching articles found."})

    articles = []
    for a in results:
        articles.append({
            "title": a.title,
            "category": a.category,
            "tags": a.tags,
            "content": a.content[:2000],  # Truncate for context window
        })

    return json.dumps({"results": articles, "count": len(articles)}, indent=2)


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

    # Fall back to search
    results = engine.search(concept, limit=3)
    if results:
        return json.dumps(
            {
                "related_articles": [
                    {"title": a.title, "category": a.category, "excerpt": a.content[:500]}
                    for a in results
                ],
                "note": f"No exact article for '{concept}', but found related content.",
            },
            indent=2,
        )

    return json.dumps(
        {"message": f"No knowledge base content found for '{concept}'. Using general knowledge."}
    )
