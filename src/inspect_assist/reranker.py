"""Reranking module — cross-encoder and LLM-based reranking for RAG results."""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger()

_cross_encoder_model = None  # lazy-loaded


def _get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Lazy-load cross-encoder model on first use to avoid slow startup."""
    global _cross_encoder_model
    if _cross_encoder_model is None:
        try:
            from sentence_transformers import CrossEncoder

            _cross_encoder_model = CrossEncoder(model_name)
            logger.info("cross_encoder_loaded", model=model_name)
        except Exception as e:
            logger.warning("cross_encoder_load_failed", error=str(e))
            return None
    return _cross_encoder_model


def rerank_cross_encoder(
    query: str,
    chunks: list[dict[str, Any]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Rerank chunks using a cross-encoder model.

    Each chunk dict must have a 'document' key with the text content.
    Returns chunks sorted by cross-encoder relevance score (descending).
    """
    if not chunks:
        return []

    model = _get_cross_encoder(model_name)
    if model is None:
        logger.warning("cross_encoder_unavailable_returning_original_order")
        return chunks

    pairs = [(query, c["document"]) for c in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    ranked = sorted(chunks, key=lambda c: c.get("rerank_score", 0), reverse=True)
    if top_n:
        ranked = ranked[:top_n]
    return ranked


async def rerank_llm(
    query: str,
    chunks: list[dict[str, Any]],
    llm_client,
    model: str = "gpt-4o-mini",
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Rerank using an LLM to judge relevance. Fallback when cross-encoder unavailable.

    Sends top chunks to the LLM and asks it to rank by relevance.
    """
    if not chunks or len(chunks) <= 1:
        return chunks

    # Build numbered list of chunk excerpts
    excerpts = []
    for i, c in enumerate(chunks):
        text = c["document"][:300]  # truncate for prompt efficiency
        excerpts.append(f"[{i}] {text}")

    prompt = (
        f"Given the query: \"{query}\"\n\n"
        f"Rank the following text passages by relevance (most relevant first).\n"
        f"Return ONLY a JSON array of passage indices, e.g. [2, 0, 4, 1, 3].\n\n"
        + "\n\n".join(excerpts)
    )

    try:
        response = await llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )
        import json

        ranking_text = response.choices[0].message.content.strip()
        # Extract JSON array from response
        start = ranking_text.find("[")
        end = ranking_text.rfind("]") + 1
        if start >= 0 and end > start:
            indices = json.loads(ranking_text[start:end])
            if isinstance(indices, list):
                reranked = []
                seen = set()
                for idx in indices:
                    if isinstance(idx, int) and 0 <= idx < len(chunks) and idx not in seen:
                        seen.add(idx)
                        chunks[idx]["rerank_score"] = len(chunks) - len(reranked)
                        reranked.append(chunks[idx])
                # Add any chunks not mentioned by LLM
                for i, c in enumerate(chunks):
                    if i not in seen:
                        c["rerank_score"] = 0
                        reranked.append(c)
                return reranked[:top_n]
    except Exception as e:
        logger.warning("llm_rerank_failed", error=str(e))

    return chunks[:top_n]
