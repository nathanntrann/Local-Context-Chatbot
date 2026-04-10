"""RAG evaluation harness.

Measures Context Precision@k, Context Recall@k, and MRR against a gold-standard
eval dataset. Run from the repo root:

    python -m tests.eval_rag [--k 5] [--dataset tests/eval_dataset.json]

Exit code 1 if Context Recall@5 < 0.8 (CI gate).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def context_precision_at_k(
    retrieved_slugs: list[str], expected_slugs: list[str], k: int
) -> float:
    """Fraction of top-k retrieved results that are relevant."""
    top_k = retrieved_slugs[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for s in top_k if s in expected_slugs)
    return hits / len(top_k)


def context_recall_at_k(
    retrieved_slugs: list[str], expected_slugs: list[str], k: int
) -> float:
    """Fraction of expected articles that appear in top-k results."""
    if not expected_slugs:
        return 1.0
    top_k = set(retrieved_slugs[:k])
    hits = sum(1 for s in expected_slugs if s in top_k)
    return hits / len(expected_slugs)


def reciprocal_rank(retrieved_slugs: list[str], expected_slugs: list[str]) -> float:
    """1 / rank of the first relevant result (0 if none found)."""
    for i, slug in enumerate(retrieved_slugs, start=1):
        if slug in expected_slugs:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_eval(
    dataset_path: Path,
    knowledge_path: Path,
    vectorstore_path: Path,
    k: int = 5,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the evaluation and return aggregate metrics."""
    from inspect_assist.config import LLMProvider, Settings
    from inspect_assist.knowledge import KnowledgeEngine
    from inspect_assist.llm.providers import create_llm_provider

    settings = Settings()
    llm = create_llm_provider(settings)

    # Resolve embedding model
    embed_model = settings.embedding_model
    if not embed_model:
        if settings.llm_provider == LLMProvider.OLLAMA:
            embed_model = "nomic-embed-text"
        else:
            embed_model = "text-embedding-3-small"

    engine = KnowledgeEngine(
        knowledge_path,
        vectorstore_path=vectorstore_path,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        parent_chunk_size=settings.parent_chunk_size,
        parent_chunk_overlap=settings.parent_chunk_overlap,
        embed_model=embed_model,
        llm_model=llm._model,
        contextual_retrieval=settings.contextual_retrieval_enabled,
        hybrid_search=settings.hybrid_search_enabled,
        rrf_k=settings.rrf_k,
        reranker_enabled=settings.reranker_enabled,
        reranker_type=settings.reranker_type,
        reranker_model=settings.reranker_model,
        rerank_top_n=settings.rerank_top_n,
        hyde_enabled=settings.hyde_enabled,
        cache_enabled=False,  # disable cache for eval
        max_context_tokens=settings.max_context_tokens,
    )

    # Build embeddings index
    engine.set_embed_client(llm._client)
    await engine.build_embeddings()

    with open(dataset_path) as f:
        dataset: list[dict] = json.load(f)

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    mrr_scores: list[float] = []
    failures: list[dict] = []

    for i, item in enumerate(dataset):
        query = item["query"]
        expected_slugs = item["expected_slugs"]

        results = await engine.semantic_search(query, limit=k, return_parent_context=False)
        retrieved_slugs = [r.get("article_slug", r.get("metadata", {}).get("article_slug", "")) for r in results]

        p = context_precision_at_k(retrieved_slugs, expected_slugs, k)
        r = context_recall_at_k(retrieved_slugs, expected_slugs, k)
        rr = reciprocal_rank(retrieved_slugs, expected_slugs)

        precision_scores.append(p)
        recall_scores.append(r)
        mrr_scores.append(rr)

        if r < 1.0:
            failures.append(
                {
                    "index": i,
                    "query": query,
                    "expected": expected_slugs,
                    "retrieved": retrieved_slugs[:k],
                    "recall": r,
                }
            )

        if verbose:
            status = "PASS" if r >= 1.0 else "FAIL"
            print(f"  [{status}] Q{i+1}: recall={r:.2f} precision={p:.2f} rr={rr:.2f} | {query[:60]}")

    n = len(dataset)
    metrics = {
        "num_queries": n,
        "context_precision_at_k": sum(precision_scores) / n if n else 0,
        "context_recall_at_k": sum(recall_scores) / n if n else 0,
        "mrr": sum(mrr_scores) / n if n else 0,
        "k": k,
        "num_failures": len(failures),
        "failures": failures,
    }
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG evaluation harness")
    parser.add_argument("--k", type=int, default=5, help="Top-k for metrics (default: 5)")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/eval_dataset.json"),
        help="Path to eval dataset JSON",
    )
    parser.add_argument(
        "--knowledge",
        type=Path,
        default=Path("knowledge"),
        help="Path to knowledge directory",
    )
    parser.add_argument(
        "--vectorstore",
        type=Path,
        default=Path("data/eval_vectorstore"),
        help="Path for evaluation vectorstore (separate from production)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-query results")
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.8,
        help="Minimum Context Recall@k to pass (default: 0.8)",
    )
    args = parser.parse_args()

    print(f"RAG Evaluation — k={args.k}, min_recall={args.min_recall}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Knowledge: {args.knowledge}")
    print()

    metrics = asyncio.run(
        run_eval(
            dataset_path=args.dataset,
            knowledge_path=args.knowledge,
            vectorstore_path=args.vectorstore,
            k=args.k,
            verbose=args.verbose,
        )
    )

    print()
    print("=" * 60)
    print(f"  Queries evaluated:     {metrics['num_queries']}")
    print(f"  Context Precision@{args.k}:  {metrics['context_precision_at_k']:.3f}")
    print(f"  Context Recall@{args.k}:     {metrics['context_recall_at_k']:.3f}")
    print(f"  MRR:                   {metrics['mrr']:.3f}")
    print(f"  Failures:              {metrics['num_failures']}")
    print("=" * 60)

    if metrics["context_recall_at_k"] < args.min_recall:
        print(
            f"\nFAIL: Context Recall@{args.k} = {metrics['context_recall_at_k']:.3f} "
            f"< {args.min_recall:.2f}"
        )
        if metrics["failures"]:
            print("\nFailed queries:")
            for f in metrics["failures"][:10]:
                print(f"  Q{f['index']+1}: recall={f['recall']:.2f} | {f['query'][:70]}")
                print(f"       expected={f['expected']}, got={f['retrieved']}")
        sys.exit(1)
    else:
        print(f"\nPASS: Context Recall@{args.k} = {metrics['context_recall_at_k']:.3f}")


if __name__ == "__main__":
    main()
