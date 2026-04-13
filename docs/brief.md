# InspectAssist — Project Brief

## What Is This

InspectAssist is a context-aware, tool-augmented LLM assistant designed as a **sidecar companion** for an industrial thermal seal inspection system. It does not replace the inspection system — it helps users understand, troubleshoot, and work with it effectively.

## Problem

1. Users don't understand what inspection parameters do or how to tune them
2. Troubleshooting is inconsistent — knowledge lives in people's heads
3. It's hard to explain why something was rejected or behaved unexpectedly
4. System changes make it harder for users to adapt
5. No centralized, searchable knowledge base for the inspection domain

## Solution

A standalone FastAPI service that:
- Analyzes thermal inspection images using GPT-4o vision
- Searches a curated domain knowledge base (Markdown + YAML frontmatter)
- Flags potential mislabels in the dataset via automated visual audit
- Provides expert-level troubleshooting and explanations
- Exposes a REST API (`POST /api/v1/chat`) for integration into any UI

## Who Uses It

| User | Needs |
| --- | --- |
| **Operators** | Explain rejections, parameter guidance, quick troubleshooting |
| **Engineers** | Dataset QA, compare images, analyze failure patterns |
| **Support/Field** | Structured troubleshooting, knowledge base search, context before arriving on-site |

## Key Design Decisions

- **Sidecar, not replacement** — reads system data, never controls it
- **Tool-augmented generation** — LLM calls real functions (11 tools), never guesses
- **Image-first** — GPT-4o vision analyzes thermal PNGs as a core feature
- **Production RAG pipeline** — two-tier chunking, hybrid search (semantic + BM25 via RRF), cross-encoder reranking, parent document retrieval, semantic caching
- **Flexible LLM backend** — abstraction layer supports Azure OpenAI, OpenAI, Ollama, and Anthropic Claude with smart routing
- **SQLite persistence** — conversations + user feedback stored in aiosqlite; ChromaDB for vector embeddings
- **API-first** — full-featured chat UI with streaming; designed to embed into the main inspection GUI

## Current Stack

- Python 3.11+, FastAPI, Pydantic Settings, structlog
- OpenAI SDK, Anthropic SDK, Ollama (OpenAI-compatible)
- ChromaDB (persistent vector store), rank-bm25 (lexical search)
- sentence-transformers (cross-encoder reranking), langchain-text-splitters (chunking)
- aiosqlite (async SQLite for conversations + feedback)
- YAML-frontmatter Markdown knowledge base
- Docker Compose for deployment
- 11 registered tools via decorator-based framework
- RAG evaluation harness (Precision@k, Recall@k, MRR)

## Integration Plan

Phase 1 (complete): Standalone service with labeled image dataset + knowledge base
Phase 2 (complete): RAG pipeline — two-tier chunking, ChromaDB vectors, hybrid search (BM25 + semantic via RRF), cross-encoder reranking, parent document retrieval, contextual retrieval, HyDE, semantic caching
Phase 3 (complete): Multi-provider LLM support (Azure OpenAI, OpenAI, Ollama, Anthropic) + smart routing + SQLite persistence + user feedback + RAG evaluation frameworkPhase 3.5 (complete): Chat UI enhancements — SSE streaming, tool activity indicators, follow-up suggestion chips, image attachment lightbox, conversation history sidebar; embeddable widget upgraded with streaming + tool indicators + suggestionsPhase 4: Connect to coworker’s inspection system via API adapters (recipes, results, logs)
Phase 5: Deeper reasoning — trend analysis, adaptive suggestions, GUI widget embedding
