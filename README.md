# InspectAssist

Context-aware LLM assistant for industrial thermal seal inspection. Uses GPT-4o function calling, vision, and a production-grade RAG pipeline to analyze thermal images, flag mislabels, and provide expert troubleshooting — designed as a sidecar companion to an existing inspection system.

## What It Does

| Capability | Example |
| --- | --- |
| **Analyze thermal images** | "Analyze PASS/img_042.png" → GPT-4o vision interprets seal quality |
| **Flag mislabels** | "Are any PASS images mislabeled?" → samples and audits with vision |
| **Dataset QA** | "What does my dataset look like?" → counts, class balance, labels |
| **Deep analytics** | "Show me dataset statistics" → dimensions, file sizes, imbalance metrics |
| **Audit reports** | "Run a full quality audit" → batch analysis with saved JSON report |
| **Compare images** | "Compare these two images" → side-by-side visual diff with explanation |
| **RAG knowledge search** | "Getting too many false positives" → hybrid semantic + keyword search with reranking |
| **Explain concepts** | "What makes a good thermal seal?" → retrieves and synthesizes domain knowledge |
| **Multi-hop retrieval** | Complex questions → agent chains multiple searches across knowledge categories |
| **Stream responses** | Real-time token streaming via Server-Sent Events |
| **Tool transparency** | Live tool activity indicators during streaming (spinner + labels) |
| **Follow-up suggestions** | Clickable suggestion chips guide users to next steps |
| **Image lightbox** | Vision tool thumbnails displayed inline with click-to-expand |
| **Conversation sidebar** | Searchable history with resume, delete, and export |
| **Persist history** | SQLite-backed conversation storage with list/load/delete/export |
| **User feedback** | Rate responses → feedback stored for RAG quality tracking |

## Architecture

```
FastAPI Service
├── Chat API (REST + SSE)        POST /api/v1/chat, /api/v1/chat/stream
├── Orchestrator                 System prompt → LLM → tool calls → loop → persist
├── LLM Providers                Azure OpenAI / OpenAI / Ollama / Anthropic Claude
├── 11 Tools                     Dataset, vision, knowledge — via @tool decorator
├── RAG Pipeline                 Chunking → ChromaDB → Hybrid Search → Rerank → Parent Retrieval
│   ├── Contextual Retrieval     LLM-generated article summaries prepended to chunks
│   ├── Hybrid Search (RRF)      Semantic (ChromaDB) + BM25 merged via Reciprocal Rank Fusion
│   ├── Cross-encoder Reranker   ms-marco-MiniLM-L-6-v2 reranking of candidates
│   ├── Parent Document Retrieval  Small search chunks → large parent context chunks
│   ├── HyDE                     Hypothetical Document Embeddings (optional)
│   └── Semantic Cache           Cosine-similarity LRU cache with TTL
├── Conversation Store           SQLite persistence (aiosqlite) + feedback table
├── Dataset Adapter              Scans PASS/FAULT image folders
├── Knowledge Engine             Full RAG over Markdown + YAML frontmatter articles
└── Chat UI                  SSE streaming, tool indicators, suggestions, image lightbox, conversation sidebar
```

## RAG Pipeline

The knowledge engine implements a multi-stage retrieval-augmented generation pipeline:

1. **Document Chunking** — Two-tier system using `langchain-text-splitters` with markdown-aware separators:
   - Small chunks (256 tokens, 32 overlap) for precise search
   - Parent chunks (1024 tokens, 128 overlap) for rich context retrieval

2. **Contextual Retrieval** — LLM generates a short article summary prepended to each chunk before embedding, improving retrieval precision for ambiguous queries.

3. **Vector Store** — ChromaDB persistent store with two collections (`knowledge_chunks` for search, `knowledge_parents` for context lookup). Incremental re-indexing via content hashing.

4. **Hybrid Search with RRF** — Combines semantic search (ChromaDB cosine similarity) with lexical search (BM25) using Reciprocal Rank Fusion: `score(d) = Σ 1/(k + rank(d))` with k=60.

5. **Cross-encoder Reranking** — Top candidates reranked by `cross-encoder/ms-marco-MiniLM-L-6-v2` for higher precision. Falls back to LLM-based reranking if the model is unavailable.

6. **Parent Document Retrieval** — After reranking, small search chunks are mapped back to their parent chunks to provide richer context to the LLM.

7. **HyDE (optional)** — Hypothetical Document Embeddings: generates a hypothetical answer to the query and searches with that embedding.

8. **Semantic Cache** — LRU cache with cosine similarity matching (threshold 0.95) and configurable TTL. Avoids redundant searches.

9. **User Feedback Loop** — Users rate responses; feedback is stored in SQLite with query and retrieved chunk metadata for continuous evaluation.

10. **RAG Evaluation Framework** — Offline harness measuring Context Precision@k, Context Recall@k, and MRR against a 30-query gold-standard dataset. CI gate fails if Recall@5 < 0.8.

## Quick Start

```bash
# Clone
git clone https://github.com/nathanntrann/Local-Context-Chatbot.git
cd Local-Context-Chatbot

# Setup
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
# source .venv/bin/activate         # macOS/Linux

pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env — add your Azure OpenAI or OpenAI API key

# Run
python -m inspect_assist
# → http://localhost:8000
```

## Docker

```bash
cp .env.example .env    # fill in API keys
docker compose up
```

## Project Structure

```
src/inspect_assist/
├── app.py                 # FastAPI app factory
├── config.py              # Settings from .env (LLM, RAG, limits)
├── orchestrator.py        # Conversation + tool dispatch + streaming + persistence
├── storage.py             # SQLite conversation + feedback persistence (aiosqlite)
├── knowledge.py           # RAG engine: chunking → embed → hybrid search → rerank → cache
├── chunking.py            # Two-tier document chunking with contextual enrichment
├── vectorstore.py         # ChromaDB wrapper (search chunks + parent chunks)
├── reranker.py            # Cross-encoder and LLM-based reranking
├── cache.py               # Semantic query cache (cosine similarity LRU)
├── llm/                   # LLM abstraction (OpenAI / Azure / Ollama / Anthropic)
├── adapters/              # Data source adapters (image dataset)
├── tools/                 # 11 tools: dataset, vision, knowledge
├── templates/             # Chat UI + widget demo
├── static/                # CSS + JS
└── api/                   # Routes + Pydantic models

knowledge/                 # Domain knowledge base (Markdown + YAML frontmatter)
├── concepts/              # Thermal inspection, optics, classification, dataset quality
├── parameters/            # Thresholds, sensitivity
├── procedures/            # Getting started, conveyor integration
├── troubleshooting/       # False positives, inconsistent results
└── known-issues/          # KB articles (ambient temp, sensor contamination)

tests/
├── test_core.py           # Unit tests
├── test_orchestrator.py   # Orchestrator tests
├── test_routes.py         # API route tests
├── test_storage.py        # Storage tests
├── eval_rag.py            # RAG evaluation harness (Precision@k, Recall@k, MRR)
└── eval_dataset.json      # 30 gold-standard Q→article eval pairs

data/
├── images/                # Labeled thermal images
│   ├── PASS/
│   └── FAULT/
├── vectorstore/           # ChromaDB persistent store (auto-created)
├── conversations.db       # SQLite persistence (auto-created)
└── reports/               # Audit reports (auto-created)
```

## API

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Chat UI |
| `/health` | GET | Health check |
| `/api/v1/chat` | POST | `{"message": "...", "conversation_id": "..."}` |
| `/api/v1/chat/stream` | POST | Stream response tokens via SSE |
| `/api/v1/tools` | GET | List registered tools |
| `/api/v1/stats` | GET | Conversation stats (active + persisted) |
| `/api/v1/models` | GET | List available models (Ollama + cloud) |
| `/api/v1/models/switch` | POST | Switch LLM provider/model at runtime |
| `/api/v1/conversations` | GET | List persisted conversations |
| `/api/v1/conversations/search?q=` | GET | Full-text search across conversations |
| `/api/v1/conversations/{id}` | GET | Load a conversation with messages |
| `/api/v1/conversations/{id}` | DELETE | Delete a conversation |
| `/api/v1/conversations/{id}/export` | GET | Export conversation as JSON report |
| `/api/v1/conversations/{id}/feedback` | POST | Submit user feedback (thumbs up/down) |
| `/api/v1/feedback/summary` | GET | Aggregate feedback statistics |

## Tools

| Tool | Module | Purpose |
| --- | --- | --- |
| `get_dataset_summary` | dataset | Image counts, class balance, available labels |
| `get_sample_images` | dataset | Random sample filenames from a label folder |
| `get_dataset_statistics` | dataset | Deep analytics: dimensions, file sizes, class balance metrics |
| `analyze_image` | vision | GPT-4o vision analysis of a thermal image |
| `compare_images` | vision | Side-by-side vision comparison of two images |
| `find_suspicious_labels` | vision | Batch mislabel detection via vision |
| `generate_audit_report` | vision | Full dataset quality audit with JSON report saved to disk |
| `search_knowledge` | knowledge | Semantic RAG search across the knowledge base |
| `search_knowledge_filtered` | knowledge | RAG search filtered by category (concepts, troubleshooting, etc.) |
| `get_article_section` | knowledge | Retrieve a specific section from a knowledge article |
| `explain_concept` | knowledge | Look up a specific concept or parameter explanation |

## RAG Evaluation

Run the offline evaluation harness against the gold-standard dataset:

```bash
python -m tests.eval_rag --verbose
# Context Precision@5, Context Recall@5, MRR
# Exits 1 if Recall@5 < 0.8 (CI gate)
```

## Tests

```bash
pytest tests/ -v
```

## Configuration

See [.env.example](.env.example) for all options. Key settings:

**LLM Provider:**
- `LLM_PROVIDER` — `azure_openai`, `openai`, `ollama`, or `anthropic`
- `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY` — Azure credentials
- `OPENAI_API_KEY` — OpenAI credentials
- `ANTHROPIC_API_KEY` — Anthropic Claude credentials
- `OLLAMA_BASE_URL` / `OLLAMA_MODEL` — Ollama config

**Smart Routing:**
- `ROUTING_ENABLED` — enable fast/strong model routing
- `FAST_PROVIDER` / `FAST_MODEL` — lightweight model for simple queries
- `STRONG_PROVIDER` / `STRONG_MODEL` — powerful model for vision/analysis

**RAG Pipeline:**
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — small chunk dimensions (default: 256/32 tokens)
- `PARENT_CHUNK_SIZE` / `PARENT_CHUNK_OVERLAP` — parent chunk dimensions (default: 1024/128 tokens)
- `EMBEDDING_MODEL` — embedding model (default: `text-embedding-3-small`)
- `CONTEXTUAL_RETRIEVAL_ENABLED` — prepend article summary to chunks (default: true)
- `HYBRID_SEARCH_ENABLED` — combine semantic + BM25 via RRF (default: true)
- `RERANKER_ENABLED` / `RERANKER_TYPE` — cross-encoder or LLM reranking (default: true/cross-encoder)
- `HYDE_ENABLED` — hypothetical document embeddings (default: false, adds latency)
- `SEMANTIC_CACHE_ENABLED` — cache similar queries (default: true)
- `MAX_CONTEXT_TOKENS` — cap on tokens injected into LLM context (default: 4096)

**Data:**
- `DATASET_PATH` — path to PASS/FAULT image folders
- `KNOWLEDGE_PATH` — path to knowledge base

## Roadmap

- [x] SSE streaming responses
- [x] SQLite conversation persistence
- [x] Runtime model switching (Ollama/OpenAI/Azure/Anthropic)
- [x] Dataset statistics tool
- [x] Audit report generation
- [x] Conversation export
- [x] Anthropic Claude provider support
- [x] Smart routing (fast/strong model selection)
- [x] RAG pipeline with two-tier document chunking
- [x] ChromaDB vector store with incremental indexing
- [x] Hybrid search (semantic + BM25 via Reciprocal Rank Fusion)
- [x] Contextual retrieval (LLM article summaries on chunks)
- [x] Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
- [x] Parent document retrieval (small search → large context)
- [x] HyDE (Hypothetical Document Embeddings)
- [x] Semantic query caching with TTL
- [x] Agentic multi-hop RAG (chained tool calls)
- [x] User feedback loop with SQLite storage
- [x] RAG evaluation framework (Precision@k, Recall@k, MRR)
- [ ] GUI knowledge base integration (waiting on coworker's GUI)
- [ ] Integration with inspection system API (recipes, runs, results)
- [ ] Embeddable widget for inspection GUI

## License

MIT
