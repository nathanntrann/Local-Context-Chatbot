# InspectAssist — Architecture

> Last updated: 2026-07-22

## System Overview

```
┌──────────────────────────────────────────────────────────────┐
│  Client (Chat UI / Widget / Future: Python GUI embed)        │
└──────────────────┬───────────────────────────────────────────┘
                   │ HTTP REST + SSE streaming
┌──────────────────▼───────────────────────────────────────────┐
│  FastAPI Application                                         │
│                                                              │
│  ┌─────────┐   ┌──────────────┐   ┌───────────────────────┐ │
│  │ Routes  │──▶│ Orchestrator │──▶│ LLM Provider          │ │
│  │ (API)   │   │              │   │ • Azure OpenAI        │ │
│  └─────────┘   │  System      │   │ • OpenAI              │ │
│                │  Prompt +    │   │ • Ollama              │ │
│                │  Tool Loop   │   │ • Anthropic Claude    │ │
│                │  + Streaming │   └───────────────────────┘ │
│                │              │                              │
│                │        ┌─────▼───┐ ┌─────────────────────┐ │
│                │        │ Tool    │ │ Conversation Store  │ │
│                │        │ Registry│ │ (SQLite / aiosqlite)│ │
│                │        │ 11 tools│ │ + Feedback table    │ │
│                └────────┴──┬──────┘ └─────────────────────┘ │
│                            │                                 │
│         ┌──────────────────┴──────────────┐                  │
│         ▼                                 ▼                  │
│  ┌──────────────────┐      ┌──────────────────────────────┐  │
│  │ Dataset Adapter  │      │ RAG Knowledge Engine         │  │
│  │ (Image Folders)  │      │ ┌──────────┐ ┌────────────┐ │  │
│  └────────┬─────────┘      │ │ ChromaDB │ │ BM25 Index │ │  │
│           │                │ │ Vectors  │ │ (rank-bm25)│ │  │
│           │                │ └──────────┘ └────────────┘ │  │
│           │                │ ┌────────────────────────┐  │  │
│           │                │ │ Reranker (cross-enc.)  │  │  │
│           │                │ └────────────────────────┘  │  │
│           │                │ ┌────────────────────────┐  │  │
│           │                │ │ Semantic Cache (LRU)   │  │  │
│           │                │ └────────────────────────┘  │  │
│           │                └──────────────┬───────────────┘  │
└───────────┼──────────────────────────────┼───────────────────┘
            ▼                              ▼
    data/images/                    knowledge/
    ├── PASS/                       ├── concepts/
    └── FAULT/                      ├── parameters/
                                    ├── procedures/
    data/conversations.db           ├── troubleshooting/
    data/vectorstore/               └── known-issues/
    data/reports/
```

## Component Details

### API Layer (`src/inspect_assist/api/`)

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Chat UI — SSE streaming, tool indicators, suggestions, image attachments, conversation sidebar |
| `/widget-demo` | GET | Widget embedding demo page |
| `/health` | GET | Health check |
| `/api/v1/chat` | POST | Send message, get response (with tool calls, attachments, suggestions) |
| `/api/v1/chat/stream` | POST | Stream response tokens via Server-Sent Events |
| `/api/v1/tools` | GET | List registered tools and descriptions |
| `/api/v1/stats` | GET | Active + persisted conversation counts |
| `/api/v1/models` | GET | List available models (Ollama local + cloud) |
| `/api/v1/models/switch` | POST | Switch LLM provider/model at runtime |
| `/api/v1/conversations` | GET | List persisted conversations |
| `/api/v1/conversations/search?q=` | GET | Full-text search across conversations |
| `/api/v1/conversations/{id}` | GET | Load a conversation with messages |
| `/api/v1/conversations/{id}` | DELETE | Delete a persisted conversation |
| `/api/v1/conversations/{id}/export` | GET | Export conversation as a downloadable JSON report |
| `/api/v1/conversations/{id}/feedback` | POST | Submit user feedback (thumbs up/down) on a response |
| `/api/v1/feedback/summary` | GET | Aggregate feedback statistics (positive, negative, rate) |

### Orchestrator (`src/inspect_assist/orchestrator.py`)

Manages multi-turn conversations with a tool dispatch loop:

1. Receives user message → appends to conversation history
2. Sends history + tool schemas to LLM
3. If LLM returns tool calls → executes them via ToolRegistry → appends results → loops back to step 2
4. If LLM returns text → extracts suggestions + attachments → returns `ChatResult`
5. Enforces turn limits and tool call limits per turn
6. Persists conversation to SQLite after every response

Key behaviors:
- **First-message disclaimer** — new conversations get a one-time advisory about AI limitations and data locality
- **Provider metadata** — every response includes the LLM provider name and data locality info
- **Suggested follow-ups** — LLM embeds `<!--suggestions:[...]-->` comments, orchestrator parses and strips them
- **Image attachments** — vision tool results include base64 thumbnails, surfaced as attachments in the response
- **Streaming** — `chat_stream()` yields SSE-formatted events (token, tool_start, tool_result, done)

Conversations are in-memory with auto-pruning (keeps last 100) + SQLite persistence.

### Chat UI (`src/inspect_assist/templates/chat.html`)

Full-featured single-page chat interface served at `/`:

- **SSE streaming** — real-time token-by-token rendering via `POST /api/v1/chat/stream` with `ReadableStream`
- **Tool activity indicators** — spinner + human-readable label (e.g. "Searching knowledge base…") during tool execution
- **Follow-up suggestions** — clickable chips rendered from the `suggestions` array in the `done` SSE event
- **Image attachments** — base64 thumbnails from vision tools displayed inline with click-to-expand lightbox
- **Conversation sidebar** — collapsible 280px panel with search, resume, delete, and export for persisted conversations
- **Dark theme** — CSS custom properties, markdown rendering via `marked.js`, syntax highlighting via `highlight.js`
- **Mobile responsive** — sidebar collapses to overlay on screens < 768px

### Embeddable Widget (`src/inspect_assist/static/widget.js`)

Self-contained chat widget for embedding into external applications:

- **SSE streaming** — same `ReadableStream` pattern as the main chat UI
- **Tool indicators** — spinner + labels during tool execution
- **Suggestion chips** — follow-up suggestions rendered as clickable buttons
- Compact 300×340px floating panel with header, message area, and input

### LLM Abstraction (`src/inspect_assist/llm/`)

- `LLMProviderProtocol` — async `chat()` and `stream()` methods
- `OpenAIProvider` — unified implementation for OpenAI, Azure OpenAI, and Ollama (all OpenAI-compatible)
- `AnthropicProvider` — Anthropic Claude support via `anthropic` SDK
- `stream()` — async generator yielding text chunks (`str`) and a final `LLMResponse` with tool calls
- `provider_name` / `data_locality` properties for transparency metadata
- Supports text + vision (multi-image) via `ImageContent` dataclass
- Runtime model switching via settings mutation + provider rebuild
- Smart routing: configurable fast/strong model pair — simple queries go to the cheaper model, vision/analysis goes to the strong model

### Conversation Store (`src/inspect_assist/storage.py`)

- Async SQLite persistence via `aiosqlite`
- `save()` — upsert conversation (id, title, model, timestamps, serialized messages)
- `load()` — returns `list[Message]` for conversation resumption
- `load_detail()` — returns full metadata + messages dict for API responses
- `list_conversations()` — ordered by most recent, with pagination
- `delete()` / `count()` — housekeeping operations
- Title auto-derived from first user message (truncated to 80 chars)
- Images (base64) intentionally excluded from serialization to keep DB small
- **Feedback table** — stores user ratings (thumbs up/down) per response with query text and retrieved chunk metadata for RAG quality tracking
- `save_feedback()` / `get_feedback_summary()` — aggregate satisfaction metrics

### Tool Framework (`src/inspect_assist/tools/__init__.py`)

- `@tool` decorator with name, description, and typed parameter definitions
- `ToolRegistry` discovers tools from modules, generates OpenAI function-calling schemas
- `ToolRegistry.call()` dispatches by name, handles JSON serialization, catches errors

### Dataset Adapter (`src/inspect_assist/adapters/dataset.py`)

- Scans `data/images/` for label-named subdirectories (PASS, FAULT, etc.)
- Caches file listings with manual invalidation
- Provides: summary stats, filtered image lists, random sampling, path/name lookup

### Knowledge Engine — RAG Pipeline (`src/inspect_assist/knowledge.py`)

Full retrieval-augmented generation pipeline over Markdown + YAML frontmatter articles:

- Loads all `.md` files recursively from `knowledge/`
- Parses YAML frontmatter (title, category, tags, custom fields)
- **Two-tier document chunking** (`chunking.py`) — small chunks (256 tokens, 32 overlap) for precise search + parent chunks (1024 tokens, 128 overlap) for rich context. Uses `langchain-text-splitters` with markdown-aware separators.
- **Contextual retrieval** — LLM generates a short article summary prepended to each chunk before embedding, improving retrieval for ambiguous queries
- **ChromaDB vector store** (`vectorstore.py`) — persistent embedding store with two collections (`knowledge_chunks` for search, `knowledge_parents` for parent context lookup). Incremental re-indexing via content hashing.
- **Hybrid search with RRF** — combines semantic search (ChromaDB cosine similarity) with lexical search (BM25 via `rank-bm25`). Merged using Reciprocal Rank Fusion: `score(d) = Σ 1/(k + rank(d))` with k=60.
- **Cross-encoder reranking** (`reranker.py`) — top candidates reranked by `cross-encoder/ms-marco-MiniLM-L-6-v2` from `sentence-transformers`. Falls back to LLM-based reranking if model unavailable.
- **Parent document retrieval** — after reranking, small search chunks mapped back to their parent chunks for richer context
- **HyDE (optional)** — Hypothetical Document Embeddings: generates a hypothetical answer and searches with that embedding
- **Semantic cache** (`cache.py`) — LRU cache with cosine similarity matching (threshold 0.95) and configurable TTL. Avoids redundant searches.
- **Legacy fallback** — keyword search with weighted scoring (title 3x > tags 2x > category 1.5x > content 0.5x) when RAG index not available

## Data Model

```
ImageInfo
├── path: Path
├── label: str          # Parent folder name (PASS / FAULT)
├── filename: str
└── size_bytes: int

DatasetSummary
├── total_images: int
├── pass_count / fault_count: int
├── pass_ratio / fault_ratio: float
├── labels: list[str]
└── path: str

KnowledgeArticle
├── slug: str           # Filename stem, e.g. "false-positives"
├── title / category / tags / content / metadata / path

Conversation (in-memory + SQLite)
├── id: str             # 12-char hex UUID
├── messages: list[Message]
├── total_tokens: int
└── tool_calls_count: int

Message
├── role: Role          # system | user | assistant | tool
├── content: str
├── tool_call_id: str?
├── tool_calls: list[ToolCallRequest]?
├── images: list[ImageContent]?
└── name: str?

ChatResult (orchestrator output)
├── response: str       # Clean text (suggestions stripped)
├── conversation_id: str
├── provider: str       # e.g. "Azure OpenAI"
├── data_locality: str  # e.g. "Cloud (Azure US East)"
├── attachments: list[dict]  # Image thumbnails from vision tools
└── suggestions: list[str]   # Parsed follow-up suggestions
```

### SQLite Schema

```sql
conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    messages TEXT NOT NULL DEFAULT '[]'  -- JSON-serialized Message list
)

feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    query TEXT NOT NULL DEFAULT '',
    retrieved_chunks TEXT NOT NULL DEFAULT '[]',  -- JSON-serialized chunk metadata
    rating INTEGER NOT NULL,                       -- 1 = thumbs up, -1 = thumbs down
    created_at TEXT NOT NULL
)
```

### Future (v2+) — When Inspection System Ships

The coworker's inspection system will provide additional data sources. The adapter pattern allows adding new adapters without changing assistant logic.

Expected additions:
- **Recipes** — named parameter sets (threshold, sensitivity, model, ROI config)
- **Inspection Runs** — batches of results with timestamps, recipe used, version info
- **Inspection Results** — per-image classification outcome, confidence score, defect type
- **System Logs** — hardware events, errors, timing data

## Tool Registry (11 tools)

| Tool | Module | Inputs | Purpose |
| --- | --- | --- | --- |
| `get_dataset_summary` | dataset_tools | — | Image counts, class balance, labels |
| `get_sample_images` | dataset_tools | label, count? | Random sample filenames from a label |
| `get_dataset_statistics` | dataset_tools | — | Deep analytics: dimensions, file sizes, class balance metrics |
| `analyze_image` | vision_tools | image_path | GPT-4o vision analysis of one image |
| `compare_images` | vision_tools | image_path_1, image_path_2 | Side-by-side vision comparison |
| `find_suspicious_labels` | vision_tools | label, sample_size? | Batch mislabel detection |
| `generate_audit_report` | vision_tools | sample_size? | Full dataset quality audit with report saved to disk |
| `search_knowledge` | knowledge_tools | query, limit? | Semantic RAG search across knowledge base |
| `search_knowledge_filtered` | knowledge_tools | query, category, limit? | RAG search filtered by category (concepts, troubleshooting, etc.) |
| `get_article_section` | knowledge_tools | article_slug, heading | Retrieve a specific section from a knowledge article |
| `explain_concept` | knowledge_tools | concept | Slug lookup or RAG search for a concept |

## Configuration

All settings via environment variables (`.env` file), loaded by Pydantic Settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | `azure_openai`, `openai`, `ollama`, or `anthropic` |
| `AZURE_OPENAI_ENDPOINT` | — | Azure resource URL |
| `AZURE_OPENAI_API_KEY` | — | Azure API key |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o` | Deployment name |
| `AZURE_OPENAI_API_VERSION` | `2024-10-21` | Azure API version |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model name |
| `ANTHROPIC_API_KEY` | — | Anthropic Claude API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Anthropic model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model name |
| `ROUTING_ENABLED` | `false` | Enable smart fast/strong model routing |
| `FAST_PROVIDER` / `FAST_MODEL` | — | Lightweight model for simple queries |
| `STRONG_PROVIDER` / `STRONG_MODEL` | — | Powerful model for vision/analysis |
| `APP_HOST` | `0.0.0.0` | FastAPI bind address |
| `APP_PORT` | `8000` | FastAPI port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DATASET_PATH` | `./data/images` | Path to labeled image folders |
| `KNOWLEDGE_PATH` | `./knowledge` | Path to knowledge base |
| `MAX_CONVERSATION_TURNS` | `50` | Max user messages per conversation |
| `MAX_TOOL_CALLS_PER_TURN` | `5` | Max tool call rounds before forcing summary |
| `VISION_MAX_IMAGE_SIZE_PX` | `1024` | Max image dimension for vision API |
| `VISION_SAMPLE_SIZE` | `8` | Default batch size for image audits |
| `CHUNK_SIZE` | `256` | Small chunk size in tokens |
| `CHUNK_OVERLAP` | `32` | Overlap between small chunks |
| `PARENT_CHUNK_SIZE` | `1024` | Parent chunk size in tokens |
| `PARENT_CHUNK_OVERLAP` | `128` | Overlap between parent chunks |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for vector search |
| `CONTEXTUAL_RETRIEVAL_ENABLED` | `true` | Prepend article summary to chunks |
| `HYBRID_SEARCH_ENABLED` | `true` | Combine semantic + BM25 via RRF |
| `RRF_K` | `60` | Reciprocal Rank Fusion parameter |
| `RERANKER_ENABLED` | `true` | Cross-encoder or LLM reranking |
| `RERANKER_TYPE` | `cross-encoder` | `cross-encoder` or `llm` |
| `HYDE_ENABLED` | `false` | Hypothetical Document Embeddings |
| `SEMANTIC_CACHE_ENABLED` | `true` | Cache similar queries |
| `MAX_CONTEXT_TOKENS` | `4096` | Max tokens injected into LLM context |

## Deployment

- **Local dev**: `pip install -e ".[dev]"` → `python -m inspect_assist`
- **Docker**: `docker compose up` — mounts `data/images` and `knowledge/` as volumes
- **Production**: Same Docker image, provide `.env` with production API keys
