# InspectAssist — Architecture

> Last updated: 2026-04-02

## System Overview

```
┌──────────────────────────────────────────────────────────────┐
│  Client (Dev Chat UI / Future: Python GUI embed)             │
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
│                │  Tool Loop   │   └───────────────────────┘ │
│                │  + Streaming │                              │
│                │              │   ┌───────────────────────┐ │
│                │        ┌─────▼───│ Conversation Store    │ │
│                │        │ Tool    │ (SQLite / aiosqlite)  │ │
│                │        │ Registry│                       │ │
│                │        │ 9 tools └───────────────────────┘ │
│                └────────┴──┬──────┬──────┘                   │
│                            │      │                          │
│         ┌──────────────────┘      └──────────────┐           │
│         ▼                                        ▼           │
│  ┌──────────────────┐                ┌────────────────────┐  │
│  │ Dataset Adapter  │                │ Knowledge Engine   │  │
│  │ (Image Folders)  │                │ (Markdown + YAML)  │  │
│  └────────┬─────────┘                └────────┬───────────┘  │
└───────────┼──────────────────────────────────┼───────────────┘
            ▼                                  ▼
    data/images/                        knowledge/
    ├── PASS/                           ├── concepts/
    └── FAULT/                          ├── parameters/
                                        ├── procedures/
    data/conversations.db               ├── troubleshooting/
    data/reports/                        └── known-issues/
```

## Component Details

### API Layer (`src/inspect_assist/api/`)

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Dev chat UI (Jinja2 HTML) |
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

### LLM Abstraction (`src/inspect_assist/llm/`)

- `LLMProviderProtocol` — async `chat()` and `stream()` methods
- `OpenAIProvider` — unified implementation for OpenAI, Azure OpenAI, and Ollama
- `stream()` — async generator yielding text chunks (`str`) and a final `LLMResponse` with tool calls
- `provider_name` / `data_locality` properties for transparency metadata
- Supports text + vision (multi-image) via `ImageContent` dataclass
- Runtime model switching via settings mutation + provider rebuild

### Conversation Store (`src/inspect_assist/storage.py`)

- Async SQLite persistence via `aiosqlite`
- `save()` — upsert conversation (id, title, model, timestamps, serialized messages)
- `load()` — returns `list[Message]` for conversation resumption
- `load_detail()` — returns full metadata + messages dict for API responses
- `list_conversations()` — ordered by most recent, with pagination
- `delete()` / `count()` — housekeeping operations
- Title auto-derived from first user message (truncated to 80 chars)
- Images (base64) intentionally excluded from serialization to keep DB small

### Tool Framework (`src/inspect_assist/tools/__init__.py`)

- `@tool` decorator with name, description, and typed parameter definitions
- `ToolRegistry` discovers tools from modules, generates OpenAI function-calling schemas
- `ToolRegistry.call()` dispatches by name, handles JSON serialization, catches errors

### Dataset Adapter (`src/inspect_assist/adapters/dataset.py`)

- Scans `data/images/` for label-named subdirectories (PASS, FAULT, etc.)
- Caches file listings with manual invalidation
- Provides: summary stats, filtered image lists, random sampling, path/name lookup

### Knowledge Engine (`src/inspect_assist/knowledge.py`)

- Loads all `.md` files recursively from `knowledge/`
- Parses YAML frontmatter (title, category, tags, custom fields)
- Keyword search with weighted scoring: title (3x) > tags (2x) > category (1.5x) > content (0.5x per hit)
- Lookup by slug or category

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
```

### Future (v2+) — When Inspection System Ships

The coworker's inspection system will provide additional data sources. The adapter pattern allows adding new adapters without changing assistant logic.

Expected additions:
- **Recipes** — named parameter sets (threshold, sensitivity, model, ROI config)
- **Inspection Runs** — batches of results with timestamps, recipe used, version info
- **Inspection Results** — per-image classification outcome, confidence score, defect type
- **System Logs** — hardware events, errors, timing data

## Tool Registry (9 tools)

| Tool | Module | Inputs | Purpose |
| --- | --- | --- | --- |
| `get_dataset_summary` | dataset_tools | — | Image counts, class balance, labels |
| `get_sample_images` | dataset_tools | label, count? | Random sample filenames from a label |
| `get_dataset_statistics` | dataset_tools | — | Deep analytics: dimensions, file sizes, class balance metrics |
| `analyze_image` | vision_tools | image_path | GPT-4o vision analysis of one image |
| `compare_images` | vision_tools | image_path_1, image_path_2 | Side-by-side vision comparison |
| `find_suspicious_labels` | vision_tools | label, sample_size? | Batch mislabel detection |
| `generate_audit_report` | vision_tools | sample_size? | Full dataset quality audit with report saved to disk |
| `search_knowledge` | knowledge_tools | query, limit? | Keyword search across knowledge base |
| `explain_concept` | knowledge_tools | concept | Slug lookup or search for a concept |

## Configuration

All settings via environment variables (`.env` file), loaded by Pydantic Settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | `azure_openai`, `openai`, or `ollama` |
| `AZURE_OPENAI_ENDPOINT` | — | Azure resource URL |
| `AZURE_OPENAI_API_KEY` | — | Azure API key |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o` | Deployment name |
| `AZURE_OPENAI_API_VERSION` | `2024-10-21` | Azure API version |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model name |
| `APP_HOST` | `0.0.0.0` | FastAPI bind address |
| `APP_PORT` | `8000` | FastAPI port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DATASET_PATH` | `./data/images` | Path to labeled image folders |
| `KNOWLEDGE_PATH` | `./knowledge` | Path to knowledge base |
| `MAX_CONVERSATION_TURNS` | `50` | Max user messages per conversation |
| `MAX_TOOL_CALLS_PER_TURN` | `5` | Max tool call rounds before forcing summary |
| `VISION_MAX_IMAGE_SIZE_PX` | `1024` | Max image dimension for vision API |
| `VISION_SAMPLE_SIZE` | `8` | Default batch size for image audits |

## Deployment

- **Local dev**: `pip install -e ".[dev]"` → `python -m inspect_assist`
- **Docker**: `docker compose up` — mounts `data/images` and `knowledge/` as volumes
- **Production**: Same Docker image, provide `.env` with production API keys
