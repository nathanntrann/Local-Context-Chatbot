# InspectAssist — Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────┐
│  Client (Dev Chat UI / Future: Inspection GUI embed)         │
└──────────────────┬───────────────────────────────────────────┘
                   │ HTTP REST / WebSocket (future)
┌──────────────────▼───────────────────────────────────────────┐
│  FastAPI Application                                         │
│                                                              │
│  ┌─────────┐   ┌──────────────┐   ┌───────────────────────┐ │
│  │ Routes  │──▶│ Orchestrator │──▶│ LLM Provider          │ │
│  │ (API)   │   │              │   │ • Azure OpenAI        │ │
│  └─────────┘   │  System      │   │ • OpenAI              │ │
│                │  Prompt +    │   │ • Ollama (future)     │ │
│                │  Tool Loop   │   └───────────────────────┘ │
│                │              │                              │
│                │        ┌─────▼──────────┐                   │
│                │        │ Tool Registry  │                   │
│                │        │ 7 tools        │                   │
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
                                        ├── troubleshooting/
                                        └── known-issues/
```

## Component Details

### API Layer (`src/inspect_assist/api/`)

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Dev chat UI (Jinja2 HTML) |
| `/health` | GET | Health check |
| `/api/v1/chat` | POST | Send message, get response (with tool calls) |
| `/api/v1/tools` | GET | List registered tools and descriptions |
| `/api/v1/stats` | GET | Active conversations, usage stats |

### Orchestrator (`src/inspect_assist/orchestrator.py`)

Manages multi-turn conversations with a tool dispatch loop:

1. Receives user message → appends to conversation history
2. Sends history + tool schemas to LLM
3. If LLM returns tool calls → executes them via ToolRegistry → appends results → loops back to step 2
4. If LLM returns text → returns to user
5. Enforces turn limits and tool call limits per turn

Conversations are in-memory with auto-pruning (keeps last 100).

### LLM Abstraction (`src/inspect_assist/llm/`)

- `LLMProviderProtocol` — async `chat()` method accepting messages + tool schemas
- `OpenAIProvider` — unified implementation for both OpenAI and Azure OpenAI
- Supports text + vision (multi-image) via `ImageContent` dataclass
- Message types: `Message`, `ToolCallRequest`, `LLMResponse`

### Tool Framework (`src/inspect_assist/tools/__init__.py`)

- `@tool` decorator with name, description, and typed parameter definitions
- `ToolRegistry` discovers tools from modules, generates OpenAI function-calling schemas
- `ToolRegistry.call()` dispatches by name, handles JSON serialization, catches errors

### Dataset Adapter (`src/inspect_assist/adapters/dataset.py`)

- Scans `data/images/` for label-named subdirectories (PASS, FAULT, etc.)
- Caches file listings with manual invalidation
- Provides: summary stats, filtered image lists, random sampling, path-based lookup

### Knowledge Engine (`src/inspect_assist/knowledge.py`)

- Loads all `.md` files recursively from `knowledge/`
- Parses YAML frontmatter (title, category, tags, custom fields)
- Keyword search with weighted scoring: title (3x) > tags (2x) > category (1.5x) > content (0.5x per hit)
- Lookup by slug or category

## Data Model

### Current (v1) — No Database

```
ImageInfo
├── path: Path          # Absolute path to image file
├── label: str          # Parent folder name (PASS / FAULT)
├── filename: str       # e.g. "img_001.png"
└── size_bytes: int

DatasetSummary
├── total_images: int
├── pass_count: int
├── fault_count: int
├── pass_ratio: float
├── fault_ratio: float
├── labels: list[str]   # All discovered label folder names
└── path: str

KnowledgeArticle
├── slug: str           # Filename stem, e.g. "false-positives"
├── title: str          # From YAML frontmatter
├── category: str       # From frontmatter or parent folder name
├── tags: list[str]     # From frontmatter
├── content: str        # Markdown body (after frontmatter)
├── metadata: dict      # Full frontmatter dict
└── path: Path

Conversation (in-memory)
├── id: str             # 12-char hex UUID
├── messages: list[Message]
├── total_tokens: int
└── tool_calls_count: int

Message
├── role: Role          # system | user | assistant | tool
├── content: str
├── tool_call_id: str?  # For tool result messages
├── tool_calls: list[ToolCallRequest]?  # For assistant tool-call messages
├── images: list[ImageContent]?         # For vision requests
└── name: str?          # Tool name for tool result messages
```

### Future (v2+) — When Inspection System Ships

The coworker's inspection system will provide additional data sources. The adapter pattern (`SystemContextProvider` protocol) allows adding new adapters without changing assistant logic.

Expected additions:
- **Recipes** — named parameter sets (threshold, sensitivity, model, ROI config)
- **Inspection Runs** — batches of results with timestamps, recipe used, version info
- **Inspection Results** — per-image classification outcome, confidence score, defect type
- **System Logs** — hardware events, errors, timing data
- **Version Info** — software versions, changelog, config diffs

These will be accessed via API adapters or file-based fallback adapters, plugged into new tools like `get_current_recipe`, `compare_recipes`, `get_recent_results`, `compare_versions`.

## Tool Registry

| Tool | Module | Inputs | Purpose |
| --- | --- | --- | --- |
| `get_dataset_summary` | dataset_tools | — | Image counts, class balance, labels |
| `get_sample_images` | dataset_tools | label, count? | Random sample filenames from a label |
| `analyze_image` | vision_tools | image_path | GPT-4o vision analysis of one image |
| `compare_images` | vision_tools | image_path_1, image_path_2 | Side-by-side vision comparison |
| `find_suspicious_labels` | vision_tools | label, sample_size? | Batch mislabel detection |
| `search_knowledge` | knowledge_tools | query, limit? | Keyword search across knowledge base |
| `explain_concept` | knowledge_tools | concept | Slug lookup or search for a concept |

## Configuration

All settings via environment variables (`.env` file), loaded by Pydantic Settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `azure_openai` | `azure_openai` or `openai` |
| `AZURE_OPENAI_ENDPOINT` | — | Azure resource URL |
| `AZURE_OPENAI_API_KEY` | — | Azure API key |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o` | Deployment name |
| `DATASET_PATH` | `./data/images` | Path to labeled image folders |
| `KNOWLEDGE_PATH` | `./knowledge` | Path to knowledge base |
| `MAX_CONVERSATION_TURNS` | `50` | Max user messages per conversation |
| `MAX_TOOL_CALLS_PER_TURN` | `5` | Max tool call rounds before forcing summary |

## Deployment

- **Local dev**: `pip install -e ".[dev]"` → `python -m inspect_assist`
- **Docker**: `docker compose up` — mounts `data/images` and `knowledge/` as read-only volumes
- **Production**: Same Docker image, provide `.env` with production API keys
