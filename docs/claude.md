# InspectAssist — Claude Context & Progress Log

> Last updated: 2026-04-02

This file provides context for Claude (or any LLM assistant) working on this codebase, and tracks implementation progress.

---

## Project Summary

InspectAssist is a **tool-augmented LLM assistant** for industrial thermal seal inspection. It is a standalone FastAPI service that reads labeled thermal images and a curated knowledge base, uses GPT-4o function calling + vision to analyze images and answer questions, and exposes a REST API for integration into a Python GUI (being built by a coworker).

**Design principles:**
- Sidecar companion — reads system data, never controls it
- Tool-grounded — LLM always calls real functions, never fabricates data
- RAG approach — no model training; foundation models + context injection
- API-first — designed for embedding, dev chat UI is temporary

---

## Repository Layout

```
├── src/inspect_assist/
│   ├── app.py                   # FastAPI app factory — wires all components
│   ├── config.py                # Pydantic Settings from .env
│   ├── orchestrator.py          # Conversation + tool dispatch + streaming + persistence
│   ├── knowledge.py             # Markdown knowledge engine (YAML frontmatter search)
│   ├── storage.py               # SQLite conversation persistence (aiosqlite)
│   ├── llm/
│   │   ├── __init__.py          # Message, Role, ToolCallRequest, ImageContent, LLMResponse
│   │   └── providers.py         # OpenAI + Azure OpenAI + Ollama (unified provider)
│   ├── adapters/
│   │   └── dataset.py           # Image dataset scanner (PASS/FAULT folders)
│   ├── tools/
│   │   ├── __init__.py          # @tool decorator, ToolRegistry, ToolDef
│   │   ├── dataset_tools.py     # get_dataset_summary, get_sample_images, get_dataset_statistics
│   │   ├── vision_tools.py      # analyze_image, compare_images, find_suspicious_labels, generate_audit_report
│   │   └── knowledge_tools.py   # search_knowledge, explain_concept
│   ├── api/
│   │   ├── models.py            # Pydantic request/response models
│   │   └── routes.py            # 15 endpoints (chat, stream, models, conversations, search, export)
│   ├── templates/
│   │   ├── chat.html            # Dev chat UI (dark theme, markdown rendering)
│   │   └── widget_demo.html     # Widget embedding demo
│   └── static/
│       ├── style.css
│       └── widget.js
├── knowledge/                   # Domain knowledge base
│   ├── concepts/                # thermal-seal-inspection, classification-labeling
│   ├── parameters/              # thresholds
│   ├── procedures/              # getting-started
│   ├── troubleshooting/         # false-positives, inconsistent-results
│   └── known-issues/            # KB-001-ambient-temperature
├── data/
│   ├── images/                  # Labeled thermal images
│   │   ├── PASS/
│   │   └── FAULT/
│   ├── conversations.db         # SQLite persistence (auto-created)
│   └── reports/                 # Audit reports (auto-created)
├── tests/test_core.py           # 19 unit tests (dataset, knowledge, tool registry)
├── tests/test_storage.py        # 17 unit tests (serialization, persistence)
├── tests/test_orchestrator.py   # 17 unit tests (chat, tools, suggestions, streaming)
├── tests/test_routes.py         # 21 unit tests (all API endpoints)
├── docs/
│   ├── brief.md                 # Project brief
│   ├── architecture.md          # Full architecture doc
│   └── claude.md                # This file
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Key Patterns

### Tool Framework
Tools are defined with `@tool` decorator in their modules. `ToolRegistry.register_module()` scans for decorated functions. The orchestrator passes tool schemas to the LLM via OpenAI function-calling format.

### Dependency Wiring
Tool modules use module-level `set_*()` functions to receive dependencies. The app factory (`app.py`) calls these during startup. Avoids circular imports and keeps tool functions testable.

### Adapter Pattern
Data sources are accessed through adapter classes (`ImageDatasetAdapter`). When the coworker's inspection system ships, new adapters will be added (API adapter, file-based adapter) implementing the same interface.

### Streaming
`chat_stream()` is an async generator yielding SSE-formatted events. `LLMProvider.stream()` yields text chunks and a final `LLMResponse`. Tool calls are executed between streaming rounds with `tool_start`/`tool_result` events.

### Persistence
`ConversationStore` uses aiosqlite for async SQLite. Messages are serialized to JSON (images excluded to save space). The orchestrator calls `_persist()` after every response. Conversations are also kept in memory for the current session.

---

## Important Constraints

1. **Read-only / advisory** — the assistant never modifies data, recipes, or system state
2. **Tool-grounded** — system prompt instructs the LLM to always use tools, never fabricate data
3. **Sidecar design** — companion to the inspection system, not a replacement
4. **API-first** — the chat UI is temporary; the service is designed for GUI embedding
5. **No model training** — uses foundation models (GPT-4o, Ollama) + RAG + context injection
6. **Images excluded from SQLite** — base64 blobs too large; not needed for conversation replay

---

## Running

```bash
# Dev
pip install -e ".[dev]"
cp .env.example .env   # Fill in API keys
python -m inspect_assist

# Test
pytest tests/ -v       # 74 tests

# Docker
docker compose up
```

---

## Implementation Progress

### Sprint 1 — Safety & Streaming ✅

| ID | Feature | Status |
|----|---------|--------|
| S1 | First-message disclaimer (AI limitations, data locality warning) | ✅ Done |
| S2 | Provider/data-locality metadata on every response | ✅ Done |
| A1 | SSE streaming via `POST /api/v1/chat/stream` | ✅ Done |
| A2 | Image thumbnails as attachments in responses (from vision tools) | ✅ Done |
| B5 | Dynamic suggested follow-ups (`<!--suggestions:[...]-->` parsing) | ✅ Done |

**Files changed:** orchestrator.py, providers.py, models.py, routes.py, vision_tools.py

### Sprint 2 — Conversation Persistence ✅

| ID | Feature | Status |
|----|---------|--------|
| A3 | SQLite conversation persistence (aiosqlite) | ✅ Done |
| — | List/load/delete conversation API endpoints | ✅ Done |
| — | Async `get_stats()` with persisted count | ✅ Done |

**Files changed:** NEW storage.py, orchestrator.py, models.py, routes.py, app.py, pyproject.toml

### Sprint 3 — Data Intelligence & Reporting ✅

| ID | Feature | Status |
|----|---------|--------|
| B4 | `get_dataset_statistics` tool (dimensions, file sizes, class balance metrics) | ✅ Done |
| B1 | `generate_audit_report` tool (full dataset quality audit, saves JSON report) | ✅ Done |
| — | Conversation export endpoint (`GET /conversations/{id}/export`) | ✅ Done |

**Files changed:** dataset_tools.py, vision_tools.py, routes.py

### Sprint 4 — Semantic Search & Polish ✅

| ID | Feature | Status |
|----|---------|--------|
| — | Vector search for knowledge base (OpenAI embeddings + cosine similarity) | ✅ Done |
| — | Embedding cache (JSON file, content-hash invalidation) | ✅ Done |
| — | Graceful fallback to keyword search when embeddings unavailable | ✅ Done |
| — | Runtime model switching (Ollama/OpenAI/Azure) | ✅ Done (shipped with Sprint 1) |
| — | GUI knowledge base integration | ⏸ Deferred (waiting on coworker's GUI) |
| — | Integration with inspection system API | 🔲 Not started (waiting on coworker) |

**Files changed:** knowledge.py, knowledge_tools.py, app.py

### Sprint 5 — Hardening & Integration Prep ✅

| ID | Feature | Status |
|----|---------|--------|
| — | Comprehensive test suite (19 → 74 tests: storage, orchestrator, routes) | ✅ Done |
| — | Conversation search endpoint (`GET /conversations/search?q=`) | ✅ Done |
| — | System prompt tuned (domain context, all 9 tools listed, conciseness) | ✅ Done |
| — | API key authentication middleware | ✅ Done |
| — | Rate limiting (10 req/min on chat endpoints) | ✅ Done |
| — | Batch analysis CLI (`python -m inspect_assist batch`) | ✅ Done |
| — | Knowledge base expanded (4 new articles) | ✅ Done |
| — | Docker build verified | ✅ Done |
| — | Integration guide for GUI developers | ✅ Done |

**Files changed:** test_storage.py (new), test_orchestrator.py (new), test_routes.py (new), routes.py, orchestrator.py, app.py, __main__.py, config.py, Dockerfile, docker-compose.yml, knowledge/ (4 new articles), docs/integration-guide.md (new)

---

## Current Stats

- **9 tools** registered (3 dataset, 4 vision, 2 knowledge)
- **15 API endpoints**
- **74 tests** all passing
- **3 LLM providers** supported (Azure OpenAI, OpenAI, Ollama)
- **Tech stack**: Python 3.13, FastAPI, OpenAI SDK, Pillow, aiosqlite, structlog
- **Semantic search**: OpenAI embeddings with cosine similarity (cached to JSON)
