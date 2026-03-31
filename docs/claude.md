# InspectAssist — Claude Context

This file provides context for Claude (or any LLM assistant) working on this codebase.

## Project Summary

InspectAssist is a **tool-augmented LLM assistant** for industrial thermal seal inspection. It is a standalone FastAPI service that reads labeled thermal images and a curated knowledge base, uses GPT-4o function calling + vision to analyze images and answer questions, and exposes a REST API for integration.

## Repository Layout

```
├── src/inspect_assist/          # Application source
│   ├── app.py                   # FastAPI app factory — wires all components
│   ├── config.py                # Pydantic Settings from .env
│   ├── orchestrator.py          # Conversation manager + tool dispatch loop
│   ├── knowledge.py             # Markdown knowledge engine (YAML frontmatter)
│   ├── llm/
│   │   ├── __init__.py          # Message types, LLMProviderProtocol, ImageContent
│   │   └── providers.py         # OpenAI + Azure OpenAI implementation
│   ├── adapters/
│   │   └── dataset.py           # Image dataset scanner (PASS/FAULT folders)
│   ├── tools/
│   │   ├── __init__.py          # @tool decorator, ToolRegistry, ToolDef
│   │   ├── dataset_tools.py     # get_dataset_summary, get_sample_images
│   │   ├── vision_tools.py      # analyze_image, compare_images, find_suspicious_labels
│   │   └── knowledge_tools.py   # search_knowledge, explain_concept
│   ├── api/
│   │   ├── models.py            # Pydantic request/response models
│   │   └── routes.py            # /health, /api/v1/chat, /api/v1/tools, /api/v1/stats
│   ├── templates/chat.html      # Dev chat UI (dark theme, markdown rendering)
│   └── static/style.css         # (minimal, styles are inline in chat.html)
├── knowledge/                   # Domain knowledge base (Markdown + YAML frontmatter)
│   ├── concepts/                # What things are (thermal inspection, labeling)
│   ├── parameters/              # What parameters do (thresholds, sensitivity)
│   ├── procedures/              # How-to guides (setup, calibration)
│   ├── troubleshooting/         # Problem diagnosis (false positives, inconsistency)
│   └── known-issues/            # Known bugs/behaviors (KB-001, KB-002, ...)
├── data/images/                 # Labeled thermal images
│   ├── PASS/                    # Good seal images
│   └── FAULT/                   # Defective seal images
├── tests/test_core.py           # Unit tests (dataset adapter, knowledge engine, tool registry)
├── docs/                        # Project documentation
├── pyproject.toml               # Dependencies + tool config
├── Dockerfile                   # Python 3.11-slim container
├── docker-compose.yml           # Single-service compose with volume mounts
└── .env.example                 # Environment variable template
```

## Key Patterns

### Tool Framework
Tools are defined with a `@tool` decorator in their respective modules. The decorator attaches a `ToolDef` to the function. `ToolRegistry.register_module()` scans modules for decorated functions. The orchestrator passes tool schemas to the LLM via OpenAI function-calling format.

### Dependency Wiring
Tool modules use module-level `set_*()` functions to receive their dependencies (dataset adapter, LLM provider, knowledge engine). The app factory in `app.py` calls these during startup. This avoids circular imports and keeps tool functions testable.

### Adapter Pattern
Data sources are accessed through adapter classes (`ImageDatasetAdapter`). When the coworker's inspection system ships, new adapters will be added (API adapter, file-based adapter) implementing the same interface pattern. Tools call adapters — they don't know where data comes from.

### Knowledge Base
Articles are Markdown files with YAML frontmatter (`---` delimited). The `KnowledgeEngine` loads them recursively, parses frontmatter for metadata, and provides keyword search with weighted scoring (title > tags > category > content).

## Important Constraints

1. **Read-only / advisory** — the assistant never modifies data, recipes, or system state
2. **Tool-grounded** — system prompt instructs the LLM to always use tools, never fabricate data
3. **Sidecar design** — this is a companion to the inspection system, not a replacement
4. **No database** — v1 uses file system only (image folders + Markdown knowledge base)
5. **API-first** — the chat UI is temporary; the service is designed for API integration

## Running

```bash
# Dev
pip install -e ".[dev]"
cp .env.example .env   # Fill in API keys
python -m inspect_assist

# Test
pytest tests/ -v

# Docker
docker compose up
```

## What's Coming (v2+)

- Integration with coworker's inspection system (recipes, runs, results via API)
- Additional tools: `get_current_recipe`, `compare_recipes`, `get_recent_results`, `compare_versions`
- Ollama local model fallback
- WebSocket streaming for real-time responses
- Persistent conversation storage
- Vector search for knowledge base
