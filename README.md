# InspectAssist

Context-aware LLM assistant for industrial thermal seal inspection. Uses GPT-4o function calling and vision to analyze thermal images, flag mislabels, and provide expert troubleshooting — designed as a sidecar companion to an existing inspection system.

## What It Does

| Capability | Example |
| --- | --- |
| **Analyze thermal images** | "Analyze PASS/img_042.png" → GPT-4o vision interprets seal quality |
| **Flag mislabels** | "Are any PASS images mislabeled?" → samples and audits with vision |
| **Dataset QA** | "What does my dataset look like?" → counts, class balance, labels |
| **Deep analytics** | "Show me dataset statistics" → dimensions, file sizes, imbalance metrics |
| **Audit reports** | "Run a full quality audit" → batch analysis with saved JSON report |
| **Compare images** | "Compare these two images" → side-by-side visual diff with explanation |
| **Troubleshoot** | "Getting too many false positives" → searches knowledge base + data |
| **Explain concepts** | "What makes a good thermal seal?" → domain knowledge articles |
| **Stream responses** | Real-time token streaming via Server-Sent Events |
| **Persist history** | SQLite-backed conversation storage with list/load/delete/export |

## Architecture

```
FastAPI Service
├── Chat API (REST + SSE)     POST /api/v1/chat, /api/v1/chat/stream
├── Orchestrator              System prompt → LLM → tool calls → loop → persist
├── LLM Providers             Azure OpenAI / OpenAI / Ollama (runtime switching)
├── 9 Tools                   Dataset, vision, knowledge — via @tool decorator
├── Conversation Store        SQLite persistence (aiosqlite)
├── Dataset Adapter           Scans PASS/FAULT image folders
├── Knowledge Engine          Markdown + YAML frontmatter search
└── Dev Chat UI               Single-page dark theme at /
```

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
├── config.py              # Settings from .env
├── orchestrator.py        # Conversation + tool dispatch + streaming + persistence
├── storage.py             # SQLite conversation persistence (aiosqlite)
├── knowledge.py           # Markdown knowledge engine
├── llm/                   # LLM abstraction (OpenAI / Azure / Ollama)
├── adapters/              # Data source adapters (image dataset)
├── tools/                 # 9 tools: dataset, vision, knowledge
├── templates/             # Chat UI + widget demo
├── static/                # CSS + JS
└── api/                   # Routes + Pydantic models

knowledge/                 # Domain knowledge base (Markdown + YAML frontmatter)
├── concepts/              # Thermal inspection, classification
├── parameters/            # Thresholds, sensitivity
├── procedures/            # Setup guides
├── troubleshooting/       # False positives, inconsistent results
└── known-issues/          # KB articles (ambient temp, etc.)

data/
├── images/                # Labeled thermal images
│   ├── PASS/
│   └── FAULT/
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
| `/api/v1/conversations/{id}` | GET | Load a conversation with messages |
| `/api/v1/conversations/{id}` | DELETE | Delete a conversation |
| `/api/v1/conversations/{id}/export` | GET | Export conversation as JSON report |

## Tools

| Tool | Purpose |
| --- | --- |
| `get_dataset_summary` | Image counts, class balance, available labels |
| `get_sample_images` | Random sample filenames from a label folder |
| `get_dataset_statistics` | Deep analytics: dimensions, file sizes, class balance metrics |
| `analyze_image` | GPT-4o vision analysis of a thermal image |
| `compare_images` | Side-by-side vision comparison of two images |
| `find_suspicious_labels` | Batch mislabel detection via vision |
| `generate_audit_report` | Full dataset quality audit with JSON report saved to disk |
| `search_knowledge` | Keyword search across knowledge base |
| `explain_concept` | Look up a specific concept or parameter |

## Tests

```bash
pytest tests/ -v    # 19 tests
```

## Configuration

See [.env.example](.env.example) for all options. Key settings:

- `LLM_PROVIDER` — `azure_openai`, `openai`, or `ollama`
- `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY` — Azure credentials
- `OPENAI_API_KEY` — OpenAI credentials
- `OLLAMA_BASE_URL` / `OLLAMA_MODEL` — Ollama config
- `DATASET_PATH` — path to PASS/FAULT image folders
- `KNOWLEDGE_PATH` — path to knowledge base

## Roadmap

- [x] SSE streaming responses
- [x] SQLite conversation persistence
- [x] Runtime model switching (Ollama/OpenAI/Azure)
- [x] Dataset statistics tool
- [x] Audit report generation
- [x] Conversation export
- [x] Vector search for knowledge base (OpenAI embeddings)
- [ ] GUI knowledge base integration (waiting on coworker's GUI)
- [ ] Integration with inspection system API (recipes, runs, results)
- [ ] Embeddable widget for inspection GUI

## License

MIT
