# InspectAssist

Context-aware LLM assistant for industrial thermal seal inspection. Uses GPT-4o function calling and vision to analyze thermal images, flag mislabels, and provide expert troubleshooting — designed as a sidecar companion to an existing inspection system.

## What It Does

| Capability | Example |
| --- | --- |
| **Analyze thermal images** | "Analyze PASS/img_042.png" → GPT-4o vision interprets seal quality |
| **Flag mislabels** | "Are any PASS images mislabeled?" → samples and audits with vision |
| **Dataset QA** | "What does my dataset look like?" → counts, class balance, labels |
| **Compare images** | "Compare these two images" → side-by-side visual diff with explanation |
| **Troubleshoot** | "Getting too many false positives" → searches knowledge base + data |
| **Explain concepts** | "What makes a good thermal seal?" → domain knowledge articles |

## Architecture

```
FastAPI Service
├── Chat API (REST)           POST /api/v1/chat
├── Orchestrator              System prompt → LLM → tool calls → loop
├── LLM Providers             Azure OpenAI / OpenAI (abstracted)
├── 7 Tools                   Dataset, vision, knowledge — via @tool decorator
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
├── orchestrator.py        # Conversation + tool dispatch loop
├── knowledge.py           # Markdown knowledge engine
├── llm/                   # LLM abstraction + OpenAI/Azure impl
├── adapters/              # Data source adapters (image dataset)
├── tools/                 # 7 tools: dataset, vision, knowledge
├── templates/chat.html    # Dev chat UI
└── api/                   # Routes + Pydantic models

knowledge/                 # Domain knowledge base (Markdown + YAML frontmatter)
├── concepts/              # Thermal inspection, classification
├── parameters/            # Thresholds, sensitivity
├── procedures/            # Setup guides
├── troubleshooting/       # False positives, inconsistent results
└── known-issues/          # KB articles (ambient temp, etc.)

data/images/               # Labeled thermal images
├── PASS/
└── FAULT/
```

## API

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Chat UI |
| `/health` | GET | Health check |
| `/api/v1/chat` | POST | `{"message": "...", "conversation_id": "..."}` |
| `/api/v1/tools` | GET | List registered tools |
| `/api/v1/stats` | GET | Conversation stats |

## Tools

| Tool | Purpose |
| --- | --- |
| `get_dataset_summary` | Image counts, class balance, available labels |
| `get_sample_images` | Random sample filenames from a label folder |
| `analyze_image` | GPT-4o vision analysis of a thermal image |
| `compare_images` | Side-by-side vision comparison of two images |
| `find_suspicious_labels` | Batch mislabel detection via vision |
| `search_knowledge` | Keyword search across knowledge base |
| `explain_concept` | Look up a specific concept or parameter |

## Tests

```bash
pytest tests/ -v
```

## Configuration

See [.env.example](.env.example) for all options. Key settings:

- `LLM_PROVIDER` — `azure_openai` or `openai`
- `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY` — Azure credentials
- `OPENAI_API_KEY` — OpenAI credentials (if using OpenAI provider)
- `DATASET_PATH` — path to PASS/FAULT image folders
- `KNOWLEDGE_PATH` — path to knowledge base

## Roadmap

- [ ] Integration with inspection system API (recipes, runs, results)
- [ ] Ollama local model fallback
- [ ] WebSocket streaming responses
- [ ] Persistent conversation storage
- [ ] Vector search for knowledge base
- [ ] Embeddable widget for inspection GUI

## License

MIT
