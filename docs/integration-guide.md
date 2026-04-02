# InspectAssist Integration Guide

> For developers integrating InspectAssist into a Python GUI or other application.

## Quick Start

```bash
# 1. Install and run the service
pip install -e "."
cp .env.example .env   # Configure LLM provider
python -m inspect_assist
# Server starts on http://localhost:8000
```

## Authentication

If `API_KEY` is set in `.env`, all `/api/` endpoints require an `X-API-Key` header:

```python
headers = {"X-API-Key": "your-secret-key"}
```

If `API_KEY` is empty (default), no authentication is required.

## Core Endpoints

### Health Check

```python
import requests

resp = requests.get("http://localhost:8000/health")
# {"status": "ok", "version": "0.1.0"}
```

### Send a Chat Message

```python
resp = requests.post("http://localhost:8000/api/v1/chat", json={
    "message": "How many images are in the dataset?",
    "conversation_id": None,  # omit for new conversation
})

data = resp.json()
print(data["response"])           # AI response text (markdown)
print(data["conversation_id"])    # Use this for follow-up messages
print(data["provider"])           # e.g. "Ollama (llama3.1:8b)"
print(data["data_locality"])      # "local" or "cloud"
print(data["suggestions"])        # ["Ask about class balance", ...]
print(data["attachments"])        # Image thumbnails (base64)
```

### Continue a Conversation

```python
resp = requests.post("http://localhost:8000/api/v1/chat", json={
    "message": "Show me a random FAULT image",
    "conversation_id": data["conversation_id"],  # from previous response
})
```

### Streaming (Server-Sent Events)

For real-time token streaming in a GUI:

```python
import requests
import json

resp = requests.post(
    "http://localhost:8000/api/v1/chat/stream",
    json={"message": "Analyze this image: PASS/img_001.png"},
    stream=True,
)

for line in resp.iter_lines():
    if line:
        text = line.decode("utf-8")
        if text.startswith("data: "):
            event = json.loads(text[6:])
            if event["type"] == "token":
                print(event["content"], end="", flush=True)  # stream to UI
            elif event["type"] == "tool_start":
                print(f"\n[Calling {event['tool']}...]")
            elif event["type"] == "tool_result":
                pass  # tool output available in event["result"]
            elif event["type"] == "done":
                print(f"\nConversation: {event['conversation_id']}")
```

## Dataset & Tools

### List Available Tools

```python
resp = requests.get("http://localhost:8000/api/v1/tools")
# [{"name": "analyze_image", "description": "..."}, ...]
```

### Get System Stats

```python
resp = requests.get("http://localhost:8000/api/v1/stats")
# {"active_conversations": 3, "total_conversations": 10, "persisted_conversations": 5}
```

## Conversation Management

### List Conversations

```python
resp = requests.get("http://localhost:8000/api/v1/conversations")
# [{"id": "abc123", "title": "Dataset overview", "model": "gpt-4o", ...}]
```

### Search Conversations

```python
resp = requests.get("http://localhost:8000/api/v1/conversations/search", params={"q": "thermal"})
# Returns matching conversations
```

### Load a Conversation

```python
resp = requests.get(f"http://localhost:8000/api/v1/conversations/{conv_id}")
# {"id": "...", "title": "...", "messages": [{"role": "user", "content": "..."}, ...]}
```

### Delete a Conversation

```python
resp = requests.delete(f"http://localhost:8000/api/v1/conversations/{conv_id}")
# {"deleted": true}
```

### Export as JSON Report

```python
resp = requests.get(f"http://localhost:8000/api/v1/conversations/{conv_id}/export")
# Downloads a JSON file with full conversation + metadata
```

## Model Switching

### List Models

```python
resp = requests.get("http://localhost:8000/api/v1/models")
# [{"id": "ollama/llama3.1:8b", "name": "llama3.1:8b", "provider": "ollama", "active": true}, ...]
```

### Switch Model

```python
# Switch to a local Ollama model
resp = requests.post("http://localhost:8000/api/v1/models/switch", json={
    "provider": "ollama",
    "model": "llama3.1:8b",
})

# Switch to OpenAI (requires API key)
resp = requests.post("http://localhost:8000/api/v1/models/switch", json={
    "provider": "openai",
    "model": "gpt-4o",
    "api_key": "sk-...",
})

# Switch to Azure OpenAI
resp = requests.post("http://localhost:8000/api/v1/models/switch", json={
    "provider": "azure_openai",
    "model": "gpt-4o",
    "api_key": "...",
    "endpoint": "https://your-resource.openai.azure.com/",
})
```

## Batch Analysis (CLI)

Run mislabel audits from the command line without starting the server:

```bash
# Audit all labels (PASS + FAULT), 8 samples each
python -m inspect_assist batch

# Audit only FAULT, 15 samples
python -m inspect_assist batch --labels FAULT --sample-size 15

# Save to custom path
python -m inspect_assist batch --output results/audit.json
```

## GUI Integration Tips

1. **Start the server as a subprocess** when your GUI launches:
   ```python
   import subprocess
   proc = subprocess.Popen(["python", "-m", "inspect_assist"], ...)
   ```

2. **Use the health endpoint** to confirm the server is ready before sending requests.

3. **Display `data_locality`** prominently so operators know if data is going to the cloud.

4. **Show `suggestions`** as clickable buttons in your chat panel — they guide users to useful next steps.

5. **Handle `attachments`** — these are base64-encoded image thumbnails returned by vision tools. Display them inline in the chat.

6. **Rate limiting** is set to 30 requests/min on chat endpoints by default. Adjust `RATE_LIMIT_PER_MINUTE` in `.env`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai`, or `azure_openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama API URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | Default Ollama model |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model |
| `AZURE_OPENAI_ENDPOINT` | — | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | — | Azure OpenAI key |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o` | Azure deployment name |
| `APP_HOST` | `0.0.0.0` | Server bind address |
| `APP_PORT` | `8000` | Server port |
| `API_KEY` | — | If set, requires `X-API-Key` header |
| `RATE_LIMIT_PER_MINUTE` | `30` | Chat endpoint rate limit (0 = unlimited) |
| `DATASET_PATH` | `./data/images` | Path to labeled image folders |
| `KNOWLEDGE_PATH` | `./knowledge` | Path to knowledge base articles |
| `MAX_CONVERSATION_TURNS` | `50` | Max user messages per conversation |
| `MAX_TOOL_CALLS_PER_TURN` | `5` | Max tool invocations per chat round |
