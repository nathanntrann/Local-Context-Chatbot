"""Tests for API routes using FastAPI TestClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inspect_assist.api.models import (
    ChatResponse,
    ConversationDetail,
    ConversationSummary,
    HealthResponse,
    StatsResponse,
)
from inspect_assist.api.routes import router
from inspect_assist.orchestrator import ChatResult
from inspect_assist.tools import ToolDef, ToolParam, ToolRegistry


def _build_app(
    orchestrator=None,
    tool_registry=None,
    settings=None,
    conversation_store=None,
) -> FastAPI:
    """Create a minimal FastAPI app with mocked state for testing."""
    app = FastAPI()
    app.include_router(router)

    if orchestrator is None:
        orchestrator = AsyncMock()
    if tool_registry is None:
        tool_registry = ToolRegistry()
    if settings is None:
        settings = MagicMock()
        settings.llm_provider = MagicMock(value="ollama")
        settings.ollama_model = "llama3"
        settings.ollama_base_url = "http://localhost:11434/v1"
        settings.openai_model = "gpt-4o"
        settings.azure_openai_deployment = "gpt-4o"

    app.state.orchestrator = orchestrator
    app.state.tool_registry = tool_registry
    app.state.settings = settings
    app.state.conversation_store = conversation_store

    return app


class TestHealth:
    def test_health(self):
        client = TestClient(_build_app())
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestChat:
    def test_chat(self):
        orch = AsyncMock()
        orch.chat.return_value = ChatResult(
            response="Hello!",
            conversation_id="abc123",
            provider="test",
            data_locality="local",
        )
        client = TestClient(_build_app(orchestrator=orch))
        resp = client.post("/api/v1/chat", json={"message": "Hi"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Hello!"
        assert data["conversation_id"] == "abc123"

    def test_chat_with_conversation_id(self):
        orch = AsyncMock()
        orch.chat.return_value = ChatResult(
            response="Continuing",
            conversation_id="existing",
            provider="test",
            data_locality="local",
        )
        client = TestClient(_build_app(orchestrator=orch))
        resp = client.post("/api/v1/chat", json={
            "message": "More",
            "conversation_id": "existing",
        })
        assert resp.status_code == 200
        orch.chat.assert_called_once_with(
            user_message="More",
            conversation_id="existing",
        )

    def test_chat_empty_message(self):
        client = TestClient(_build_app())
        resp = client.post("/api/v1/chat", json={"message": ""})
        assert resp.status_code == 422  # Validation error

    def test_chat_with_attachments(self):
        orch = AsyncMock()
        orch.chat.return_value = ChatResult(
            response="Got image",
            conversation_id="c1",
            provider="test",
            data_locality="local",
            attachments=[{"type": "image", "data": "abc123", "media_type": "image/png", "label": "thermal"}],
        )
        client = TestClient(_build_app(orchestrator=orch))
        resp = client.post("/api/v1/chat", json={"message": "show image"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["attachments"]) == 1
        assert data["attachments"][0]["data"] == "abc123"


class TestTools:
    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(ToolDef(
            name="test_tool",
            description="A test tool",
            params=[ToolParam(name="x", type="string", description="input")],
        ))
        client = TestClient(_build_app(tool_registry=registry))
        resp = client.get("/api/v1/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "test_tool"

    def test_list_tools_empty(self):
        client = TestClient(_build_app())
        resp = client.get("/api/v1/tools")
        assert resp.status_code == 200
        assert resp.json() == []


class TestStats:
    def test_stats(self):
        orch = AsyncMock()
        orch.get_stats.return_value = {
            "active_conversations": 3,
            "total_conversations": 10,
            "persisted_conversations": 5,
        }
        client = TestClient(_build_app(orchestrator=orch))
        resp = client.get("/api/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_conversations"] == 3
        assert data["total_conversations"] == 10


class TestConversations:
    def test_list_conversations(self):
        store = AsyncMock()
        store.list_conversations.return_value = [
            {"id": "c1", "title": "Chat 1", "model": "gpt-4o", "created_at": "2024-01-01", "updated_at": "2024-01-01"},
        ]
        client = TestClient(_build_app(conversation_store=store))
        resp = client.get("/api/v1/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == "c1"

    def test_list_conversations_no_store(self):
        client = TestClient(_build_app(conversation_store=None))
        resp = client.get("/api/v1/conversations")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_search_conversations(self):
        store = AsyncMock()
        store.search.return_value = [
            {"id": "c1", "title": "Thermal analysis", "model": "gpt-4o", "created_at": "2024-01-01", "updated_at": "2024-01-01"},
        ]
        client = TestClient(_build_app(conversation_store=store))
        resp = client.get("/api/v1/conversations/search", params={"q": "thermal"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        store.search.assert_called_once_with("thermal")

    def test_search_conversations_empty_query(self):
        store = AsyncMock()
        client = TestClient(_build_app(conversation_store=store))
        resp = client.get("/api/v1/conversations/search", params={"q": ""})
        assert resp.status_code == 200
        assert resp.json() == []
        store.search.assert_not_called()

    def test_search_conversations_no_store(self):
        client = TestClient(_build_app(conversation_store=None))
        resp = client.get("/api/v1/conversations/search", params={"q": "test"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_conversation(self):
        store = AsyncMock()
        store.load_detail.return_value = {
            "id": "c1",
            "title": "Chat",
            "model": "gpt-4o",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
        }
        client = TestClient(_build_app(conversation_store=store))
        resp = client.get("/api/v1/conversations/c1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "c1"
        assert len(data["messages"]) == 2

    def test_get_conversation_not_found(self):
        store = AsyncMock()
        store.load_detail.return_value = None
        client = TestClient(_build_app(conversation_store=store))
        resp = client.get("/api/v1/conversations/missing")
        assert resp.status_code == 404

    def test_get_conversation_no_store(self):
        client = TestClient(_build_app(conversation_store=None))
        resp = client.get("/api/v1/conversations/c1")
        assert resp.status_code == 404

    def test_delete_conversation(self):
        store = AsyncMock()
        store.delete.return_value = True
        client = TestClient(_build_app(conversation_store=store))
        resp = client.delete("/api/v1/conversations/c1")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    def test_delete_conversation_not_found(self):
        store = AsyncMock()
        store.delete.return_value = False
        client = TestClient(_build_app(conversation_store=store))
        resp = client.delete("/api/v1/conversations/missing")
        assert resp.status_code == 404

    def test_export_conversation(self):
        store = AsyncMock()
        store.load_detail.return_value = {
            "id": "c1",
            "title": "Export Test",
            "model": "gpt-4o",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        client = TestClient(_build_app(conversation_store=store))
        resp = client.get("/api/v1/conversations/c1/export")
        assert resp.status_code == 200
        data = resp.json()
        assert data["export_type"] == "conversation_report"
        assert "Content-Disposition" in resp.headers

    def test_export_conversation_not_found(self):
        store = AsyncMock()
        store.load_detail.return_value = None
        client = TestClient(_build_app(conversation_store=store))
        resp = client.get("/api/v1/conversations/missing/export")
        assert resp.status_code == 404


class TestModels:
    @patch("httpx.AsyncClient")
    def test_list_models(self, mock_httpx_class):
        """List models includes cloud providers even when Ollama is down."""
        # Simulate Ollama connection error
        import httpx

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        mock_httpx_class.return_value = mock_client

        client = TestClient(_build_app())
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        # Should still have cloud models
        assert any(m["provider"] == "openai" for m in data)
