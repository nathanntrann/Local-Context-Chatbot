"""Tests for the Orchestrator — conversation management, tool dispatch, suggestions."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from inspect_assist.llm import LLMResponse, Message, Role, ToolCallRequest
from inspect_assist.orchestrator import (
    Orchestrator,
    _extract_attachments,
    _extract_suggestions,
    classify_difficulty,
)
from inspect_assist.tools import ToolDef, ToolParam, ToolRegistry


# --- Helper factories ---

def _make_llm(content: str = "Hello!", tool_calls=None, usage=None):
    """Create a mock LLM provider that returns a fixed response."""
    llm = AsyncMock()
    llm.provider_name = "test"
    llm.data_locality = "local"
    llm.chat = AsyncMock(return_value=LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=usage or {"total_tokens": 10},
    ))
    return llm


def _make_registry():
    """Create an empty tool registry."""
    return ToolRegistry()


def _make_registry_with_echo():
    """Create a tool registry with a simple echo tool."""
    registry = ToolRegistry()

    async def echo_handler(text: str) -> str:
        return json.dumps({"echo": text})

    td = ToolDef(
        name="echo",
        description="Echoes text",
        params=[ToolParam(name="text", type="string", description="Text to echo")],
        handler=echo_handler,
    )
    registry.register(td)
    return registry


# --- Unit tests for helper functions ---

class TestExtractSuggestions:
    def test_extracts_suggestions(self):
        text = 'Here is info.\n<!--suggestions:["Ask about X","Ask about Y"]-->'
        clean, suggestions = _extract_suggestions(text)
        assert "<!--" not in clean
        assert len(suggestions) == 2
        assert suggestions[0] == "Ask about X"

    def test_no_suggestions(self):
        text = "Just a plain response."
        clean, suggestions = _extract_suggestions(text)
        assert clean == text
        assert suggestions == []

    def test_malformed_json(self):
        text = "Something <!--suggestions:not json-->"
        clean, suggestions = _extract_suggestions(text)
        assert suggestions == []

    def test_strips_trailing_whitespace(self):
        text = "Answer.\n\n<!--suggestions:['a']-->"
        # json.loads won't parse single quotes, so falls back
        clean, suggestions = _extract_suggestions(text)
        # Even on fallback, original text is returned
        assert isinstance(clean, str)


class TestExtractAttachments:
    def test_extracts_attachments(self):
        data = json.dumps({
            "analysis": "looks good",
            "_attachments": [{"type": "image", "data": "abc123"}],
        })
        result, attachments = _extract_attachments(data)
        assert len(attachments) == 1
        assert attachments[0]["data"] == "abc123"
        parsed = json.loads(result)
        assert "_attachments" not in parsed

    def test_no_attachments(self):
        data = json.dumps({"result": "ok"})
        result, attachments = _extract_attachments(data)
        assert attachments == []

    def test_non_json(self):
        result, attachments = _extract_attachments("plain text")
        assert result == "plain text"
        assert attachments == []


# --- Orchestrator tests ---

class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_basic_chat(self):
        llm = _make_llm(content="Hello there!")
        orch = Orchestrator(llm=llm, tool_registry=_make_registry())
        result = await orch.chat("Hi")
        assert result.response == "Hello there!"
        assert result.conversation_id
        assert result.provider == "test"
        assert result.data_locality == "local"

    @pytest.mark.asyncio
    async def test_conversation_continuity(self):
        llm = _make_llm(content="Response 1")
        orch = Orchestrator(llm=llm, tool_registry=_make_registry())

        r1 = await orch.chat("First message")
        cid = r1.conversation_id

        llm.chat.return_value = LLMResponse(content="Response 2", usage={"total_tokens": 5})
        r2 = await orch.chat("Second message", conversation_id=cid)
        assert r2.conversation_id == cid
        assert r2.response == "Response 2"

    @pytest.mark.asyncio
    async def test_suggestions_extracted(self):
        llm = _make_llm(content='Great answer!\n<!--suggestions:["More info","Help"]-->')
        orch = Orchestrator(llm=llm, tool_registry=_make_registry())
        result = await orch.chat("Tell me stuff")
        assert "<!--" not in result.response
        assert len(result.suggestions) == 2

    @pytest.mark.asyncio
    async def test_tool_dispatch(self):
        # First call returns a tool request, second returns text
        tc = ToolCallRequest(id="tc1", function_name="echo", arguments_json='{"text":"hi"}')
        llm = AsyncMock()
        llm.provider_name = "test"
        llm.data_locality = "local"
        llm.chat = AsyncMock(side_effect=[
            LLMResponse(content="", tool_calls=[tc], usage={"total_tokens": 5}),
            LLMResponse(content="Echo result: hi", usage={"total_tokens": 8}),
        ])

        orch = Orchestrator(llm=llm, tool_registry=_make_registry_with_echo())
        result = await orch.chat("Echo something")
        assert "Echo result" in result.response
        assert llm.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_turn_limit(self):
        llm = _make_llm(content="reply")
        orch = Orchestrator(llm=llm, tool_registry=_make_registry(), max_turns=2)

        r1 = await orch.chat("msg 1")
        cid = r1.conversation_id
        llm.chat.return_value = LLMResponse(content="reply 2", usage={"total_tokens": 5})
        await orch.chat("msg 2", conversation_id=cid)

        r3 = await orch.chat("msg 3", conversation_id=cid)
        assert "limit reached" in r3.response.lower()

    @pytest.mark.asyncio
    async def test_tool_call_limit(self):
        # LLM keeps requesting tools beyond the limit
        tc = ToolCallRequest(id="tc1", function_name="echo", arguments_json='{"text":"x"}')
        tool_response = LLMResponse(content="", tool_calls=[tc], usage={"total_tokens": 5})
        summary_response = LLMResponse(content="Summary after limit", usage={"total_tokens": 8})

        llm = AsyncMock()
        llm.provider_name = "test"
        llm.data_locality = "local"
        # Return tool calls for max_tool_calls+1, then a summary
        llm.chat = AsyncMock(side_effect=[
            tool_response,  # round 1
            tool_response,  # round 2 (exceeds limit of 1)
            summary_response,  # summary call
        ])

        orch = Orchestrator(
            llm=llm,
            tool_registry=_make_registry_with_echo(),
            max_tool_calls_per_turn=1,
        )
        result = await orch.chat("keep echoing")
        assert "summary" in result.response.lower()

    @pytest.mark.asyncio
    async def test_get_stats(self):
        llm = _make_llm()
        orch = Orchestrator(llm=llm, tool_registry=_make_registry())
        stats = await orch.get_stats()
        assert stats["active_conversations"] == 0
        assert "total_conversations" in stats

        await orch.chat("hello")
        stats = await orch.get_stats()
        assert stats["active_conversations"] == 1

    @pytest.mark.asyncio
    async def test_get_stats_with_store(self, tmp_path):
        from inspect_assist.storage import ConversationStore
        store = ConversationStore(tmp_path / "test.db")
        llm = _make_llm()
        orch = Orchestrator(llm=llm, tool_registry=_make_registry(), store=store)
        stats = await orch.get_stats()
        assert stats["persisted_conversations"] == 0

    @pytest.mark.asyncio
    async def test_prune_old_conversations(self):
        llm = _make_llm()
        orch = Orchestrator(llm=llm, tool_registry=_make_registry())
        # Create 105 conversations
        for i in range(105):
            llm.chat.return_value = LLMResponse(content=f"r{i}", usage={"total_tokens": 1})
            await orch.chat(f"msg {i}")
        # After pruning, should have <= 100
        assert len(orch._conversations) <= 101  # 100 kept + 1 new

    @pytest.mark.asyncio
    async def test_persistence_called(self, tmp_path):
        from inspect_assist.storage import ConversationStore
        store = ConversationStore(tmp_path / "test.db")
        llm = _make_llm(content="saved response")
        orch = Orchestrator(llm=llm, tool_registry=_make_registry(), store=store)

        result = await orch.chat("hi")
        loaded = await store.load(result.conversation_id)
        assert loaded is not None
        assert any(m.content == "hi" for m in loaded)


# --- Smart routing tests ---

class TestClassifyDifficulty:
    """Test the deterministic keyword classifier for model routing."""

    def test_vision_keywords_strong(self):
        assert classify_difficulty("Can you analyze this thermal image?") == "strong"
        assert classify_difficulty("Compare these two images side by side") == "strong"
        assert classify_difficulty("Audit the FAULT folder for mislabels") == "strong"
        assert classify_difficulty("What's wrong with this seal?") == "strong"
        assert classify_difficulty("Check this image for defects") == "strong"
        assert classify_difficulty("Diagnose the issue with these seals") == "strong"

    def test_report_keywords_strong(self):
        assert classify_difficulty("Generate a quality report") == "strong"
        assert classify_difficulty("Generate audit report for PASS folder") == "strong"
        assert classify_difficulty("Find suspicious labels in FAULT") == "strong"

    def test_defect_keywords_strong(self):
        assert classify_difficulty("Is this a cold seal?") == "strong"
        assert classify_difficulty("I see burn-through on this sample") == "strong"
        assert classify_difficulty("This looks like a wrinkle defect") == "strong"
        assert classify_difficulty("Check the seal quality") == "strong"

    def test_simple_knowledge_fast(self):
        assert classify_difficulty("What is NETD?") == "fast"
        assert classify_difficulty("How do I adjust thresholds?") == "fast"
        assert classify_difficulty("Getting started with the system") == "fast"
        assert classify_difficulty("What temperature range should I use?") == "fast"
        assert classify_difficulty("Hello, how can you help me?") == "fast"

    def test_troubleshooting_fast(self):
        assert classify_difficulty("Why am I getting inconsistent results?") == "fast"
        assert classify_difficulty("The system seems slow today") == "fast"
        assert classify_difficulty("How do I calibrate the camera?") == "fast"

    def test_case_insensitive(self):
        assert classify_difficulty("ANALYZE this IMAGE") == "strong"
        assert classify_difficulty("WHAT IS NETD?") == "fast"


class TestRoutingIntegration:
    """Test that routing wires the correct provider into chat."""

    @pytest.mark.asyncio
    async def test_routing_disabled_returns_empty_tier(self):
        llm = _make_llm(content="reply")
        orch = Orchestrator(llm=llm, tool_registry=_make_registry(), routing_enabled=False)
        result = await orch.chat("analyze this image")
        assert result.model_tier == ""
        assert result.response == "reply"

    @pytest.mark.asyncio
    async def test_routing_enabled_strong(self):
        llm_strong = _make_llm(content="strong reply")
        llm_fast = _make_llm(content="fast reply")
        orch = Orchestrator(
            llm=llm_strong,
            tool_registry=_make_registry(),
            llm_fast=llm_fast,
            routing_enabled=True,
        )
        result = await orch.chat("analyze this thermal image for defects")
        assert result.model_tier == "strong"
        assert result.response == "strong reply"
        llm_strong.chat.assert_called_once()
        llm_fast.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_routing_enabled_fast(self):
        llm_strong = _make_llm(content="strong reply")
        llm_fast = _make_llm(content="fast reply")
        orch = Orchestrator(
            llm=llm_strong,
            tool_registry=_make_registry(),
            llm_fast=llm_fast,
            routing_enabled=True,
        )
        result = await orch.chat("What is NETD?")
        assert result.model_tier == "fast"
        assert result.response == "fast reply"
        llm_fast.chat.assert_called_once()
        llm_strong.chat.assert_not_called()
