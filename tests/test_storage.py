"""Tests for SQLite conversation storage."""

import pytest

from inspect_assist.llm import Message, Role, ToolCallRequest
from inspect_assist.storage import (
    ConversationStore,
    _derive_title,
    _deserialize_messages,
    _serialize_messages,
)


class TestSerialisation:
    def test_roundtrip_simple_messages(self):
        msgs = [
            Message(role=Role.SYSTEM, content="You are a bot."),
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi!"),
        ]
        data = _serialize_messages(msgs)
        restored = _deserialize_messages(data)
        assert len(restored) == 3
        assert restored[0].role == Role.SYSTEM
        assert restored[1].content == "Hello"
        assert restored[2].role == Role.ASSISTANT

    def test_roundtrip_tool_calls(self):
        tc = ToolCallRequest(id="tc_1", function_name="my_tool", arguments_json='{"q": "x"}')
        msgs = [
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
            Message(role=Role.TOOL, content='{"result": 42}', tool_call_id="tc_1", name="my_tool"),
        ]
        data = _serialize_messages(msgs)
        restored = _deserialize_messages(data)
        assert restored[0].tool_calls is not None
        assert restored[0].tool_calls[0].function_name == "my_tool"
        assert restored[1].tool_call_id == "tc_1"
        assert restored[1].name == "my_tool"

    def test_empty_list(self):
        assert _deserialize_messages(_serialize_messages([])) == []


class TestDeriveTitle:
    def test_derives_from_first_user_message(self):
        msgs = [
            Message(role=Role.SYSTEM, content="system"),
            Message(role=Role.USER, content="What is thermal seal inspection?"),
        ]
        assert _derive_title(msgs) == "What is thermal seal inspection?"

    def test_truncates_long_messages(self):
        msgs = [Message(role=Role.USER, content="x" * 200)]
        title = _derive_title(msgs)
        assert len(title) <= 83  # 80 + "..."
        assert title.endswith("...")

    def test_no_user_messages(self):
        msgs = [Message(role=Role.SYSTEM, content="system")]
        assert _derive_title(msgs) == "New conversation"

    def test_empty_messages(self):
        assert _derive_title([]) == "New conversation"


class TestConversationStore:
    @pytest.mark.asyncio
    async def test_save_and_load(self, tmp_path):
        store = ConversationStore(tmp_path / "test.db")
        msgs = [
            Message(role=Role.SYSTEM, content="You are a bot."),
            Message(role=Role.USER, content="Hello"),
        ]
        await store.save("conv_1", msgs, model="test-model")
        loaded = await store.load("conv_1")
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[1].content == "Hello"

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, tmp_path):
        store = ConversationStore(tmp_path / "test.db")
        assert await store.load("nonexistent") is None

    @pytest.mark.asyncio
    async def test_save_updates_existing(self, tmp_path):
        store = ConversationStore(tmp_path / "test.db")
        msgs1 = [Message(role=Role.USER, content="first")]
        await store.save("c1", msgs1)

        msgs2 = [Message(role=Role.USER, content="first"), Message(role=Role.ASSISTANT, content="reply")]
        await store.save("c1", msgs2)

        loaded = await store.load("c1")
        assert len(loaded) == 2

    @pytest.mark.asyncio
    async def test_list_conversations(self, tmp_path):
        store = ConversationStore(tmp_path / "test.db")
        await store.save("c1", [Message(role=Role.USER, content="first")])
        await store.save("c2", [Message(role=Role.USER, content="second")])
        convs = await store.list_conversations()
        assert len(convs) == 2
        assert all(isinstance(c, dict) for c in convs)
        assert {c["id"] for c in convs} == {"c1", "c2"}

    @pytest.mark.asyncio
    async def test_delete(self, tmp_path):
        store = ConversationStore(tmp_path / "test.db")
        await store.save("c1", [Message(role=Role.USER, content="hi")])
        assert await store.delete("c1") is True
        assert await store.load("c1") is None
        assert await store.delete("c1") is False

    @pytest.mark.asyncio
    async def test_count(self, tmp_path):
        store = ConversationStore(tmp_path / "test.db")
        assert await store.count() == 0
        await store.save("c1", [Message(role=Role.USER, content="a")])
        await store.save("c2", [Message(role=Role.USER, content="b")])
        assert await store.count() == 2

    @pytest.mark.asyncio
    async def test_load_detail(self, tmp_path):
        store = ConversationStore(tmp_path / "test.db")
        await store.save("c1", [Message(role=Role.USER, content="hello")], model="gpt-4o")
        detail = await store.load_detail("c1")
        assert detail is not None
        assert detail["id"] == "c1"
        assert detail["model"] == "gpt-4o"
        assert detail["title"] == "hello"
        assert len(detail["messages"]) == 1

    @pytest.mark.asyncio
    async def test_load_detail_nonexistent(self, tmp_path):
        store = ConversationStore(tmp_path / "test.db")
        assert await store.load_detail("nope") is None

    @pytest.mark.asyncio
    async def test_search(self, tmp_path):
        store = ConversationStore(tmp_path / "test.db")
        await store.save("c1", [Message(role=Role.USER, content="thermal seal inspection")])
        await store.save("c2", [Message(role=Role.USER, content="calibration guide")])
        results = await store.search("thermal")
        assert len(results) == 1
        assert results[0]["id"] == "c1"

    @pytest.mark.asyncio
    async def test_search_no_match(self, tmp_path):
        store = ConversationStore(tmp_path / "test.db")
        await store.save("c1", [Message(role=Role.USER, content="hello")])
        results = await store.search("zzzzyyy")
        assert results == []
