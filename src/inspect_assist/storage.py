"""SQLite-backed conversation persistence using aiosqlite."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from inspect_assist.llm import Message, Role, ToolCallRequest

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    messages TEXT NOT NULL DEFAULT '[]'
);
"""


def _serialize_messages(messages: list[Message]) -> str:
    """Convert Message dataclass list to JSON string for storage."""
    serialized: list[dict[str, Any]] = []
    for m in messages:
        entry: dict[str, Any] = {
            "role": m.role.value,
            "content": m.content,
        }
        if m.tool_call_id is not None:
            entry["tool_call_id"] = m.tool_call_id
        if m.name is not None:
            entry["name"] = m.name
        if m.tool_calls:
            entry["tool_calls"] = [asdict(tc) for tc in m.tool_calls]
        # Intentionally skip images — base64 blobs are too large for SQLite
        # and were already sent to the LLM; they don't need to be replayed
        serialized.append(entry)
    return json.dumps(serialized)


def _deserialize_messages(data: str) -> list[Message]:
    """Restore Message list from JSON string."""
    raw: list[dict[str, Any]] = json.loads(data)
    messages: list[Message] = []
    for entry in raw:
        tool_calls = None
        if "tool_calls" in entry:
            tool_calls = [
                ToolCallRequest(
                    id=tc["id"],
                    function_name=tc["function_name"],
                    arguments_json=tc["arguments_json"],
                )
                for tc in entry["tool_calls"]
            ]
        messages.append(
            Message(
                role=Role(entry["role"]),
                content=entry.get("content", ""),
                tool_call_id=entry.get("tool_call_id"),
                tool_calls=tool_calls,
                name=entry.get("name"),
            )
        )
    return messages


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _derive_title(messages: list[Message]) -> str:
    """Derive a short title from the first user message."""
    for m in messages:
        if m.role == Role.USER and m.content:
            text = m.content.strip()
            return text[:80] + ("..." if len(text) > 80 else "")
    return "New conversation"


class ConversationStore:
    """Async SQLite store for conversations."""

    def __init__(self, db_path: Path | str = "conversations.db") -> None:
        self._db_path = str(db_path)
        self._initialized = False

    async def _ensure_db(self) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self._db_path)
        if not self._initialized:
            await db.execute(_CREATE_TABLE)
            await db.commit()
            self._initialized = True
        return db

    async def save(
        self,
        conversation_id: str,
        messages: list[Message],
        model: str = "",
    ) -> None:
        """Insert or update a conversation."""
        db = await self._ensure_db()
        try:
            now = _utcnow()
            title = _derive_title(messages)
            data = _serialize_messages(messages)
            await db.execute(
                """
                INSERT INTO conversations (id, title, model, created_at, updated_at, messages)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    model = excluded.model,
                    updated_at = excluded.updated_at,
                    messages = excluded.messages
                """,
                (conversation_id, title, model, now, now, data),
            )
            await db.commit()
        finally:
            await db.close()

    async def load(self, conversation_id: str) -> list[Message] | None:
        """Load messages for a conversation. Returns None if not found."""
        db = await self._ensure_db()
        try:
            cursor = await db.execute(
                "SELECT messages FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return _deserialize_messages(row[0])
        finally:
            await db.close()

    async def load_detail(self, conversation_id: str) -> dict[str, Any] | None:
        """Load full conversation record including metadata and messages."""
        db = await self._ensure_db()
        try:
            cursor = await db.execute(
                "SELECT id, title, model, created_at, updated_at, messages FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            msgs = json.loads(row[5])
            return {
                "id": row[0],
                "title": row[1],
                "model": row[2],
                "created_at": row[3],
                "updated_at": row[4],
                "messages": [{"role": m["role"], "content": m.get("content", "")} for m in msgs],
            }
        finally:
            await db.close()

    async def list_conversations(
        self, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """List conversations ordered by most recent first."""
        db = await self._ensure_db()
        try:
            cursor = await db.execute(
                """
                SELECT id, title, model, created_at, updated_at
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = await cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "title": r[1],
                    "model": r[2],
                    "created_at": r[3],
                    "updated_at": r[4],
                }
                for r in rows
            ]
        finally:
            await db.close()

    async def delete(self, conversation_id: str) -> bool:
        """Delete a conversation. Returns True if it existed."""
        db = await self._ensure_db()
        try:
            cursor = await db.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            await db.commit()
            return cursor.rowcount > 0
        finally:
            await db.close()

    async def count(self) -> int:
        """Return total number of stored conversations."""
        db = await self._ensure_db()
        try:
            cursor = await db.execute("SELECT COUNT(*) FROM conversations")
            row = await cursor.fetchone()
            return row[0] if row else 0
        finally:
            await db.close()

    async def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text search across conversation titles and message content."""
        db = await self._ensure_db()
        try:
            pattern = f"%{query}%"
            cursor = await db.execute(
                """
                SELECT id, title, model, created_at, updated_at
                FROM conversations
                WHERE title LIKE ? OR messages LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (pattern, pattern, limit),
            )
            rows = await cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "title": r[1],
                    "model": r[2],
                    "created_at": r[3],
                    "updated_at": r[4],
                }
                for r in rows
            ]
        finally:
            await db.close()
