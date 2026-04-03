"""Orchestrator — manages conversations, system prompt, and tool dispatch loop."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

import structlog

from inspect_assist.llm import LLMResponse, Message, Role, ToolCallRequest
from inspect_assist.llm.providers import OpenAIProvider
from inspect_assist.storage import ConversationStore
from inspect_assist.tools import ToolRegistry

logger = structlog.get_logger()

SYSTEM_PROMPT = """\
You are InspectAssist, an expert AI assistant for thermal seal inspection systems \
used in packaging quality control.

## Domain context
Thermal seal inspection uses infrared imaging to evaluate heat-sealed packaging. \
An IR camera captures the thermal signature after the sealing bar applies heat; \
software classifies each seal as PASS (acceptable) or FAULT (defective). \
Good seals show uniform temperature and consistent width. \
Common defect types: cold seal, partial seal, wrinkled seal, burn-through, contamination.

## Your role
- Help users understand their inspection dataset (counts, class balance, sample images)
- Analyze thermal images to assess seal quality and flag potential defects
- Identify potential mislabels in the dataset with reasoned confidence levels
- Explain inspection concepts, parameters (thresholds, sensitivity), and procedures
- Provide troubleshooting guidance for false positives, inconsistent results, or ambient temperature issues
- Compare images side-by-side to teach what distinguishes PASS from FAULT
- Generate audit reports summarising dataset quality

## Behavior rules
1. Always call tools for real data — NEVER fabricate image analysis results or statistics
2. When asked about the dataset, call get_dataset_summary first
3. When asked to analyze an image, call analyze_image with the correct path
4. When asked to compare images, call compare_images with both paths
5. Cite specific numbers and observations from tool results
6. If a tool returns an error, explain the issue clearly and suggest next steps
7. You are advisory only — you do not modify the dataset, thresholds, or system settings
8. Ask clarifying questions when the request is ambiguous (e.g. which image, which label folder)
9. Format responses with markdown — use tables for statistics, bullet lists for findings
10. When flagging mislabels, state your confidence (low / medium / high) and reasoning
11. Keep responses concise — lead with the key finding, then supporting detail
12. At the very end of every response, include 2-3 contextual follow-up suggestions \
the user might want to ask next. Format them as a hidden HTML comment on its own line: \
<!--suggestions:["suggestion 1","suggestion 2","suggestion 3"]-->

## Available tools
- **get_dataset_summary** — scan the image dataset for counts and class balance
- **get_sample_images** — get random sample filenames from a label folder
- **get_dataset_statistics** — detailed statistics (file sizes, resolution distribution)
- **analyze_image** — use GPT-4o vision to analyze a thermal inspection image
- **compare_images** — compare two images side by side with vision
- **find_suspicious_labels** — audit a label folder for potential mislabels
- **generate_audit_report** — produce a full dataset quality report
- **search_knowledge** — search the knowledge base for articles and guides
- **explain_concept** — look up a specific concept or parameter explanation
"""

DISCLAIMER = (
    "\u26a0\ufe0f **Advisory Notice:** InspectAssist is an AI-powered advisory tool. "
    "Its analysis may contain errors and should not be used as the sole basis "
    "for quality decisions. Always verify results against inspection data. "
    "When using cloud models (OpenAI, Azure), images and conversations are "
    "sent to external servers."
)

_SUGGESTIONS_RE = re.compile(r"<!--suggestions:(.*?)-->", re.DOTALL)

# Keywords that signal a question needs the strong (expensive) model
_STRONG_KEYWORDS = re.compile(
    r"\b("
    r"analy[sz]e|inspect|diagnos[ei]|audit|suspicious|mislabel|mis-label"
    r"|compare\s+image|compare\s+these|side.by.side"
    r"|what(?:'|')?s\s+wrong|check\s+this|look\s+at\s+this|examine"
    r"|generate\s+report|quality\s+report|audit\s+report"
    r"|defect|fault.*image|image.*fault"
    r"|seal\s+quality|burn.through|cold\s+seal|wrinkle"
    r")\b",
    re.IGNORECASE,
)


def classify_difficulty(user_message: str) -> Literal["fast", "strong"]:
    """Classify whether a message needs the strong or fast model.

    Returns "strong" for vision/analysis tasks, "fast" for knowledge/general questions.
    """
    if _STRONG_KEYWORDS.search(user_message):
        return "strong"
    return "fast"


def _extract_suggestions(text: str) -> tuple[str, list[str]]:
    """Parse and strip <!--suggestions:[...]-->  from response text."""
    match = _SUGGESTIONS_RE.search(text)
    if not match:
        return text, []
    try:
        suggestions = json.loads(match.group(1))
        if isinstance(suggestions, list):
            clean = text[: match.start()].rstrip()
            return clean, [str(s) for s in suggestions]
    except (json.JSONDecodeError, TypeError):
        pass
    return text, []


def _extract_attachments(tool_result: str) -> tuple[str, list[dict]]:
    """Pull _attachments from a tool result JSON, returning cleaned result + attachments."""
    try:
        data = json.loads(tool_result)
        if isinstance(data, dict) and "_attachments" in data:
            attachments = data.pop("_attachments")
            return json.dumps(data, indent=2), attachments
    except (json.JSONDecodeError, TypeError):
        pass
    return tool_result, []


@dataclass
class ChatResult:
    """Rich result from a chat turn."""

    response: str
    conversation_id: str
    provider: str = ""
    data_locality: str = ""
    attachments: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    model_tier: str = ""  # "fast" or "strong" when routing is enabled


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    messages: list[Message] = field(default_factory=list)
    total_tokens: int = 0
    tool_calls_count: int = 0


class Orchestrator:
    """Manages multi-turn conversations with tool dispatch."""

    def __init__(
        self,
        llm: OpenAIProvider,
        tool_registry: ToolRegistry,
        max_turns: int = 50,
        max_tool_calls_per_turn: int = 5,
        store: ConversationStore | None = None,
        llm_fast: OpenAIProvider | None = None,
        routing_enabled: bool = False,
    ) -> None:
        self._llm = llm  # default / strong provider
        self._llm_fast = llm_fast or llm  # cheap provider (same as strong when routing off)
        self._routing_enabled = routing_enabled
        self._tools = tool_registry
        self._max_turns = max_turns
        self._max_tool_calls = max_tool_calls_per_turn
        self._conversations: dict[str, Conversation] = {}
        self._store = store

    def _pick_llm(self, user_message: str) -> tuple[OpenAIProvider, str]:
        """Choose the appropriate LLM provider based on message difficulty."""
        if not self._routing_enabled:
            return self._llm, ""
        tier = classify_difficulty(user_message)
        if tier == "strong":
            return self._llm, "strong"
        return self._llm_fast, "fast"

    async def get_or_create_conversation(self, conversation_id: str | None = None) -> Conversation:
        if conversation_id and conversation_id in self._conversations:
            return self._conversations[conversation_id]

        # Try loading from persistent storage
        if conversation_id and self._store:
            messages = await self._store.load(conversation_id)
            if messages is not None:
                conv = Conversation(id=conversation_id, messages=messages)
                self._conversations[conv.id] = conv
                return conv

        conv = Conversation()
        conv.messages.append(Message(role=Role.SYSTEM, content=SYSTEM_PROMPT))
        conv.messages.append(Message(role=Role.ASSISTANT, content=DISCLAIMER))
        self._conversations[conv.id] = conv
        return conv

    async def _persist(self, conv: Conversation) -> None:
        """Save conversation to persistent storage if available."""
        if self._store:
            await self._store.save(
                conv.id,
                conv.messages,
                model=self._llm.provider_name,
            )

    def _prune_old_conversations(self, keep: int = 100) -> None:
        if len(self._conversations) > keep:
            # Remove oldest conversations
            ids = list(self._conversations.keys())
            for cid in ids[: len(ids) - keep]:
                del self._conversations[cid]

    async def chat(self, user_message: str, conversation_id: str | None = None) -> ChatResult:
        """
        Process a user message and return a ChatResult.

        Handles the full tool-calling loop: send to LLM → if tool calls, execute them
        → send results back → repeat until LLM gives a text response.
        """
        self._prune_old_conversations()
        conv = await self.get_or_create_conversation(conversation_id)

        # Enforce turn limit
        user_msgs = [m for m in conv.messages if m.role == Role.USER]
        if len(user_msgs) >= self._max_turns:
            return ChatResult(
                response="Conversation limit reached. Please start a new conversation.",
                conversation_id=conv.id,
                provider=self._llm.provider_name,
                data_locality=self._llm.data_locality,
            )

        # Pick model based on routing
        llm, model_tier = self._pick_llm(user_message)

        conv.messages.append(Message(role=Role.USER, content=user_message))

        tool_schemas = self._tools.openai_schemas()
        tool_rounds = 0
        attachments: list[dict] = []

        while True:
            log = logger.bind(conversation_id=conv.id, tool_round=tool_rounds, model_tier=model_tier)

            response: LLMResponse = await llm.chat(
                messages=conv.messages,
                tools=tool_schemas if tool_schemas else None,
            )

            # Track usage
            conv.total_tokens += response.usage.get("total_tokens", 0)

            if not response.has_tool_calls:
                # Final text response — extract suggestions
                clean_text, suggestions = _extract_suggestions(response.content)
                conv.messages.append(Message(role=Role.ASSISTANT, content=clean_text))
                log.info("assistant_response", tokens=response.usage)
                await self._persist(conv)
                return ChatResult(
                    response=clean_text,
                    conversation_id=conv.id,
                    provider=llm.provider_name,
                    data_locality=llm.data_locality,
                    attachments=attachments,
                    suggestions=suggestions,
                    model_tier=model_tier,
                )

            # Process tool calls
            tool_rounds += 1
            if tool_rounds > self._max_tool_calls:
                conv.messages.append(
                    Message(role=Role.ASSISTANT, content="I've reached the tool call limit for this turn. Let me summarize what I found so far.")
                )
                # One more LLM call without tools to get summary
                summary = await llm.chat(messages=conv.messages)
                clean_text, suggestions = _extract_suggestions(summary.content)
                conv.messages.append(Message(role=Role.ASSISTANT, content=clean_text))
                await self._persist(conv)
                return ChatResult(
                    response=clean_text,
                    conversation_id=conv.id,
                    provider=llm.provider_name,
                    data_locality=llm.data_locality,
                    attachments=attachments,
                    suggestions=suggestions,
                    model_tier=model_tier,
                )

            # Add the assistant's tool-call message
            conv.messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

            # Execute each tool call and add results
            for tc in response.tool_calls:
                log.info("tool_call", tool=tc.function_name, args=tc.arguments_json)
                result = await self._tools.call(tc.function_name, tc.arguments_json)

                # Extract image attachments before storing in conversation
                result, new_attachments = _extract_attachments(result)
                attachments.extend(new_attachments)

                conv.messages.append(
                    Message(role=Role.TOOL, content=result, tool_call_id=tc.id, name=tc.function_name)
                )
                conv.tool_calls_count += 1

    async def chat_stream(
        self, user_message: str, conversation_id: str | None = None
    ) -> AsyncIterator[str]:
        """Async generator that yields SSE-formatted event strings."""
        self._prune_old_conversations()
        conv = await self.get_or_create_conversation(conversation_id)

        user_msgs = [m for m in conv.messages if m.role == Role.USER]
        if len(user_msgs) >= self._max_turns:
            yield _sse({"type": "token", "content": "Conversation limit reached. Please start a new conversation."})
            yield _sse({"type": "done", "conversation_id": conv.id})
            return

        # Pick model based on routing
        llm, model_tier = self._pick_llm(user_message)

        conv.messages.append(Message(role=Role.USER, content=user_message))

        tool_schemas = self._tools.openai_schemas()
        tool_rounds = 0
        attachments: list[dict] = []

        while True:
            final_response: LLMResponse | None = None

            async for chunk in llm.stream(
                messages=conv.messages,
                tools=tool_schemas if tool_schemas else None,
            ):
                if isinstance(chunk, str):
                    yield _sse({"type": "token", "content": chunk})
                elif isinstance(chunk, LLMResponse):
                    final_response = chunk

            if final_response is None:
                break

            if not final_response.has_tool_calls:
                # Suggestions are embedded as HTML comments — invisible in markdown
                clean_text, suggestions = _extract_suggestions(final_response.content)
                conv.messages.append(Message(role=Role.ASSISTANT, content=clean_text))
                await self._persist(conv)
                yield _sse({
                    "type": "done",
                    "conversation_id": conv.id,
                    "provider": llm.provider_name,
                    "data_locality": llm.data_locality,
                    "suggestions": suggestions,
                    "attachments": attachments,
                    "model_tier": model_tier,
                })
                return

            # Tool-calling round
            tool_rounds += 1
            if tool_rounds > self._max_tool_calls:
                conv.messages.append(
                    Message(role=Role.ASSISTANT, content="I've reached the tool call limit. Let me summarize.")
                )
                summary_suggestions: list[str] = []
                async for chunk in llm.stream(messages=conv.messages):
                    if isinstance(chunk, str):
                        yield _sse({"type": "token", "content": chunk})
                    elif isinstance(chunk, LLMResponse):
                        clean_text, summary_suggestions = _extract_suggestions(chunk.content)
                        conv.messages.append(Message(role=Role.ASSISTANT, content=clean_text))
                await self._persist(conv)
                yield _sse({
                    "type": "done",
                    "conversation_id": conv.id,
                    "provider": llm.provider_name,
                    "data_locality": llm.data_locality,
                    "suggestions": summary_suggestions,
                    "attachments": attachments,
                    "model_tier": model_tier,
                })
                return

            # Record the assistant's tool-call message
            conv.messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=final_response.content,
                    tool_calls=final_response.tool_calls,
                )
            )

            # Execute tool calls
            for tc in final_response.tool_calls:
                yield _sse({"type": "tool_start", "name": tc.function_name})
                result = await self._tools.call(tc.function_name, tc.arguments_json)
                result, new_attachments = _extract_attachments(result)
                attachments.extend(new_attachments)
                conv.messages.append(
                    Message(role=Role.TOOL, content=result, tool_call_id=tc.id, name=tc.function_name)
                )
                conv.tool_calls_count += 1
                yield _sse({"type": "tool_result", "name": tc.function_name})

    async def get_stats(self) -> dict:
        stats = {
            "active_conversations": len(self._conversations),
            "total_conversations": len(self._conversations),
        }
        if self._store:
            stats["persisted_conversations"] = await self._store.count()
        return stats


def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Event line."""
    return f"data: {json.dumps(data)}\n\n"
