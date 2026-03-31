"""Orchestrator — manages conversations, system prompt, and tool dispatch loop."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import structlog

from inspect_assist.llm import LLMResponse, Message, Role, ToolCallRequest
from inspect_assist.llm.providers import OpenAIProvider
from inspect_assist.tools import ToolRegistry

logger = structlog.get_logger()

SYSTEM_PROMPT = """\
You are InspectAssist, an expert AI assistant for industrial thermal inspection systems.

Your role:
- Help users understand their inspection dataset and thermal images
- Analyze thermal images to assess seal quality (good vs defective)
- Flag potential mislabels in the dataset
- Explain inspection concepts, parameters, and procedures
- Provide troubleshooting guidance
- Compare images to teach what distinguishes PASS from FAULT

Behavior rules:
1. Always use your tools to get real data — NEVER guess or fabricate image analysis results
2. When asked about the dataset, call get_dataset_summary first
3. When asked to analyze an image, call analyze_image with the correct path
4. When asked to compare images, call compare_images
5. Cite specific data from tool results in your responses
6. If a tool returns an error, explain the issue clearly
7. You are advisory only — you do not modify the dataset or system
8. If you don't have enough information, ask clarifying questions
9. Format responses with markdown for readability
10. When flagging mislabels, explain your reasoning and confidence level

You have access to these tools:
- get_dataset_summary: Scan the image dataset for counts and class balance
- get_sample_images: Get random sample filenames from a label folder
- analyze_image: Use GPT-4o vision to analyze a thermal inspection image
- compare_images: Compare two images side by side with vision
- find_suspicious_labels: Audit a label folder for potential mislabels
- search_knowledge: Search the knowledge base for articles and guides
- explain_concept: Look up a specific concept or parameter explanation
"""


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
    ) -> None:
        self._llm = llm
        self._tools = tool_registry
        self._max_turns = max_turns
        self._max_tool_calls = max_tool_calls_per_turn
        self._conversations: dict[str, Conversation] = {}

    def get_or_create_conversation(self, conversation_id: str | None = None) -> Conversation:
        if conversation_id and conversation_id in self._conversations:
            return self._conversations[conversation_id]
        conv = Conversation()
        conv.messages.append(Message(role=Role.SYSTEM, content=SYSTEM_PROMPT))
        self._conversations[conv.id] = conv
        return conv

    def _prune_old_conversations(self, keep: int = 100) -> None:
        if len(self._conversations) > keep:
            # Remove oldest conversations
            ids = list(self._conversations.keys())
            for cid in ids[: len(ids) - keep]:
                del self._conversations[cid]

    async def chat(self, user_message: str, conversation_id: str | None = None) -> tuple[str, str]:
        """
        Process a user message and return (response_text, conversation_id).

        Handles the full tool-calling loop: send to LLM → if tool calls, execute them
        → send results back → repeat until LLM gives a text response.
        """
        self._prune_old_conversations()
        conv = self.get_or_create_conversation(conversation_id)

        # Enforce turn limit
        user_msgs = [m for m in conv.messages if m.role == Role.USER]
        if len(user_msgs) >= self._max_turns:
            return "Conversation limit reached. Please start a new conversation.", conv.id

        conv.messages.append(Message(role=Role.USER, content=user_message))

        tool_schemas = self._tools.openai_schemas()
        tool_rounds = 0

        while True:
            log = logger.bind(conversation_id=conv.id, tool_round=tool_rounds)

            response: LLMResponse = await self._llm.chat(
                messages=conv.messages,
                tools=tool_schemas if tool_schemas else None,
            )

            # Track usage
            conv.total_tokens += response.usage.get("total_tokens", 0)

            if not response.has_tool_calls:
                # Final text response
                conv.messages.append(Message(role=Role.ASSISTANT, content=response.content))
                log.info("assistant_response", tokens=response.usage)
                return response.content, conv.id

            # Process tool calls
            tool_rounds += 1
            if tool_rounds > self._max_tool_calls:
                conv.messages.append(
                    Message(role=Role.ASSISTANT, content="I've reached the tool call limit for this turn. Let me summarize what I found so far.")
                )
                # One more LLM call without tools to get summary
                summary = await self._llm.chat(messages=conv.messages)
                conv.messages.append(Message(role=Role.ASSISTANT, content=summary.content))
                return summary.content, conv.id

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
                conv.messages.append(
                    Message(role=Role.TOOL, content=result, tool_call_id=tc.id, name=tc.function_name)
                )
                conv.tool_calls_count += 1

    def get_stats(self) -> dict:
        return {
            "active_conversations": len(self._conversations),
            "total_conversations": len(self._conversations),
        }
