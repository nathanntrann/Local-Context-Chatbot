"""Tool framework: decorator-based registration, schema generation, and dispatch."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, get_type_hints


@dataclass
class ToolParam:
    name: str
    type: str
    description: str
    required: bool = True
    enum: list[str] | None = None


@dataclass
class ToolDef:
    name: str
    description: str
    params: list[ToolParam] = field(default_factory=list)
    handler: Callable[..., Coroutine[Any, Any, Any]] | None = None

    def openai_schema(self) -> dict[str, Any]:
        """Generate OpenAI function-calling tool schema."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for p in self.params:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# Global type mapping for JSON schema
_PY_TO_JSON_TYPE = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


def tool(
    name: str,
    description: str,
    params: list[ToolParam] | None = None,
) -> Callable:
    """Decorator to register an async function as an assistant tool."""

    def decorator(fn: Callable[..., Coroutine[Any, Any, Any]]) -> Callable:
        tool_def = ToolDef(
            name=name,
            description=description,
            params=params or [],
            handler=fn,
        )
        # Attach metadata to the function
        fn._tool_def = tool_def  # type: ignore[attr-defined]
        return fn

    return decorator


class ToolRegistry:
    """Registry of available tools. Discovers tools from modules."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool_def: ToolDef) -> None:
        self._tools[tool_def.name] = tool_def

    def register_module(self, module: Any) -> None:
        """Scan a module for @tool-decorated functions and register them."""
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if callable(obj) and hasattr(obj, "_tool_def"):
                self.register(getattr(obj, "_tool_def"))

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    @property
    def all_tools(self) -> list[ToolDef]:
        return list(self._tools.values())

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [t.openai_schema() for t in self._tools.values()]

    async def call(self, name: str, arguments_json: str) -> str:
        """Execute a tool by name with JSON arguments. Returns JSON string result."""
        tool_def = self._tools.get(name)
        if not tool_def or not tool_def.handler:
            return json.dumps({"error": f"Unknown tool: {name}"})

        try:
            args = json.loads(arguments_json) if arguments_json else {}
            result = await tool_def.handler(**args)
            if isinstance(result, str):
                return result
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": f"Tool '{name}' failed: {str(e)}"})
