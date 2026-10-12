from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ToolContext:
    conversation_id: int | None
    tool_call_id: str | None
    confirmed: bool = False


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON Schema
    permission: Literal["read", "write"]
    source: Literal["builtin", "mcp"]  # audit tool_source
    mcp_server: str | None
    handler: Callable
    present: Callable | None = None
