from __future__ import annotations

from enum import Enum


class ClaudeContentBlockType(str, Enum):
    Text = "text"
    Image = "image"
    ToolUse = "tool_use"
    ToolResult = "tool_result"
    Thinking = "thinking"
    UnsupportedContent = "unsupported_content"


class NonClaudeMessageRole(str, Enum):
    System = "system"
    Developer = "developer"
    Tool = "tool"
