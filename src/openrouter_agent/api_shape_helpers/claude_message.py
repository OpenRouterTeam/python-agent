from __future__ import annotations

from typing import Any, Dict, List, TypedDict, Union


class ClaudeContentBlock(TypedDict, total=False):
    type: str
    text: str
    id: str
    name: str
    input: Any
    content: Any
    tool_use_id: str
    source: Any


class ClaudeMessageParam(TypedDict):
    role: str
    content: Union[str, List[ClaudeContentBlock]]


class ClaudeMessage(TypedDict, total=False):
    role: str
    content: List[ClaudeContentBlock]
    stop_reason: str
    usage: Dict[str, Any]
