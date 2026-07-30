"""Claude message shape types.

Ported for 1:1 module parity with upstream `api-shape-helpers/claude-message.ts`.
Upstream imports these types into `anthropic-compat`, `claude-type-guards`, and
`stream-transformers`; this port declares the equivalent shapes inline in
`anthropic_compat.py`, so nothing here is currently reachable.

Kept deliberately: the porting contract (`.upstreamer/upstreamer.md`) requires one
Python module per upstream module, so deleting this would be a parity regression
that the next sync re-creates. Excluded from the coverage floor in
`pyproject.toml` rather than deleted. Do not re-litigate.
"""

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
