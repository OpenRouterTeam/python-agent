"""Tool-name matching for tool-scoped hook entries. Mirrors `hooks-matchers.ts`."""

from __future__ import annotations

import re
from typing import Optional

from .hooks_types import ToolMatcher


def matches_tool(matcher: Optional[ToolMatcher], tool_name: str) -> bool:
    """Evaluate a ToolMatcher against a tool name.

    - `None` -> wildcard, matches all tools
    - `str` -> exact match
    - compiled regex pattern -> `.search(tool_name)` is truthy
    - callable -> arbitrary predicate (coerced to bool)
    """
    if matcher is None:
        return True
    if isinstance(matcher, str):
        return matcher == tool_name
    if isinstance(matcher, re.Pattern):
        return matcher.search(tool_name) is not None
    return bool(matcher(tool_name))
