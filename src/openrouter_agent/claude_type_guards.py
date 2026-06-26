from __future__ import annotations

from typing import Any, Mapping, Sequence


def is_claude_style_messages(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    return all(isinstance(item, Mapping) and item.get("role") in {"user", "assistant"} for item in value)
