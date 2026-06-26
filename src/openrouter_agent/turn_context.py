from __future__ import annotations

from typing import Any, Dict, List


def normalize_input_to_array(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    return [value]


def build_turn_context(number_of_turns: int = 0, tool_call: Any = None, turn_request: Any = None) -> Dict[str, Any]:
    ctx = {"number_of_turns": number_of_turns}
    if tool_call is not None:
        ctx["tool_call"] = tool_call
    if turn_request is not None:
        ctx["turn_request"] = turn_request
    return ctx
