from __future__ import annotations

from typing import Any, Dict, List

from ._utils import get_field


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


def is_input_item(value: Any) -> bool:
    """Whether ``value`` is shaped like a single item of a Responses ``input``
    array: an object carrying a string item ``type`` or message ``role``.

    Response output items are fed back as input on the next turn; this guard is
    how an output item crosses over. Accepts mappings and SDK model objects.
    """
    if value is None or isinstance(value, (str, bytes, int, float, bool, list, tuple)):
        return False
    return isinstance(get_field(value, "type"), str) or isinstance(get_field(value, "role"), str)


def output_to_input_items(output: Any) -> List[Any]:
    """Convert a response's output into input items for the next request,
    dropping anything that is not item-shaped."""
    items = output if isinstance(output, (list, tuple)) else [output]
    return [item for item in items if is_input_item(item)]
