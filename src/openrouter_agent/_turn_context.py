"""Turn context building and input normalization."""

from __future__ import annotations

from typing import Any

from ._types import FunctionCallItem, TurnContext


def build_turn_context(
    number_of_turns: int = 0,
    tool_call: FunctionCallItem | None = None,
    turn_request: dict[str, Any] | None = None,
) -> TurnContext:
    """Build a TurnContext for the current turn."""
    return TurnContext(
        tool_call=tool_call,
        number_of_turns=number_of_turns,
        turn_request=turn_request,
    )


def normalize_input_to_array(
    input_data: str | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize input to an array of message items.

    Handles string input (converted to user message) and list input (passed through).
    """
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]
    return list(input_data)
