"""Ports `packages/agent/tests/unit/turn-context.test.ts` plus the
`is_input_item` / `output_to_input_items` helpers added to turn-context.ts.

Python's turn context uses snake_case keys (`number_of_turns`, `tool_call`,
`turn_request`) for upstream's `numberOfTurns` / `toolCall` / `turnRequest`.
"""

from __future__ import annotations

from typing import Any, Dict, List

from openrouter_agent.turn_context import (
    build_turn_context,
    is_input_item,
    normalize_input_to_array,
    output_to_input_items,
)


def test_build_turn_context_minimal() -> None:
    context = build_turn_context(number_of_turns=0)

    assert context == {"number_of_turns": 0}
    assert "tool_call" not in context
    assert "turn_request" not in context


def test_build_turn_context_includes_tool_call() -> None:
    tool_call = {"type": "function_call", "callId": "call_1", "name": "search", "arguments": "{}", "id": "call_1"}

    context = build_turn_context(number_of_turns=2, tool_call=tool_call)

    assert context["number_of_turns"] == 2
    assert context["tool_call"] is tool_call
    assert "turn_request" not in context


def test_build_turn_context_includes_turn_request() -> None:
    turn_request: Dict[str, Any] = {"model": "openai/gpt-4", "input": []}

    context = build_turn_context(number_of_turns=1, turn_request=turn_request)

    assert context["turn_request"] is turn_request


def test_build_turn_context_includes_both() -> None:
    tool_call = {"type": "function_call", "callId": "call_2", "name": "browse", "arguments": "{}", "id": "call_2"}
    turn_request = {"model": "openai/gpt-4"}

    context = build_turn_context(number_of_turns=3, tool_call=tool_call, turn_request=turn_request)

    assert context == {"number_of_turns": 3, "tool_call": tool_call, "turn_request": turn_request}


def test_normalize_input_converts_string_to_single_user_message() -> None:
    assert normalize_input_to_array("Hello!") == [{"role": "user", "content": "Hello!"}]


def test_normalize_input_passes_list_through_by_reference() -> None:
    items: List[Any] = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

    assert normalize_input_to_array(items) is items


def test_normalize_input_handles_empty_string() -> None:
    assert normalize_input_to_array("") == [{"role": "user", "content": ""}]


def test_is_input_item_accepts_typed_or_role_bearing_objects() -> None:
    assert is_input_item({"type": "function_call"}) is True
    assert is_input_item({"role": "user", "content": "x"}) is True


def test_is_input_item_rejects_non_items() -> None:
    assert is_input_item(None) is False
    assert is_input_item("message") is False
    assert is_input_item(42) is False
    assert is_input_item([{"type": "message"}]) is False
    assert is_input_item({}) is False
    assert is_input_item({"type": 1, "role": None}) is False


def test_is_input_item_reads_attributes_of_model_objects() -> None:
    class OutputItem:
        type = "message"

    assert is_input_item(OutputItem()) is True


def test_output_to_input_items_filters_and_wraps() -> None:
    message = {"type": "message", "role": "assistant", "content": []}
    call = {"type": "function_call", "callId": "c1", "name": "x", "arguments": "{}"}

    assert output_to_input_items([message, None, "junk", call]) == [message, call]
    assert output_to_input_items(message) == [message]
    assert output_to_input_items(None) == []
