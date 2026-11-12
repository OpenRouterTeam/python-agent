"""Ports `packages/agent/src/lib/chat-compat.test.ts` (fromChatMessages cases,
including the assistant tool-call regression for upstream issue #11).

Deliberate divergence, asserted throughout: this port's message items carry an
explicit ``"type": "message"`` (pre-existing shape of `chat_compat.py`, also
asserted in tests/unit/test_compat.py), where upstream emits bare
``{role, content}`` EasyInputMessages. Both are valid Responses input items.
"""

from __future__ import annotations

from typing import Any, Dict, List

from openrouter_agent import from_chat_messages


def _message(role: str, content: str) -> Dict[str, Any]:
    return {"type": "message", "role": role, "content": content}


def _function_call(call_id: str, name: str, arguments: str) -> Dict[str, Any]:
    return {
        "type": "function_call",
        "callId": call_id,
        "id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def _tool_call(call_id: str, name: str, arguments: str) -> Dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


# -- basic message conversion ----------------------------------------------------


def test_converts_each_role_with_string_content() -> None:
    for role in ("user", "assistant", "system", "developer"):
        assert from_chat_messages([{"role": role, "content": "Hello"}]) == [_message(role, "Hello")]


def test_converts_multiple_messages_in_order() -> None:
    result = from_chat_messages(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
    )
    assert result == [
        _message("system", "You are helpful."),
        _message("user", "Hi"),
        _message("assistant", "Hello!"),
    ]


# -- tool response conversion -----------------------------------------------------


def test_converts_tool_message_to_function_call_output() -> None:
    assert from_chat_messages([{"role": "tool", "content": "Sunny", "tool_call_id": "call_1"}]) == [
        {"type": "function_call_output", "callId": "call_1", "output": "Sunny"}
    ]


def test_converts_tool_message_with_object_content_by_stringifying() -> None:
    result = from_chat_messages([{"role": "tool", "content": {"temp": 22}, "toolCallId": "call_1"}])
    assert result == [{"type": "function_call_output", "callId": "call_1", "output": '{"temp":22}'}]


# -- null and empty content -------------------------------------------------------


def test_null_assistant_content_without_tool_calls_round_trips_as_empty_message() -> None:
    assert from_chat_messages([{"role": "assistant", "content": None}]) == [_message("assistant", "")]


def test_empty_string_content() -> None:
    assert from_chat_messages([{"role": "user", "content": ""}]) == [_message("user", "")]


def test_empty_messages_list() -> None:
    assert from_chat_messages([]) == []


# -- assistant tool call conversion (#11) -----------------------------------------


def test_emits_function_call_for_assistant_with_null_content_and_one_tool_call() -> None:
    result = from_chat_messages(
        [
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("call_123", "get_weather", '{"location":"Paris"}')],
            },
            {"role": "tool", "content": "Sunny, 22C", "tool_call_id": "call_123"},
        ]
    )

    assert result == [
        _message("user", "What is the weather in Paris?"),
        _function_call("call_123", "get_weather", '{"location":"Paris"}'),
        {"type": "function_call_output", "callId": "call_123", "output": "Sunny, 22C"},
    ]


def test_emits_message_then_function_call_when_assistant_has_text_and_tool_calls() -> None:
    result = from_chat_messages(
        [
            {
                "role": "assistant",
                "content": "Let me check the weather for you.",
                "toolCalls": [_tool_call("call_456", "get_weather", '{"location":"London"}')],
            }
        ]
    )

    assert result == [
        _message("assistant", "Let me check the weather for you."),
        _function_call("call_456", "get_weather", '{"location":"London"}'),
    ]


def test_emits_one_function_call_per_parallel_tool_call_in_order() -> None:
    result = from_chat_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _tool_call("call_a", "get_weather", '{"location":"Paris"}'),
                    _tool_call("call_b", "get_time", '{"tz":"UTC"}'),
                ],
            }
        ]
    )

    assert result == [
        _function_call("call_a", "get_weather", '{"location":"Paris"}'),
        _function_call("call_b", "get_time", '{"tz":"UTC"}'),
    ]


def test_does_not_re_stringify_serialized_tool_call_arguments() -> None:
    result = from_chat_messages(
        [{"role": "assistant", "content": None, "tool_calls": [_tool_call("call_raw", "noop", '{"a":1}')]}]
    )

    # Would be '"{\\"a\\":1}"' if json.dumps were applied a second time.
    assert len(result) == 1
    assert result[0]["arguments"] == '{"a":1}'


def test_mapping_arguments_are_serialized_once() -> None:
    # Port-specific: a pre-parsed dict (not valid chat format, but common in
    # Python callers) is serialized exactly once rather than forwarded raw.
    result = from_chat_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c", "type": "function", "function": {"name": "n", "arguments": {"a": 1}}}],
            }
        ]
    )
    assert result == [_function_call("c", "n", '{"a":1}')]


def test_empty_tool_calls_list_emits_only_the_message() -> None:
    result = from_chat_messages([{"role": "assistant", "content": "No tools needed.", "tool_calls": []}])

    assert result == [_message("assistant", "No tools needed.")]


def test_full_conversation_items_are_all_input_item_shaped() -> None:
    # Upstream #41 is a compile-time assignability check (`Item[]`); the runtime
    # analogue is that every emitted item is a Responses input item.
    items: List[Dict[str, Any]] = from_chat_messages(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": None, "tool_calls": [_tool_call("call_typed", "get_weather", "{}")]},
            {"role": "tool", "content": "ok", "tool_call_id": "call_typed"},
        ]
    )

    assert [item["type"] for item in items] == ["message", "message", "function_call", "function_call_output"]
