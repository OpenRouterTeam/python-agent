"""Port of upstream `src/lib/chat-compat.test.ts` -- assistant tool-call
conversion (#11) and the Item[] return shape (#41).

Divergence (pre-existing, kept deliberately): this port's message items carry
an explicit ``"type": "message"`` discriminator, while upstream emits bare
``{role, content}`` easy-input messages. Both are accepted by the Responses API.
"""

from __future__ import annotations

from openrouter_agent import call_model, from_chat_messages
from tests._fixtures import QueuedClient, text_response


def test_assistant_null_content_with_one_tool_call_emits_function_call_item() -> None:
    result = from_chat_messages(
        [
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "content": "Sunny, 22C", "tool_call_id": "call_123"},
        ]
    )
    assert result == [
        {"type": "message", "role": "user", "content": "What is the weather in Paris?"},
        {
            "type": "function_call",
            "callId": "call_123",
            "id": "call_123",
            "name": "get_weather",
            "arguments": '{"location":"Paris"}',
            "status": "completed",
        },
        {"type": "function_call_output", "callId": "call_123", "output": "Sunny, 22C"},
    ]


def test_assistant_text_and_tool_calls_emit_both_items_in_order() -> None:
    result = from_chat_messages(
        [
            {
                "role": "assistant",
                "content": "Let me check the weather for you.",
                "toolCalls": [
                    {
                        "id": "call_456",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location":"London"}'},
                    }
                ],
            }
        ]
    )
    assert result == [
        {"type": "message", "role": "assistant", "content": "Let me check the weather for you."},
        {
            "type": "function_call",
            "callId": "call_456",
            "id": "call_456",
            "name": "get_weather",
            "arguments": '{"location":"London"}',
            "status": "completed",
        },
    ]


def test_parallel_tool_calls_emit_one_function_call_each() -> None:
    result = from_chat_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}},
                    {"id": "call_b", "type": "function", "function": {"name": "get_time", "arguments": '{"tz":"UTC"}'}},
                ],
            }
        ]
    )
    assert [item["callId"] for item in result] == ["call_a", "call_b"]
    assert [item["name"] for item in result] == ["get_weather", "get_time"]
    assert all(item["type"] == "function_call" for item in result)


def test_already_serialized_arguments_are_not_restringified() -> None:
    result = from_chat_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_raw", "type": "function", "function": {"name": "noop", "arguments": '{"a":1}'}}],
            }
        ]
    )
    assert result[0]["arguments"] == '{"a":1}'


def test_empty_tool_calls_array_emits_only_the_message() -> None:
    result = from_chat_messages([{"role": "assistant", "content": "No tools needed.", "tool_calls": []}])
    assert result == [{"type": "message", "role": "assistant", "content": "No tools needed."}]


def test_empty_assistant_message_without_tool_calls_still_round_trips() -> None:
    result = from_chat_messages([{"role": "assistant", "content": None}])
    assert result == [{"type": "message", "role": "assistant", "content": ""}]


async def test_converted_history_is_accepted_as_call_model_input() -> None:
    """#41: the converter output is directly usable as `call_model` input and
    reaches the wire with the function_call item intact."""
    items = from_chat_messages(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_typed", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "content": "ok", "tool_call_id": "call_typed"},
        ]
    )
    client = QueuedClient([text_response("resp_1", "done")])
    result = call_model(client, {"model": "m", "input": items})
    assert await result.get_text() == "done"
    sent = client.requests[0]["input"]
    # call_id: snake_case at the transport boundary (model_result.py `_send`).
    assert [item.get("type") for item in sent] == ["message", "message", "function_call", "function_call_output"]
    assert sent[2]["call_id"] == "call_typed"
    assert sent[3]["call_id"] == "call_typed"
