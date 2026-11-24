from __future__ import annotations

from openrouter_agent import from_chat_messages, from_claude_messages, to_chat_message, to_claude_message
from openrouter_agent.stream_transformers import (
    build_tool_call_stream,
    get_unsupported_content_summary,
    has_unsupported_content,
)


def test_chat_conversion_rounds_common_messages() -> None:
    converted = from_chat_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "call_1", "content": {"ok": True}},
        ]
    )
    assert converted[0] == {"type": "message", "role": "user", "content": "hi"}
    assert converted[1]["type"] == "function_call_output"
    response = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "hello"}]}]}
    assert to_chat_message(response) == {"role": "assistant", "content": "hello"}


def test_claude_conversion_preserves_tool_use_and_unsupported_content() -> None:
    converted = from_claude_messages(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "x"}},
                ],
            }
        ]
    )
    assert converted[0]["type"] == "message"
    assert converted[1]["type"] == "function_call"

    claude = to_claude_message(
        {"output": [{"type": "function_call", "callId": "call_1", "name": "lookup", "arguments": {"q": "x"}}]}
    )
    assert claude["stop_reason"] == "tool_use"
    assert claude["content"][0]["type"] == "tool_use"

    value = {"unsupported_content": [{"type": "image_generation_call"}]}
    assert has_unsupported_content(value)
    assert get_unsupported_content_summary(value) == {"image_generation_call": 1}


def test_claude_image_blocks_become_responses_input_images() -> None:
    converted = from_claude_messages(
        [
            {
                "role": "user",
                "content": [{"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}}],
            }
        ]
    )

    assert converted[0]["content"][0] == {
        "type": "input_image",
        "image_url": "https://example.com/cat.png",
        "detail": "auto",
    }


async def test_build_tool_call_stream_reconstructs_function_arguments() -> None:
    async def events():
        yield {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": "item_1", "callId": "call_1", "name": "lookup"},
        }
        yield {"type": "response.function_call_arguments.delta", "itemId": "item_1", "delta": '{"q"'}
        yield {"type": "response.function_call_arguments.delta", "itemId": "item_1", "delta": ':"x"}'}
        yield {"type": "response.function_call_arguments.done", "itemId": "item_1"}

    calls = [call async for call in build_tool_call_stream(events())]

    assert calls[0].id == "call_1"
    assert calls[0].name == "lookup"
    assert calls[0].arguments == {"q": "x"}


# Ported from upstream src/lib/chat-compat.test.ts (#11, #41): assistant tool
# calls become function_call items instead of being dropped.


def test_from_chat_messages_emits_function_call_items_for_assistant_tool_calls() -> None:
    converted = from_chat_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Austin"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp":30}'},
        ]
    )

    # The content-less assistant message is replaced by its tool call; the
    # already-serialized arguments are forwarded as-is.
    assert converted == [
        {
            "type": "function_call",
            "callId": "call_1",
            "id": "call_1",
            "name": "get_weather",
            "arguments": '{"city":"Austin"}',
            "status": "completed",
        },
        {"type": "function_call_output", "callId": "call_1", "output": '{"temp":30}'},
    ]


def test_from_chat_messages_keeps_text_and_tool_calls_in_order() -> None:
    converted = from_chat_messages(
        [
            {
                "role": "assistant",
                "content": "Let me check.",
                "toolCalls": [
                    {"id": "call_a", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "call_b", "type": "function", "function": {"name": "b", "arguments": '{"x":1}'}},
                ],
            }
        ]
    )

    assert [(item["type"], item.get("callId")) for item in converted] == [
        ("message", None),
        ("function_call", "call_a"),
        ("function_call", "call_b"),
    ]
    assert converted[0]["content"] == "Let me check."


def test_from_chat_messages_keeps_an_empty_assistant_message_without_tool_calls() -> None:
    assert from_chat_messages([{"role": "assistant", "content": None}]) == [
        {"type": "message", "role": "assistant", "content": ""}
    ]
