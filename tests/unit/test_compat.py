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
