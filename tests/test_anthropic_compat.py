"""Tests for Claude/Anthropic format conversion."""

from openrouter_agent import from_claude_messages, to_claude_message


def test_from_claude_user_text():
    messages = [{"role": "user", "content": "Hello"}]
    result = from_claude_messages(messages)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Hello"


def test_from_claude_user_content_blocks():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ],
        }
    ]
    result = from_claude_messages(messages)
    assert any(item.get("content") == "Hello world" for item in result)


def test_from_claude_assistant_tool_use():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me search"},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "search",
                    "input": {"query": "hello"},
                },
            ],
        }
    ]
    result = from_claude_messages(messages)
    # Should have a message item and a function_call item
    msg_items = [i for i in result if i.get("type") == "message"]
    fc_items = [i for i in result if i.get("type") == "function_call"]
    assert len(msg_items) == 1
    assert len(fc_items) == 1
    assert fc_items[0]["name"] == "search"


def test_from_claude_tool_result():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "content": "search results here",
                }
            ],
        }
    ]
    result = from_claude_messages(messages)
    output_items = [i for i in result if i.get("type") == "function_call_output"]
    assert len(output_items) == 1
    assert output_items[0]["call_id"] == "tu_1"


def test_to_claude_message_text():
    response = {
        "output": [
            {"type": "output_text", "text": "Hello there"},
        ]
    }
    msg = to_claude_message(response)
    assert msg["role"] == "assistant"
    assert len(msg["content"]) == 1
    assert msg["content"][0]["type"] == "text"
    assert msg["content"][0]["text"] == "Hello there"


def test_to_claude_message_tool_call():
    response = {
        "output": [
            {
                "type": "function_call",
                "call_id": "fc_1",
                "name": "search",
                "arguments": '{"query": "test"}',
            }
        ]
    }
    msg = to_claude_message(response)
    tool_uses = [b for b in msg["content"] if b["type"] == "tool_use"]
    assert len(tool_uses) == 1
    assert tool_uses[0]["name"] == "search"
    assert tool_uses[0]["input"] == {"query": "test"}
