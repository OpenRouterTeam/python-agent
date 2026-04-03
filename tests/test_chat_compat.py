"""Tests for Chat format conversion."""

from openrouter_agent import from_chat_messages, to_chat_message


def test_from_chat_user_message():
    messages = [{"role": "user", "content": "Hello"}]
    result = from_chat_messages(messages)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Hello"


def test_from_chat_assistant_with_tool_calls():
    messages = [
        {
            "role": "assistant",
            "content": "I'll search for that",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "test"}',
                    },
                }
            ],
        }
    ]
    result = from_chat_messages(messages)
    msg_items = [i for i in result if i.get("type") == "message"]
    fc_items = [i for i in result if i.get("type") == "function_call"]
    assert len(msg_items) == 1
    assert len(fc_items) == 1
    assert fc_items[0]["name"] == "search"


def test_from_chat_tool_response():
    messages = [
        {"role": "tool", "tool_call_id": "tc_1", "content": "results here"}
    ]
    result = from_chat_messages(messages)
    assert len(result) == 1
    assert result[0]["type"] == "function_call_output"
    assert result[0]["call_id"] == "tc_1"


def test_to_chat_message_text():
    response = {
        "output": [
            {"type": "output_text", "text": "Hello!"},
        ]
    }
    msg = to_chat_message(response)
    assert msg["role"] == "assistant"
    assert msg["content"] == "Hello!"


def test_to_chat_message_tool_calls():
    response = {
        "output": [
            {
                "type": "function_call",
                "call_id": "fc_1",
                "name": "search",
                "arguments": '{"q": "test"}',
            }
        ]
    }
    msg = to_chat_message(response)
    assert msg["role"] == "assistant"
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "search"


def test_roundtrip_chat_text():
    """Convert to chat then back, text should be preserved."""
    response = {
        "output": [
            {"type": "output_text", "text": "The answer is 42"},
        ]
    }
    chat_msg = to_chat_message(response)
    assert chat_msg["content"] == "The answer is 42"
