"""Ports `packages/agent/tests/unit/claude-type-guards.test.ts`."""

from __future__ import annotations

from openrouter_agent import is_claude_style_messages
from openrouter_agent.claude_constants import ClaudeContentBlockType

TOOL_RESULT = ClaudeContentBlockType.ToolResult.value
IMAGE = ClaudeContentBlockType.Image.value
TOOL_USE = ClaudeContentBlockType.ToolUse.value


def test_returns_false_for_non_array_input() -> None:
    assert is_claude_style_messages("hello") is False
    assert is_claude_style_messages(None) is False
    assert is_claude_style_messages(42) is False
    assert is_claude_style_messages({"role": "user"}) is False


def test_returns_false_for_an_empty_array() -> None:
    assert is_claude_style_messages([]) is False


def test_detects_tool_result_content_blocks() -> None:
    messages = [{"role": "user", "content": [{"type": TOOL_RESULT, "tool_use_id": "x", "content": "ok"}]}]
    assert is_claude_style_messages(messages) is True


def test_detects_image_blocks_with_a_source() -> None:
    messages = [
        {
            "role": "user",
            "content": [{"type": IMAGE, "source": {"type": "base64", "media_type": "image/png", "data": "abc"}}],
        }
    ]
    assert is_claude_style_messages(messages) is True


def test_rejects_image_blocks_without_a_source_object() -> None:
    assert is_claude_style_messages([{"role": "user", "content": [{"type": IMAGE}]}]) is False
    assert is_claude_style_messages([{"role": "user", "content": [{"type": IMAGE, "source": "base64"}]}]) is False


def test_detects_tool_use_blocks_with_a_string_id() -> None:
    messages = [{"role": "assistant", "content": [{"type": TOOL_USE, "id": "toolu_1", "name": "x", "input": {}}]}]
    assert is_claude_style_messages(messages) is True


def test_rejects_tool_use_blocks_with_a_non_string_id() -> None:
    assert is_claude_style_messages([{"role": "assistant", "content": [{"type": TOOL_USE, "id": 123}]}]) is False


def test_returns_false_for_plain_openai_style_string_content() -> None:
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"}]
    assert is_claude_style_messages(messages) is False


def test_returns_false_when_any_message_has_a_non_claude_role() -> None:
    assert is_claude_style_messages([{"role": "system", "content": [{"type": TOOL_RESULT}]}]) is False
    assert is_claude_style_messages([{"role": "developer", "content": [{"type": TOOL_RESULT}]}]) is False
    assert is_claude_style_messages([{"role": "tool", "content": [{"type": IMAGE, "source": {}}]}]) is False


def test_non_claude_role_after_a_claude_block_does_not_override_detection() -> None:
    # Upstream scans in order and returns at the first decisive message.
    messages = [
        {"role": "user", "content": [{"type": TOOL_RESULT}]},
        {"role": "system", "content": "late"},
    ]
    assert is_claude_style_messages(messages) is True


def test_skips_non_record_entries_and_messages_without_a_role() -> None:
    messages = [
        "not-a-message",
        {"content": [{"type": TOOL_RESULT}]},  # no role
        {"role": "user", "content": [{"type": TOOL_RESULT}]},
    ]
    assert is_claude_style_messages(messages) is True


def test_skips_messages_that_carry_a_top_level_type() -> None:
    messages = [{"role": "user", "type": "message", "content": [{"type": TOOL_RESULT}]}]
    assert is_claude_style_messages(messages) is False


def test_returns_false_when_content_blocks_are_all_plain_text() -> None:
    assert is_claude_style_messages([{"role": "user", "content": [{"type": "text", "text": "hello"}]}]) is False


def test_ignores_non_record_blocks_inside_content_arrays() -> None:
    assert is_claude_style_messages([{"role": "user", "content": ["just-a-string", None, 42]}]) is False


def test_detects_claude_blocks_in_later_messages_even_if_earlier_ones_are_plain() -> None:
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": TOOL_USE, "id": "toolu_9", "name": "x", "input": {}}]},
    ]
    assert is_claude_style_messages(messages) is True


def test_accepts_enum_members_as_block_types() -> None:
    # ClaudeContentBlockType is a str Enum, so members compare equal to values.
    messages = [{"role": "user", "content": [{"type": ClaudeContentBlockType.ToolResult}]}]
    assert is_claude_style_messages(messages) is True
