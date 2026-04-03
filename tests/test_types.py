"""Tests for core types and type guards."""

from pydantic import BaseModel

from openrouter_agent import (
    APIError,
    ConversationStatus,
    ManualTool,
    ManualToolFunction,
    ToolFunctionWithExecute,
    ToolType,
    ToolWithExecute,
    TurnContext,
    TurnEndEvent,
    TurnStartEvent,
    has_execute_function,
    is_claude_style_messages,
    is_generator_tool,
    is_manual_tool,
    is_regular_execute_tool,
    is_turn_end_event,
    is_turn_start_event,
)


class SearchInput(BaseModel):
    query: str


class SearchOutput(BaseModel):
    results: list[str]


class SearchEvent(BaseModel):
    progress: float


def _make_execute_tool() -> ToolWithExecute:
    return ToolWithExecute(
        type=ToolType.FUNCTION,
        function=ToolFunctionWithExecute(
            name="search",
            description="Search the web",
            input_schema=SearchInput,
            output_schema=SearchOutput,
            execute=lambda params, ctx: SearchOutput(results=["result1"]),
        ),
    )


def _make_manual_tool() -> ManualTool:
    return ManualTool(
        type=ToolType.FUNCTION,
        function=ManualToolFunction(
            name="approve",
            input_schema=SearchInput,
        ),
    )


def test_tool_type_enum():
    assert ToolType.FUNCTION == "function"


def test_conversation_status_enum():
    assert ConversationStatus.COMPLETE == "complete"
    assert ConversationStatus.AWAITING_APPROVAL == "awaiting_approval"


def test_turn_context_defaults():
    ctx = TurnContext()
    assert ctx.number_of_turns == 0
    assert ctx.tool_call is None


def test_type_guards_execute_tool():
    t = _make_execute_tool()
    assert has_execute_function(t)
    assert is_regular_execute_tool(t)
    assert not is_generator_tool(t)
    assert not is_manual_tool(t)


def test_type_guards_manual_tool():
    t = _make_manual_tool()
    assert not has_execute_function(t)
    assert is_manual_tool(t)
    assert not is_regular_execute_tool(t)


def test_turn_event_guards():
    start = TurnStartEvent(turn_number=0, timestamp=0.0)
    end = TurnEndEvent(turn_number=0, timestamp=0.0)
    assert is_turn_start_event(start)
    assert is_turn_end_event(end)
    assert not is_turn_start_event(end)
    assert not is_turn_end_event(start)
    # Dict form
    assert is_turn_start_event({"type": "turn.start"})
    assert is_turn_end_event({"type": "turn.end"})


# ---------------------------------------------------------------------------
# is_claude_style_messages – should NOT false-positive on OpenAI format
# ---------------------------------------------------------------------------

def test_is_claude_style_messages_with_tool_use():
    """Claude-exclusive 'tool_use' blocks are detected."""
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tu_1", "name": "search", "input": {}}],
        }
    ]
    assert is_claude_style_messages(messages) is True


def test_is_claude_style_messages_with_tool_result():
    """Claude-exclusive 'tool_result' blocks are detected."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"}],
        }
    ]
    assert is_claude_style_messages(messages) is True


def test_is_claude_style_messages_text_blocks_not_false_positive():
    """'text' content blocks exist in both formats – must NOT trigger."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
    ]
    assert is_claude_style_messages(messages) is False


def test_is_claude_style_messages_image_blocks_not_false_positive():
    """'image' content blocks exist in both formats – must NOT trigger."""
    messages = [
        {"role": "user", "content": [{"type": "image", "source": {"url": "http://example.com/img.png"}}]}
    ]
    assert is_claude_style_messages(messages) is False


def test_is_claude_style_messages_plain_string():
    """Plain string content is never Claude-style."""
    messages = [{"role": "user", "content": "Hello"}]
    assert is_claude_style_messages(messages) is False


def test_is_claude_style_messages_non_list():
    """Non-list input returns False."""
    assert is_claude_style_messages("just a string") is False


# ---------------------------------------------------------------------------
# APIError
# ---------------------------------------------------------------------------

def test_api_error_is_exception():
    assert issubclass(APIError, Exception)


def test_api_error_preserves_cause():
    original = ValueError("connection reset")
    try:
        raise APIError("API call failed: connection reset") from original
    except APIError as exc:
        assert exc.__cause__ is original
        assert "connection reset" in str(exc)
