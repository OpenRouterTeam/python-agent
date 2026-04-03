"""Tests for core types and type guards."""

from pydantic import BaseModel

from openrouter_agent import (
    ChatAssistantMessage,
    ConversationStatus,
    InferToolContext,
    InferToolEvent,
    InferToolInput,
    InferToolOutput,
    ManualTool,
    ManualToolFunction,
    OpenResponsesResult,
    StreamableOutputItem,
    ToolFunctionWithExecute,
    ToolType,
    ToolWithExecute,
    TurnContext,
    TurnEndEvent,
    TurnStartEvent,
    has_execute_function,
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


# --- Concrete TypedDict types ---


def test_chat_assistant_message():
    msg: ChatAssistantMessage = {"role": "assistant", "content": "Hello"}
    assert msg["role"] == "assistant"
    assert msg["content"] == "Hello"


def test_chat_assistant_message_with_tool_calls():
    msg: ChatAssistantMessage = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        ],
    }
    assert msg["tool_calls"] is not None
    assert len(msg["tool_calls"]) == 1


def test_streamable_output_item():
    item: StreamableOutputItem = {"type": "text", "text": "hello"}
    assert item["type"] == "text"
    assert item["text"] == "hello"


def test_streamable_output_item_tool_call():
    item: StreamableOutputItem = {
        "type": "function_call", "id": "c1", "name": "search", "arguments": "{}",
    }
    assert item["type"] == "function_call"
    assert item["name"] == "search"


def test_open_responses_result():
    result: OpenResponsesResult = {
        "id": "resp_123",
        "output": [{"type": "message", "content": []}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "finish_reason": "stop",
        "model": "gpt-4",
    }
    assert result["id"] == "resp_123"
    assert result["finish_reason"] == "stop"


def test_open_responses_result_minimal():
    result: OpenResponsesResult = {"id": "resp_456"}
    assert result["id"] == "resp_456"


# --- Type inference aliases ---


def test_infer_type_vars_exist():
    """Verify that the type inference aliases are importable TypeVars."""
    # These are TypeVars -- they exist as documentation stubs
    assert InferToolInput is not None
    assert InferToolOutput is not None
    assert InferToolContext is not None
    assert InferToolEvent is not None
