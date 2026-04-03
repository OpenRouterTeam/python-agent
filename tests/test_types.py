"""Tests for core types and type guards."""

from pydantic import BaseModel

from openrouter_agent import (
    CallModelInput,
    ConversationStatus,
    ManualTool,
    ManualToolFunction,
    RequestOptions,
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


# ---------------------------------------------------------------------------
# CallModelInput / RequestOptions TypedDict tests
# ---------------------------------------------------------------------------


def test_call_model_input_minimal():
    """CallModelInput works with only a subset of keys."""
    req: CallModelInput = {"model": "openai/gpt-4o", "input": "hello"}
    assert req["model"] == "openai/gpt-4o"
    assert req["input"] == "hello"


def test_call_model_input_with_api_fields():
    """Standard API fields are accepted."""
    req: CallModelInput = {
        "model": "openai/gpt-4o",
        "input": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "max_output_tokens": 1024,
        "top_p": 0.9,
        "top_k": 40,
        "instructions": "Be helpful",
        "previous_response_id": "resp_abc",
    }
    assert req["temperature"] == 0.7
    assert req["max_output_tokens"] == 1024


def test_call_model_input_with_sdk_fields():
    """SDK-specific fields (tools, stop_when, etc.) are accepted."""
    req: CallModelInput = {
        "model": "openai/gpt-4o",
        "input": "hello",
        "approve_tool_calls": ["call_1"],
        "reject_tool_calls": ["call_2"],
    }
    assert req["approve_tool_calls"] == ["call_1"]
    assert req["reject_tool_calls"] == ["call_2"]


def test_call_model_input_backward_compat_with_dict():
    """CallModelInput is a dict subtype — plain dicts still work at runtime."""
    plain: dict = {"model": "openai/gpt-4o", "input": "hi", "custom_field": True}
    # Should be passable wherever CallModelInput | dict[str, Any] is accepted
    assert isinstance(plain, dict)


def test_request_options_minimal():
    """RequestOptions works with only a subset of keys."""
    opts: RequestOptions = {"timeout": 30.0}
    assert opts["timeout"] == 30.0


def test_request_options_full():
    """RequestOptions accepts headers and timeout."""
    opts: RequestOptions = {
        "headers": {"Authorization": "Bearer sk-xxx"},
        "timeout": None,
    }
    assert opts["headers"]["Authorization"] == "Bearer sk-xxx"
    assert opts["timeout"] is None
