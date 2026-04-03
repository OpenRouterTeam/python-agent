"""Tests for core types and type guards."""

from pydantic import BaseModel

from openrouter_agent import (
    ConversationStatus,
    ManualTool,
    ManualToolFunction,
    StepResult,
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


def test_step_result_experimental_provider_metadata_defaults_to_none():
    step = StepResult(step_type="initial")
    assert step.experimental_provider_metadata is None


def test_step_result_experimental_provider_metadata_accepts_value():
    meta = {"openai": {"reasoning_tokens": 42}}
    step = StepResult(step_type="initial", experimental_provider_metadata=meta)
    assert step.experimental_provider_metadata == meta
