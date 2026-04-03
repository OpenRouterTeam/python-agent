"""Tests for tool execution."""

import pytest
from pydantic import BaseModel

from openrouter_agent import (
    ParsedToolCall,
    ToolContextStore,
    ToolWithExecute,
    TurnContext,
    execute_regular_tool,
    parse_tool_call_arguments,
    tool,
    tool_to_api_format,
)


class MathInput(BaseModel):
    a: int
    b: int


class MathOutput(BaseModel):
    result: int


def _make_add_tool() -> ToolWithExecute:
    t = tool(
        name="add",
        description="Add two numbers",
        input_schema=MathInput,
        output_schema=MathOutput,
        execute=lambda params, ctx: MathOutput(result=params.a + params.b),
    )
    assert isinstance(t, ToolWithExecute)
    return t


def test_tool_to_api_format():
    t = _make_add_tool()
    api = tool_to_api_format(t)
    assert api.name == "add"
    assert api.description == "Add two numbers"
    assert api.type == "function"
    assert "properties" in api.parameters
    assert "a" in api.parameters["properties"]
    assert "b" in api.parameters["properties"]


def test_parse_tool_call_arguments_valid():
    result = parse_tool_call_arguments('{"a": 1, "b": 2}', MathInput)
    assert result.is_ok()
    parsed = result.unwrap()
    assert parsed.a == 1
    assert parsed.b == 2


def test_parse_tool_call_arguments_invalid_json():
    result = parse_tool_call_arguments("not json", MathInput)
    assert result.is_err()
    assert "Invalid JSON" in result.unwrap_err()


def test_parse_tool_call_arguments_validation_error():
    result = parse_tool_call_arguments('{"a": "not_a_number", "b": 2}', MathInput)
    assert result.is_err()
    assert "Validation error" in result.unwrap_err()


@pytest.mark.anyio
async def test_execute_regular_tool():
    t = _make_add_tool()
    tc = ParsedToolCall(id="call_1", name="add", arguments={"a": 3, "b": 4})
    ctx = TurnContext(number_of_turns=0)
    store = ToolContextStore()
    result = await execute_regular_tool(t, tc, ctx, store)
    assert result.error is None
    assert result.result == {"result": 7}
    assert result.tool_call_id == "call_1"


@pytest.mark.anyio
async def test_execute_regular_tool_validation_error():
    t = _make_add_tool()
    tc = ParsedToolCall(id="call_1", name="add", arguments={"a": "bad"})
    ctx = TurnContext(number_of_turns=0)
    store = ToolContextStore()
    result = await execute_regular_tool(t, tc, ctx, store)
    assert result.error is not None
    assert "validation error" in result.error.lower() or "Input validation" in result.error
