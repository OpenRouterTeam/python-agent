"""Tests for the tool() factory function."""

from pydantic import BaseModel

from openrouter_agent import (
    ManualTool,
    ToolWithExecute,
    ToolWithGenerator,
    is_generator_tool,
    is_manual_tool,
    is_regular_execute_tool,
    tool,
)


class CalcInput(BaseModel):
    expression: str


class CalcOutput(BaseModel):
    result: float


class CalcEvent(BaseModel):
    step: str


def test_tool_creates_execute_tool():
    t = tool(
        name="calculator",
        description="Evaluate math",
        input_schema=CalcInput,
        output_schema=CalcOutput,
        execute=lambda params, ctx: CalcOutput(result=42.0),
    )
    assert isinstance(t, ToolWithExecute)
    assert is_regular_execute_tool(t)
    assert t.function.name == "calculator"
    assert t.function.description == "Evaluate math"


def test_tool_creates_manual_tool():
    t = tool(
        name="confirm",
        description="Get user confirmation",
        input_schema=CalcInput,
    )
    assert isinstance(t, ManualTool)
    assert is_manual_tool(t)
    assert t.function.name == "confirm"


async def _generator(params: CalcInput, ctx: object) -> None:
    yield CalcEvent(step="step1")
    yield CalcOutput(result=42.0)


def test_tool_creates_generator_tool():
    t = tool(
        name="long_calc",
        description="Long calculation with progress",
        input_schema=CalcInput,
        event_schema=CalcEvent,
        output_schema=CalcOutput,
        execute=_generator,
    )
    assert isinstance(t, ToolWithGenerator)
    assert is_generator_tool(t)
    assert t.function.name == "long_calc"


def test_tool_with_approval():
    t = tool(
        name="dangerous",
        input_schema=CalcInput,
        execute=lambda params, ctx: CalcOutput(result=0.0),
        require_approval=True,
    )
    assert isinstance(t, ToolWithExecute)
    assert t.function.require_approval is True
