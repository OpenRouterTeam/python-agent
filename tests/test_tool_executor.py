"""Tests for tool execution."""

from collections.abc import AsyncGenerator

import pytest
from pydantic import BaseModel

from openrouter_agent import (
    ParsedToolCall,
    ToolContextStore,
    ToolWithExecute,
    TurnContext,
    execute_generator_tool,
    execute_regular_tool,
    parse_tool_call_arguments,
    tool,
    tool_to_api_format,
)
from openrouter_agent._types import ToolWithGenerator


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


# ---------------------------------------------------------------------------
# Generator tool tests
# ---------------------------------------------------------------------------


class SearchInput(BaseModel):
    query: str


class SearchEvent(BaseModel):
    progress: int


class SearchOutput(BaseModel):
    results: list[str]


def _make_search_generator_tool(
    gen_fn: object | None = None,
) -> ToolWithGenerator:
    async def default_gen(params: SearchInput, ctx: object) -> AsyncGenerator:
        yield SearchEvent(progress=50)
        yield SearchEvent(progress=100)
        yield SearchOutput(results=["a", "b"])

    t = tool(
        name="search",
        description="Search things",
        input_schema=SearchInput,
        event_schema=SearchEvent,
        output_schema=SearchOutput,
        execute=gen_fn or default_gen,
    )
    assert isinstance(t, ToolWithGenerator)
    return t


@pytest.mark.anyio
async def test_execute_generator_tool_basic():
    """Events are collected and final output is returned."""
    t = _make_search_generator_tool()
    tc = ParsedToolCall(id="call_g1", name="search", arguments={"query": "test"})
    ctx = TurnContext(number_of_turns=0)
    store = ToolContextStore()

    result = await execute_generator_tool(t, tc, ctx, store)
    assert result.error is None
    assert result.result == {"results": ["a", "b"]}
    assert result.preliminary_results is not None
    assert len(result.preliminary_results) == 2
    assert result.preliminary_results[0] == {"progress": 50}
    assert result.preliminary_results[1] == {"progress": 100}


@pytest.mark.anyio
async def test_generator_tool_isinstance_takes_precedence_over_validation():
    """When a value satisfies both schemas, isinstance determines its role.

    This test creates schemas where every event also validates as a valid
    output. The isinstance check must treat it as an event, not an output.
    """

    class AmbiguousInput(BaseModel):
        x: int

    # Both schemas share the same field so any event validates as output too
    class AmbiguousEvent(BaseModel):
        value: int

    class AmbiguousOutput(BaseModel):
        value: int

    events_collected: list[dict] = []

    async def ambiguous_gen(params: AmbiguousInput, ctx: object) -> AsyncGenerator:
        # These are events -- isinstance(v, AmbiguousEvent) is True
        yield AmbiguousEvent(value=1)
        yield AmbiguousEvent(value=2)
        # This is the final output -- isinstance(v, AmbiguousOutput) is True
        yield AmbiguousOutput(value=42)

    t = tool(
        name="ambiguous",
        description="Ambiguous schemas",
        input_schema=AmbiguousInput,
        event_schema=AmbiguousEvent,
        output_schema=AmbiguousOutput,
        execute=ambiguous_gen,
    )
    assert isinstance(t, ToolWithGenerator)

    def on_prelim(call_id: str, data: object, ts: float) -> None:
        events_collected.append(data)  # type: ignore[arg-type]

    tc = ParsedToolCall(id="call_a1", name="ambiguous", arguments={"x": 1})
    ctx = TurnContext(number_of_turns=0)
    store = ToolContextStore()

    result = await execute_generator_tool(t, tc, ctx, store, on_preliminary_result=on_prelim)

    # Events must be treated as events, not swallowed as output
    assert result.error is None
    assert result.preliminary_results is not None
    assert len(result.preliminary_results) == 2
    assert result.preliminary_results[0] == {"value": 1}
    assert result.preliminary_results[1] == {"value": 2}
    # Final output is the last yielded AmbiguousOutput
    assert result.result == {"value": 42}
    # on_preliminary_result callback was invoked for each event
    assert len(events_collected) == 2


@pytest.mark.anyio
async def test_generator_tool_fallback_for_dict_values():
    """Plain dicts that match output_schema are handled via the fallback path."""

    class DictInput(BaseModel):
        x: int

    class DictEvent(BaseModel):
        status: str

    class DictOutput(BaseModel):
        value: int

    async def dict_gen(params: DictInput, ctx: object) -> AsyncGenerator:
        yield {"status": "working"}  # not an instance of either schema
        yield {"value": 99}  # validates against output schema in fallback

    t = tool(
        name="dictgen",
        description="Dict generator",
        input_schema=DictInput,
        event_schema=DictEvent,
        output_schema=DictOutput,
        execute=dict_gen,
    )
    assert isinstance(t, ToolWithGenerator)

    tc = ParsedToolCall(id="call_d1", name="dictgen", arguments={"x": 1})
    ctx = TurnContext(number_of_turns=0)
    store = ToolContextStore()

    result = await execute_generator_tool(t, tc, ctx, store)
    assert result.error is None
    # The dict {"value": 99} validates as output in fallback
    assert result.result == {"value": 99}
    # The dict {"status": "working"} fails output validation, treated as event
    assert result.preliminary_results is not None
    assert len(result.preliminary_results) == 1
