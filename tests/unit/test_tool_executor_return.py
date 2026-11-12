"""Ports tests/unit/tool-executor-return.test.ts.

Upstream (DEV-817) stopped accumulating every yield into a per-call
`preliminaryResults` array: yields are broadcast live through
`onPreliminaryResult` only, and the terminal result no longer carries them.

DIVERGENCE (contract, Idiomatic Divergences #3): Python async generators cannot
`return` a value. Upstream's "return value" cases are ported as a final
*output-shaped yield*, which is how this port's generator tools deliver their
final result (tool_executor.py `execute_generator_tool`).
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Tuple

from pydantic import BaseModel

from openrouter_agent import call_model, tool
from openrouter_agent.tool_executor import execute_generator_tool, execute_tool
from tests._fixtures import QueuedClient, text_response, tool_call_response
from openrouter_agent.tool_types import ParsedToolCall

MOCK_CONTEXT: Dict[str, Any] = {"number_of_turns": 1}


class InputParams(BaseModel):
    input: str


class Empty(BaseModel):
    pass


class Status(BaseModel):
    status: str


class ResultOut(BaseModel):
    result: str


class Index(BaseModel):
    index: int


class Done(BaseModel):
    done: bool


class FinalValue(BaseModel):
    finalValue: str


class Completed(BaseModel):
    completed: bool


def _no_preliminary_history(result: Dict[str, Any]) -> None:
    # Upstream: `expect(result.preliminaryResults).toBeUndefined()`. Check both
    # casings so a regression in either naming style is caught.
    assert "preliminary_results" not in result
    assert "preliminaryResults" not in result


async def test_captures_the_final_output_yield_as_the_final_result() -> None:
    async def execute(params: Any, ctx: Any) -> AsyncIterator[Dict[str, Any]]:
        yield {"status": "working"}
        yield {"status": "almost done"}
        # Upstream RETURNs here; Python yields the output-shaped final value.
        yield {"result": f"Done: {params.input}"}

    generator_tool = tool(
        name="return_value_tool",
        input_schema=InputParams,
        event_schema=Status,
        output_schema=ResultOut,
        execute=execute,
    )
    tool_call = ParsedToolCall(id="test-call-1", name="return_value_tool", arguments={"input": "test"})

    result = await execute_generator_tool(generator_tool, tool_call, MOCK_CONTEXT)

    assert result.get("error") is None
    _no_preliminary_history(result)
    assert result["result"].model_dump() == {"result": "Done: test"}


async def test_broadcasts_many_preliminary_results_without_retaining_them_on_the_terminal_result() -> None:
    yield_count = 10_000

    async def execute(params: Any, ctx: Any) -> AsyncIterator[Dict[str, Any]]:
        for index in range(yield_count):
            yield {"index": index}
        yield {"done": True}

    generator_tool = tool(
        name="streaming_tool",
        input_schema=Empty,
        event_schema=Index,
        output_schema=Done,
        execute=execute,
    )
    tool_call = ParsedToolCall(id="streaming-call", name="streaming_tool", arguments={})
    broadcasts: List[Tuple[str, Any]] = []

    def on_preliminary_result(call_id: str, value: Any) -> None:
        broadcasts.append((call_id, value))

    result = await execute_generator_tool(generator_tool, tool_call, MOCK_CONTEXT, on_preliminary_result)

    assert len(broadcasts) == yield_count
    # Delivered in yield order, each tagged with the executing call's id.
    assert {call_id for call_id, _ in broadcasts} == {"streaming-call"}
    assert [value.index for _, value in broadcasts[:3]] == [0, 1, 2]
    assert broadcasts[-1][1].index == yield_count - 1
    assert result["result"].model_dump() == {"done": True}
    _no_preliminary_history(result)


async def test_captures_the_final_value_even_with_no_event_yields() -> None:
    async def execute(params: Any, ctx: Any) -> AsyncIterator[Dict[str, Any]]:
        # Upstream: no yields, only `return`. Python: one output-shaped yield.
        yield {"result": f"Direct: {params.input}"}

    generator_tool = tool(
        name="return_only_tool",
        input_schema=InputParams,
        event_schema=Status,
        output_schema=ResultOut,
        execute=execute,
    )
    tool_call = ParsedToolCall(id="test-call-2", name="return_only_tool", arguments={"input": "test"})
    broadcasts: List[Any] = []

    result = await execute_generator_tool(
        generator_tool, tool_call, MOCK_CONTEXT, lambda call_id, value: broadcasts.append(value)
    )

    assert result.get("error") is None
    _no_preliminary_history(result)
    assert broadcasts == []
    assert result["result"].model_dump() == {"result": "Direct: test"}


async def test_prefers_the_output_shaped_value_over_the_last_event_yield() -> None:
    async def execute(params: Any, ctx: Any) -> AsyncIterator[Dict[str, Any]]:
        yield {"status": "progress 1"}
        yield {"status": "progress 2"}
        yield {"finalValue": f"Final: {params.input}"}

    generator_tool = tool(
        name="both_yield_return_tool",
        input_schema=InputParams,
        event_schema=Status,
        output_schema=FinalValue,
        execute=execute,
    )
    tool_call = ParsedToolCall(id="test-call-3", name="both_yield_return_tool", arguments={"input": "test"})
    broadcasts: List[Any] = []

    result = await execute_generator_tool(
        generator_tool, tool_call, MOCK_CONTEXT, lambda call_id, value: broadcasts.append(value.status)
    )

    assert result.get("error") is None
    _no_preliminary_history(result)
    assert broadcasts == ["progress 1", "progress 2"]
    assert result["result"].model_dump() == {"finalValue": "Final: test"}


async def test_still_works_with_yield_only_generators() -> None:
    async def execute(params: Any, ctx: Any) -> AsyncIterator[Dict[str, Any]]:
        yield {"status": "working"}
        yield {"completed": True}

    generator_tool = tool(
        name="yield_only_tool",
        input_schema=Empty,
        event_schema=Status,
        output_schema=Completed,
        execute=execute,
    )
    tool_call = ParsedToolCall(id="test-call-4", name="yield_only_tool", arguments={})

    result = await execute_generator_tool(generator_tool, tool_call, MOCK_CONTEXT)

    assert result.get("error") is None
    # The last yield matching output_schema but not event_schema is the result.
    assert result["result"].model_dump() == {"completed": True}


async def test_dispatcher_result_carries_no_preliminary_history() -> None:
    async def execute(params: Any, ctx: Any) -> AsyncIterator[Dict[str, Any]]:
        yield {"status": "one"}
        yield {"done": True}

    generator_tool = tool(name="gen", input_schema=Empty, event_schema=Status, output_schema=Done, execute=execute)
    seen: List[Any] = []

    result = await execute_tool(
        generator_tool,
        ParsedToolCall(id="c1", name="gen", arguments={}),
        MOCK_CONTEXT,
        lambda call_id, value: seen.append((call_id, value.status)),
    )

    assert result is not None
    _no_preliminary_history(result)
    assert seen == [("c1", "one")]
    assert result["result"].model_dump() == {"done": True}


async def test_yields_are_delivered_live_through_the_tool_stream_and_absent_from_the_result_event() -> None:
    async def execute(params: Any, ctx: Any) -> AsyncIterator[Dict[str, Any]]:
        yield {"status": "one"}
        yield {"status": "two"}
        yield {"done": True}

    generator_tool = tool(name="gen", input_schema=Empty, event_schema=Status, output_schema=Done, execute=execute)
    client = QueuedClient([tool_call_response("resp_1", "gen", call_id="call_gen"), text_response("resp_2", "ok")])

    result = call_model(client, {"model": "m", "input": "go", "tools": [generator_tool]})
    events = [event async for event in result.get_tool_stream()]

    tool_events = [e for e in events if e["type"] in {"preliminary_result", "tool_result"}]
    assert [e["type"] for e in tool_events] == ["preliminary_result", "preliminary_result", "tool_result"]
    assert [e["result"].status for e in tool_events[:2]] == ["one", "two"]
    assert all(e["toolCallId"] == "call_gen" for e in tool_events)
    assert "preliminary_results" not in tool_events[2]
    assert "preliminaryResults" not in tool_events[2]
    assert len(client.requests) == 2
