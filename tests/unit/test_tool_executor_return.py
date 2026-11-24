"""Ported from upstream tests/unit/tool-executor-return.test.ts (generator
return capture after #110 dropped `preliminaryResults`) and the
execute-context `toolCall` threading from hitl-tool.test.ts (#91)."""

from __future__ import annotations

import json
from typing import Any, List

from pydantic import BaseModel

from openrouter_agent import call_model, tool
from openrouter_agent.tool_executor import execute_tool
from openrouter_agent.tool_types import ParsedToolCall
from tests._fixtures import MemoryStateAccessor, QueuedClient, text_response, tool_call_response


class _In(BaseModel):
    input: str


class _Out(BaseModel):
    result: str


class _Event(BaseModel):
    status: str


async def test_generator_yields_are_broadcast_live_but_not_accumulated_on_the_result() -> None:
    async def execute(params: _In, _ctx: Any) -> Any:
        yield {"status": "working"}
        yield {"status": "almost done"}
        yield {"result": f"Done: {params.input}"}

    generator = tool(name="gen", input_schema=_In, event_schema=_Event, output_schema=_Out, execute=execute)
    live: List[Any] = []

    result = await execute_tool(
        generator,
        ParsedToolCall(id="call_1", name="gen", arguments={"input": "test"}),
        on_preliminary_result=lambda call_id, value: live.append((call_id, value.status)),
    )

    assert result is not None
    assert "error" not in result
    assert live == [("call_1", "working"), ("call_1", "almost done")]
    assert "preliminary_results" not in result
    assert result["result"] == _Out(result="Done: test")


async def test_tool_result_event_does_not_carry_the_yield_history() -> None:
    async def execute(_params: Any, _ctx: Any) -> Any:
        yield {"status": "working"}
        yield {"result": "done"}

    generator = tool(name="gen", input_schema=dict, event_schema=_Event, output_schema=_Out, execute=execute)
    client = QueuedClient([tool_call_response("r1", "gen", call_id="call_g"), text_response("r2", "ok")])
    result = call_model(client, {"model": "m", "input": "go", "tools": [generator]})

    events = [event async for event in result.get_full_responses_stream()]

    preliminary = [e for e in events if e.get("type") == "tool.preliminary_result"]
    results = [e for e in events if e.get("type") == "tool.result"]
    assert len(preliminary) == 1
    assert len(results) == 1
    assert "preliminaryResults" not in results[0]
    assert "preliminary_results" not in results[0]


async def test_execute_context_carries_the_executed_tool_call_in_wire_shape() -> None:
    seen: List[Any] = []

    def execute(_params: Any, ctx: Any) -> Any:
        seen.append(ctx["tool_call"])
        return {"ok": True}

    echo = tool(name="echo", input_schema=dict, execute=execute)
    client = QueuedClient(
        [tool_call_response("r1", "echo", call_id="call_e", arguments='{"x":1}'), text_response("r2", "ok")]
    )

    await call_model(client, {"model": "m", "input": "go", "tools": [echo]}).get_text()

    assert len(seen) == 1
    assert seen[0]["type"] == "function_call"
    assert seen[0]["callId"] == "call_e"
    assert seen[0]["id"] == "call_e"
    assert seen[0]["name"] == "echo"
    assert json.loads(seen[0]["arguments"]) == {"x": 1}


async def test_execute_context_carries_the_tool_call_on_the_approval_resume_path() -> None:
    seen: List[Any] = []

    def execute(_params: Any, ctx: Any) -> Any:
        seen.append(ctx["tool_call"])
        return {"ok": True}

    gated = tool(name="gated", input_schema=dict, execute=execute, require_approval=True)
    state = MemoryStateAccessor()
    await call_model(
        QueuedClient([tool_call_response("r1", "gated", call_id="call_g", arguments='{"v":2}')]),
        {"model": "m", "input": "go", "tools": [gated], "state": state},
    ).get_response()

    await call_model(
        QueuedClient([text_response("r2", "ok")]),
        {"model": "m", "input": [], "tools": [gated], "state": state, "approve_tool_calls": ["call_g"]},
    ).get_text()

    assert [(c["callId"], json.loads(c["arguments"])) for c in seen] == [("call_g", {"v": 2})]


async def test_a_caller_provided_tool_call_wins() -> None:
    seen: List[Any] = []

    def execute(_params: Any, ctx: Any) -> Any:
        seen.append(ctx["tool_call"])
        return {}

    echo = tool(name="echo", input_schema=dict, execute=execute)
    provided = {"type": "function_call", "callId": "call_x", "name": "echo", "arguments": "{}", "status": "completed"}

    await execute_tool(echo, ParsedToolCall(id="call_x", name="echo", arguments={}), {"tool_call": provided})

    assert seen == [provided]
