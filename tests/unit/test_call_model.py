from __future__ import annotations

from typing import Any, Dict, List

from openrouter_agent import call_model, step_count_is, tool
from tests._fixtures import MemoryStateAccessor, QueuedClient, text_response, tool_call_response, usage_block


def double_turns() -> List[Dict[str, Any]]:
    """The two-turn exchange every tool test in this file drives: call, then "4"."""
    return [
        tool_call_response("resp_1", "double", call_id="call_1", arguments='{"value": 2}'),
        text_response("resp_2", "4", usage=usage_block(total_tokens=3)),
    ]


async def test_call_model_uses_responses_api_and_executes_tool_loop() -> None:
    client = QueuedClient(double_turns())
    double = tool(name="double", input_schema=dict, execute=lambda params, ctx: {"value": params["value"] * 2})

    result = call_model(client, {"model": "test/model", "input": "double 2", "tools": [double]})

    assert await result.get_text() == "4"
    assert len(client.requests) == 2
    assert client.requests[0]["stream"] is True
    assert client.requests[0]["tools"][0]["name"] == "double"
    assert client.requests[1]["input"][-1]["type"] == "function_call_output"
    assert [call.name for call in await result.get_tool_calls()] == ["double"]


async def test_call_model_forwards_request_options_to_responses_api() -> None:
    client = QueuedClient([text_response("resp_1", "hello")])

    result = call_model(
        client,
        {"model": "test/model", "input": "hello"},
        {"headers": {"x-test": "yes"}, "timeout_ms": 1234},
    )

    await result.get_response()
    assert client.requests[0]["http_headers"]["x-test"] == "yes"
    assert client.requests[0]["http_headers"]["x-openrouter-callmodel"] == "true"
    assert client.requests[0]["timeout_ms"] == 1234


async def test_model_result_stream_consumers_are_reusable() -> None:
    client = QueuedClient(double_turns())
    noop = tool(name="double", input_schema=dict, execute=lambda params, ctx: {"value": 4})
    result = call_model(client, {"model": "test/model", "input": "double 2", "tools": [noop]})

    assert [chunk async for chunk in result.get_text_stream()] == ["4"]
    assert [chunk async for chunk in result.get_text_stream()] == ["4"]
    assert await result.get_text() == "4"


async def test_allow_final_response_executes_pending_tool_before_no_tools_turn() -> None:
    client = QueuedClient(double_turns())
    double = tool(name="double", input_schema=dict, execute=lambda params, ctx: {"value": params["value"] * 2})

    result = call_model(
        client,
        {
            "model": "test/model",
            "input": "double 2",
            "tools": [double],
            "stop_when": step_count_is(1),
            "allow_final_response": "summarize",
        },
    )

    assert await result.get_text() == "4"
    assert len(client.requests) == 2
    # Tools stay in the request (prompt-cache prefix preserved); calling is
    # forbidden via tool_choice instead of stripping the tools block.
    assert "tools" in client.requests[1]
    assert client.requests[1]["tool_choice"] == "none"
    second_input = client.requests[1]["input"]
    types = [item.get("type") for item in second_input]
    assert types.index("function_call") < types.index("function_call_output")
    assert second_input[-1] == {"role": "user", "content": "summarize"}


async def test_next_turn_params_are_applied_to_followup_request() -> None:
    client = QueuedClient(double_turns())
    double = tool(
        name="double",
        input_schema=dict,
        execute=lambda params, ctx: {"value": params["value"] * 2},
        next_turn_params={"temperature": lambda args, ctx: 0.2},
    )

    result = call_model(client, {"model": "test/model", "input": "double 2", "tools": [double]})

    await result.get_response()
    assert client.requests[1]["temperature"] == 0.2


async def test_user_input_persists_with_response_in_state() -> None:
    accessor = MemoryStateAccessor()
    client = QueuedClient([text_response("resp_text", "hi")])

    result = call_model(client, {"model": "test/model", "input": "hello", "state": accessor})
    await result.get_response()

    assert accessor.stored is not None
    assert accessor.stored.status == "complete"
    assert accessor.stored.messages[0] == {"role": "user", "content": "hello"}
    assert accessor.stored.messages[-1]["type"] == "message"
