"""Port of upstream tests/unit/max-output-tokens-truncation.test.ts.

A turn the provider stopped at `max_output_tokens` carries a cut-off
`function_call` batch. None of it executes, and the loop does not re-request
on the same exhausted budget.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from openrouter_agent import call_model, step_count_is, tool
from openrouter_agent.stream_transformers import extract_tool_calls_from_response, response_has_tool_calls
from tests._fixtures import QueuedClient, function_call_item, make_response


def truncated_tool_call_response(output: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    if output is None:
        cut = function_call_item("call_1", "run_shell", '{"commands":')
        cut["status"] = "incomplete"
        output = [cut]
    return make_response(
        "resp_truncated",
        output,
        status="incomplete",
        incompleteDetails={"reason": "max_output_tokens"},
        usage={"input_tokens": 4529, "output_tokens": 3002, "total_tokens": 7531},
    )


def truncated_batch_response() -> Dict[str, Any]:
    items = [
        function_call_item(f"call_weather_{index}", "get_weather", json.dumps({"city": city}))
        for index, city in enumerate(["Paris", "London", "Tokyo"])
    ]
    cut = function_call_item("call_shell", "run_shell", '{"commands":')
    cut["status"] = "incomplete"
    return truncated_tool_call_response([*items, cut])


class ShellInput(BaseModel):
    commands: List[str]


class WeatherInput(BaseModel):
    city: str


def test_extracts_no_tool_calls_from_a_response_truncated_at_max_output_tokens() -> None:
    response = truncated_tool_call_response()
    assert response_has_tool_calls(response) is False
    assert extract_tool_calls_from_response(response) == []


def test_other_incomplete_reasons_still_yield_tool_calls() -> None:
    response = make_response(
        "resp_filtered",
        [function_call_item("call_1", "run_shell", '{"commands":["ls"]}')],
        status="incomplete",
        incompleteDetails={"reason": "content_filter"},
    )
    assert response_has_tool_calls(response) is True
    assert [call.id for call in extract_tool_calls_from_response(response)] == ["call_1"]


async def test_finalizes_on_the_truncated_turn_without_executing_or_requesting_again() -> None:
    executed: List[Any] = []

    def run_shell(params: ShellInput, ctx: Any) -> Dict[str, Any]:
        executed.append(params)
        return {"ok": True}

    client = QueuedClient([truncated_tool_call_response()])
    result = call_model(
        client,
        {
            "model": "test-model",
            "input": "Run echo hello.",
            "stop_when": step_count_is(3),
            "tools": [tool(name="run_shell", input_schema=ShellInput, execute=run_shell)],
        },
    )
    response = await result.get_response()

    assert executed == []
    assert len(client.requests) == 1
    assert response["id"] == "resp_truncated"
    assert response["status"] == "incomplete"


async def test_does_not_execute_calls_that_completed_before_the_cut_off_either() -> None:
    executed: List[Any] = []

    def get_weather(params: WeatherInput, ctx: Any) -> Dict[str, Any]:
        executed.append(params)
        return {"temperature": 22}

    def run_shell(params: ShellInput, ctx: Any) -> Dict[str, Any]:
        executed.append(params)
        return {"ok": True}

    client = QueuedClient([truncated_batch_response()])
    result = call_model(
        client,
        {
            "model": "test-model",
            "input": "Weather in three cities, then run echo hello.",
            "stop_when": step_count_is(3),
            "tools": [
                tool(name="get_weather", input_schema=WeatherInput, execute=get_weather),
                tool(name="run_shell", input_schema=ShellInput, execute=run_shell),
            ],
        },
    )
    response = await result.get_response()

    assert executed == []
    assert len(client.requests) == 1
    assert response["status"] == "incomplete"
    # The caller gets the whole turn, cut-off item included, to resume from.
    assert [item.get("callId", item["type"]) for item in response["output"]] == [
        "call_weather_0",
        "call_weather_1",
        "call_weather_2",
        "call_shell",
    ]
