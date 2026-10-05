"""Port of upstream `tests/unit/max-output-tokens-truncation.test.ts`.

A turn the provider stopped at `max_output_tokens` mid tool call must end the
run: no execution of the cut-off call (nor of calls completed before the
cut-off in the same batch), and no re-request on the same exhausted budget.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from pydantic import BaseModel

from openrouter_agent import call_model, step_count_is, tool
from openrouter_agent.stream_transformers import extract_tool_calls_from_response, response_has_tool_calls
from tests._fixtures import QueuedClient, make_response


def _truncated(output: List[Dict[str, Any]]) -> Dict[str, Any]:
    return make_response(
        "resp_truncated",
        output,
        status="incomplete",
        incompleteDetails={"reason": "max_output_tokens"},
        usage={"input_tokens": 4529, "output_tokens": 3002, "total_tokens": 7531},
    )


def _shell_call() -> Dict[str, Any]:
    return {
        "type": "function_call",
        "id": "fc_1",
        "callId": "call_1",
        "name": "run_shell",
        "arguments": '{"commands":',
        "status": "incomplete",
    }


def _batch() -> Dict[str, Any]:
    items = [
        {
            "type": "function_call",
            "id": f"fc_weather_{i}",
            "callId": f"call_weather_{i}",
            "name": "get_weather",
            "arguments": json.dumps({"city": city}),
            "status": "completed",
        }
        for i, city in enumerate(["Paris", "London", "Tokyo"])
    ]
    items.append({**_shell_call(), "id": "fc_shell", "callId": "call_shell"})
    return _truncated(items)


class ShellIn(BaseModel):
    commands: List[str]


class WeatherIn(BaseModel):
    city: str


def test_extracts_no_tool_calls_from_truncated_response() -> None:
    response = _truncated([_shell_call()])
    assert response_has_tool_calls(response) is False
    assert extract_tool_calls_from_response(response) == []


def test_complete_response_with_call_still_reports_tool_calls() -> None:
    response = make_response("r", [_shell_call()])
    assert response_has_tool_calls(response) is True
    assert [c.id for c in extract_tool_calls_from_response(response)] == ["call_1"]


async def test_finalizes_on_truncated_turn_without_executing_or_requesting_again() -> None:
    executed: List[Any] = []

    def run_shell(params: ShellIn, ctx: Any) -> Dict[str, Any]:
        executed.append(params)
        return {"ok": True}

    client = QueuedClient([_truncated([_shell_call()])])
    result = call_model(
        client,
        {
            "model": "test-model",
            "input": "Run echo hello.",
            "stop_when": step_count_is(3),
            "tools": [tool(name="run_shell", input_schema=ShellIn, execute=run_shell)],
        },
    )
    response = await result.get_response()
    assert executed == []
    assert len(client.requests) == 1
    assert response["id"] == "resp_truncated"
    assert response["status"] == "incomplete"


async def test_does_not_execute_calls_completed_before_the_cutoff_either() -> None:
    executed: List[Any] = []

    def get_weather(params: WeatherIn, ctx: Any) -> Dict[str, Any]:
        executed.append(params)
        return {"temperature": 22}

    def run_shell(params: ShellIn, ctx: Any) -> Dict[str, Any]:
        executed.append(params)
        return {"ok": True}

    client = QueuedClient([_batch()])
    result = call_model(
        client,
        {
            "model": "test-model",
            "input": "Weather in three cities, then run echo hello.",
            "stop_when": step_count_is(3),
            "tools": [
                tool(name="get_weather", input_schema=WeatherIn, execute=get_weather),
                tool(name="run_shell", input_schema=ShellIn, execute=run_shell),
            ],
        },
    )
    response = await result.get_response()
    assert executed == []
    assert len(client.requests) == 1
    assert response["status"] == "incomplete"
    # The caller gets the whole turn, cut-off item included, to resume from.
    assert [item["callId"] for item in response["output"]] == [
        "call_weather_0",
        "call_weather_1",
        "call_weather_2",
        "call_shell",
    ]
