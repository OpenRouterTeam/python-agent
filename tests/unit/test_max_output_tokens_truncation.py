"""Ported from upstream tests/unit/max-output-tokens-truncation.test.ts (#125).

A response the provider stopped at `max_output_tokens` is an unfinished batch:
none of its function calls run -- not even the ones completed before the
cut-off -- and the loop does not re-request on the exhausted budget.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from openrouter_agent import call_model, step_count_is, tool
from openrouter_agent.stream_transformers import extract_tool_calls_from_response, response_has_tool_calls
from tests._fixtures import QueuedClient, make_response, recording


def _truncated(output: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    items = output or [
        {
            "type": "function_call",
            "id": "fc_shell",
            "callId": "call_shell",
            "name": "run_shell",
            "arguments": '{"commands":["echo hel',
            "status": "incomplete",
        }
    ]
    return make_response(
        "resp_truncated", items, status="incomplete", incompleteDetails={"reason": "max_output_tokens"}
    )


def _truncated_batch() -> Dict[str, Any]:
    weather = [
        {
            "type": "function_call",
            "id": f"fc_weather_{index}",
            "callId": f"call_weather_{index}",
            "name": "get_weather",
            "arguments": json.dumps({"city": city}),
            "status": "completed",
        }
        for index, city in enumerate(["Paris", "London", "Tokyo"])
    ]
    shell = {
        "type": "function_call",
        "id": "fc_shell",
        "callId": "call_shell",
        "name": "run_shell",
        "arguments": '{"commands":',
        "status": "incomplete",
    }
    return _truncated([*weather, shell])


class _Shell(BaseModel):
    commands: List[str]


class _City(BaseModel):
    city: str


def test_extracts_no_tool_calls_from_a_response_truncated_at_max_output_tokens() -> None:
    response = _truncated()

    assert response_has_tool_calls(response) is False
    assert extract_tool_calls_from_response(response) == []


def test_a_completed_response_still_reports_its_tool_calls() -> None:
    response = make_response(
        "resp_ok",
        [{"type": "function_call", "id": "fc", "callId": "call_1", "name": "run_shell", "arguments": "{}"}],
    )

    assert response_has_tool_calls(response) is True
    assert [c.id for c in extract_tool_calls_from_response(response)] == ["call_1"]


async def test_finalizes_on_the_truncated_turn_without_executing_or_requesting_again() -> None:
    executed: List[Any] = []
    shell = tool(
        name="run_shell",
        description="Run shell commands.",
        input_schema=_Shell,
        execute=recording(executed, {"ok": True}),
    )
    client = QueuedClient([_truncated()])

    response = await call_model(
        client,
        {"model": "test-model", "input": "Run echo hello.", "stop_when": step_count_is(3), "tools": [shell]},
    ).get_response()

    assert executed == []
    assert len(client.requests) == 1
    assert response["id"] == "resp_truncated"
    assert response["status"] == "incomplete"


async def test_does_not_execute_the_calls_that_completed_before_the_cut_off_either() -> None:
    executed: List[Any] = []
    weather = tool(
        name="get_weather",
        input_schema=_City,
        execute=recording(executed, {"temperature": 22}),
    )
    shell = tool(
        name="run_shell",
        input_schema=_Shell,
        execute=recording(executed, {"ok": True}),
    )
    client = QueuedClient([_truncated_batch()])

    response = await call_model(
        client,
        {
            "model": "test-model",
            "input": "Weather in three cities, then run echo hello.",
            "stop_when": step_count_is(3),
            "tools": [weather, shell],
        },
    ).get_response()

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
