from __future__ import annotations

from typing import Any, Dict

from openrouter_agent import call_model, step_count_is, tool
from openrouter_agent.model_result import DEFAULT_FINAL_RESPONSE_DIRECTIVE
from tests._fixtures import QueuedClient, text_response
from tests._fixtures import tool_call_response as _tool_call_response


def tool_call_response(response_id: str) -> Dict[str, Any]:
    """The one weather call every test in this file queues as its first turn."""
    return _tool_call_response(response_id, "get_weather", call_id="call_weather", arguments='{"city":"nyc"}')


weather_tool = tool(
    name="get_weather",
    input_schema=dict,
    output_schema=dict,
    execute=lambda params, ctx: {"temperature": 22},
)


async def test_bare_true_appends_default_directive() -> None:
    client = QueuedClient([tool_call_response("resp_1"), text_response("resp_2", "Final summary.")])

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": "weather?",
            "tools": [weather_tool],
            "stop_when": step_count_is(0),
            "allow_final_response": True,
        },
    ).get_text()

    assert text == "Final summary."
    second_request = client.requests[1]
    assert "tools" in second_request
    assert second_request["tool_choice"] == "none"
    last_item = second_request["input"][-1]
    assert last_item == {"role": "user", "content": DEFAULT_FINAL_RESPONSE_DIRECTIVE}


async def test_omitted_allow_final_response_defaults_to_enabled_with_directive() -> None:
    client = QueuedClient([tool_call_response("resp_1"), text_response("resp_2", "Final summary.")])

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": "weather?",
            "tools": [weather_tool],
            "stop_when": step_count_is(0),
            # allow_final_response deliberately omitted
        },
    ).get_text()

    assert text == "Final summary."
    second_request = client.requests[1]
    assert second_request["tool_choice"] == "none"
    assert second_request["input"][-1] == {"role": "user", "content": DEFAULT_FINAL_RESPONSE_DIRECTIVE}


async def test_non_empty_string_overrides_directive() -> None:
    client = QueuedClient([tool_call_response("resp_1"), text_response("resp_2", "Final summary.")])

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "weather?",
            "tools": [weather_tool],
            "stop_when": step_count_is(0),
            "allow_final_response": "Summarize now.",
        },
    ).get_text()

    second_request = client.requests[1]
    assert second_request["input"][-1] == {"role": "user", "content": "Summarize now."}


async def test_empty_string_appends_no_message() -> None:
    client = QueuedClient([tool_call_response("resp_1"), text_response("resp_2", "Final summary.")])

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "weather?",
            "tools": [weather_tool],
            "stop_when": step_count_is(0),
            "allow_final_response": "",
        },
    ).get_text()

    second_request = client.requests[1]
    last_item = second_request["input"][-1]
    assert last_item.get("type") == "function_call_output"


async def test_false_disables_the_final_turn_entirely() -> None:
    client = QueuedClient([tool_call_response("resp_1")])

    result = call_model(
        client,
        {
            "model": "test-model",
            "input": "weather?",
            "tools": [weather_tool],
            "stop_when": step_count_is(0),
            "allow_final_response": False,
        },
    )
    response = await result.get_response()

    assert response["id"] == "resp_1"
    assert len(client.requests) == 1
