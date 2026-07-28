from __future__ import annotations

from typing import Any, Dict, List

from openrouter_agent import call_model, step_count_is, tool
from openrouter_agent.model_result import DEFAULT_FINAL_RESPONSE_DIRECTIVE


class QueuedResponses:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.requests: List[Dict[str, Any]] = []

    async def send_async(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return self._responses.pop(0)


class QueuedClient:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.beta = type("Beta", (), {"responses": QueuedResponses(responses)})()


def function_call_item(call_id: str, name: str, arguments: str) -> Dict[str, Any]:
    return {"type": "function_call", "id": f"fc_{call_id}", "callId": call_id, "name": name, "arguments": arguments}


def text_response(response_id: str, text: str) -> Dict[str, Any]:
    return {
        "id": response_id,
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
    }


def tool_call_response(response_id: str) -> Dict[str, Any]:
    return {"id": response_id, "output": [function_call_item("call_weather", "get_weather", '{"city":"nyc"}')]}


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
    second_request = client.beta.responses.requests[1]
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
    second_request = client.beta.responses.requests[1]
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

    second_request = client.beta.responses.requests[1]
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

    second_request = client.beta.responses.requests[1]
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
    assert len(client.beta.responses.requests) == 1
