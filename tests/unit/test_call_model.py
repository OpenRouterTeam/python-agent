from __future__ import annotations

from openrouter_agent import call_model, step_count_is, tool


class Responses:
    def __init__(self) -> None:
        self.requests = []

    async def send_async(self, **kwargs):
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            return {
                "id": "resp_1",
                "output": [
                    {
                        "type": "function_call",
                        "id": "item_1",
                        "callId": "call_1",
                        "name": "double",
                        "arguments": '{"value": 2}',
                    }
                ],
            }
        return {
            "id": "resp_2",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "4"}]}],
            "usage": {"total_tokens": 3},
        }


class Beta:
    def __init__(self) -> None:
        self.responses = Responses()


class Client:
    def __init__(self) -> None:
        self.beta = Beta()
        self.chat = object()


async def test_call_model_uses_responses_api_and_executes_tool_loop() -> None:
    client = Client()
    double = tool(name="double", input_schema=dict, execute=lambda params, ctx: {"value": params["value"] * 2})

    result = call_model(client, {"model": "test/model", "input": "double 2", "tools": [double]})

    assert await result.get_text() == "4"
    assert len(client.beta.responses.requests) == 2
    assert client.beta.responses.requests[0]["stream"] is True
    assert client.beta.responses.requests[0]["tools"][0]["name"] == "double"
    assert client.beta.responses.requests[1]["input"][-1]["type"] == "function_call_output"
    assert [call.name for call in await result.get_tool_calls()] == ["double"]


async def test_call_model_forwards_request_options_to_responses_api() -> None:
    client = Client()

    result = call_model(
        client,
        {"model": "test/model", "input": "hello"},
        {"headers": {"x-test": "yes"}, "timeout_ms": 1234},
    )

    await result.get_response()
    assert client.beta.responses.requests[0]["http_headers"]["x-test"] == "yes"
    assert client.beta.responses.requests[0]["http_headers"]["x-openrouter-callmodel"] == "true"
    assert client.beta.responses.requests[0]["timeout_ms"] == 1234


async def test_model_result_stream_consumers_are_reusable() -> None:
    client = Client()
    noop = tool(name="double", input_schema=dict, execute=lambda params, ctx: {"value": 4})
    result = call_model(client, {"model": "test/model", "input": "double 2", "tools": [noop]})

    assert [chunk async for chunk in result.get_text_stream()] == ["4"]
    assert [chunk async for chunk in result.get_text_stream()] == ["4"]
    assert await result.get_text() == "4"


async def test_allow_final_response_executes_pending_tool_before_no_tools_turn() -> None:
    client = Client()
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
    assert len(client.beta.responses.requests) == 2
    assert "tools" not in client.beta.responses.requests[1]
    second_input = client.beta.responses.requests[1]["input"]
    types = [item.get("type") for item in second_input]
    assert types.index("function_call") < types.index("function_call_output")
    assert second_input[-1] == {"role": "user", "content": "summarize"}


async def test_next_turn_params_are_applied_to_followup_request() -> None:
    client = Client()
    double = tool(
        name="double",
        input_schema=dict,
        execute=lambda params, ctx: {"value": params["value"] * 2},
        next_turn_params={"temperature": lambda args, ctx: 0.2},
    )

    result = call_model(client, {"model": "test/model", "input": "double 2", "tools": [double]})

    await result.get_response()
    assert client.beta.responses.requests[1]["temperature"] == 0.2


async def test_user_input_persists_with_response_in_state() -> None:
    class State:
        def __init__(self) -> None:
            self.current = None

        async def load(self):
            return self.current

        async def save(self, state):
            self.current = state

    class TextResponses:
        async def send_async(self, **kwargs):
            return {
                "id": "resp_text",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}],
            }

    class TextClient:
        def __init__(self) -> None:
            self.beta = type("Beta", (), {"responses": TextResponses()})()

    state = State()
    client = TextClient()

    result = call_model(client, {"model": "test/model", "input": "hello", "state": state})
    await result.get_response()

    assert state.current.status == "complete"
    assert state.current.messages[0] == {"role": "user", "content": "hello"}
    assert state.current.messages[-1]["type"] == "message"
