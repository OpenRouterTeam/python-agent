from __future__ import annotations

import asyncio

import httpx
from pydantic import BaseModel

from openrouter_agent import OpenRouter, call_model, from_claude_messages, max_tokens_used, to_claude_message, tool
from openrouter_agent.conversation_state import create_initial_state, update_state
from openrouter_agent.model_result import ModelResult
from openrouter_agent.stop_conditions import is_stop_condition_met
from openrouter_agent.tool_executor import apply_on_response_received_hooks, execute_tool
from openrouter_agent.tool_types import ParsedToolCall


class Responses:
    def __init__(self) -> None:
        self.requests = []

    async def send_async(self, **kwargs):
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            return {
                "id": "resp_tool",
                "output": [
                    {
                        "type": "function_call",
                        "id": "item_1",
                        "callId": "call_1",
                        "name": "double",
                        "arguments": '{"value": 2}',
                    }
                ],
                "usage": {"total_tokens": 5, "input_tokens": 2, "output_tokens": 3},
            }
        return {
            "id": "resp_done",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}],
        }


class Beta:
    def __init__(self) -> None:
        self.responses = Responses()


class Client:
    def __init__(self) -> None:
        self.beta = Beta()


async def test_max_tokens_used_matches_upstream_total_tokens_only() -> None:
    steps = [{"usage": {"total_tokens": 5, "input_tokens": 5, "output_tokens": 1}}]

    assert await is_stop_condition_met([max_tokens_used(5)], steps)
    assert not await is_stop_condition_met([max_tokens_used(11)], steps)


async def test_full_and_tool_streams_include_turn_and_tool_events() -> None:
    client = Client()
    double = tool(name="double", input_schema=dict, execute=lambda params, ctx: {"value": params["value"] * 2})
    result = ModelResult(
        {"client": client, "request": {"model": "test", "input": "go", "tools": [double]}, "tools": [double]}
    )

    full_events = [event async for event in result.get_full_responses_stream()]
    assert "turn.start" in [event["type"] for event in full_events]
    assert "turn.end" in [event["type"] for event in full_events]
    assert "tool.result" in [event["type"] for event in full_events]
    assert "tool.call_output" in [event["type"] for event in full_events]

    result2 = ModelResult(
        {"client": Client(), "request": {"model": "test", "input": "go", "tools": [double]}, "tools": [double]}
    )
    tool_events = [event async for event in result2.get_tool_stream()]
    assert "turn.start" in [event["type"] for event in tool_events]
    assert "turn.end" in [event["type"] for event in tool_events]
    assert "tool_result" in [event["type"] for event in tool_events]
    assert "tool_call_output" in [event["type"] for event in tool_events]


async def test_generator_tool_emits_preliminary_event_before_completion() -> None:
    seen = asyncio.Event()
    release = asyncio.Event()

    async def execute(params, ctx):
        yield {"progress": 0.5}
        seen.set()
        await release.wait()
        yield {"answer": params["value"] * 2}

    def output_schema(value):
        if "answer" not in value:
            raise ValueError("missing answer")
        return value

    generated = tool(name="gen", input_schema=dict, event_schema=dict, output_schema=output_schema, execute=execute)
    prelim = []
    task = asyncio.create_task(
        execute_tool(
            generated,
            ParsedToolCall(id="call_1", name="gen", arguments={"value": 3}),
            on_preliminary_result=lambda call_id, value: prelim.append((call_id, value)),
        )
    )
    await asyncio.wait_for(seen.wait(), timeout=1)
    assert prelim == [("call_1", {"progress": 0.5})]
    release.set()
    result = await task
    assert result is not None
    assert result["result"] == {"answer": 6}


def test_openrouter_normalizes_hooks_into_sdk_configuration() -> None:
    calls = []

    class Hook:
        def before_request(self, ctx, request):
            calls.append(("before", request.url.path))
            return request

        def after_success(self, ctx, response):
            calls.append(("after", response.status_code))
            return response

    def handler(request):
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "test/model",
                "output": [],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            },
        )

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenRouter(api_key="sk-test", server_url="https://mock.local", async_client=async_client, hooks=Hook())

    try:
        asyncio.run(client.beta.responses.send_async(model="test/model", input="hi"))
    except Exception:
        pass
    finally:
        asyncio.run(async_client.aclose())

    assert ("before", "/responses") in calls
    assert ("after", 200) in calls


def test_claude_conversion_preserves_metadata_and_structured_unsupported_content() -> None:
    response = {
        "id": "resp_1",
        "model": "openai/test",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "hello"}]},
            {"type": "image_generation_call", "id": "img_1", "status": "completed", "result": "opaque"},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }

    claude = to_claude_message(response)
    assert claude["id"] == "resp_1"
    assert claude["type"] == "message"
    assert claude["model"] == "openai/test"
    assert claude["unsupported_content"][0]["original_type"] == "image_generation_call"
    assert set(claude["unsupported_content"][0]) == {"original_type", "data", "reason"}

    round_tripped = from_claude_messages([claude])
    assert any(item.get("type") == "image_generation_call" for item in round_tripped)


async def test_hitl_on_response_received_only_applies_to_fresh_outputs() -> None:
    calls = []

    def on_response_received(raw, ctx):
        calls.append(raw)
        return raw

    hitl = tool(
        name="approve",
        input_schema=dict,
        output_schema=dict,
        on_tool_called=lambda params, ctx: None,
        on_response_received=on_response_received,
    )
    state = update_state(
        create_initial_state(),
        {
            "messages": [
                {"type": "function_call", "id": "item_1", "callId": "call_1", "name": "approve", "arguments": "{}"},
                {"type": "function_call_output", "callId": "call_1", "output": '{"ok":true}'},
            ]
        },
    )

    class StateAccessor:
        async def load(self):
            return state

        async def save(self, new_state):
            self.saved = new_state

    client = Client()
    result = call_model(client, {"model": "test/model", "input": "next", "tools": [hitl], "state": StateAccessor()})

    await result.get_response()
    assert calls == []


async def test_new_messages_stream_filters_unknown_manual_tool_calls() -> None:
    class OneShotResponses:
        async def send_async(self, **kwargs):
            return {
                "id": "resp_1",
                "output": [
                    {
                        "type": "function_call",
                        "id": "ghost_item",
                        "callId": "ghost_call",
                        "name": "ghost",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call",
                        "id": "real_item",
                        "callId": "real_call",
                        "name": "real",
                        "arguments": "{}",
                    },
                ],
            }

    class OneShotClient:
        def __init__(self) -> None:
            self.beta = type("Beta", (), {"responses": OneShotResponses()})()

    real = tool(name="real", input_schema=dict, execute=False)
    result = call_model(OneShotClient(), {"model": "test/model", "input": "call tools", "tools": [real]})

    messages = [item async for item in result.get_new_messages_stream()]
    assert [item["name"] for item in messages] == ["real"]


async def test_hitl_without_response_hook_validates_caller_output() -> None:
    class Output(BaseModel):
        ok: bool

    hitl = tool(name="approve", input_schema=dict, output_schema=Output, on_tool_called=lambda params, ctx: None)
    items = [
        {"type": "function_call", "id": "item_1", "callId": "call_1", "name": "approve", "arguments": "{}"},
        {"type": "function_call_output", "callId": "call_1", "output": '{"ok":"nope"}'},
    ]

    rewritten = await apply_on_response_received_hooks(items, [hitl])

    assert rewritten is not items
    assert "error" in rewritten[1]["output"]
    assert "originalOutput" in rewritten[1]["output"]


class MemoryState:
    def __init__(self):
        self.current = None
        self.saved = []

    async def load(self):
        return self.current

    async def save(self, new_state):
        self.current = new_state
        self.saved.append(new_state)


class PauseThenDoneResponses:
    def __init__(self, tool_name: str, pause_first: bool = True) -> None:
        self.tool_name = tool_name
        self.pause_first = pause_first
        self.requests = []

    async def send_async(self, **kwargs):
        self.requests.append(kwargs)
        if self.pause_first and len(self.requests) == 1:
            return {
                "id": f"resp_{self.tool_name}",
                "output": [
                    {
                        "type": "function_call",
                        "id": "item_1",
                        "callId": "call_1",
                        "name": self.tool_name,
                        "arguments": "{}",
                    }
                ],
            }
        return {
            "id": "resp_done",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}],
        }


class PauseClient:
    def __init__(self, tool_name: str, pause_first: bool = True) -> None:
        self.beta = type("Beta", (), {"responses": PauseThenDoneResponses(tool_name, pause_first)})()


async def test_approval_pause_persists_tool_call_turn_and_resume_orders_output_after_call() -> None:
    state = MemoryState()
    delete = tool(
        name="delete",
        input_schema=dict,
        output_schema=dict,
        execute=lambda params, ctx: {"ok": True},
        require_approval=True,
    )

    first_client = PauseClient("delete")
    first = call_model(first_client, {"model": "test", "input": "delete it", "tools": [delete], "state": state})
    await first.get_response()

    paused = state.current
    assert paused.status == "awaiting_approval"
    assert paused.previous_response_id == "resp_delete"
    assert [item.get("type") for item in paused.messages][-1:] == ["function_call"]

    resume_client = PauseClient("delete", pause_first=False)
    resumed = call_model(
        resume_client,
        {"model": "test", "input": [], "tools": [delete], "state": state, "approve_tool_calls": ["call_1"]},
    )
    await resumed.get_response()

    sent_input = resume_client.beta.responses.requests[0]["input"]
    types = [item.get("type") for item in sent_input]
    assert types.index("function_call") < types.index("function_call_output")
    assert state.saved[-1].previous_response_id == "resp_done"


async def test_hitl_pause_persists_tool_call_turn_and_resume_orders_output_after_call() -> None:
    state = MemoryState()
    calls = 0

    def decide(params, ctx):
        nonlocal calls
        calls += 1
        return None if calls == 1 else {"ok": True}

    approve = tool(name="approve", input_schema=dict, output_schema=dict, on_tool_called=decide)

    first_client = PauseClient("approve")
    first = call_model(first_client, {"model": "test", "input": "approve it", "tools": [approve], "state": state})
    await first.get_response()

    paused = state.current
    assert paused.status == "awaiting_hitl"
    assert paused.previous_response_id == "resp_approve"
    assert [item.get("type") for item in paused.messages][-1:] == ["function_call"]

    resume_client = PauseClient("approve", pause_first=False)
    resumed = call_model(
        resume_client,
        {"model": "test", "input": [], "tools": [approve], "state": state, "approve_tool_calls": ["call_1"]},
    )
    await resumed.get_response()

    sent_input = resume_client.beta.responses.requests[0]["input"]
    types = [item.get("type") for item in sent_input]
    assert types.index("function_call") < types.index("function_call_output")


async def test_model_result_tool_calls_stream_reconstructs_streamed_argument_deltas() -> None:
    class StreamResponses:
        async def send_async(self, **kwargs):
            async def events():
                yield {
                    "type": "response.output_item.added",
                    "item": {"type": "function_call", "id": "item_1", "callId": "call_1", "name": "lookup"},
                }
                yield {"type": "response.function_call_arguments.delta", "itemId": "item_1", "delta": '{"q"'}
                yield {"type": "response.function_call_arguments.delta", "itemId": "item_1", "delta": ':"x"}'}
                yield {"type": "response.function_call_arguments.done", "itemId": "item_1"}
                yield {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_stream",
                        "output": [
                            {
                                "type": "function_call",
                                "id": "item_1",
                                "callId": "call_1",
                                "name": "lookup",
                                "arguments": '{"q":"x"}',
                            }
                        ],
                    },
                }

            return events()

    class StreamClient:
        def __init__(self) -> None:
            self.beta = type("Beta", (), {"responses": StreamResponses()})()

    manual = tool(name="lookup", input_schema=dict, execute=False)
    result = call_model(StreamClient(), {"model": "test", "input": "lookup", "tools": [manual]})

    calls = [call async for call in result.get_tool_calls_stream()]

    assert calls == [ParsedToolCall(id="call_1", name="lookup", arguments={"q": "x"})]
