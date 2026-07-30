from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import httpx
from pydantic import BaseModel

from openrouter_agent import OpenRouter, call_model, from_claude_messages, max_tokens_used, to_claude_message, tool
from openrouter_agent.conversation_state import create_initial_state, update_state
from openrouter_agent.model_result import ModelResult
from openrouter_agent.stop_conditions import is_stop_condition_met
from openrouter_agent.tool_executor import apply_on_response_received_hooks, execute_tool
from openrouter_agent.tool_types import ParsedToolCall
from tests._fixtures import (
    MemoryStateAccessor,
    QueuedClient,
    function_call_item,
    make_response,
    text_response,
    tool_call_response,
    usage_block,
)


def double_then_done() -> List[Dict[str, Any]]:
    """One `double` tool call, then a terminal "done"."""
    return [
        tool_call_response(
            "resp_tool",
            "double",
            call_id="call_1",
            arguments='{"value": 2}',
            usage=usage_block(total_tokens=5, input_tokens=2, output_tokens=3),
        ),
        text_response("resp_done", "done"),
    ]


def pause_then_done(tool_name: str, pause_first: bool = True) -> List[Dict[str, Any]]:
    """A pausing tool call (approval/HITL) followed by the terminal turn.

    `pause_first=False` builds the resume client, whose only queued turn is the
    completion -- so an unexpected extra request fails loudly.
    """
    turns: List[Dict[str, Any]] = []
    if pause_first:
        turns.append(tool_call_response(f"resp_{tool_name}", tool_name, call_id="call_1"))
    turns.append(text_response("resp_done", "done"))
    return turns


async def test_max_tokens_used_matches_upstream_total_tokens_only() -> None:
    steps = [{"usage": {"total_tokens": 5, "input_tokens": 5, "output_tokens": 1}}]

    assert await is_stop_condition_met([max_tokens_used(5)], steps)
    assert not await is_stop_condition_met([max_tokens_used(11)], steps)


async def test_full_and_tool_streams_include_turn_and_tool_events() -> None:
    client = QueuedClient(double_then_done())
    double = tool(name="double", input_schema=dict, execute=lambda params, ctx: {"value": params["value"] * 2})
    result = ModelResult(
        {"client": client, "request": {"model": "test", "input": "go", "tools": [double]}, "tools": [double]}
    )

    full_events = [event async for event in result.get_full_responses_stream()]
    full_types = [event["type"] for event in full_events]
    # Order and count, not membership: a membership check passes even when
    # turn.end fires twice, never fires, or precedes turn.start. See
    # tests/unit/test_turn_end_race_condition.py for why that matters here.
    assert full_types.count("turn.start") == full_types.count("turn.end")
    assert full_types.index("turn.start") < full_types.index("turn.end")
    for expected in ("tool.result", "tool.call_output"):
        assert full_types.count(expected) == 1, f"{expected}: {full_types}"
    # The tool round's events land inside the turn that produced them.
    assert full_types.index("turn.start") < full_types.index("tool.result")

    result2 = ModelResult(
        {
            "client": QueuedClient(double_then_done()),
            "request": {"model": "test", "input": "go", "tools": [double]},
            "tools": [double],
        }
    )
    tool_events = [event async for event in result2.get_tool_stream()]
    tool_types = [event["type"] for event in tool_events]
    assert tool_types.count("turn.start") == tool_types.count("turn.end")
    assert tool_types.index("turn.start") < tool_types.index("turn.end")
    for expected in ("tool_result", "tool_call_output"):
        assert tool_types.count(expected) == 1, f"{expected}: {tool_types}"


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

    accessor = MemoryStateAccessor()
    accessor.stored = state

    client = QueuedClient(double_then_done())
    result = call_model(client, {"model": "test/model", "input": "next", "tools": [hitl], "state": accessor})

    await result.get_response()
    assert calls == []


async def test_new_messages_stream_filters_unknown_manual_tool_calls() -> None:
    client = QueuedClient(
        [
            make_response(
                "resp_1",
                [
                    function_call_item("ghost_call", "ghost"),
                    function_call_item("real_call", "real"),
                ],
            )
        ]
    )

    real = tool(name="real", input_schema=dict, execute=False)
    result = call_model(client, {"model": "test/model", "input": "call tools", "tools": [real]})

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


async def test_approval_pause_persists_tool_call_turn_and_resume_orders_output_after_call() -> None:
    state = MemoryStateAccessor()
    delete = tool(
        name="delete",
        input_schema=dict,
        output_schema=dict,
        execute=lambda params, ctx: {"ok": True},
        require_approval=True,
    )

    first_client = QueuedClient(pause_then_done("delete"))
    first = call_model(first_client, {"model": "test", "input": "delete it", "tools": [delete], "state": state})
    await first.get_response()

    paused = state.stored
    assert paused is not None
    assert paused.status == "awaiting_approval"
    assert paused.previous_response_id == "resp_delete"
    assert [item.get("type") for item in paused.messages][-1:] == ["function_call"]

    resume_client = QueuedClient(pause_then_done("delete", pause_first=False))
    resumed = call_model(
        resume_client,
        {"model": "test", "input": [], "tools": [delete], "state": state, "approve_tool_calls": ["call_1"]},
    )
    await resumed.get_response()

    sent_input = resume_client.requests[0]["input"]
    types = [item.get("type") for item in sent_input]
    assert types.index("function_call") < types.index("function_call_output")
    assert state.saved[-1].previous_response_id == "resp_done"


async def test_hitl_pause_persists_tool_call_turn_and_resume_orders_output_after_call() -> None:
    state = MemoryStateAccessor()
    calls = 0

    def decide(params, ctx):
        nonlocal calls
        calls += 1
        return None if calls == 1 else {"ok": True}

    approve = tool(name="approve", input_schema=dict, output_schema=dict, on_tool_called=decide)

    first_client = QueuedClient(pause_then_done("approve"))
    first = call_model(first_client, {"model": "test", "input": "approve it", "tools": [approve], "state": state})
    await first.get_response()

    paused = state.stored
    assert paused is not None
    assert paused.status == "awaiting_hitl"
    assert paused.previous_response_id == "resp_approve"
    assert [item.get("type") for item in paused.messages][-1:] == ["function_call"]

    resume_client = QueuedClient(pause_then_done("approve", pause_first=False))
    resumed = call_model(
        resume_client,
        {"model": "test", "input": [], "tools": [approve], "state": state, "approve_tool_calls": ["call_1"]},
    )
    await resumed.get_response()

    sent_input = resume_client.requests[0]["input"]
    types = [item.get("type") for item in sent_input]
    assert types.index("function_call") < types.index("function_call_output")


async def test_model_result_tool_calls_stream_reconstructs_streamed_argument_deltas() -> None:
    # Bespoke: this must yield an SSE *event* sequence, not a single result, so
    # it cannot be a QueuedClient. Its terminal payload still uses the shared
    # builders so it carries every required response field.
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
                    "response": tool_call_response("resp_stream", "lookup", call_id="call_1", arguments='{"q":"x"}'),
                }

            return events()

    class StreamClient:
        def __init__(self) -> None:
            self.beta = type("Beta", (), {"responses": StreamResponses()})()

    manual = tool(name="lookup", input_schema=dict, execute=False)
    result = call_model(StreamClient(), {"model": "test", "input": "lookup", "tools": [manual]})

    calls = [call async for call in result.get_tool_calls_stream()]

    assert calls == [ParsedToolCall(id="call_1", name="lookup", arguments={"q": "x"})]
