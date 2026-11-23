"""The tool call shown by get_tool_calls_stream() must be the one that is saved and run."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest
from pydantic import BaseModel

from openrouter_agent import call_model, tool
from tests._fixtures import MemoryStateAccessor, QueuedClient, function_call_item, make_response, text_response


class Amount(BaseModel):
    amount: int


def _stream(events: List[Dict[str, Any]]) -> Any:
    async def gen() -> AsyncIterator[Dict[str, Any]]:
        for event in events:
            yield event

    return gen()


def _tool_call_events(
    streamed_amount: int,
    terminal_amount: int,
    *,
    item_id: str = "fc_1",
    terminal_name: str = "pay",
    call_id_key: str = "callId",
) -> List[Dict[str, Any]]:
    terminal_item = function_call_item("call_1", terminal_name, json.dumps({"amount": terminal_amount}))
    return [
        {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": item_id, call_id_key: "call_1", "name": "pay", "arguments": ""},
        },
        {"type": "response.function_call_arguments.delta", "itemId": item_id, "delta": '{"amount":'},
        {"type": "response.function_call_arguments.delta", "itemId": item_id, "delta": f"{streamed_amount}}}"},
        {"type": "response.function_call_arguments.done", "itemId": item_id},
        {"type": "response.completed", "response": make_response("r1", [terminal_item])},
    ]


def _pay_tool(executed: List[int]) -> Any:
    def execute(params: Amount, _context: Any) -> Dict[str, bool]:
        executed.append(params.amount)
        return {"ok": True}

    return tool(name="pay", input_schema=Amount, output_schema=dict, execute=execute, require_approval=True)


async def _run(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    state = MemoryStateAccessor()
    executed: List[int] = []
    pay = _pay_tool(executed)
    client = QueuedClient(
        [_stream(events), _stream([{"type": "response.completed", "response": text_response("r2", "done")}])]
    )
    shown: List[Any] = []
    error: Optional[BaseException] = None
    try:
        result = call_model(client, {"model": "m", "input": "pay", "tools": [pay], "state": state})
        async for call in result.get_tool_calls_stream():
            shown.append((call.id, call.name, call.arguments))
    except ValueError as exc:
        error = exc
    pending = [(c.id, c.arguments) for c in (state.stored.pending_tool_calls or [])] if state.stored else []
    if error is None:
        await call_model(
            client, {"model": "m", "input": "go", "tools": [pay], "state": state, "approve_tool_calls": ["call_1"]}
        ).get_response()
    return {"shown": shown, "pending": pending, "executed": executed, "error": error, "requests": len(client.requests)}


async def test_matching_stream_and_terminal_call_runs_after_approval() -> None:
    out = await _run(_tool_call_events(1, 1))

    assert out["error"] is None
    assert out["shown"] == [("call_1", "pay", {"amount": 1})]
    assert out["pending"] == [("call_1", {"amount": 1})]
    assert out["executed"] == [1]


async def test_terminal_arguments_that_differ_from_the_stream_are_refused() -> None:
    out = await _run(_tool_call_events(1, 1000))

    assert isinstance(out["error"], ValueError)
    assert "call_1" in str(out["error"])
    assert out["pending"] == []
    assert out["executed"] == []
    assert out["requests"] == 1


async def test_terminal_name_that_differs_from_the_stream_is_refused() -> None:
    out = await _run(_tool_call_events(1, 1, terminal_name="refund"))

    assert isinstance(out["error"], ValueError)
    assert out["pending"] == []
    assert out["executed"] == []


async def test_mismatch_is_refused_when_stream_and_terminal_item_ids_differ() -> None:
    out = await _run(_tool_call_events(1, 1000, item_id="stream_item", call_id_key="call_id"))

    assert isinstance(out["error"], ValueError)
    assert out["pending"] == []
    assert out["executed"] == []


async def test_stream_reports_snake_case_call_id_instead_of_item_id() -> None:
    out = await _run(_tool_call_events(1, 1, call_id_key="call_id"))

    assert out["error"] is None
    assert out["shown"] == [("call_1", "pay", {"amount": 1})]
    assert out["executed"] == [1]


@pytest.mark.parametrize("terminal_amount", [1, 1000])
async def test_resume_after_refused_response_runs_nothing(terminal_amount: int) -> None:
    state = MemoryStateAccessor()
    executed: List[int] = []
    pay = _pay_tool(executed)
    client = QueuedClient(
        [
            _stream(_tool_call_events(1, terminal_amount)),
            _stream([{"type": "response.completed", "response": text_response("r2", "done")}]),
        ]
    )
    try:
        await call_model(client, {"model": "m", "input": "pay", "tools": [pay], "state": state}).get_response()
    except ValueError:
        pass
    try:
        await call_model(
            client, {"model": "m", "input": "go", "tools": [pay], "state": state, "approve_tool_calls": ["call_1"]}
        ).get_response()
    except Exception:
        pass

    assert executed == ([1] if terminal_amount == 1 else [])
