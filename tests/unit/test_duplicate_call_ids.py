"""Tool calls that share a call_id must not let one approval run another call."""

from __future__ import annotations

import dataclasses
from typing import Any, List

import pytest

from openrouter_agent import call_model, step_count_is, tool
from tests._fixtures import MemoryStateAccessor, QueuedClient, function_call_item, make_response, text_response


def _tools(executed: List[str]) -> List[Any]:
    def read(_params: Any, _context: Any) -> Any:
        executed.append("read_config")
        return {"ok": True}

    def delete(_params: Any, _context: Any) -> Any:
        executed.append("delete_record")
        return {"deleted": True}

    return [
        tool(name="read_config", input_schema=dict, output_schema=dict, execute=read, require_approval=True),
        tool(name="delete_record", input_schema=dict, output_schema=dict, execute=delete, require_approval=True),
    ]


def _batch(first_id: str, second_id: str) -> dict:
    return make_response(
        "r1",
        [
            function_call_item(first_id, "read_config", '{"path":"/public"}'),
            function_call_item(second_id, "delete_record", '{"id":"rec"}'),
        ],
    )


async def test_approving_the_harmless_call_cannot_run_a_protected_call_with_the_same_id() -> None:
    executed: List[str] = []
    tools = _tools(executed)
    state = MemoryStateAccessor()
    client = QueuedClient([_batch("call_x", "call_x"), text_response("r2", "done")])

    with pytest.raises(ValueError, match='more than one function_call with call_id "call_x"'):
        await call_model(client, {"model": "m", "input": "go", "tools": tools, "state": state}).get_response()

    assert executed == []
    assert len(client.requests) == 1
    assert state.stored is None or not state.stored.pending_tool_calls


async def test_distinct_ids_still_run_only_the_approved_call() -> None:
    executed: List[str] = []
    tools = _tools(executed)
    state = MemoryStateAccessor()
    client = QueuedClient([_batch("call_a", "call_b"), text_response("r2", "done")])

    await call_model(client, {"model": "m", "input": "go", "tools": tools, "state": state}).get_response()
    assert state.stored is not None
    assert [c.id for c in state.stored.pending_tool_calls or []] == ["call_a", "call_b"]

    await call_model(
        client, {"model": "m", "input": "continue", "tools": tools, "state": state, "approve_tool_calls": ["call_a"]}
    ).get_response()

    assert executed == ["read_config"]


async def test_auto_executed_batch_with_duplicate_ids_runs_nothing() -> None:
    executed: List[str] = []

    def run(_params: Any, _context: Any) -> Any:
        executed.append("run")
        return {"ok": True}

    auto = tool(name="lookup", input_schema=dict, output_schema=dict, execute=run)
    client = QueuedClient(
        [
            make_response("r1", [function_call_item("call_x", "lookup"), function_call_item("call_x", "lookup")]),
            text_response("r2", "done"),
        ]
    )

    with pytest.raises(ValueError, match="call_x"):
        await call_model(client, {"model": "m", "input": "go", "tools": [auto]}).get_response()

    assert executed == []
    assert len(client.requests) == 1


async def test_allow_final_response_path_refuses_duplicate_ids() -> None:
    executed: List[str] = []
    tools = _tools(executed)
    client = QueuedClient([_batch("call_x", "call_x"), text_response("r2", "done")])

    with pytest.raises(ValueError, match="call_x"):
        await call_model(
            client,
            {
                "model": "m",
                "input": "go",
                "tools": tools,
                "state": MemoryStateAccessor(),
                "stop_when": step_count_is(0),
                "allow_final_response": True,
            },
        ).get_response()

    assert executed == []
    assert len(client.requests) == 1


async def test_manual_only_batch_with_duplicate_ids_is_returned_to_the_caller() -> None:
    manual = tool(name="exec_command", input_schema=dict, execute=False)
    state = MemoryStateAccessor()
    client = QueuedClient(
        [
            make_response(
                "r1", [function_call_item("call_x", "exec_command"), function_call_item("call_x", "exec_command")]
            )
        ]
    )

    result = call_model(client, {"model": "m", "input": "go", "tools": [manual], "state": state})
    await result.get_response()

    assert [c.id for c in await result.get_pending_tool_calls()] == ["call_x", "call_x"]


async def test_resume_refuses_saved_state_with_duplicate_pending_ids() -> None:
    executed: List[str] = []
    tools = _tools(executed)
    state = MemoryStateAccessor()
    client = QueuedClient([_batch("call_a", "call_b"), text_response("r2", "done")])
    await call_model(client, {"model": "m", "input": "go", "tools": tools, "state": state}).get_response()
    assert state.stored is not None
    pending = state.stored.pending_tool_calls or []
    state.stored = dataclasses.replace(
        state.stored, pending_tool_calls=[pending[0], dataclasses.replace(pending[1], id="call_a")]
    )

    with pytest.raises(ValueError, match='more than one tool call with call_id "call_a"'):
        await call_model(
            client,
            {"model": "m", "input": "continue", "tools": tools, "state": state, "approve_tool_calls": ["call_a"]},
        ).get_response()

    assert executed == []
    assert len(client.requests) == 1
