from __future__ import annotations

from typing import Any, Dict, List

from openrouter_agent import call_model, tool
from tests._fixtures import MemoryStateAccessor, QueuedClient, function_call_item, make_response, text_response

auto_tool = tool(
    name="auto_search",
    input_schema=dict,
    output_schema=dict,
    execute=lambda params, ctx: {"result": "found it"},
)

# No `execute` -- the client is responsible for running this tool.
manual_tool = tool(name="exec_command", input_schema=dict, execute=False)


async def test_all_manual_round_stops_loop_with_awaiting_client_tools() -> None:
    client = QueuedClient(
        [make_response("resp_manual", [function_call_item("call_manual_1", "exec_command", '{"command":"ls"}')])]
    )
    accessor = MemoryStateAccessor()

    result = call_model(
        client,
        {"model": "test-model", "input": "run ls", "tools": [manual_tool], "state": accessor},
    )

    pending = await result.get_pending_tool_calls()
    assert len(pending) == 1
    assert pending[0].id == "call_manual_1"
    assert pending[0].name == "exec_command"

    state = await result.get_state()
    assert state.status == "awaiting_client_tools"
    assert len(state.pending_tool_calls) == 1

    # No follow-up request -- the loop stopped after the unresolved manual call.
    assert len(client.requests) == 1
    assert accessor.stored is not None
    assert accessor.stored.status == "awaiting_client_tools"


async def test_mixed_auto_and_manual_round_persists_auto_output_and_pauses_manual() -> None:
    client = QueuedClient(
        [
            make_response(
                "resp_mixed",
                [
                    function_call_item("call_auto_1", "auto_search", '{"query":"docs"}'),
                    function_call_item("call_manual_1", "exec_command", '{"command":"ls"}'),
                ],
            )
        ]
    )
    accessor = MemoryStateAccessor()

    result = call_model(
        client,
        {
            "model": "test-model",
            "input": "do both",
            "tools": [auto_tool, manual_tool],
            "state": accessor,
        },
    )

    pending = await result.get_pending_tool_calls()
    assert len(pending) == 1
    assert pending[0].id == "call_manual_1"

    # No follow-up request: it would contain exec_command's function_call with
    # no matching function_call_output, which providers reject.
    assert len(client.requests) == 1

    state = await result.get_state()
    assert state.status == "awaiting_client_tools"

    auto_output = next(
        (m for m in state.messages if m.get("type") == "function_call_output" and m.get("callId") == "call_auto_1"),
        None,
    )
    assert auto_output is not None
    assert "found it" in auto_output["output"]
    manual_output = next(
        (m for m in state.messages if m.get("type") == "function_call_output" and m.get("callId") == "call_manual_1"),
        None,
    )
    assert manual_output is None


async def test_no_state_accessor_nothing_persisted_but_response_readable() -> None:
    client = QueuedClient(
        [make_response("resp_manual", [function_call_item("call_manual_1", "exec_command", '{"command":"ls"}')])]
    )

    result = call_model(client, {"model": "test-model", "input": "run ls", "tools": [manual_tool]})

    response = await result.get_response()
    assert response["id"] == "resp_manual"
    assert len(client.requests) == 1

    pending = await result.get_pending_tool_calls()
    assert pending == []


async def test_clears_pending_manual_calls_only_after_a_resume_succeeds() -> None:
    accessor = MemoryStateAccessor()
    client = QueuedClient(
        [
            make_response("resp_manual", [function_call_item("call_manual_1", "exec_command", '{"command":"pwd"}')]),
            text_response("resp_done", "done"),
        ]
    )

    await call_model(
        client, {"model": "test-model", "input": "run pwd", "tools": [manual_tool], "state": accessor}
    ).get_pending_tool_calls()

    await call_model(
        client,
        {
            "model": "test-model",
            "input": [{"type": "function_call_output", "callId": "call_manual_1", "output": '{"stdout":"/tmp"}'}],
            "tools": [manual_tool],
            "state": accessor,
        },
    ).get_response()

    assert accessor.stored is not None
    assert accessor.stored.status == "complete"
    assert not accessor.stored.pending_tool_calls


async def test_keeps_pending_manual_calls_when_a_resume_request_fails() -> None:
    accessor = MemoryStateAccessor()

    class FlakyResponses:
        def __init__(self) -> None:
            self.calls = 0
            self.requests: List[Dict[str, Any]] = []

        async def send_async(self, **kwargs: Any) -> Any:
            self.requests.append(kwargs)
            self.calls += 1
            if self.calls == 1:
                return make_response(
                    "resp_manual", [function_call_item("call_manual_1", "exec_command", '{"command":"pwd"}')]
                )
            raise RuntimeError("temporary failure")

    client = type("Client", (), {"beta": type("Beta", (), {"responses": FlakyResponses()})()})()

    await call_model(
        client, {"model": "test-model", "input": "run pwd", "tools": [manual_tool], "state": accessor}
    ).get_pending_tool_calls()

    try:
        await call_model(
            client,
            {
                "model": "test-model",
                "input": [{"type": "function_call_output", "callId": "call_manual_1", "output": '{"stdout":"/tmp"}'}],
                "tools": [manual_tool],
                "state": accessor,
            },
        ).get_response()
        raised = False
    except RuntimeError:
        raised = True

    assert raised
    assert accessor.stored is not None
    assert accessor.stored.status == "awaiting_client_tools"
    assert accessor.stored.pending_tool_calls is not None
    assert accessor.stored.pending_tool_calls[0].id == "call_manual_1"


async def test_hitl_pause_still_yields_awaiting_hitl_no_regression() -> None:
    client = QueuedClient([make_response("resp_hitl", [function_call_item("call_hitl_1", "approve", '{"amount":5}')])])
    accessor = MemoryStateAccessor()

    approve = tool(
        name="approve",
        input_schema=dict,
        output_schema=dict,
        on_tool_called=lambda params, ctx: None,
    )

    result = call_model(client, {"model": "test-model", "input": "approve 5", "tools": [approve], "state": accessor})

    await result.get_response()
    pending = await result.get_pending_tool_calls()
    assert len(pending) == 1
    assert pending[0].id == "call_hitl_1"

    state = await result.get_state()
    assert state.status == "awaiting_hitl"
    assert state.status != "awaiting_client_tools"
