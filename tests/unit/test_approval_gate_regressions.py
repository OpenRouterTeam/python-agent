"""Approval-gate regressions ported from upstream PR #94."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, List, Mapping

import pytest
from pydantic import BaseModel

from openrouter_agent import HookEntry, HookName, HooksManager, call_model, step_count_is, tool
from tests._fixtures import (
    MemoryStateAccessor,
    QueuedClient,
    function_call_item,
    make_response,
    text_response,
    tool_call_response,
)


async def test_allow_final_response_pauses_instead_of_executing_an_approval_gated_tool() -> None:
    state = MemoryStateAccessor()
    executed: List[Any] = []

    def execute(params: Any, _context: Any) -> Any:
        executed.append(params)
        return {"ok": True}

    gated = tool(
        name="deploy",
        input_schema=dict,
        output_schema=dict,
        execute=execute,
        require_approval=True,
    )
    client = QueuedClient([tool_call_response("r1", "deploy")])

    response = await call_model(
        client,
        {
            "model": "test-model",
            "input": "deploy",
            "tools": [gated],
            "state": state,
            "stop_when": step_count_is(0),
            "allow_final_response": True,
        },
    ).get_response()

    assert response["id"] == "r1"
    assert executed == []
    assert len(client.requests) == 1
    assert state.stored is not None
    assert state.stored.status == "awaiting_approval"
    assert [call.name for call in state.stored.pending_tool_calls or []] == ["deploy"]


async def test_allow_final_response_honors_permission_request_deny() -> None:
    executed: List[Any] = []
    permission_requests: List[Any] = []
    hooks = HooksManager()

    def deny(payload: Any, _context: Any) -> Any:
        permission_requests.append(payload)
        return {"decision": "deny", "reason": "policy denied"}

    hooks.on(HookName.PermissionRequest.value, HookEntry(handler=deny))

    def execute(params: Any, _context: Any) -> Any:
        executed.append(params)
        return {"ok": True}

    gated = tool(
        name="deploy",
        input_schema=dict,
        output_schema=dict,
        execute=execute,
        require_approval=True,
    )
    client = QueuedClient([tool_call_response("r1", "deploy"), text_response("r2", "not deployed")])

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": "deploy",
            "tools": [gated],
            "hooks": hooks,
            "stop_when": step_count_is(0),
            "allow_final_response": True,
        },
    ).get_text()

    assert text == "not deployed"
    assert executed == []
    assert len(permission_requests) == 1
    rejection = next(item for item in client.requests[1]["input"] if item.get("type") == "function_call_output")
    assert "policy denied" in rejection["output"]


async def test_allow_final_response_persists_ungated_results_before_approval_pause() -> None:
    state = MemoryStateAccessor()
    executions = {"lookup": 0, "deploy": 0}

    def execute_lookup(_params: Any, _context: Any) -> Any:
        executions["lookup"] += 1
        return {"value": 42}

    def execute_deploy(_params: Any, _context: Any) -> Any:
        executions["deploy"] += 1
        return {"ok": True}

    lookup = tool(name="lookup", input_schema=dict, output_schema=dict, execute=execute_lookup)
    deploy = tool(
        name="deploy",
        input_schema=dict,
        output_schema=dict,
        execute=execute_deploy,
        require_approval=True,
    )
    response = make_response(
        "r1",
        [function_call_item("call_lookup", "lookup"), function_call_item("call_deploy", "deploy")],
    )

    await call_model(
        QueuedClient([response]),
        {
            "model": "test-model",
            "input": "lookup then deploy",
            "tools": [lookup, deploy],
            "state": state,
            "stop_when": step_count_is(0),
        },
    ).get_response()

    assert executions == {"lookup": 1, "deploy": 0}
    assert state.stored is not None
    assert [call.name for call in state.stored.pending_tool_calls or []] == ["deploy"]
    assert [result.name for result in state.stored.unsent_tool_results or []] == ["lookup"]


async def test_tool_approval_predicate_receives_schema_defaults() -> None:
    class DeployInput(BaseModel):
        dangerous: bool = True

    state = MemoryStateAccessor()
    observed: List[bool] = []
    executed: List[Any] = []

    def needs_approval(params: Mapping[str, Any], _context: Any) -> bool:
        observed.append(params["dangerous"])
        return params["dangerous"]

    def execute(params: Any, _context: Any) -> Any:
        executed.append(params)
        return {"ok": True}

    gated = tool(
        name="deploy",
        input_schema=DeployInput,
        output_schema=dict,
        execute=execute,
        require_approval=needs_approval,
    )

    await call_model(
        QueuedClient([tool_call_response("r1", "deploy", arguments="{}")]),
        {"model": "test-model", "input": "deploy", "tools": [gated], "state": state},
    ).get_response()

    assert observed == [True]
    assert executed == []
    assert state.stored is not None and state.stored.status == "awaiting_approval"


async def test_call_level_approval_predicate_receives_schema_normalized_call() -> None:
    class DeployInput(BaseModel):
        dangerous: bool = True

    state = MemoryStateAccessor()
    observed: List[bool] = []

    def needs_approval(call: Any, _context: Any) -> bool:
        observed.append(call.arguments["dangerous"])
        return call.arguments["dangerous"]

    deploy = tool(
        name="deploy",
        input_schema=DeployInput,
        output_schema=dict,
        execute=lambda params, ctx: {"ok": True},
    )

    await call_model(
        QueuedClient([tool_call_response("r1", "deploy", arguments="{}")]),
        {
            "model": "test-model",
            "input": "deploy",
            "tools": [deploy],
            "state": state,
            "require_approval": needs_approval,
        },
    ).get_response()

    assert observed == [True]
    assert state.stored is not None and state.stored.status == "awaiting_approval"


async def test_pre_tool_use_mutation_is_approved_once_and_persists_across_resume() -> None:
    class DeployInput(BaseModel):
        dangerous: bool = False

    state = MemoryStateAccessor()
    hooks = HooksManager()
    pre_tool_use_calls: List[Any] = []
    predicate_values: List[bool] = []
    executed: List[DeployInput] = []

    def mutate(payload: Any, _context: Any) -> Any:
        pre_tool_use_calls.append(payload)
        return {"mutated_input": {"dangerous": True}}

    def needs_approval(params: Mapping[str, Any], _context: Any) -> bool:
        predicate_values.append(params["dangerous"])
        return params["dangerous"]

    def execute(params: DeployInput, _context: Any) -> Any:
        executed.append(params)
        return {"ok": True}

    hooks.on(HookName.PreToolUse.value, HookEntry(handler=mutate))
    gated = tool(
        name="deploy",
        input_schema=DeployInput,
        output_schema=dict,
        execute=execute,
        require_approval=needs_approval,
    )

    await call_model(
        QueuedClient([tool_call_response("r1", "deploy", arguments='{"dangerous": false}')]),
        {"model": "test-model", "input": "deploy", "tools": [gated], "state": state, "hooks": hooks},
    ).get_response()

    assert predicate_values == [True]
    assert len(pre_tool_use_calls) == 1
    assert state.stored is not None
    pending = state.stored.pending_tool_calls or []
    assert pending[0].arguments == {"dangerous": True}
    assert pending[0].pre_tool_use_applied is True

    resumed = call_model(
        QueuedClient([text_response("r2", "deployed")]),
        {
            "model": "test-model",
            "input": "continue",
            "tools": [gated],
            "state": state,
            "hooks": hooks,
            "approve_tool_calls": [pending[0].id],
        },
    )
    assert await resumed.get_text() == "deployed"
    assert len(pre_tool_use_calls) == 1
    assert len(executed) == 1 and executed[0].dangerous is True


async def test_invalid_tool_input_fails_closed_before_predicate() -> None:
    class DeployInput(BaseModel):
        dangerous: bool

    state = MemoryStateAccessor()
    predicate_calls: List[Any] = []

    def needs_approval(params: Any, _context: Any) -> bool:
        predicate_calls.append(params)
        return False

    gated = tool(
        name="deploy",
        input_schema=DeployInput,
        output_schema=dict,
        execute=lambda params, ctx: {"ok": True},
        require_approval=needs_approval,
    )

    await call_model(
        QueuedClient([tool_call_response("r1", "deploy", arguments="{}")]),
        {"model": "test-model", "input": "deploy", "tools": [gated], "state": state},
    ).get_response()

    assert predicate_calls == []
    assert state.stored is not None and state.stored.status == "awaiting_approval"


# ---------------------------------------------------------------------------
# Approval decisions are validated before anything runs, and a paused
# conversation resumed without decisions stays paused (upstream #131,
# typescript-sdk#517/#518).
# ---------------------------------------------------------------------------


class _GatedInput(BaseModel):
    v: str


def _function_call_outputs(request: Mapping[str, Any]) -> List[Any]:
    # Outgoing items use snake_case `call_id`, not upstream's `callId`:
    # ModelResult._send normalizes at the transport boundary
    # (model_result.py `_send`).
    return [
        item for item in request["input"] if isinstance(item, Mapping) and item.get("type") == "function_call_output"
    ]


async def _pause_on_gated_calls(call_ids: List[str]) -> Any:
    executed: List[Any] = []

    def execute(params: _GatedInput, _context: Any) -> Any:
        executed.append(params.v)
        return {"ok": True}

    gated = tool(name="gated", input_schema=_GatedInput, output_schema=dict, require_approval=True, execute=execute)
    state = MemoryStateAccessor()
    await call_model(
        QueuedClient(
            [make_response("resp_pause", [function_call_item(cid, "gated", '{"v":"x"}') for cid in call_ids])]
        ),
        {"model": "test-model", "input": "run the gated action", "tools": [gated], "state": state},
    ).get_response()
    assert state.stored is not None and state.stored.status == "awaiting_approval"
    assert executed == []
    return gated, state, executed


async def test_executes_an_approved_call_once_when_its_id_is_listed_twice() -> None:
    gated, state, executed = await _pause_on_gated_calls(["call_1"])
    client = QueuedClient([text_response("resp_done", "Done.")])

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": [],
            "tools": [gated],
            "state": state,
            "approve_tool_calls": ["call_1", "call_1"],
        },
    ).get_text()

    assert text == "Done."
    assert executed == ["x"]
    assert [item["call_id"] for item in _function_call_outputs(client.requests[0])] == ["call_1"]


async def test_records_a_rejection_once_when_its_id_is_listed_twice() -> None:
    gated, state, executed = await _pause_on_gated_calls(["call_1"])
    client = QueuedClient([text_response("resp_done", "Done.")])

    await call_model(
        client,
        {
            "model": "test-model",
            "input": [],
            "tools": [gated],
            "state": state,
            "reject_tool_calls": ["call_1", "call_1"],
        },
    ).get_response()

    assert executed == []
    assert [item["call_id"] for item in _function_call_outputs(client.requests[0])] == ["call_1"]


async def test_refuses_an_id_that_is_both_approved_and_rejected_before_anything_runs() -> None:
    gated, state, executed = await _pause_on_gated_calls(["call_1", "call_2"])
    before = state.stored
    saves_before = len(state.saved)
    client = QueuedClient([])

    with pytest.raises(ValueError, match="both approved and rejected: call_2"):
        call_model(
            client,
            {
                "model": "test-model",
                "input": [],
                "tools": [gated],
                "state": state,
                "approve_tool_calls": ["call_1", "call_2"],
                "reject_tool_calls": ["call_2"],
            },
        )

    assert executed == []
    assert client.requests == []
    assert state.stored is before
    assert len(state.saved) == saves_before


async def test_stays_awaiting_approval_and_sends_no_request_when_resumed_without_decisions() -> None:
    gated, state, executed = await _pause_on_gated_calls(["call_1"])
    before = state.stored
    saves_before = len(state.saved)
    client = QueuedClient([])

    result = call_model(
        client, {"model": "test-model", "input": "run the gated action", "tools": [gated], "state": state}
    )

    with pytest.raises(RuntimeError, match="awaiting approval.*call_1"):
        await result.get_response()
    with pytest.raises(RuntimeError, match="awaiting approval"):
        await result.get_text()
    assert await result.requires_approval() is True
    assert [call.id for call in await result.get_pending_tool_calls()] == ["call_1"]
    assert client.requests == []
    assert executed == []
    assert state.stored is before
    assert len(state.saved) == saves_before

    # The pause still resolves normally once a decision is supplied.
    await call_model(
        QueuedClient([text_response("resp_done", "Done.")]),
        {"model": "test-model", "input": [], "tools": [gated], "state": state, "approve_tool_calls": ["call_1"]},
    ).get_response()
    assert executed == ["x"]
    assert state.stored is not None and state.stored.status == "complete"


async def test_reports_the_pause_when_decisions_cover_only_some_pending_calls() -> None:
    gated, state, executed = await _pause_on_gated_calls(["call_1", "call_2"])
    client = QueuedClient([])

    result = call_model(
        client, {"model": "test-model", "input": [], "tools": [gated], "state": state, "approve_tool_calls": ["call_1"]}
    )

    with pytest.raises(RuntimeError, match="awaiting approval.*call_2"):
        await result.get_response()
    assert executed == ["x"]
    assert client.requests == []
    assert state.stored is not None and state.stored.status == "awaiting_approval"
    assert [call.id for call in state.stored.pending_tool_calls or []] == ["call_2"]
    # The executed call's output is kept for the eventual resume request.
    assert [r.call_id for r in state.stored.unsent_tool_results or []] == ["call_1"]


class _Answer(BaseModel):
    answer: str


async def _pause_on_hitl_call() -> Any:
    called: List[Any] = []
    received: List[Any] = []

    def on_tool_called(params: _GatedInput, _context: Any) -> Any:
        called.append(params.v)
        return None

    def on_response_received(raw: Any, _context: Any) -> Any:
        received.append(raw)
        return {"answer": str(raw["answer"]).upper()}

    review = tool(
        name="gated",
        input_schema=_GatedInput,
        output_schema=_Answer,
        on_tool_called=on_tool_called,
        on_response_received=on_response_received,
    )
    state = MemoryStateAccessor()
    await call_model(
        QueuedClient([tool_call_response("resp_hitl", "gated", call_id="call_hitl", arguments='{"v":"x"}')]),
        {"model": "test-model", "input": "review", "tools": [review], "state": state},
    ).get_response()
    assert state.stored is not None and state.stored.status == "awaiting_hitl"
    assert called == ["x"]
    return review, state, called, received


def _caller_output(call_id: str, output: Any) -> dict:
    return {"type": "function_call_output", "id": f"output_{call_id}", "callId": call_id, "output": json.dumps(output)}


async def test_stays_awaiting_hitl_and_sends_no_request_when_resumed_without_decisions() -> None:
    review, state, called, received = await _pause_on_hitl_call()
    before = state.stored

    for input_value in ["a new user message", [_caller_output("call_other", {"answer": "yes"})]]:
        client = QueuedClient([])
        result = call_model(client, {"model": "test-model", "input": input_value, "tools": [review], "state": state})
        with pytest.raises(
            RuntimeError,
            match="awaiting HITL input.*call_hitl.*function_call_output.*input supplied with this call was not sent",
        ):
            await result.get_response()
        assert client.requests == []

    assert called == ["x"]
    assert received == []
    assert state.stored is before


async def test_resumes_awaiting_hitl_when_input_carries_a_function_call_output_for_the_paused_call() -> None:
    review, state, called, received = await _pause_on_hitl_call()
    client = QueuedClient([text_response("resp_done", "Done.")])

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": [_caller_output("call_hitl", {"answer": "yes"})],
            "tools": [review],
            "state": state,
        },
    ).get_text()

    assert text == "Done."
    assert received == [{"answer": "yes"}]
    assert called == ["x"]  # the paused call was not re-run
    assert len(client.requests) == 1
    outputs = _function_call_outputs(client.requests[0])
    assert [(o["call_id"], o["output"]) for o in outputs] == [("call_hitl", '{"answer":"YES"}')]
    assert state.stored is not None and state.stored.status == "complete"


async def test_resolves_a_function_valued_input_once_when_it_resumes_awaiting_hitl() -> None:
    review, state, _called, received = await _pause_on_hitl_call()
    calls: List[Any] = []

    def input_fn(_ctx: Any) -> Any:
        calls.append(_ctx)
        return [_caller_output("call_hitl", {"answer": "yes"})]

    text = await call_model(
        QueuedClient([text_response("resp_done", "Done.")]),
        {"model": "test-model", "input": input_fn, "tools": [review], "state": state},
    ).get_text()

    assert text == "Done."
    assert len(calls) == 1
    assert len(received) == 1


async def test_stays_awaiting_approval_when_input_carries_a_function_call_output_for_the_gated_call() -> None:
    gated, state, executed = await _pause_on_gated_calls(["call_1"])
    before = state.stored
    client = QueuedClient([])

    result = call_model(
        client,
        {"model": "test-model", "input": [_caller_output("call_1", {"ok": True})], "tools": [gated], "state": state},
    )

    with pytest.raises(RuntimeError, match="awaiting approval.*call_1.*input supplied with this call was not sent"):
        await result.get_response()
    assert client.requests == []
    assert executed == []
    assert state.stored is before


async def test_rejects_every_response_and_stream_getter_with_the_awaiting_error_on_a_paused_resume() -> None:
    gated, state, _executed = await _pause_on_gated_calls(["call_1"])
    client = QueuedClient([])
    result = call_model(client, {"model": "test-model", "input": [], "tools": [gated], "state": state})

    async def first(stream: Any) -> Any:
        return await stream.__anext__()

    consumers = {
        "get_response": lambda: result.get_response(),
        "get_text": lambda: result.get_text(),
        "get_tool_calls": lambda: result.get_tool_calls(),
        "get_text_stream": lambda: first(result.get_text_stream()),
        "get_reasoning_stream": lambda: first(result.get_reasoning_stream()),
        "get_full_responses_stream": lambda: first(result.get_full_responses_stream()),
        "get_items_stream": lambda: first(result.get_items_stream()),
        "get_new_messages_stream": lambda: first(result.get_new_messages_stream()),
        "get_tool_stream": lambda: first(result.get_tool_stream()),
        "get_tool_calls_stream": lambda: first(result.get_tool_calls_stream()),
    }
    for name, consume in consumers.items():
        with pytest.raises(RuntimeError, match="awaiting approval.*call_1"):
            await consume()
        assert client.requests == [], name

    assert await result.requires_approval() is True
    assert [call.id for call in await result.get_pending_tool_calls()] == ["call_1"]
    assert (await result.get_state()).status == "awaiting_approval"


async def test_continues_a_paused_state_whose_pending_tool_calls_is_empty() -> None:
    cases: List[Any] = [("awaiting_approval", []), ("awaiting_approval", None), ("awaiting_hitl", [])]
    for status, pending in cases:
        gated, state, executed = await _pause_on_gated_calls(["call_1"])
        assert state.stored is not None
        state.stored = dataclasses.replace(state.stored, status=status, pending_tool_calls=pending)
        client = QueuedClient([text_response("resp_done", "Done.")])

        text = await call_model(
            client, {"model": "test-model", "input": [], "tools": [gated], "state": state}
        ).get_text()

        assert text == "Done."
        assert len(client.requests) == 1
        assert executed == []
        assert state.stored.status == "complete"


async def test_caps_the_pending_ids_listed_in_the_awaiting_error_and_omits_the_input_note_without_input() -> None:
    gated, state, _executed = await _pause_on_gated_calls([f"call_{i + 1}" for i in range(25)])
    result = call_model(QueuedClient([]), {"model": "test-model", "input": [], "tools": [gated], "state": state})

    with pytest.raises(RuntimeError) as caught:
        await result.get_response()

    message = str(caught.value)
    assert "call_1, call_2," in message
    assert "call_10, ... (15 more)" in message
    assert "call_11" not in message
    assert "was not sent" not in message


async def test_a_duplicated_id_cannot_bypass_re_approval_after_a_pre_tool_use_mutation() -> None:
    executed: List[Any] = []

    def execute(params: _GatedInput, _context: Any) -> Any:
        executed.append(params.v)
        return {"ok": True}

    gated = tool(
        name="gated",
        input_schema=_GatedInput,
        output_schema=dict,
        require_approval=lambda _params, _ctx: True,
        execute=execute,
    )
    state = MemoryStateAccessor()
    await call_model(
        QueuedClient([tool_call_response("resp_pause", "gated", call_id="call_1", arguments='{"v":"reviewed"}')]),
        {"model": "test-model", "input": "go", "tools": [gated], "state": state},
    ).get_response()
    assert state.stored is not None and state.stored.status == "awaiting_approval"

    hooks = HooksManager()
    hooks.on(HookName.PreToolUse.value, HookEntry(handler=lambda _p, _c: {"mutated_input": {"v": "mutated"}}))
    client = QueuedClient([])
    result = call_model(
        client,
        {
            "model": "test-model",
            "input": [],
            "tools": [gated],
            "hooks": hooks,
            "state": state,
            "approve_tool_calls": ["call_1", "call_1"],
        },
    )

    # The human approved {"v": "reviewed"}; the mutated arguments need their
    # own approval, so nothing executes and the call is pending again.
    with pytest.raises(RuntimeError, match="awaiting approval.*call_1"):
        await result.get_response()
    assert executed == []
    assert client.requests == []
    assert state.stored.status == "awaiting_approval"
    pending = state.stored.pending_tool_calls or []
    assert [(c.id, c.name, c.arguments, c.pre_tool_use_applied) for c in pending] == [
        ("call_1", "gated", {"v": "mutated"}, True)
    ]


def _mutating_hooks(mutated: Any, failures: List[Any]) -> HooksManager:
    hooks = HooksManager()
    hooks.on(HookName.PreToolUse.value, HookEntry(handler=lambda _p, _c: {"mutated_input": mutated}))
    hooks.on(HookName.PostToolUseFailure.value, HookEntry(handler=lambda payload, _c: failures.append(payload)))
    return hooks


async def test_blanket_require_approval_runs_the_approved_call_after_a_pre_tool_use_mutation() -> None:
    # Only input-dependent checks are re-run on mutated arguments: a blanket
    # `require_approval=True` was satisfied by the human's approval.
    gated, state, executed = await _pause_on_gated_calls(["call_1"])
    client = QueuedClient([text_response("resp_done", "Done.")])

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": [],
            "tools": [gated],
            "hooks": _mutating_hooks({"v": "mutated"}, []),
            "state": state,
            "approve_tool_calls": ["call_1"],
        },
    ).get_text()

    assert text == "Done."
    assert executed == ["mutated"]
    assert len(client.requests) == 1
    assert state.stored is not None and state.stored.status == "complete"


async def test_a_mutation_that_fails_the_input_schema_is_blocked_on_resume() -> None:
    gated, state, executed = await _pause_on_gated_calls(["call_1"])
    failures: List[Any] = []
    client = QueuedClient([text_response("resp_done", "Done.")])

    await call_model(
        client,
        {
            "model": "test-model",
            "input": [],
            "tools": [gated],
            "hooks": _mutating_hooks({"v": 123}, failures),
            "state": state,
            "approve_tool_calls": ["call_1"],
        },
    ).get_text()

    assert executed == []
    outputs = _function_call_outputs(client.requests[0])
    assert [(o["call_id"], json.loads(o["output"])) for o in outputs] == [
        ("call_1", {"error": 'PreToolUse produced invalid input for "gated"'})
    ]
    assert [f["error"] for f in failures] == ['PreToolUse produced invalid input for "gated"']


async def test_rejecting_a_held_call_emits_no_post_tool_use_failure() -> None:
    # Upstream emits nothing on an ordinary reject of a call held at the
    # initial gate (PreToolUse never ran for it there).
    gated, state, executed = await _pause_on_gated_calls(["call_1"])
    failures: List[Any] = []
    hooks = HooksManager()
    hooks.on(HookName.PostToolUseFailure.value, HookEntry(handler=lambda payload, _c: failures.append(payload)))
    client = QueuedClient([text_response("resp_done", "Done.")])

    await call_model(
        client,
        {
            "model": "test-model",
            "input": [],
            "tools": [gated],
            "hooks": hooks,
            "state": state,
            "reject_tool_calls": ["call_1"],
        },
    ).get_text()

    assert executed == []
    assert failures == []
    outputs = _function_call_outputs(client.requests[0])
    assert [json.loads(o["output"]) for o in outputs] == [{"error": "Rejected by user"}]


async def test_a_permission_request_deny_on_mutated_arguments_blocks_and_reports_failure() -> None:
    executed: List[Any] = []

    def execute(params: _GatedInput, _context: Any) -> Any:
        executed.append(params.v)
        return {"ok": True}

    gated = tool(
        name="gated",
        input_schema=_GatedInput,
        output_schema=dict,
        require_approval=lambda params, _ctx: params["v"] != "safe",
        execute=execute,
    )
    state = MemoryStateAccessor()
    await call_model(
        QueuedClient([tool_call_response("resp_pause", "gated", call_id="call_1", arguments='{"v":"x"}')]),
        {"model": "test-model", "input": "go", "tools": [gated], "state": state},
    ).get_response()
    assert state.stored is not None and state.stored.status == "awaiting_approval"

    failures: List[Any] = []
    hooks = _mutating_hooks({"v": "dangerous"}, failures)
    hooks.on(HookName.PermissionRequest.value, HookEntry(handler=lambda _p, _c: {"decision": "deny", "reason": "no"}))
    client = QueuedClient([text_response("resp_done", "Done.")])

    await call_model(
        client,
        {
            "model": "test-model",
            "input": [],
            "tools": [gated],
            "hooks": hooks,
            "state": state,
            "approve_tool_calls": ["call_1"],
        },
    ).get_text()

    assert executed == []
    assert [(f["tool_name"], f["error"]) for f in failures] == [("gated", "no")]
    outputs = _function_call_outputs(client.requests[0])
    assert [json.loads(o["output"]) for o in outputs] == [{"error": "no"}]
