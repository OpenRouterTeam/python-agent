"""Approval-gate regressions ported from upstream PR #94."""

from __future__ import annotations

from typing import Any, List

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

    def needs_approval(params: DeployInput, _context: Any) -> bool:
        observed.append(params.dangerous)
        return params.dangerous

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
        observed.append(call.arguments.dangerous)
        return call.arguments.dangerous

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

    def needs_approval(params: DeployInput, _context: Any) -> bool:
        predicate_values.append(params.dangerous)
        return params.dangerous

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
