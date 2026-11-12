"""Port of upstream `tests/unit/approval-gate-regressions.test.ts`.

Pins the approval-gate regression invariants:

- approval predicates see the same schema-parsed arguments `execute` gets,
  and schema-invalid arguments fail closed;
- every engine execute path validates before running the tool body;
- the `allow_final_response` path (stop_when halting mid-tool-call) runs the
  approval gate, and each call occurrence is gated at most once per run;
- approve/reject decisions are de-duplicated and validated before anything
  runs, and a paused conversation resumed without decisions stays paused.

Upstream drives `ModelResult` with a mocked `betaResponsesSend`; this port
drives `call_model` with the shared `QueuedClient`. Upstream cases that depend
on engine features this port does not implement (unified `run` tools) are kept
as named `skip`s so the gap stays visible.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

import pytest
from pydantic import BaseModel, field_validator

from openrouter_agent import HookEntry, HookName, HooksManager, call_model, step_count_is, tool
from openrouter_agent.conversation_state import tool_requires_approval
from openrouter_agent.tool_executor import execute_generator_tool, execute_hitl_tool, execute_regular_tool
from openrouter_agent.tool_types import ParsedToolCall, Tool
from tests._fixtures import MemoryStateAccessor, QueuedClient, function_call_item, make_response, text_response

CONTEXT: Dict[str, Any] = {"number_of_turns": 1}


class Recorder:
    """Counts invocations and records arguments; returns a fixed value.

    A class (not `lambda: calls.append(...) or value`) so the return value is
    explicit and mypy-clean -- see tests/unit/test_tool_execution_once.py.
    """

    def __init__(self, value: Any = None) -> None:
        self.value = value
        self.calls: List[Tuple[Any, ...]] = []

    def __call__(self, *args: Any) -> Any:
        self.calls.append(args)
        return self.value

    @property
    def count(self) -> int:
        return len(self.calls)


def _dumped_first_args(recorder: Recorder) -> List[Any]:
    return [call[0].model_dump() if isinstance(call[0], BaseModel) else call[0] for call in recorder.calls]


def _function_call_outputs(client: QueuedClient, index: int) -> List[Dict[str, Any]]:
    sent = client.requests[index].get("input") or []
    return [item for item in sent if isinstance(item, dict) and item.get("type") == "function_call_output"]


def _deep_nested(levels: int = 200) -> Dict[str, Any]:
    deep: Dict[str, Any] = {}
    for _ in range(levels):
        deep = {"child": deep}
    return deep


def _done(response_id: str, text: str = "Done.") -> Dict[str, Any]:
    return text_response(response_id, text)


# ---------------------------------------------------------------------------
# Bug 1: the approval predicate must see the SAME arguments `execute` gets.
# ---------------------------------------------------------------------------


class DangerousDefaultArgs(BaseModel):
    dangerous: bool = True


class AmountArgs(BaseModel):
    amount: int


class DestructiveDefaultArgs(BaseModel):
    destructive: bool = True


class TargetArgs(BaseModel):
    target: str


class OkOutput(BaseModel):
    ok: bool


async def test_applies_input_schema_defaults_before_function_require_approval() -> None:
    predicate = Recorder()

    def require(params: Any, ctx: Any) -> bool:
        predicate(params, ctx)
        return params["dangerous"] is True

    conditional = tool(
        name="conditional_action",
        input_schema=DangerousDefaultArgs,
        require_approval=require,
        execute=lambda params, ctx: {},
    )

    # The model emitted `{}`; the schema default makes `dangerous` True by the
    # time execute runs, so the predicate must see it True too.
    requires = await tool_requires_approval(
        ParsedToolCall(id="1", name="conditional_action", arguments={}), [conditional], CONTEXT
    )

    # Pydantic models are model_dump()'d for the predicate (conversation_state.py
    # _parse_tool_arguments): the predicate contract is a mapping, as upstream.
    assert [call[0] for call in predicate.calls] == [{"dangerous": True}]
    assert requires is True


async def test_applies_input_schema_coercions_before_function_require_approval() -> None:
    predicate = Recorder()

    def require(params: Any, ctx: Any) -> bool:
        predicate(params, ctx)
        return bool(params["amount"] > 100)

    transfer = tool(
        name="transfer",
        input_schema=AmountArgs,
        require_approval=require,
        execute=lambda params, ctx: {},
    )

    # String amount is coerced (Pydantic lax mode stands in for z.coerce), so
    # the comparison is numeric, not lexicographic.
    requires = await tool_requires_approval(
        ParsedToolCall(id="2", name="transfer", arguments={"amount": "500"}), [transfer], CONTEXT
    )

    assert [call[0] for call in predicate.calls] == [{"amount": 500}]
    assert requires is True


async def test_applies_schema_defaults_for_call_level_checks_without_mutating_call() -> None:
    defaulted = tool(
        name="call_level_defaulted_action",
        input_schema=DestructiveDefaultArgs,
        execute=lambda params, ctx: {},
    )
    tool_call = ParsedToolCall(id="call-level-default", name="call_level_defaulted_action", arguments={})
    call_level_check = Recorder(True)

    requires = await tool_requires_approval(tool_call, [defaulted], CONTEXT, call_level_check)

    assert call_level_check.calls == [(dataclasses.replace(tool_call, arguments={"destructive": True}), CONTEXT)]
    assert tool_call.arguments == {}
    assert requires is True


@pytest.mark.parametrize("decision", [False, True])
async def test_honors_call_level_check_when_arguments_fail_schema_validation(decision: bool) -> None:
    strict = tool(name="call_level_invalid_action", input_schema=TargetArgs, execute=lambda params, ctx: {})
    invalid_call = ParsedToolCall(id=f"call-level-invalid-{decision}", name="call_level_invalid_action", arguments={})
    call_level_check = Recorder(decision)

    assert await tool_requires_approval(invalid_call, [strict], CONTEXT, call_level_check) is decision
    # Parsing failed, so the call-level check receives the raw call.
    assert call_level_check.calls == [(invalid_call, CONTEXT)]


async def test_parses_original_wire_arguments_once_for_predicate_and_once_for_execution() -> None:
    transform_calls = {"n": 0}

    class TransformedArgs(BaseModel):
        value: str

        @field_validator("value")
        @classmethod
        def _suffix(cls, value: str) -> str:
            transform_calls["n"] += 1
            return f"{value}:{transform_calls['n']}"

    predicate = Recorder(False)
    execute = Recorder({})
    transformed = tool(
        name="transformed_action",
        input_schema=TransformedArgs,
        require_approval=predicate,
        execute=execute,
    )
    tool_call = ParsedToolCall(id="non-idempotent-transform", name="transformed_action", arguments={"value": "wire"})

    assert await tool_requires_approval(tool_call, [transformed], CONTEXT) is False
    await execute_regular_tool(transformed, tool_call, CONTEXT)

    assert predicate.calls == [({"value": "wire:1"}, CONTEXT)]
    assert _dumped_first_args(execute) == [{"value": "wire:2"}]
    assert transform_calls["n"] == 2
    assert tool_call.arguments == {"value": "wire"}


async def test_fails_closed_when_arguments_fail_schema_validation() -> None:
    predicate = Recorder(False)
    strict = tool(name="strict_action", input_schema=TargetArgs, require_approval=predicate, execute=lambda p, c: {})

    requires = await tool_requires_approval(
        ParsedToolCall(id="3", name="strict_action", arguments={}), [strict], CONTEXT
    )

    assert requires is True
    assert predicate.count == 0


async def test_still_fails_closed_when_schema_parses_to_non_object_value() -> None:
    predicate = Recorder(False)
    weird = tool(name="weird_action", input_schema=TargetArgs, require_approval=predicate, execute=lambda p, c: {})
    # Mirrors upstream swapping in a schema whose parse succeeds with a
    # non-record payload: a callable schema is honored by validate_schema.
    weird["function"]["input_schema"] = lambda value: ["not", "a", "record"]

    requires = await tool_requires_approval(
        ParsedToolCall(id="4", name="weird_action", arguments={"target": "x"}), [weird], CONTEXT
    )

    assert requires is True
    assert predicate.count == 0


async def test_still_fails_closed_on_schema_invalid_arguments_for_manual_tools() -> None:
    predicate = Recorder(False)
    # A manual tool: no execute / on_tool_called. The host runs it as-is, so a
    # malformed call must not be waved past the gate.
    manual_tool: Tool = {
        "type": "function",
        "function": {"name": "manual_action", "input_schema": TargetArgs, "require_approval": predicate},
    }

    requires = await tool_requires_approval(
        ParsedToolCall(id="5", name="manual_action", arguments={}), [manual_tool], CONTEXT
    )

    assert requires is True
    assert predicate.count == 0


# ---------------------------------------------------------------------------
# Approval vs. PreToolUse-mutated arguments.
# ---------------------------------------------------------------------------


class GuardedArgs(BaseModel):
    dangerous: bool


async def test_does_not_let_a_hook_rewrite_safe_arguments_into_unapproved_dangerous_call() -> None:
    predicate = Recorder()

    def require(params: Any, ctx: Any) -> bool:
        predicate(params, ctx)
        return bool(params["dangerous"])

    execute = Recorder({"ok": True})
    guarded = tool(
        name="guarded_action",
        input_schema=GuardedArgs,
        output_schema=OkOutput,
        require_approval=require,
        execute=execute,
    )
    pre_tool_use = Recorder({"mutated_input": {"dangerous": True}})
    hooks = HooksManager()
    hooks.on(HookName.PreToolUse.value, HookEntry(handler=pre_tool_use))
    accessor = MemoryStateAccessor()
    client = QueuedClient(
        [make_response("resp_mutation", [function_call_item("call_guarded", "guarded_action", '{"dangerous":false}')])]
    )

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "run the guarded action",
            "tools": [guarded],
            "hooks": hooks,
            "state": accessor,
        },
    ).get_response()

    # Initial gate on the wire arguments, then the mutated-input re-gate.
    # Upstream asserts `toHaveBeenCalledWith(mutatedInput, ...)`; the full
    # ordered call list pins that the re-gate is the second (and last) check.
    assert predicate.calls == [
        ({"dangerous": False}, {"number_of_turns": 0}),
        ({"dangerous": True}, {"number_of_turns": 0}),
    ]
    assert pre_tool_use.count == 1
    assert execute.count == 0
    assert len(client.requests) == 1
    assert accessor.stored is not None
    assert accessor.stored.status == "awaiting_approval"
    assert accessor.stored.pending_tool_calls == [
        ParsedToolCall(
            id="call_guarded", name="guarded_action", arguments={"dangerous": True}, pre_tool_use_applied=True
        )
    ]


async def test_does_not_abort_approval_bookkeeping_for_circular_hook_mutations() -> None:
    circular: Dict[str, Any] = {}
    circular["self"] = circular

    class SelfArgs(BaseModel):
        # Pydantic v2's __init__ takes `self` positional-only, so a field
        # named `self` mirrors upstream's `z.object({ self: z.unknown() })`.
        self: Any = None

    predicate = Recorder(False)
    execute = Recorder({"ok": True})
    guarded = tool(
        name="circular_mutation",
        input_schema=SelfArgs,
        output_schema=OkOutput,
        require_approval=predicate,
        execute=execute,
    )
    hooks = HooksManager()
    hooks.on(HookName.PreToolUse.value, HookEntry(handler=lambda p, c: {"mutated_input": circular}))
    client = QueuedClient(
        [
            make_response(
                "resp_circular_mutation", [function_call_item("call_circular_mutation", "circular_mutation", "{}")]
            ),
            # Upstream's `mockResolvedValue` repeats the empty response; the
            # port's empty-final tolerance retries once, so queue two.
            make_response("resp_circular_done", []),
            make_response("resp_circular_done_retry", []),
        ]
    )

    response = await call_model(
        client, {"model": "test-model", "input": "run", "tools": [guarded], "hooks": hooks}
    ).get_response()

    assert response["id"] == "resp_circular_done_retry"
    # The mutated-input re-gate received the circular input (dict equality
    # short-circuits on the self-reference identity).
    assert predicate.calls[-1] == (circular, {"number_of_turns": 0})
    assert predicate.count == 2
    assert execute.count == 0


async def test_lets_normal_validation_handle_deeply_nested_model_input() -> None:
    execute = Recorder({"ok": True})
    strict = tool(
        name="deep_invalid",
        input_schema=TargetArgs,
        output_schema=OkOutput,
        require_approval=lambda params, ctx: False,
        execute=execute,
    )
    accessor = MemoryStateAccessor()
    client = QueuedClient(
        [
            make_response(
                "resp_deep_invalid",
                [function_call_item("call_deep_invalid", "deep_invalid", json.dumps(_deep_nested()))],
            )
        ]
    )

    response = await call_model(
        client, {"model": "test-model", "input": "run", "tools": [strict], "state": accessor}
    ).get_response()

    assert response["id"] == "resp_deep_invalid"
    assert accessor.stored is not None
    assert accessor.stored.status == "awaiting_approval"
    assert execute.count == 0
    assert len(client.requests) == 1


async def test_gates_identical_post_hook_calls_independently_across_responses() -> None:
    predicate = Recorder()

    def require(params: Any, ctx: Any) -> bool:
        predicate(params, ctx)
        return bool(params["dangerous"])

    execute = Recorder({"ok": True})
    guarded = tool(
        name="guarded_action",
        input_schema=GuardedArgs,
        output_schema=OkOutput,
        require_approval=require,
        execute=execute,
    )
    hooks = HooksManager()
    hooks.on(HookName.PreToolUse.value, HookEntry(handler=lambda p, c: {"mutated_input": {"dangerous": True}}))
    permission_request = Recorder({"decision": "allow"})
    hooks.on(HookName.PermissionRequest.value, HookEntry(handler=permission_request))
    # Two distinct response objects with identical ids and calls.
    client = QueuedClient(
        [
            make_response(
                "resp_mutation_reused", [function_call_item("call_reused", "guarded_action", '{"dangerous":false}')]
            ),
            make_response(
                "resp_mutation_reused", [function_call_item("call_reused", "guarded_action", '{"dangerous":false}')]
            ),
            _done("resp_done"),
        ]
    )

    await call_model(
        client, {"model": "test-model", "input": "run twice", "tools": [guarded], "hooks": hooks}
    ).get_response()

    assert [call[0] for call in predicate.calls if call[0]["dangerous"] is True] == [
        {"dangerous": True},
        {"dangerous": True},
    ]
    assert permission_request.count == 2
    assert [call[0]["tool_input"] for call in permission_request.calls] == [{"dangerous": True}, {"dangerous": True}]
    assert execute.count == 2
    assert len(client.requests) == 3


async def test_keeps_hook_mutated_arguments_raw_across_approval_and_execution_transforms() -> None:
    class NormalizedArgs(BaseModel):
        value: str

        @field_validator("value")
        @classmethod
        def _normalize(cls, value: str) -> str:
            return f"{value}:normalized"

    class ValueOutput(BaseModel):
        value: str

    predicate = Recorder()

    def require(params: Any, ctx: Any) -> bool:
        predicate(params, ctx)
        return bool(params["value"] == "hook:normalized")

    execute = Recorder()

    def run(params: NormalizedArgs, ctx: Any) -> Dict[str, Any]:
        execute(params, ctx)
        return {"value": params.value}

    guarded = tool(
        name="transformed_mutation",
        input_schema=NormalizedArgs,
        output_schema=ValueOutput,
        require_approval=require,
        execute=run,
    )
    hooks = HooksManager()
    hooks.on(HookName.PreToolUse.value, HookEntry(handler=lambda p, c: {"mutated_input": {"value": "hook"}}))
    accessor = MemoryStateAccessor()
    client = QueuedClient(
        [
            make_response(
                "resp_transformed_mutation",
                [function_call_item("call_transformed_mutation", "transformed_mutation", '{"value":"wire"}')],
            )
        ]
    )

    await call_model(
        client, {"model": "test-model", "input": "run", "tools": [guarded], "hooks": hooks, "state": accessor}
    ).get_response()

    # The predicate judges the schema-transformed post-hook input...
    assert predicate.calls[-1] == ({"value": "hook:normalized"}, {"number_of_turns": 0})
    assert execute.count == 0
    # ...but the pending call keeps the RAW post-hook input, so execution
    # transforms it exactly once.
    assert accessor.stored is not None
    assert accessor.stored.pending_tool_calls == [
        ParsedToolCall(
            id="call_transformed_mutation",
            name="transformed_mutation",
            arguments={"value": "hook"},
            pre_tool_use_applied=True,
        )
    ]


async def test_executes_a_type_changing_hook_mutation_from_raw_input_exactly_once() -> None:
    class LengthArgs(BaseModel):
        value: int

        @field_validator("value", mode="before")
        @classmethod
        def _length(cls, value: Any) -> int:
            if not isinstance(value, str):
                raise ValueError("value must be a string")
            return len(value)

    class LengthOutput(BaseModel):
        value: int

    execute = Recorder()

    def run(params: LengthArgs, ctx: Any) -> Dict[str, Any]:
        execute(params, ctx)
        return {"value": params.value}

    transformed = tool(
        name="type_changing_mutation",
        input_schema=LengthArgs,
        output_schema=LengthOutput,
        require_approval=lambda params, ctx: False,
        execute=run,
    )
    hooks = HooksManager()
    hooks.on(HookName.PreToolUse.value, HookEntry(handler=lambda p, c: {"mutated_input": {"value": "hook"}}))
    client = QueuedClient(
        [
            make_response(
                "resp_type_changing_mutation",
                [function_call_item("call_type_changing_mutation", "type_changing_mutation", '{"value":"wire"}')],
            ),
            # Upstream's `mockResolvedValue` repeats the empty response; the
            # port's empty-final tolerance retries once, so queue two.
            make_response("resp_type_changing_done", []),
            make_response("resp_type_changing_done_retry", []),
        ]
    )

    await call_model(
        client, {"model": "test-model", "input": "run", "tools": [transformed], "hooks": hooks}
    ).get_response()

    assert execute.count == 1
    # Executed from the hook's raw `{"value": "hook"}` -> len("hook") == 4.
    assert _dumped_first_args(execute) == [{"value": 4}]


async def test_gates_schema_invalid_model_arguments_before_running_pre_tool_use() -> None:
    class DangerousArgs(BaseModel):
        dangerous: bool

    pre_tool_use = Recorder({"mutated_input": {"dangerous": True}})
    execute = Recorder({"ok": True})
    guarded = tool(
        name="guarded_action",
        input_schema=DangerousArgs,
        require_approval=lambda params, ctx: params["dangerous"],
        execute=execute,
    )
    hooks = HooksManager()
    hooks.on(HookName.PreToolUse.value, HookEntry(handler=pre_tool_use))
    accessor = MemoryStateAccessor()
    client = QueuedClient(
        [make_response("resp_invalid_original", [function_call_item("call_invalid_original", "guarded_action", "{}")])]
    )

    await call_model(
        client, {"model": "test-model", "input": "run", "tools": [guarded], "hooks": hooks, "state": accessor}
    ).get_response()

    assert pre_tool_use.count == 0
    assert execute.count == 0
    assert accessor.stored is not None
    assert accessor.stored.status == "awaiting_approval"
    assert len(client.requests) == 1


async def test_persists_unconditional_approvals_and_does_not_reapply_hook_on_resume() -> None:
    class ValueArgs(BaseModel):
        value: str

    class ValueOutput(BaseModel):
        value: str

    pre_tool_use = Recorder()

    def prepare(payload: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
        pre_tool_use(payload, ctx)
        return {"mutated_input": {"value": f"{payload['tool_input']['value']}-prepared"}}

    permission_request = Recorder({"decision": "ask_user"})
    execute = Recorder()

    def run(params: ValueArgs, ctx: Any) -> Dict[str, Any]:
        execute(params, ctx)
        return {"value": params.value}

    guarded = tool(
        name="always_guarded",
        input_schema=ValueArgs,
        output_schema=ValueOutput,
        require_approval=True,
        execute=run,
    )
    hooks = HooksManager()
    hooks.on(HookName.PreToolUse.value, HookEntry(handler=prepare))
    hooks.on(HookName.PermissionRequest.value, HookEntry(handler=permission_request))
    accessor = MemoryStateAccessor()

    def request(state: MemoryStateAccessor, **extra: Any) -> Dict[str, Any]:
        return {
            "model": "test-model",
            "input": "run the guarded action",
            "tools": [guarded],
            "hooks": hooks,
            "state": state,
            **extra,
        }

    first = QueuedClient(
        [
            make_response(
                "resp_unconditional",
                [function_call_item("call_unconditional", "always_guarded", '{"value":"original"}')],
            )
        ]
    )
    await call_model(first, request(accessor)).get_response()

    assert permission_request.count == 1
    assert permission_request.calls[0][0]["tool_input"] == {"value": "original"}
    assert accessor.stored is not None
    assert accessor.stored.pending_tool_calls == [
        ParsedToolCall(id="call_unconditional", name="always_guarded", arguments={"value": "original"})
    ]
    # Upstream deletes `preToolUseApplied` to build a "legacy" state; the
    # port's equivalent is the default (False) marker.
    legacy_state = copy.deepcopy(accessor.stored)
    legacy_state = dataclasses.replace(
        legacy_state,
        pending_tool_calls=[
            dataclasses.replace(call, pre_tool_use_applied=False) for call in legacy_state.pending_tool_calls or []
        ],
    )

    resume = QueuedClient([_done("resp_after_approval")])
    await call_model(resume, request(accessor, approve_tool_calls=["call_unconditional"])).get_response()

    assert pre_tool_use.count == 1
    assert permission_request.count == 1
    assert execute.count == 1
    assert _dumped_first_args(execute) == [{"value": "original-prepared"}]
    assert len(resume.requests) == 1

    legacy_accessor = MemoryStateAccessor()
    legacy_accessor.stored = legacy_state
    legacy_resume = QueuedClient([_done("resp_after_legacy_approval")])
    await call_model(legacy_resume, request(legacy_accessor, approve_tool_calls=["call_unconditional"])).get_response()

    assert pre_tool_use.count == 2
    assert execute.count == 2
    assert _dumped_first_args(execute)[-1] == {"value": "original-prepared"}


# ---------------------------------------------------------------------------
# Every engine execute path validates before running the tool body.
# ---------------------------------------------------------------------------


async def test_regular_execute_tools_validate_before_running() -> None:
    execute = Recorder({})
    regular = tool(name="regular", input_schema=TargetArgs, execute=execute)

    result = await execute_regular_tool(regular, ParsedToolCall(id="c1", name="regular", arguments={}), CONTEXT)

    assert execute.count == 0
    assert result.get("error") is not None


async def test_generator_tools_validate_before_running() -> None:
    started = Recorder()

    class Progress(BaseModel):
        progress: int

    class Done(BaseModel):
        done: bool

    async def run(params: Any, ctx: Any) -> AsyncIterator[Dict[str, Any]]:
        started(params, ctx)
        yield {"progress": 1}
        yield {"done": True}

    generator = tool(name="generator", input_schema=TargetArgs, event_schema=Progress, output_schema=Done, execute=run)

    result = await execute_generator_tool(generator, ParsedToolCall(id="c2", name="generator", arguments={}), CONTEXT)

    assert started.count == 0
    assert result.get("error") is not None


async def test_hitl_on_tool_called_tools_validate_before_running() -> None:
    on_tool_called = Recorder({})
    hitl = tool(name="hitl", input_schema=TargetArgs, output_schema=OkOutput, on_tool_called=on_tool_called)

    result = await execute_hitl_tool(hitl, ParsedToolCall(id="c3", name="hitl", arguments={}), CONTEXT)

    assert on_tool_called.count == 0
    assert result is not None
    assert result.get("error") is not None


@pytest.mark.skip(
    reason="Not implemented in the Python port: unified `run` tools / prepareUnifiedInvocation "
    "(tool() has no `run` parameter; tool.py)."
)
async def test_unified_run_tools_validate_before_running() -> None:
    """Upstream: prepareUnifiedInvocation returns an error without calling `run`."""


# ---------------------------------------------------------------------------
# Bug 2: the allow_final_response path enforces the approval gate (#54).
# ---------------------------------------------------------------------------


def _safe_and_danger_tools() -> Tuple[Recorder, Recorder, List[Tool]]:
    safe_execute = Recorder({"ok": True})
    danger_execute = Recorder({"ok": True})
    safe = tool(name="safe", input_schema=TargetArgs, output_schema=OkOutput, execute=safe_execute)
    danger = tool(
        name="danger", input_schema=TargetArgs, output_schema=OkOutput, require_approval=True, execute=danger_execute
    )
    return safe_execute, danger_execute, [safe, danger]


def _safe_then_danger_client(final_text: str = "Done.") -> QueuedClient:
    return QueuedClient(
        [
            make_response("resp_turn_0", [function_call_item("call_safe", "safe", '{"target":"staging"}')]),
            make_response("resp_turn_1", [function_call_item("call_danger", "danger", '{"target":"prod"}')]),
            _done("resp_final", final_text),
        ]
    )


async def _assert_gate_holds_on_allow_final_response_path(stop_when: Any) -> None:
    safe_execute, danger_execute, tools = _safe_and_danger_tools()
    client = _safe_then_danger_client()
    accessor = MemoryStateAccessor()

    result = call_model(
        client,
        {
            "model": "test-model",
            "input": "do the thing",
            "tools": tools,
            "state": accessor,
            "stop_when": stop_when,
            "allow_final_response": True,
        },
    )
    await result.get_response()

    # The ungated tool ran in the normal loop round, exactly once.
    assert safe_execute.count == 1
    # The gate holds on the allow_final_response path: the gated tool never
    # runs without approval, and the run pauses like the in-loop gate.
    assert danger_execute.count == 0
    saved = accessor.stored
    assert saved is not None
    assert saved.status == "awaiting_approval"
    assert [call.id for call in saved.pending_tool_calls or []] == ["call_danger"]
    assert await result.requires_approval() is True
    # No final text-coercion request while paused: initial + one follow-up.
    assert len(client.requests) == 2


async def test_does_not_execute_require_approval_tool_when_stop_when_halts_the_loop() -> None:
    # Stop conditions see executed tool rounds (upstream shouldStopExecution):
    # step_count_is(1) fires on turn 1's response, the one carrying the gated
    # call, which reaches the post-loop allow_final_response path.
    await _assert_gate_holds_on_allow_final_response_path(step_count_is(1))


async def test_still_runs_non_gated_tools_on_the_allow_final_response_path() -> None:
    execute = Recorder({"ok": True})
    safe = tool(name="safe", input_schema=TargetArgs, output_schema=OkOutput, execute=execute)
    client = QueuedClient(
        [
            make_response("resp_turn_0", [function_call_item("call_safe", "safe", '{"target":"staging"}')]),
            _done("resp_final"),
        ]
    )

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": "do the safe thing",
            "tools": [safe],
            "state": MemoryStateAccessor(),
            "stop_when": step_count_is(0),
            "allow_final_response": True,
        },
    ).get_text()

    assert execute.count == 1
    assert text == "Done."
    assert len(client.requests) == 2


async def _assert_permission_deny_on_allow_final_response_path(stop_when: Any) -> None:
    safe_execute, danger_execute, tools = _safe_and_danger_tools()
    hooks = HooksManager()
    permission_handler = Recorder({"decision": "deny", "reason": "blocked by policy"})
    hooks.on(HookName.PermissionRequest.value, HookEntry(handler=permission_handler))
    client = _safe_then_danger_client("Denied.")
    accessor = MemoryStateAccessor()

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "do the thing",
            "tools": tools,
            "hooks": hooks,
            "state": accessor,
            "stop_when": stop_when,
            "allow_final_response": True,
        },
    ).get_response()

    # The hook was consulted for the gated call exactly once.
    assert permission_handler.count == 1
    assert danger_execute.count == 0
    assert safe_execute.count == 1
    # A hook deny resolves the gate without a human: no approval pause.
    saved = accessor.stored
    assert saved is not None
    assert saved.status != "awaiting_approval"
    # The rejection is recorded in state as a synthesized output carrying the
    # hook's reason. State items keep upstream's camelCase `callId`
    # (model_result.py _rejected_output); only _send converts to `call_id`.
    outputs = [
        message
        for message in saved.messages
        if isinstance(message, dict)
        and message.get("type") == "function_call_output"
        and message.get("callId") == "call_danger"
    ]
    assert len(outputs) == 1
    assert "blocked by policy" in json.dumps(outputs[0]["output"])


async def test_honors_permission_request_deny_on_the_allow_final_response_path() -> None:
    await _assert_permission_deny_on_allow_final_response_path(step_count_is(1))


async def test_fails_closed_on_schema_invalid_arguments_that_no_hook_repairs() -> None:
    execute = Recorder({"ok": True})
    strict = tool(
        name="strict",
        input_schema=TargetArgs,
        output_schema=OkOutput,
        require_approval=lambda params, ctx: len(params["target"]) > 3,
        execute=execute,
    )
    client = QueuedClient(
        [
            make_response("resp_turn_0", [function_call_item("call_strict", "strict", "{}")]),
            _done("resp_final", "Recovered."),
        ]
    )
    accessor = MemoryStateAccessor()

    await call_model(
        client, {"model": "test-model", "input": "do the strict thing", "tools": [strict], "state": accessor}
    ).get_response()

    saved = accessor.stored
    assert saved is not None
    assert saved.status == "awaiting_approval"
    assert saved.pending_tool_calls == [ParsedToolCall(id="call_strict", name="strict", arguments={})]
    assert execute.count == 0
    assert len(client.requests) == 1


def _danger_tool(execute: Recorder, input_schema: Any = TargetArgs) -> Dict[str, Any]:
    return tool(
        name="danger", input_schema=input_schema, output_schema=OkOutput, require_approval=True, execute=execute
    )


async def test_gates_duplicate_identical_calls_separately_but_only_once_across_visits() -> None:
    danger_execute = Recorder({"ok": True})
    hooks = HooksManager()
    permission_handler = Recorder({"decision": "allow"})
    hooks.on(HookName.PermissionRequest.value, HookEntry(handler=permission_handler))
    client = QueuedClient(
        [
            make_response(
                "resp_duplicates",
                [
                    function_call_item("call_duplicate", "danger", '{"target":"prod"}'),
                    function_call_item("call_duplicate", "danger", '{"target":"prod"}'),
                ],
            ),
            _done("resp_final"),
        ]
    )

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "do both things",
            "tools": [_danger_tool(danger_execute)],
            "hooks": hooks,
            "state": MemoryStateAccessor(),
            "stop_when": step_count_is(0),
            "allow_final_response": True,
        },
    ).get_text()

    # Two occurrences -> two gates, despite the pre-loop and the
    # allow_final_response gate both visiting the response.
    assert permission_handler.count == 2
    assert danger_execute.count == 2


async def test_reuses_occurrence_identities_for_uncanonicalizable_response_calls() -> None:
    # Upstream calls the private handleApprovalCheck three times (same response
    # twice, then a later response). Driven through call_model instead: the
    # first response (two deep-nested duplicates) is visited by both the
    # pre-loop and in-loop gates; the later response carries one more
    # occurrence. Each occurrence is gated exactly once -> 3 emits.
    class ChildArgs(BaseModel):
        child: Any = None

    danger_execute = Recorder({"ok": True})
    hooks = HooksManager()
    permission_handler = Recorder({"decision": "allow"})
    hooks.on(HookName.PermissionRequest.value, HookEntry(handler=permission_handler))
    deep = json.dumps(_deep_nested())
    client = QueuedClient(
        [
            make_response(
                "resp_deep_duplicates",
                [function_call_item("call_deep", "danger", deep), function_call_item("call_deep", "danger", deep)],
            ),
            make_response("resp_deep_later", [function_call_item("call_deep", "danger", deep)]),
            _done("resp_final"),
        ]
    )

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "do the thing",
            "tools": [_danger_tool(danger_execute, ChildArgs)],
            "hooks": hooks,
        },
    ).get_response()

    assert permission_handler.count == 3
    assert danger_execute.count == 3
    assert len(client.requests) == 3


@pytest.mark.skip(
    reason="Private-API-only upstream case: it calls handleApprovalCheck with an empty `calls` subset "
    "and relies on it deriving occurrences from response.output. The port's _handle_approval_check "
    "gates exactly the calls passed (model_result.py:874-905) and every engine call site passes the "
    "full extracted list, so the subset contract has no publicly observable counterpart."
)
async def test_derives_duplicate_occurrences_from_full_response_when_passed_a_subset() -> None:
    """Upstream: growing response.output between two gate visits emits only for the new occurrence."""


async def test_gates_initial_response_only_once_when_stop_fires_on_first_iteration() -> None:
    danger_execute = Recorder({"ok": True})
    hooks = HooksManager()
    permission_handler = Recorder({"decision": "allow"})
    hooks.on(HookName.PermissionRequest.value, HookEntry(handler=permission_handler))
    client = QueuedClient(
        [
            make_response("resp_turn_0", [function_call_item("call_danger", "danger", '{"target":"prod"}')]),
            _done("resp_final"),
        ]
    )

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": "do the thing",
            "tools": [_danger_tool(danger_execute)],
            "hooks": hooks,
            "state": MemoryStateAccessor(),
            "stop_when": step_count_is(0),
            "allow_final_response": True,
        },
    ).get_text()

    # One PermissionRequest emit for the single gated call -- not one per gate
    # visit -- and the allowed call executes exactly once.
    assert permission_handler.count == 1
    assert danger_execute.count == 1
    assert text == "Done."
    assert len(client.requests) == 2


# ---------------------------------------------------------------------------
# Approval decision validation and paused resumes (#517, #518).
# ---------------------------------------------------------------------------


class VArgs(BaseModel):
    v: str


class AnswerOutput(BaseModel):
    answer: str


def _caller_output(call_id: str, output: Any) -> Dict[str, Any]:
    return {
        "type": "function_call_output",
        "id": f"output_{call_id}",
        "callId": call_id,
        "output": json.dumps(output, separators=(",", ":")),
    }


def _gated_request(tools: Sequence[Tool], accessor: MemoryStateAccessor, **extra: Any) -> Dict[str, Any]:
    return {"model": "test-model", "input": "run the gated action", "tools": list(tools), "state": accessor, **extra}


async def _pause_on_gated_calls(call_ids: Sequence[str]) -> Tuple[Recorder, List[Tool], MemoryStateAccessor]:
    execute = Recorder({"ok": True})
    gated = tool(name="gated", input_schema=VArgs, output_schema=OkOutput, require_approval=True, execute=execute)
    tools: List[Tool] = [gated]
    client = QueuedClient(
        [make_response("resp_pause", [function_call_item(call_id, "gated", '{"v":"x"}') for call_id in call_ids])]
    )
    accessor = MemoryStateAccessor()
    await call_model(client, _gated_request(tools, accessor)).get_response()
    assert accessor.stored is not None
    assert accessor.stored.status == "awaiting_approval"
    assert execute.count == 0
    return execute, tools, accessor


async def _pause_on_hitl_call() -> Tuple[Recorder, Recorder, List[Tool], MemoryStateAccessor]:
    on_tool_called = Recorder(None)
    on_response_received = Recorder()

    def transform(raw: Any, ctx: Any) -> Dict[str, Any]:
        on_response_received(raw, ctx)
        return {"answer": str(raw.get("answer")).upper()}

    review = tool(
        name="gated",
        input_schema=VArgs,
        output_schema=AnswerOutput,
        on_tool_called=on_tool_called,
        on_response_received=transform,
    )
    tools: List[Tool] = [review]
    client = QueuedClient([make_response("resp_hitl", [function_call_item("call_hitl", "gated", '{"v":"x"}')])])
    accessor = MemoryStateAccessor()
    await call_model(client, _gated_request(tools, accessor)).get_response()
    assert accessor.stored is not None
    assert accessor.stored.status == "awaiting_hitl"
    assert on_tool_called.count == 1
    return on_tool_called, on_response_received, tools, accessor


async def test_executes_an_approved_call_once_when_its_id_is_listed_twice() -> None:
    execute, tools, accessor = await _pause_on_gated_calls(["call_1"])
    client = QueuedClient([_done("resp_done")])

    text = await call_model(client, _gated_request(tools, accessor, approve_tool_calls=["call_1", "call_1"])).get_text()

    assert text == "Done."
    assert execute.count == 1
    # Upstream asserts `callId`. Outgoing items use snake_case `call_id`:
    # ModelResult._send normalizes at the transport boundary
    # (model_result.py:213-231). Deliberate -- do not change back to callId.
    assert [item["call_id"] for item in _function_call_outputs(client, 0)] == ["call_1"]


async def test_records_a_rejection_once_when_its_id_is_listed_twice() -> None:
    execute, tools, accessor = await _pause_on_gated_calls(["call_1"])
    client = QueuedClient([_done("resp_done")])

    await call_model(client, _gated_request(tools, accessor, reject_tool_calls=["call_1", "call_1"])).get_response()

    assert execute.count == 0
    outputs = _function_call_outputs(client, 0)
    # snake_case `call_id` on the wire: see model_result.py:213-231 (_send).
    assert [item["call_id"] for item in outputs] == ["call_1"]
    assert json.loads(outputs[0]["output"]) == {"error": "Rejected by user"}


async def test_refuses_an_id_that_is_both_approved_and_rejected_before_anything_runs() -> None:
    execute, tools, accessor = await _pause_on_gated_calls(["call_1", "call_2"])
    before = accessor.stored
    saves_before = len(accessor.saved)
    client = QueuedClient([])

    with pytest.raises(ValueError, match="both approved and rejected: call_2"):
        call_model(
            client,
            _gated_request(tools, accessor, approve_tool_calls=["call_1", "call_2"], reject_tool_calls=["call_2"]),
        )

    assert execute.count == 0
    assert client.requests == []
    assert accessor.stored is before
    assert len(accessor.saved) == saves_before


async def test_stays_awaiting_approval_and_sends_no_request_when_resumed_without_decisions() -> None:
    execute, tools, accessor = await _pause_on_gated_calls(["call_1"])
    before = accessor.stored
    saves_before = len(accessor.saved)
    client = QueuedClient([_done("resp_should_not_be_requested")])

    result = call_model(client, _gated_request(tools, accessor))

    with pytest.raises(RuntimeError, match=r"awaiting approval.*call_1"):
        await result.get_response()
    with pytest.raises(RuntimeError, match="awaiting approval"):
        await result.get_text()
    assert await result.requires_approval() is True
    assert [call.id for call in await result.get_pending_tool_calls()] == ["call_1"]

    assert client.requests == []
    assert execute.count == 0
    assert accessor.stored is before
    assert len(accessor.saved) == saves_before

    # The pause still resolves normally once a decision is supplied.
    resume = QueuedClient([_done("resp_done")])
    await call_model(resume, _gated_request(tools, accessor, approve_tool_calls=["call_1"])).get_response()
    assert execute.count == 1
    assert accessor.stored is not None
    assert accessor.stored.status == "complete"
    assert len(resume.requests) == 1


async def test_stays_awaiting_hitl_and_sends_no_request_when_resumed_without_decisions() -> None:
    on_tool_called = Recorder(None)
    review = tool(name="gated", input_schema=VArgs, output_schema=OkOutput, on_tool_called=on_tool_called)
    tools: List[Tool] = [review]
    accessor = MemoryStateAccessor()
    first = QueuedClient([make_response("resp_hitl", [function_call_item("call_hitl", "gated", '{"v":"x"}')])])
    await call_model(first, _gated_request(tools, accessor)).get_response()
    assert accessor.stored is not None
    assert accessor.stored.status == "awaiting_hitl"

    client = QueuedClient([_done("resp_should_not_be_requested")])
    result = call_model(client, _gated_request(tools, accessor))

    with pytest.raises(RuntimeError, match=r"awaiting HITL input.*call_hitl"):
        await result.get_response()
    assert client.requests == []
    assert on_tool_called.count == 1
    assert accessor.stored.status == "awaiting_hitl"


async def test_reports_the_pause_when_decisions_cover_only_some_pending_calls() -> None:
    execute, tools, accessor = await _pause_on_gated_calls(["call_1", "call_2"])
    client = QueuedClient([])

    result = call_model(client, _gated_request(tools, accessor, approve_tool_calls=["call_1"]))

    with pytest.raises(RuntimeError, match=r"awaiting approval.*call_2"):
        await result.get_response()
    assert execute.count == 1
    assert client.requests == []
    assert accessor.stored is not None
    assert accessor.stored.status == "awaiting_approval"
    assert [call.id for call in accessor.stored.pending_tool_calls or []] == ["call_2"]


async def test_resumes_awaiting_hitl_when_input_carries_output_for_the_paused_call() -> None:
    on_tool_called, on_response_received, tools, accessor = await _pause_on_hitl_call()
    client = QueuedClient([_done("resp_done")])

    text = await call_model(
        client,
        {**_gated_request(tools, accessor), "input": [_caller_output("call_hitl", {"answer": "yes"})]},
    ).get_text()

    assert text == "Done."
    assert on_response_received.count == 1
    assert on_response_received.calls[0][0] == {"answer": "yes"}
    # The paused call was not re-run; the model got the hook-transformed output.
    assert on_tool_called.count == 1
    assert len(client.requests) == 1
    outputs = _function_call_outputs(client, 0)
    # snake_case `call_id` on the wire: see model_result.py:213-231 (_send).
    assert [(item["call_id"], item["output"]) for item in outputs] == [("call_hitl", '{"answer":"YES"}')]
    assert accessor.stored is not None
    assert accessor.stored.status == "complete"


async def test_resolves_a_function_valued_input_once_when_it_resumes_awaiting_hitl() -> None:
    _, on_response_received, tools, accessor = await _pause_on_hitl_call()
    client = QueuedClient([_done("resp_done")])
    input_fn = Recorder([_caller_output("call_hitl", {"answer": "yes"})])

    text = await call_model(client, {**_gated_request(tools, accessor), "input": input_fn}).get_text()

    assert text == "Done."
    assert input_fn.count == 1
    assert on_response_received.count == 1
    assert len(client.requests) == 1


@pytest.mark.parametrize(
    "input_value",
    ["a new user message", [_caller_output("call_other", {"answer": "yes"})]],
    ids=["string", "output-for-other-call"],
)
async def test_stays_awaiting_hitl_when_input_has_no_output_for_a_pending_call(input_value: Any) -> None:
    _, on_response_received, tools, accessor = await _pause_on_hitl_call()
    before = accessor.stored
    saves_before = len(accessor.saved)
    client = QueuedClient([])

    result = call_model(client, {**_gated_request(tools, accessor), "input": input_value})

    with pytest.raises(
        RuntimeError,
        match=r"awaiting HITL input.*call_hitl.*function_call_output.*input supplied with this call was not sent",
    ):
        await result.get_response()
    assert client.requests == []
    assert on_response_received.count == 0
    assert accessor.stored is before
    assert len(accessor.saved) == saves_before


async def test_stays_awaiting_approval_when_input_carries_output_for_the_gated_call() -> None:
    execute, tools, accessor = await _pause_on_gated_calls(["call_1"])
    before = accessor.stored
    saves_before = len(accessor.saved)
    client = QueuedClient([])

    result = call_model(client, {**_gated_request(tools, accessor), "input": [_caller_output("call_1", {"ok": True})]})

    with pytest.raises(RuntimeError, match=r"awaiting approval.*call_1.*input supplied with this call was not sent"):
        await result.get_response()
    assert client.requests == []
    assert execute.count == 0
    assert accessor.stored is before
    assert len(accessor.saved) == saves_before


async def test_rejects_every_response_and_stream_getter_on_a_paused_resume() -> None:
    _, tools, accessor = await _pause_on_gated_calls(["call_1"])
    before = accessor.stored
    saves_before = len(accessor.saved)
    client = QueuedClient([])
    result = call_model(client, _gated_request(tools, accessor))

    consumers: Dict[str, Callable[[], Awaitable[Any]]] = {
        "get_response": result.get_response,
        "get_text": result.get_text,
        "get_tool_calls": result.get_tool_calls,
        "get_text_stream": lambda: result.get_text_stream().__anext__(),
        "get_reasoning_stream": lambda: result.get_reasoning_stream().__anext__(),
        "get_full_responses_stream": lambda: result.get_full_responses_stream().__anext__(),
        "get_items_stream": lambda: result.get_items_stream().__anext__(),
        "get_new_messages_stream": lambda: result.get_new_messages_stream().__anext__(),
        "get_tool_stream": lambda: result.get_tool_stream().__anext__(),
        "get_tool_calls_stream": lambda: result.get_tool_calls_stream().__anext__(),
    }
    failures: List[str] = []
    for name, consume in consumers.items():
        error: Optional[BaseException] = None
        try:
            await consume()
        except Exception as caught:  # noqa: BLE001 - asserting the type below
            error = caught
        if not (isinstance(error, RuntimeError) and re.search(r"awaiting approval.*call_1", str(error))):
            failures.append(f"{name}: {error!r}")
    assert failures == []

    # Inspection APIs still work on the paused conversation.
    assert await result.requires_approval() is True
    assert [call.id for call in await result.get_pending_tool_calls()] == ["call_1"]
    assert (await result.get_state()).status == "awaiting_approval"

    assert client.requests == []
    assert accessor.stored is before
    assert len(accessor.saved) == saves_before


@pytest.mark.parametrize(
    ("status", "pending_tool_calls"),
    [("awaiting_approval", []), ("awaiting_approval", None), ("awaiting_hitl", [])],
    ids=["awaiting_approval-empty", "awaiting_approval-none", "awaiting_hitl-empty"],
)
async def test_continues_a_paused_state_with_nothing_pending(
    status: str, pending_tool_calls: Optional[List[ParsedToolCall]]
) -> None:
    # A paused status with nothing pending has no decision to wait for;
    # pausing would strand the conversation, so it continues normally.
    execute, tools, paused = await _pause_on_gated_calls(["call_1"])
    assert paused.stored is not None
    accessor = MemoryStateAccessor()
    accessor.stored = dataclasses.replace(paused.stored, status=status, pending_tool_calls=pending_tool_calls)
    client = QueuedClient([_done("resp_done")])

    text = await call_model(client, _gated_request(tools, accessor)).get_text()

    assert text == "Done."
    assert len(client.requests) == 1
    assert execute.count == 0
    assert accessor.stored is not None
    assert accessor.stored.status == "complete"


async def test_caps_pending_ids_in_awaiting_error_and_omits_input_note_without_input() -> None:
    call_ids = [f"call_{index + 1}" for index in range(25)]
    _, tools, accessor = await _pause_on_gated_calls(call_ids)
    client = QueuedClient([])

    with pytest.raises(RuntimeError) as caught:
        await call_model(client, {**_gated_request(tools, accessor), "input": []}).get_response()

    message = str(caught.value)
    assert "call_1, call_2," in message
    assert "call_10, ... (15 more)" in message
    assert "call_11" not in message
    assert "was not sent" not in message
    assert client.requests == []


# Before de-duplication, the second copy of a duplicated ID skipped the
# re-approval that a PreToolUse argument mutation requires: the first copy
# re-gated the call (marking it pre_tool_use_applied), the second executed it.
async def test_does_not_let_a_duplicated_id_bypass_reapproval_after_pre_tool_use_mutation() -> None:
    execute = Recorder({"ok": True})
    gated = tool(
        name="gated", input_schema=VArgs, output_schema=OkOutput, require_approval=lambda p, c: True, execute=execute
    )
    tools: List[Tool] = [gated]
    accessor = MemoryStateAccessor()
    first = QueuedClient([make_response("resp_pause", [function_call_item("call_1", "gated", '{"v":"reviewed"}')])])
    await call_model(first, _gated_request(tools, accessor)).get_response()
    assert accessor.stored is not None
    assert accessor.stored.status == "awaiting_approval"

    hooks = HooksManager()
    hooks.on(HookName.PreToolUse.value, HookEntry(handler=lambda p, c: {"mutated_input": {"v": "mutated"}}))
    client = QueuedClient([])
    result = call_model(client, _gated_request(tools, accessor, hooks=hooks, approve_tool_calls=["call_1", "call_1"]))

    # The human approved {"v":"reviewed"}; the mutated arguments need their
    # own approval, so nothing executes and the call is pending again.
    with pytest.raises(RuntimeError, match=r"awaiting approval.*call_1"):
        await result.get_response()
    assert execute.count == 0
    assert client.requests == []
    assert accessor.stored.status == "awaiting_approval"
    assert accessor.stored.pending_tool_calls == [
        ParsedToolCall(id="call_1", name="gated", arguments={"v": "mutated"}, pre_tool_use_applied=True)
    ]


# ---------------------------------------------------------------------------
# Port-added coverage for the two-phase PreToolUse preparation (upstream
# c610b6e). Upstream ships no dedicated cases for these branches of
# prepareAfterInitialApproval / processApprovalDecisions; each test pins one.
# ---------------------------------------------------------------------------


def _mutating_hooks(mutated: Dict[str, Any]) -> Tuple[HooksManager, Recorder, Recorder]:
    pre_tool_use = Recorder({"mutated_input": mutated})
    failures = Recorder()
    hooks = HooksManager()
    hooks.on(HookName.PreToolUse.value, HookEntry(handler=pre_tool_use))
    hooks.on(HookName.PostToolUseFailure.value, HookEntry(handler=failures))
    return hooks, pre_tool_use, failures


async def _pause_on_mutated_gate(hooks: HooksManager, execute: Recorder) -> Tuple[List[Tool], MemoryStateAccessor]:
    def require(params: Any, ctx: Any) -> bool:
        return bool(params["v"] == "mutated")

    gated = tool(name="gated", input_schema=VArgs, output_schema=OkOutput, require_approval=require, execute=execute)
    tools: List[Tool] = [gated]
    accessor = MemoryStateAccessor()
    client = QueuedClient([make_response("resp_pause", [function_call_item("call_1", "gated", '{"v":"wire"}')])])
    await call_model(client, _gated_request(tools, accessor, hooks=hooks)).get_response()
    assert accessor.stored is not None
    assert accessor.stored.status == "awaiting_approval"
    assert accessor.stored.pending_tool_calls == [
        ParsedToolCall(id="call_1", name="gated", arguments={"v": "mutated"}, pre_tool_use_applied=True)
    ]
    return tools, accessor


async def test_approving_a_prepared_call_executes_its_effective_input_without_rerunning_pre_tool_use() -> None:
    execute = Recorder({"ok": True})
    hooks, pre_tool_use, failures = _mutating_hooks({"v": "mutated"})
    tools, accessor = await _pause_on_mutated_gate(hooks, execute)
    assert pre_tool_use.count == 1

    client = QueuedClient([_done("resp_done")])
    text = await call_model(
        client, _gated_request(tools, accessor, hooks=hooks, approve_tool_calls=["call_1"])
    ).get_text()

    assert text == "Done."
    # The marker means PreToolUse already ran: it does not run a second time.
    assert pre_tool_use.count == 1
    assert execute.count == 1
    assert _dumped_first_args(execute) == [{"v": "mutated"}]
    assert failures.count == 0
    assert len(client.requests) == 1


async def test_rejecting_a_prepared_call_emits_post_tool_use_failure() -> None:
    execute = Recorder({"ok": True})
    hooks, pre_tool_use, failures = _mutating_hooks({"v": "mutated"})
    tools, accessor = await _pause_on_mutated_gate(hooks, execute)

    client = QueuedClient([_done("resp_done")])
    await call_model(client, _gated_request(tools, accessor, hooks=hooks, reject_tool_calls=["call_1"])).get_response()

    assert execute.count == 0
    assert pre_tool_use.count == 1
    # The call's lifecycle entered PreToolUse, so the user rejection is a
    # lifecycle failure (hooks_schemas.PostToolUseFailurePayload).
    assert [(call[0]["tool_input"], call[0]["error"]) for call in failures.calls] == [
        ({"v": "mutated"}, "Rejected by user")
    ]
    outputs = _function_call_outputs(client, 0)
    assert [(item["call_id"], json.loads(item["output"])) for item in outputs] == [
        ("call_1", {"error": "Rejected by user"})
    ]


async def test_rejecting_an_unprepared_call_does_not_emit_post_tool_use_failure() -> None:
    execute, tools, accessor = await _pause_on_gated_calls(["call_1"])
    hooks, pre_tool_use, failures = _mutating_hooks({"v": "mutated"})

    client = QueuedClient([_done("resp_done")])
    await call_model(client, _gated_request(tools, accessor, hooks=hooks, reject_tool_calls=["call_1"])).get_response()

    # Initial approval rejections never enter the tool lifecycle.
    assert pre_tool_use.count == 0
    assert failures.count == 0
    assert execute.count == 0


async def test_blocks_a_schema_invalid_pre_tool_use_mutation() -> None:
    execute = Recorder({"ok": True})
    permission_request = Recorder({"decision": "allow"})
    hooks, pre_tool_use, failures = _mutating_hooks({"v": 42})
    hooks.on(HookName.PermissionRequest.value, HookEntry(handler=permission_request))
    gated = tool(
        name="gated", input_schema=VArgs, output_schema=OkOutput, require_approval=lambda p, c: False, execute=execute
    )
    client = QueuedClient(
        [make_response("resp_invalid", [function_call_item("call_1", "gated", '{"v":"wire"}')]), _done("resp_done")]
    )

    text = await call_model(
        client, {"model": "test-model", "input": "run", "tools": [gated], "hooks": hooks}
    ).get_text()

    reason = 'PreToolUse produced invalid input for "gated"'
    assert text == "Done."
    assert execute.count == 0
    assert pre_tool_use.count == 1
    # Never re-gated: the invalid input is blocked before approval.
    assert permission_request.count == 0
    assert [(call[0]["tool_input"], call[0]["error"]) for call in failures.calls] == [({"v": 42}, reason)]
    outputs = _function_call_outputs(client, 1)
    assert [(item["call_id"], json.loads(item["output"])) for item in outputs] == [("call_1", {"error": reason})]


async def test_permission_request_deny_on_the_mutated_gate_blocks_without_pausing() -> None:
    execute = Recorder({"ok": True})
    permission_request = Recorder({"decision": "deny", "reason": "mutation not allowed"})
    hooks, pre_tool_use, failures = _mutating_hooks({"v": "mutated"})
    hooks.on(HookName.PermissionRequest.value, HookEntry(handler=permission_request))

    def require(params: Any, ctx: Any) -> bool:
        return bool(params["v"] == "mutated")

    gated = tool(name="gated", input_schema=VArgs, output_schema=OkOutput, require_approval=require, execute=execute)
    accessor = MemoryStateAccessor()
    client = QueuedClient(
        [make_response("resp_deny", [function_call_item("call_1", "gated", '{"v":"wire"}')]), _done("resp_done")]
    )

    text = await call_model(client, _gated_request([gated], accessor, hooks=hooks)).get_text()

    assert text == "Done."
    assert execute.count == 0
    # Only the mutated input needed approval, so one PermissionRequest, on it.
    assert [call[0]["tool_input"] for call in permission_request.calls] == [{"v": "mutated"}]
    assert [(call[0]["tool_input"], call[0]["error"]) for call in failures.calls] == [
        ({"v": "mutated"}, "mutation not allowed")
    ]
    outputs = _function_call_outputs(client, 1)
    assert [(item["call_id"], json.loads(item["output"])) for item in outputs] == [
        ("call_1", {"error": "mutation not allowed"})
    ]
    assert accessor.stored is not None
    assert accessor.stored.status == "complete"


async def test_call_level_require_approval_re_gates_mutated_input() -> None:
    execute = Recorder({"ok": True})
    hooks, pre_tool_use, _ = _mutating_hooks({"v": "mutated"})
    call_level = Recorder()

    def require(call: ParsedToolCall, ctx: Any) -> bool:
        call_level(call, ctx)
        return bool(call.arguments["v"] == "mutated")

    # No tool-level require_approval: only the call-level source can depend
    # on input, and it alone re-gates the mutation.
    plain = tool(name="gated", input_schema=VArgs, output_schema=OkOutput, execute=execute)
    accessor = MemoryStateAccessor()
    client = QueuedClient([make_response("resp_call_level", [function_call_item("call_1", "gated", '{"v":"wire"}')])])

    await call_model(client, _gated_request([plain], accessor, hooks=hooks, require_approval=require)).get_response()

    assert [call[0].arguments for call in call_level.calls] == [{"v": "wire"}, {"v": "mutated"}]
    assert pre_tool_use.count == 1
    assert execute.count == 0
    assert accessor.stored is not None
    assert accessor.stored.pending_tool_calls == [
        ParsedToolCall(id="call_1", name="gated", arguments={"v": "mutated"}, pre_tool_use_applied=True)
    ]


async def test_static_require_approval_true_does_not_re_gate_mutated_input() -> None:
    execute = Recorder({"ok": True})
    permission_request = Recorder({"decision": "allow"})
    hooks, pre_tool_use, _ = _mutating_hooks({"v": "mutated"})
    hooks.on(HookName.PermissionRequest.value, HookEntry(handler=permission_request))
    gated = tool(name="gated", input_schema=VArgs, output_schema=OkOutput, require_approval=True, execute=execute)
    client = QueuedClient(
        [make_response("resp_static", [function_call_item("call_1", "gated", '{"v":"wire"}')]), _done("resp_done")]
    )

    await call_model(client, {"model": "test-model", "input": "run", "tools": [gated], "hooks": hooks}).get_text()

    # A blanket `True` cannot depend on input: one PermissionRequest (on the
    # wire input), then PreToolUse exactly once, then execution of the
    # mutated input exactly once.
    assert [call[0]["tool_input"] for call in permission_request.calls] == [{"v": "wire"}]
    assert pre_tool_use.count == 1
    assert execute.count == 1
    assert _dumped_first_args(execute) == [{"v": "mutated"}]


async def test_pre_tool_use_block_during_preparation_skips_execution_once() -> None:
    execute = Recorder({"ok": True})
    pre_tool_use = Recorder({"block": "nope"})
    hooks = HooksManager()
    hooks.on(HookName.PreToolUse.value, HookEntry(handler=pre_tool_use))
    safe = tool(name="safe", input_schema=VArgs, output_schema=OkOutput, execute=execute)
    client = QueuedClient(
        [make_response("resp_block", [function_call_item("call_1", "safe", '{"v":"x"}')]), _done("resp_done")]
    )

    await call_model(client, {"model": "test-model", "input": "run", "tools": [safe], "hooks": hooks}).get_text()

    assert pre_tool_use.count == 1
    assert execute.count == 0
    outputs = _function_call_outputs(client, 1)
    assert [(item["call_id"], json.loads(item["output"])) for item in outputs] == [("call_1", {"error": "nope"})]


async def test_pre_tool_use_runs_exactly_once_per_call_across_a_mixed_pause_and_resume() -> None:
    safe_execute, danger_execute, tools = _safe_and_danger_tools()
    pre_tool_use = Recorder()
    hooks = HooksManager()
    hooks.on(HookName.PreToolUse.value, HookEntry(handler=pre_tool_use))
    accessor = MemoryStateAccessor()
    first = QueuedClient(
        [
            make_response(
                "resp_mixed",
                [
                    function_call_item("call_safe", "safe", '{"target":"staging"}'),
                    function_call_item("call_danger", "danger", '{"target":"prod"}'),
                ],
            )
        ]
    )
    await call_model(first, _gated_request(tools, accessor, hooks=hooks)).get_response()

    # The auto-approved sibling was prepared and executed while pausing; the
    # gated call is still pending on its initial approval, so its PreToolUse
    # has not run and it carries no marker.
    assert [call[0]["tool_name"] for call in pre_tool_use.calls] == ["safe"]
    assert safe_execute.count == 1
    assert danger_execute.count == 0
    assert accessor.stored is not None
    assert accessor.stored.pending_tool_calls == [
        ParsedToolCall(id="call_danger", name="danger", arguments={"target": "prod"})
    ]

    resume = QueuedClient([_done("resp_done")])
    await call_model(
        resume, _gated_request(tools, accessor, hooks=hooks, approve_tool_calls=["call_danger"])
    ).get_text()

    assert [call[0]["tool_name"] for call in pre_tool_use.calls] == ["safe", "danger"]
    assert safe_execute.count == 1
    assert danger_execute.count == 1
    assert [item["call_id"] for item in _function_call_outputs(resume, 0)] == ["call_safe", "call_danger"]
