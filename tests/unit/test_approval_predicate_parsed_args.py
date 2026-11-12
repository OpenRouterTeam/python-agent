"""Approval predicates see the same schema-parsed arguments `execute` gets,
and schema-invalid arguments fail closed. Subset of upstream
`tests/unit/approval-gate-regressions.test.ts`.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Tuple

import pytest
from pydantic import BaseModel, field_validator

from openrouter_agent import tool
from openrouter_agent.conversation_state import tool_requires_approval
from openrouter_agent.tool_executor import execute_regular_tool
from openrouter_agent.tool_types import ParsedToolCall, Tool

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
