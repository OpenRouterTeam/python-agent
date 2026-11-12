"""Ports `packages/agent/tests/unit/stop-conditions.test.ts`."""

from __future__ import annotations

from typing import Any, List, Optional

import pytest

from openrouter_agent import finish_reason_is, has_tool_call, max_cost, max_tokens_used, step_count_is
from openrouter_agent.stop_conditions import is_stop_condition_met
from openrouter_agent.tool_types import ParsedToolCall, StepResult

_UNSET: Any = object()


def make_step(
    *,
    tool_call_names: Optional[List[str]] = None,
    usage: Any = _UNSET,
    finish_reason: Optional[str] = None,
) -> StepResult:
    """A minimal StepResult; stop conditions only read tool_calls, usage and
    finish_reason, so the remaining fields are inert placeholders."""
    return StepResult(
        step_type="initial",
        text="",
        tool_calls=[ParsedToolCall(f"call_{i}", name, {}) for i, name in enumerate(tool_call_names or [])],
        tool_results=[],
        response={},
        usage=None if usage is _UNSET else usage,
        finish_reason=finish_reason,
    )


# -- step_count_is ---------------------------------------------------------------


def test_step_count_is_false_below_threshold() -> None:
    assert step_count_is(2)({"steps": [make_step()]}) is False


def test_step_count_is_true_at_threshold_exactly() -> None:
    assert step_count_is(2)({"steps": [make_step(), make_step()]}) is True


def test_step_count_is_true_when_exceeded() -> None:
    assert step_count_is(1)({"steps": [make_step(), make_step(), make_step()]}) is True


def test_step_count_is_true_for_zero_steps_with_zero_threshold() -> None:
    assert step_count_is(0)({"steps": []}) is True


# -- has_tool_call ---------------------------------------------------------------


def test_has_tool_call_false_when_no_step_contains_the_tool() -> None:
    steps = [make_step(tool_call_names=["search", "browse"])]
    assert has_tool_call("email")({"steps": steps}) is False


def test_has_tool_call_true_when_any_step_contains_the_tool() -> None:
    steps = [make_step(tool_call_names=["search"]), make_step(tool_call_names=["email"])]
    assert has_tool_call("email")({"steps": steps}) is True


def test_has_tool_call_false_for_empty_steps() -> None:
    assert has_tool_call("search")({"steps": []}) is False


def test_has_tool_call_accepts_mapping_steps_and_calls() -> None:
    # Port-specific: steps may also be plain mappings (camelCase or snake_case).
    assert has_tool_call("email")({"steps": [{"toolCalls": [{"name": "email"}]}]}) is True
    assert has_tool_call("email")({"steps": [{"tool_calls": [{"name": "search"}]}]}) is False


# -- max_tokens_used -------------------------------------------------------------


def test_max_tokens_used_sums_total_tokens_across_steps() -> None:
    steps = [make_step(usage={"totalTokens": 400}), make_step(usage={"totalTokens": 700})]
    assert max_tokens_used(1000)({"steps": steps}) is True
    assert max_tokens_used(1100)({"steps": steps}) is True
    assert max_tokens_used(1101)({"steps": steps}) is False


def test_max_tokens_used_treats_missing_usage_as_zero() -> None:
    steps = [make_step(), make_step(usage=None)]
    assert max_tokens_used(1)({"steps": steps}) is False


def test_max_tokens_used_treats_missing_total_tokens_as_zero() -> None:
    assert max_tokens_used(0)({"steps": [make_step(usage={})]}) is True


# -- max_cost --------------------------------------------------------------------


def test_max_cost_sums_cost_across_steps() -> None:
    steps = [make_step(usage={"cost": 0.3}), make_step(usage={"cost": 0.25})]
    assert max_cost(0.5)({"steps": steps}) is True
    assert max_cost(0.6)({"steps": steps}) is False


def test_max_cost_treats_missing_cost_as_zero() -> None:
    assert max_cost(0.01)({"steps": [make_step(usage={})]}) is False


# -- finish_reason_is ------------------------------------------------------------


def test_finish_reason_is_true_when_any_step_has_the_reason() -> None:
    steps = [make_step(finish_reason="stop"), make_step(finish_reason="length")]
    assert finish_reason_is("length")({"steps": steps}) is True


def test_finish_reason_is_false_when_no_step_has_the_reason() -> None:
    steps = [make_step(finish_reason="stop"), make_step()]
    assert finish_reason_is("length")({"steps": steps}) is False


# -- is_stop_condition_met -------------------------------------------------------


async def test_is_stop_condition_met_false_without_conditions() -> None:
    assert await is_stop_condition_met([], []) is False


async def test_is_stop_condition_met_false_when_all_conditions_false() -> None:
    assert await is_stop_condition_met([step_count_is(5), has_tool_call("search")], [make_step()]) is False


async def test_is_stop_condition_met_true_when_any_condition_true() -> None:
    assert await is_stop_condition_met([step_count_is(5), step_count_is(1)], [make_step()]) is True


async def test_is_stop_condition_met_supports_async_conditions() -> None:
    async def async_condition(_options: Any) -> bool:
        return True

    assert await is_stop_condition_met([async_condition], []) is True


async def test_is_stop_condition_met_treats_none_results_as_not_stopping() -> None:
    assert await is_stop_condition_met([lambda _options: None], []) is False


async def test_is_stop_condition_met_requires_exactly_true() -> None:
    # Upstream compares `result === true`; a truthy non-bool does not stop.
    assert await is_stop_condition_met([lambda _options: 1, lambda _options: "yes"], []) is False


async def test_is_stop_condition_met_evaluates_every_condition_once() -> None:
    # Upstream evaluates all conditions with Promise.all — no short-circuit.
    calls: List[str] = []

    def first(_options: Any) -> bool:
        calls.append("first")
        return True

    async def second(_options: Any) -> bool:
        calls.append("second")
        return False

    assert await is_stop_condition_met([first, second], []) is True
    assert calls == ["first", "second"]


async def test_is_stop_condition_met_propagates_failing_conditions() -> None:
    async def failing(_options: Any) -> bool:
        raise RuntimeError("condition exploded")

    with pytest.raises(RuntimeError, match="condition exploded"):
        await is_stop_condition_met([failing], [])
