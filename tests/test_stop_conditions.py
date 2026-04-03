"""Tests for stop conditions."""

import pytest

from openrouter_agent import (
    ParsedToolCall,
    StepResult,
    Usage,
    finish_reason_is,
    has_tool_call,
    is_stop_condition_met,
    max_tokens_used,
    step_count_is,
)


def _make_step(
    tool_names: list[str] | None = None,
    total_tokens: int = 0,
    finish_reason: str | None = None,
) -> StepResult:
    tool_calls = [
        ParsedToolCall(id=f"call_{n}", name=n, arguments={})
        for n in (tool_names or [])
    ]
    return StepResult(
        step_type="continue",
        text="",
        tool_calls=tool_calls,
        usage=Usage(total_tokens=total_tokens) if total_tokens else None,
        finish_reason=finish_reason,
    )


def test_step_count_is():
    cond = step_count_is(3)
    assert not cond([_make_step(), _make_step()])
    assert cond([_make_step(), _make_step(), _make_step()])


def test_has_tool_call():
    cond = has_tool_call("search")
    assert not cond([_make_step(tool_names=["calculate"])])
    assert cond([_make_step(tool_names=["search"])])
    assert cond([_make_step(tool_names=["calculate"]), _make_step(tool_names=["search"])])


def test_max_tokens_used():
    cond = max_tokens_used(1000)
    assert not cond([_make_step(total_tokens=500)])
    assert cond([_make_step(total_tokens=500), _make_step(total_tokens=600)])


def test_finish_reason_is():
    cond = finish_reason_is("stop")
    assert not cond([_make_step(finish_reason="tool_calls")])
    assert cond([_make_step(finish_reason="stop")])


@pytest.mark.anyio
async def test_is_stop_condition_met_or_logic():
    steps = [_make_step(tool_names=["search"])]
    conditions = [step_count_is(5), has_tool_call("search")]
    assert await is_stop_condition_met(conditions, steps)


@pytest.mark.anyio
async def test_is_stop_condition_met_none_met():
    steps = [_make_step()]
    conditions = [step_count_is(5), has_tool_call("search")]
    assert not await is_stop_condition_met(conditions, steps)
