"""Tests for ModelResult accessor methods."""

from __future__ import annotations

import pytest

from openrouter_agent._model_result import ModelResult
from openrouter_agent._tool_event_broadcaster import ToolEventBroadcaster
from openrouter_agent._types import (
    ConversationState,
    ConversationStatus,
    ParsedToolCall,
    ResponseStreamEvent,
    StepResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_broadcaster(*, done: bool = True) -> ToolEventBroadcaster[ResponseStreamEvent]:
    b: ToolEventBroadcaster[ResponseStreamEvent] = ToolEventBroadcaster()
    if done:
        b.complete()
    return b


def _make_step(
    *,
    text: str = "",
    tool_calls: list[ParsedToolCall] | None = None,
    response: dict | None = None,
) -> StepResult:
    return StepResult(
        step_type="initial",
        text=text,
        tool_calls=tool_calls or [],
        response=response or {},
    )


def _make_tool_call(name: str = "my_tool", call_id: str = "tc_1") -> ParsedToolCall:
    return ParsedToolCall(id=call_id, name=name, arguments={"key": "value"})


# ---------------------------------------------------------------------------
# get_text
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_text_single_step() -> None:
    result = ModelResult(
        steps=[_make_step(text="hello world")],
        broadcaster=_make_broadcaster(),
    )
    assert await result.get_text() == "hello world"


@pytest.mark.anyio
async def test_get_text_multiple_steps() -> None:
    result = ModelResult(
        steps=[_make_step(text="hello "), _make_step(text="world")],
        broadcaster=_make_broadcaster(),
    )
    assert await result.get_text() == "hello world"


@pytest.mark.anyio
async def test_get_text_empty_steps() -> None:
    result = ModelResult(steps=[], broadcaster=_make_broadcaster())
    assert await result.get_text() == ""


# ---------------------------------------------------------------------------
# get_response
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_response_returns_last_step() -> None:
    resp1 = {"id": "r1", "model": "test"}
    resp2 = {"id": "r2", "model": "test"}
    result = ModelResult(
        steps=[_make_step(response=resp1), _make_step(response=resp2)],
        broadcaster=_make_broadcaster(),
    )
    assert await result.get_response() == resp2


@pytest.mark.anyio
async def test_get_response_empty_steps() -> None:
    result = ModelResult(steps=[], broadcaster=_make_broadcaster())
    assert await result.get_response() == {}


# ---------------------------------------------------------------------------
# get_tool_calls
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_tool_calls_aggregates_across_steps() -> None:
    tc1 = _make_tool_call("tool_a", "tc_1")
    tc2 = _make_tool_call("tool_b", "tc_2")
    result = ModelResult(
        steps=[
            _make_step(tool_calls=[tc1]),
            _make_step(tool_calls=[tc2]),
        ],
        broadcaster=_make_broadcaster(),
    )
    calls = await result.get_tool_calls()
    assert len(calls) == 2
    assert calls[0].name == "tool_a"
    assert calls[1].name == "tool_b"


@pytest.mark.anyio
async def test_get_tool_calls_empty() -> None:
    result = ModelResult(steps=[_make_step()], broadcaster=_make_broadcaster())
    assert await result.get_tool_calls() == []


# ---------------------------------------------------------------------------
# get_pending_tool_calls
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_pending_tool_calls_with_state() -> None:
    pending = [_make_tool_call("pending_tool", "tc_p")]
    state = ConversationState(
        id="conv_1",
        status=ConversationStatus.AWAITING_APPROVAL,
        pending_tool_calls=pending,
    )
    result = ModelResult(
        steps=[_make_step()],
        broadcaster=_make_broadcaster(),
        state=state,
    )
    calls = await result.get_pending_tool_calls()
    assert len(calls) == 1
    assert calls[0].name == "pending_tool"


@pytest.mark.anyio
async def test_get_pending_tool_calls_no_state() -> None:
    result = ModelResult(steps=[_make_step()], broadcaster=_make_broadcaster())
    assert await result.get_pending_tool_calls() == []


@pytest.mark.anyio
async def test_get_pending_tool_calls_empty_pending() -> None:
    state = ConversationState(
        id="conv_2",
        status=ConversationStatus.COMPLETE,
        pending_tool_calls=[],
    )
    result = ModelResult(
        steps=[_make_step()],
        broadcaster=_make_broadcaster(),
        state=state,
    )
    assert await result.get_pending_tool_calls() == []


# ---------------------------------------------------------------------------
# get_state
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_state_returns_state() -> None:
    state = ConversationState(id="conv_3", status=ConversationStatus.COMPLETE)
    result = ModelResult(
        steps=[_make_step()],
        broadcaster=_make_broadcaster(),
        state=state,
    )
    assert await result.get_state() is state


@pytest.mark.anyio
async def test_get_state_returns_none_when_absent() -> None:
    result = ModelResult(steps=[_make_step()], broadcaster=_make_broadcaster())
    assert await result.get_state() is None


# ---------------------------------------------------------------------------
# steps property (sync)
# ---------------------------------------------------------------------------


def test_steps_property_is_sync() -> None:
    steps = [_make_step(text="a"), _make_step(text="b")]
    result = ModelResult(steps=steps, broadcaster=_make_broadcaster())
    assert result.steps is steps
    assert len(result.steps) == 2
