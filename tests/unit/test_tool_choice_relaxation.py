"""Ported from upstream tests/unit/tool-choice-relaxation.test.ts (#100, DEV-785).

A caller-forced `tool_choice` (`"required"`, a specific tool, or
`allowed_tools` with `mode: "required"`) is relaxed to `"auto"` once it has
produced a tool call, including follow-ups resumed after an approval or a
manual client-tool pause, so the model can answer in text instead of calling
tools until the step budget runs out.

Not ported: the two "re-arms a dynamic forced choice" cases. This port resolves
function-valued request parameters once, at turn 0, rather than per turn (a
pre-existing divergence), so a choice whose value changes between turns cannot
be expressed through `call_model` here.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from pydantic import BaseModel

from openrouter_agent import call_model, step_count_is, tool
from tests._fixtures import MemoryStateAccessor, QueuedClient, text_response, tool_call_response


class _City(BaseModel):
    location: str


def _weather(**overrides: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": _City,
        "execute": lambda _params, _ctx: {"temperature": 22},
    }
    kwargs.update(overrides)
    return tool(**kwargs)


def _call(response_id: str, call_id: str) -> Dict[str, Any]:
    return tool_call_response(response_id, "get_weather", call_id=call_id, arguments='{"location":"Tokyo"}')


def _choices(client: QueuedClient) -> List[Any]:
    return [request.get("tool_choice") for request in client.requests]


async def test_relaxes_required_to_auto_after_a_successful_tool_round() -> None:
    client = QueuedClient([_call("resp_1", "call_1"), text_response("resp_2", "It is 22 degrees.")])

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": "What is the weather in Tokyo?",
            "tools": [_weather()],
            "tool_choice": "required",
            "stop_when": step_count_is(5),
        },
    ).get_text()

    assert text == "It is 22 degrees."
    assert _choices(client) == ["required", "auto"]
    # Tools stay available so the model may call another tool OR answer.
    assert client.requests[1]["tools"]


async def test_relaxes_a_forced_specific_tool_choice_after_a_successful_tool_round() -> None:
    client = QueuedClient([_call("resp_1", "call_1"), text_response("resp_2", "Done.")])

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "weather?",
            "tools": [_weather()],
            "tool_choice": {"type": "function", "name": "get_weather"},
            "stop_when": step_count_is(5),
        },
    ).get_text()

    assert _choices(client) == [{"type": "function", "name": "get_weather"}, "auto"]


async def test_relaxes_allowed_tools_required_mode_to_auto_keeping_the_tool_set() -> None:
    allowed = [{"type": "function", "name": "get_weather"}]
    client = QueuedClient([_call("resp_1", "call_1"), text_response("resp_2", "Done.")])

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "weather?",
            "tools": [_weather()],
            "tool_choice": {"type": "allowed_tools", "mode": "required", "tools": allowed},
            "stop_when": step_count_is(5),
        },
    ).get_text()

    assert client.requests[1]["tool_choice"] == {"type": "allowed_tools", "mode": "auto", "tools": allowed}


async def test_keeps_auto_unchanged_on_follow_up_turns() -> None:
    client = QueuedClient([_call("resp_1", "call_1"), text_response("resp_2", "Done.")])

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "weather?",
            "tools": [_weather()],
            "tool_choice": "auto",
            "stop_when": step_count_is(5),
        },
    ).get_text()

    assert _choices(client) == ["auto", "auto"]


async def test_still_forces_the_final_no_tools_turn_when_the_budget_is_exhausted() -> None:
    client = QueuedClient(
        [_call("resp_1", "call_1"), _call("resp_2", "call_2"), text_response("resp_3", "Final summary.")]
    )

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": "weather?",
            "tools": [_weather()],
            "tool_choice": "required",
            "stop_when": step_count_is(2),
            "allow_final_response": True,
        },
    ).get_text()

    assert text == "Final summary."
    assert _choices(client) == ["required", "auto", "none"]


async def test_keeps_an_unchanged_consumed_choice_relaxed_across_multiple_follow_up_turns() -> None:
    client = QueuedClient(
        [_call("resp_1", "call_1"), _call("resp_2", "call_2"), text_response("resp_3", "Final summary.")]
    )

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": "weather?",
            "tools": [_weather()],
            "tool_choice": "required",
            "stop_when": step_count_is(5),
        },
    ).get_text()

    assert text == "Final summary."
    assert _choices(client) == ["required", "auto", "auto"]


async def test_a_forced_choice_is_not_consumed_by_a_response_without_tool_calls() -> None:
    # A choice is consumed only when the response that carried it called a
    # tool; a text-only answer leaves nothing to relax.
    state = MemoryStateAccessor()
    await call_model(
        QueuedClient([text_response("resp_1", "no tools")]),
        {"model": "test-model", "input": "hi", "tools": [_weather()], "tool_choice": "required", "state": state},
    ).get_text()

    assert state.stored is not None
    assert state.stored.consumed_forced_tool_choice_key is None


async def test_persists_satisfaction_across_approval_resume_and_resets_it_after_completion() -> None:
    approval_weather = _weather(require_approval=True)
    state = MemoryStateAccessor()
    first = QueuedClient([_call("resp_1", "call_1")])

    await call_model(
        first,
        {
            "model": "test-model",
            "input": "What is the weather in Tokyo?",
            "tools": [approval_weather],
            "tool_choice": "required",
            "state": state,
        },
    ).get_response()

    assert _choices(first) == ["required"]
    assert state.stored is not None
    assert state.stored.status == "awaiting_approval"
    # Same canonical key upstream persists (JCS of the choice).
    assert state.stored.consumed_forced_tool_choice_key == json.dumps("required")

    resumed = QueuedClient([text_response("resp_2", "It is 22 degrees.")])
    text = await call_model(
        resumed,
        {
            "model": "test-model",
            "input": [],
            "tools": [approval_weather],
            "tool_choice": "required",
            "state": state,
            "approve_tool_calls": ["call_1"],
        },
    ).get_text()

    assert text == "It is 22 degrees."
    assert _choices(resumed) == ["auto"]
    assert state.stored.status == "complete"
    assert state.stored.consumed_forced_tool_choice_key is None

    fresh = QueuedClient([_call("resp_3", "call_2")])
    await call_model(
        fresh,
        {
            "model": "test-model",
            "input": "Start a new weather request.",
            "tools": [approval_weather],
            "tool_choice": "required",
            "state": state,
        },
    ).get_response()

    assert _choices(fresh) == ["required"]


async def test_relaxes_the_first_model_request_after_a_manual_client_tool_resume() -> None:
    manual_weather = tool(name="get_weather", input_schema=_City, execute=False)
    state = MemoryStateAccessor()

    await call_model(
        QueuedClient([_call("resp_1", "call_1")]),
        {
            "model": "test-model",
            "input": "What is the weather in Tokyo?",
            "tools": [manual_weather],
            "tool_choice": "required",
            "state": state,
        },
    ).get_response()

    assert state.stored is not None
    assert state.stored.status == "awaiting_client_tools"
    assert state.stored.consumed_forced_tool_choice_key == json.dumps("required")

    resumed = QueuedClient([text_response("resp_2", "It is 22 degrees.")])
    text = await call_model(
        resumed,
        {
            "model": "test-model",
            "input": [{"type": "function_call_output", "callId": "call_1", "output": json.dumps({"temperature": 22})}],
            "tools": [manual_weather],
            "tool_choice": "required",
            "state": state,
        },
    ).get_text()

    assert text == "It is 22 degrees."
    assert _choices(resumed) == ["auto"]
