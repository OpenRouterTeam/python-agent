"""Port of upstream `tests/unit/tool-choice-relaxation.test.ts` (DEV-785).

A concrete forced tool choice is one-shot until its resolved semantic value
changes. Reapplying an unchanged choice on every follow-up forbids the model
from ever answering in text, while suppressing a newly resolved dynamic choice
prevents callers from intentionally forcing a later turn.

Python request key is snake_case `tool_choice`; callable request params are
re-resolved per turn with `ctx["number_of_turns"]`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import pytest
from pydantic import BaseModel

from openrouter_agent import call_model, step_count_is, tool
from openrouter_agent.doom_loop import canonicalize_key_material
from tests._fixtures import MemoryStateAccessor, QueuedClient, text_response
from tests._fixtures import tool_call_response as _tool_call_response


class WeatherInput(BaseModel):
    location: str


class WeatherOutput(BaseModel):
    temperature: float


def _weather_execute(params: Any, ctx: Any) -> Dict[str, Any]:
    return {"temperature": 22}


def tool_call_response(response_id: str, call_id: str) -> Dict[str, Any]:
    return _tool_call_response(response_id, "get_weather", call_id=call_id, arguments='{"location":"Tokyo"}')


weather_tool = tool(
    name="get_weather",
    description="Get the weather for a location.",
    input_schema=WeatherInput,
    output_schema=WeatherOutput,
    execute=_weather_execute,
)

approval_weather_tool = tool(
    name="get_weather",
    description="Get the weather for a location.",
    input_schema=WeatherInput,
    output_schema=WeatherOutput,
    execute=_weather_execute,
    require_approval=True,
)

manual_weather_tool = tool(
    name="get_weather",
    description="Get the weather for a location.",
    input_schema=WeatherInput,
    output_schema=WeatherOutput,
    execute=False,
)


def tool_choices(client: QueuedClient) -> List[Any]:
    return [request.get("tool_choice") for request in client.requests]


async def test_relaxes_required_to_auto_after_a_successful_tool_round() -> None:
    client = QueuedClient([tool_call_response("resp_1", "call_1"), text_response("resp_text", "It is 22 degrees.")])

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": "What is the weather in Tokyo?",
            "tools": [weather_tool],
            "tool_choice": "required",
            "stop_when": step_count_is(5),
        },
    ).get_text()

    assert text == "It is 22 degrees."
    assert len(client.requests) == 2
    # Initial turn: the caller's forced choice reaches the wire intact.
    assert client.requests[0]["tool_choice"] == "required"
    # Follow-up: relaxed to auto, with tools still available.
    assert client.requests[1]["tool_choice"] == "auto"
    assert client.requests[1].get("tools") is not None
    assert [t["name"] for t in client.requests[1]["tools"]] == ["get_weather"]


async def test_relaxes_a_forced_specific_tool_choice_after_a_successful_tool_round() -> None:
    client = QueuedClient([tool_call_response("resp_1", "call_1"), text_response("resp_text", "Done.")])

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": "What is the weather in Tokyo?",
            "tools": [weather_tool],
            "tool_choice": {"type": "function", "name": "get_weather"},
            "stop_when": step_count_is(5),
        },
    ).get_text()

    assert text == "Done."
    assert len(client.requests) == 2
    assert client.requests[0]["tool_choice"] == {"type": "function", "name": "get_weather"}
    assert client.requests[1]["tool_choice"] == "auto"


async def test_relaxes_allowed_tools_mode_required_to_mode_auto_keeping_the_tool_set() -> None:
    client = QueuedClient([tool_call_response("resp_1", "call_1"), text_response("resp_text", "Done.")])
    allowed_tools = [{"type": "function", "name": "get_weather"}]

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": "What is the weather in Tokyo?",
            "tools": [weather_tool],
            "tool_choice": {"type": "allowed_tools", "mode": "required", "tools": allowed_tools},
            "stop_when": step_count_is(5),
        },
    ).get_text()

    assert text == "Done."
    assert len(client.requests) == 2
    assert client.requests[0]["tool_choice"] == {"type": "allowed_tools", "mode": "required", "tools": allowed_tools}
    assert client.requests[1]["tool_choice"] == {"type": "allowed_tools", "mode": "auto", "tools": allowed_tools}


async def test_keeps_tool_choice_auto_unchanged_on_follow_up_turns() -> None:
    client = QueuedClient([tool_call_response("resp_1", "call_1"), text_response("resp_text", "Done.")])

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "What is the weather in Tokyo?",
            "tools": [weather_tool],
            "tool_choice": "auto",
            "stop_when": step_count_is(5),
        },
    ).get_text()

    assert tool_choices(client) == ["auto", "auto"]


async def test_still_forces_the_final_no_tools_turn_when_the_budget_is_genuinely_exhausted() -> None:
    client = QueuedClient(
        [
            tool_call_response("resp_1", "call_1"),
            tool_call_response("resp_2", "call_2"),
            text_response("resp_text", "Final summary."),
        ]
    )

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": "What is the weather in Tokyo?",
            "tools": [weather_tool],
            "tool_choice": "required",
            "stop_when": step_count_is(1),
            "allow_final_response": True,
        },
    ).get_text()

    assert text == "Final summary."
    assert len(client.requests) == 3
    assert tool_choices(client) == ["required", "auto", "none"]


async def test_keeps_an_unchanged_consumed_choice_relaxed_across_multiple_follow_up_turns() -> None:
    client = QueuedClient(
        [
            tool_call_response("resp_1", "call_1"),
            tool_call_response("resp_2", "call_2"),
            text_response("resp_text", "Final summary."),
        ]
    )

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": "What is the weather in Tokyo?",
            "tools": [weather_tool],
            "tool_choice": "required",
            "stop_when": step_count_is(5),
        },
    ).get_text()

    assert text == "Final summary."
    assert tool_choices(client) == ["required", "auto", "auto"]


async def test_re_arms_when_a_dynamic_forced_choice_resolves_to_a_different_semantic_value() -> None:
    client = QueuedClient(
        [
            tool_call_response("resp_1", "call_1"),
            tool_call_response("resp_2", "call_2"),
            text_response("resp_text", "Done."),
        ]
    )
    seen_turns: List[int] = []

    def dynamic_choice(ctx: Mapping[str, Any]) -> Any:
        seen_turns.append(ctx["number_of_turns"])
        return "required" if ctx["number_of_turns"] == 0 else {"type": "function", "name": "get_weather"}

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": "What is the weather in Tokyo?",
            "tools": [weather_tool],
            "tool_choice": dynamic_choice,
            "stop_when": step_count_is(5),
        },
    ).get_text()

    assert text == "Done."
    assert tool_choices(client) == ["required", {"type": "function", "name": "get_weather"}, "auto"]
    # Re-resolved exactly once per dispatched turn.
    assert seen_turns == [0, 1, 2]


async def test_re_arms_the_same_dynamic_forced_choice_after_an_unforced_turn() -> None:
    client = QueuedClient(
        [
            tool_call_response("resp_1", "call_1"),
            tool_call_response("resp_2", "call_2"),
            tool_call_response("resp_3", "call_3"),
            text_response("resp_text", "Done."),
        ]
    )

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": "What is the weather in Tokyo?",
            "tools": [weather_tool],
            "tool_choice": lambda ctx: "auto" if ctx["number_of_turns"] == 1 else "required",
            "stop_when": step_count_is(5),
        },
    ).get_text()

    assert text == "Done."
    assert tool_choices(client) == ["required", "auto", "required", "auto"]


async def test_persists_satisfaction_across_approval_resume_and_resets_it_after_completion() -> None:
    accessor = MemoryStateAccessor()
    client = QueuedClient(
        [
            tool_call_response("resp_1", "call_1"),
            text_response("resp_text", "It is 22 degrees."),
            tool_call_response("resp_2", "call_2"),
        ]
    )

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "What is the weather in Tokyo?",
            "tools": [approval_weather_tool],
            "tool_choice": "required",
            "state": accessor,
        },
    ).get_response()

    assert len(client.requests) == 1
    assert client.requests[0]["tool_choice"] == "required"
    assert accessor.stored is not None
    assert accessor.stored.status == "awaiting_approval"
    assert accessor.stored.consumed_forced_tool_choice_key == canonicalize_key_material("required")

    resumed_text = await call_model(
        client,
        {
            "model": "test-model",
            "input": [],
            "tools": [approval_weather_tool],
            "tool_choice": "required",
            "state": accessor,
            "approve_tool_calls": ["call_1"],
        },
    ).get_text()

    assert resumed_text == "It is 22 degrees."
    assert len(client.requests) == 2
    assert client.requests[1]["tool_choice"] == "auto"
    assert accessor.stored.status == "complete"
    # Upstream asserts `toBeUndefined()`; the Python dataclass field is None.
    assert accessor.stored.consumed_forced_tool_choice_key is None

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "Start a new weather request.",
            "tools": [approval_weather_tool],
            "tool_choice": "required",
            "state": accessor,
        },
    ).get_response()

    assert len(client.requests) == 3
    assert client.requests[2]["tool_choice"] == "required"


@pytest.mark.skip(
    reason=(
        "Not implemented in Python: mid-run interruption via a reloaded state's "
        "`interrupted_by` (cancellation signal). Upstream re-loads state during the run and "
        "settles to status 'interrupted'; model_result.py loads state once in `_load_state` "
        "and has no 'interrupted' status path."
    )
)
async def test_clears_consumed_choice_state_when_an_interrupted_run_becomes_a_fresh_request() -> None:
    raise AssertionError("unreachable: skipped")


async def test_relaxes_the_first_model_request_after_a_manual_client_tool_resume() -> None:
    accessor = MemoryStateAccessor()
    client = QueuedClient([tool_call_response("resp_1", "call_1"), text_response("resp_text", "It is 22 degrees.")])

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "What is the weather in Tokyo?",
            "tools": [manual_weather_tool],
            "tool_choice": "required",
            "state": accessor,
        },
    ).get_response()

    assert len(client.requests) == 1
    assert accessor.stored is not None
    assert accessor.stored.status == "awaiting_client_tools"
    assert accessor.stored.consumed_forced_tool_choice_key == canonicalize_key_material("required")

    resumed_text = await call_model(
        client,
        {
            "model": "test-model",
            "input": [{"type": "function_call_output", "callId": "call_1", "output": '{"temperature": 22}'}],
            "tools": [manual_weather_tool],
            "tool_choice": "required",
            "state": accessor,
        },
    ).get_text()

    assert resumed_text == "It is 22 degrees."
    assert len(client.requests) == 2
    assert client.requests[1]["tool_choice"] == "auto"
