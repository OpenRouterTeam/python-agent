"""Ports tests/unit/tool-name-events.test.ts.

Upstream drives `executeToolsIfNeeded()` on a ModelResult with stubbed private
internals; this port has no such method (the tool loop is inlined in
`ModelResult._run`), so the same events are observed through `call_model`'s
`get_full_responses_stream()` / `get_tool_stream()` (contract, Test Parity).

Each upstream case is split in two: event type, order, count, `toolCallId`,
`source`, `result`; and the `toolName` field every tool event carries.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Sequence

import pytest
from pydantic import BaseModel

from openrouter_agent import call_model, tool
from openrouter_agent.tool_types import is_tool_preliminary_result_event, is_tool_result_event
from tests._fixtures import QueuedClient, function_call_item, make_response, text_response


class TextIn(BaseModel):
    text: str


class TextOut(BaseModel):
    text: str


class Empty(BaseModel):
    pass


class Stage(BaseModel):
    stage: str


class Done(BaseModel):
    done: bool


class Q(BaseModel):
    q: str


class Answer(BaseModel):
    answer: str


class Ok(BaseModel):
    ok: bool


def _dump(value: Any) -> Any:
    return value.model_dump() if hasattr(value, "model_dump") else value


async def _echo(params: Any, ctx: Any) -> Dict[str, Any]:
    return {"text": params.text}


async def _progress(params: Any, ctx: Any) -> AsyncIterator[Dict[str, Any]]:
    yield {"stage": "one"}
    yield {"stage": "two"}
    yield {"done": True}


async def _hitl(params: Any, ctx: Any) -> Dict[str, Any]:
    return {"answer": "42"}


async def _boom(params: Any, ctx: Any) -> Dict[str, Any]:
    raise RuntimeError("explode")


def echo_tool() -> Dict[str, Any]:
    return tool(name="echo", input_schema=TextIn, output_schema=TextOut, execute=_echo)


def progress_tool() -> Dict[str, Any]:
    return tool(name="progress_tool", input_schema=Empty, event_schema=Stage, output_schema=Done, execute=_progress)


def hitl_tool() -> Dict[str, Any]:
    return tool(name="hitl_tool", input_schema=Q, output_schema=Answer, on_tool_called=_hitl)


def boom_tool() -> Dict[str, Any]:
    return tool(name="boom", input_schema=Empty, output_schema=Ok, execute=_boom)


SCENARIOS: Dict[str, Dict[str, Any]] = {
    "regular": {"tool": echo_tool, "call_id": "call_echo", "name": "echo", "args": {"text": "hi"}},
    "generator": {"tool": progress_tool, "call_id": "call_progress", "name": "progress_tool", "args": {}},
    "hitl": {"tool": hitl_tool, "call_id": "call_hitl", "name": "hitl_tool", "args": {"q": "life?"}},
    "error": {"tool": boom_tool, "call_id": "call_boom", "name": "boom", "args": {}},
}


async def _run(scenario: str) -> List[Any]:
    spec = SCENARIOS[scenario]
    client = QueuedClient(
        [
            make_response("resp_test", [function_call_item(spec["call_id"], spec["name"], json.dumps(spec["args"]))]),
            text_response("resp_final", "done"),
        ]
    )
    result = call_model(client, {"model": "test-model", "input": "hello", "tools": [spec["tool"]()]})
    events = [event async for event in result.get_full_responses_stream()]
    # Exactly one tool round then the final turn.
    assert len(client.requests) == 2
    return events


def _tool_events(events: Sequence[Any]) -> List[Dict[str, Any]]:
    return [
        e for e in events if isinstance(e, dict) and (is_tool_result_event(e) or is_tool_preliminary_result_event(e))
    ]


# -- parts of each upstream case this port implements -------------------------


async def test_tool_result_for_regular_execute_tools() -> None:
    events = await _run("regular")

    tool_results = [e for e in events if isinstance(e, dict) and is_tool_result_event(e)]
    assert len(tool_results) == 1
    assert tool_results[0]["type"] == "tool.result"
    assert tool_results[0]["toolCallId"] == "call_echo"
    assert tool_results[0]["source"] == "client"
    assert _dump(tool_results[0]["result"]) == {"text": "hi"}


async def test_preliminary_and_final_generator_events() -> None:
    events = await _run("generator")

    tool_events = _tool_events(events)
    assert [e["type"] for e in tool_events] == [
        "tool.preliminary_result",
        "tool.preliminary_result",
        "tool.result",
    ]
    prelims, final = tool_events[:2], tool_events[2]
    assert [e["toolCallId"] for e in tool_events] == ["call_progress"] * 3
    assert [_dump(e["result"]) for e in prelims] == [{"stage": "one"}, {"stage": "two"}]
    assert _dump(final["result"]) == {"done": True}
    assert "preliminaryResults" not in final
    assert "preliminary_results" not in final


async def test_legacy_tool_stream_projection_for_generator_events() -> None:
    client = QueuedClient(
        [
            make_response("resp_test", [function_call_item("call_progress", "progress_tool", "{}")]),
            text_response("resp_final", "done"),
        ]
    )
    result = call_model(client, {"model": "test-model", "input": "hello", "tools": [progress_tool()]})

    projected = [e async for e in result.get_tool_stream() if e["type"] in {"preliminary_result", "tool_result"}]

    assert [e["type"] for e in projected] == ["preliminary_result", "preliminary_result", "tool_result"]
    assert [e["toolCallId"] for e in projected] == ["call_progress"] * 3


async def test_tool_result_for_hitl_auto_resolve_path() -> None:
    events = await _run("hitl")

    tool_results = [e for e in events if isinstance(e, dict) and is_tool_result_event(e)]
    assert len(tool_results) == 1
    assert tool_results[0]["toolCallId"] == "call_hitl"
    assert _dump(tool_results[0]["result"]) == {"answer": "42"}


async def test_tool_result_for_rejected_or_error_tools() -> None:
    events = await _run("error")

    tool_results = [e for e in events if isinstance(e, dict) and is_tool_result_event(e)]
    assert len(tool_results) == 1
    assert tool_results[0]["type"] == "tool.result"
    assert tool_results[0]["toolCallId"] == "call_boom"
    assert tool_results[0]["result"] == {"error": "explode"}


def test_preserves_literal_names_on_manual_regular_generator_and_hitl_factory_tools() -> None:
    async def gen(params: Any, ctx: Any) -> AsyncIterator[Dict[str, Any]]:
        yield {"ok": True}

    regular = tool(name="alpha", input_schema=Empty, execute=lambda params, ctx: 1)
    generator = tool(name="beta", input_schema=Empty, event_schema=Stage, output_schema=Ok, execute=gen)
    manual = tool(name="gamma", input_schema=Empty, execute=False)
    hitl = tool(name="delta", input_schema=Empty, output_schema=Done, on_tool_called=lambda params, ctx: None)

    assert [t["function"]["name"] for t in (regular, generator, manual, hitl)] == ["alpha", "beta", "gamma", "delta"]


# -- toolName ---------------------------------------------------------


@pytest.mark.parametrize("scenario", ["regular", "generator", "hitl", "error"])
async def test_tool_events_carry_tool_name(scenario: str) -> None:
    events = await _run(scenario)

    tool_events = _tool_events(events)
    expected = SCENARIOS[scenario]["name"]
    # Exact event sequence per scenario: the generator emits two preliminary
    # results before its single terminal result; the others emit one result.
    expected_types = (
        ["tool.preliminary_result", "tool.preliminary_result", "tool.result"]
        if scenario == "generator"
        else ["tool.result"]
    )
    assert [e["type"] for e in tool_events] == expected_types
    assert [e.get("toolName") for e in tool_events] == [expected] * len(expected_types)


async def test_legacy_tool_stream_projection_carries_tool_name() -> None:
    client = QueuedClient(
        [
            make_response("resp_test", [function_call_item("call_progress", "progress_tool", "{}")]),
            text_response("resp_final", "done"),
        ]
    )
    result = call_model(client, {"model": "test-model", "input": "hello", "tools": [progress_tool()]})

    projected = [e async for e in result.get_tool_stream() if e["type"] in {"preliminary_result", "tool_result"}]

    assert len(projected) == 3
    assert [e.get("toolName") for e in projected] == ["progress_tool"] * 3
