"""Ported from upstream tests/unit/stop-when-before-approval.test.ts (#143).

With `allow_final_response=False` and `stop_when` already met on the first
response, pausing for an approval-gated call must not execute its auto-approved
siblings; they are saved as skipped results instead.
"""

from __future__ import annotations

from typing import Any, List, Tuple

from pydantic import BaseModel

from openrouter_agent import call_model, tool
from tests._fixtures import MemoryStateAccessor, QueuedClient, function_call_item, make_response


class _Input(BaseModel):
    v: str


def _tools() -> Tuple[List[Any], List[str], List[str]]:
    auto_calls: List[str] = []
    gated_calls: List[str] = []

    def auto_execute(params: _Input, _ctx: Any) -> Any:
        auto_calls.append(params.v)
        return {"ok": True}

    def gated_execute(params: _Input, _ctx: Any) -> Any:
        gated_calls.append(params.v)
        return {"ok": True}

    auto = tool(name="auto", input_schema=_Input, output_schema=dict, execute=auto_execute)
    gated = tool(name="gated", input_schema=_Input, output_schema=dict, require_approval=True, execute=gated_execute)
    return [auto, gated], auto_calls, gated_calls


def _response() -> Any:
    return make_response(
        "resp_1",
        [function_call_item("call_auto", "auto", '{"v":"x"}'), function_call_item("call_gated", "gated", '{"v":"x"}')],
    )


async def test_does_not_run_an_auto_executable_sibling_when_stop_when_is_already_met() -> None:
    tools, auto_calls, gated_calls = _tools()
    state = MemoryStateAccessor()
    client = QueuedClient([_response()])

    result = call_model(
        client,
        {
            "model": "test-model",
            "input": "go",
            "tools": tools,
            "state": state,
            "stop_when": lambda _ctx: True,
            "allow_final_response": False,
        },
    )
    await result.get_response()

    assert auto_calls == []
    assert gated_calls == []
    assert len(client.requests) == 1
    saved = state.stored
    assert saved is not None
    assert saved.status == "awaiting_approval"
    assert [c.id for c in saved.pending_tool_calls or []] == ["call_gated"]
    # The skipped sibling still gets a matching output for the resume request.
    assert [(r.call_id, r.name, r.output, r.error) for r in saved.unsent_tool_results or []] == [
        ("call_auto", "auto", None, "Tool execution skipped: stop condition met")
    ]
    assert await result.requires_approval() is True


async def test_still_runs_the_auto_executable_sibling_when_stop_when_is_not_met() -> None:
    tools, auto_calls, _gated_calls = _tools()
    state = MemoryStateAccessor()

    await call_model(
        QueuedClient([_response()]),
        {
            "model": "test-model",
            "input": "go",
            "tools": tools,
            "state": state,
            "stop_when": lambda _ctx: False,
            "allow_final_response": False,
        },
    ).get_response()

    assert auto_calls == ["x"]
    saved = state.stored
    assert saved is not None
    assert saved.status == "awaiting_approval"
    assert [c.id for c in saved.pending_tool_calls or []] == ["call_gated"]
    assert [r.call_id for r in saved.unsent_tool_results or []] == ["call_auto"]
