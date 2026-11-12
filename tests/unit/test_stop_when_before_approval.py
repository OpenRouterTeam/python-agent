"""Port of upstream `tests/unit/stop-when-before-approval.test.ts`.

With `allow_final_response=False`, a stop_when that is already met on the
first response means no tool from that response runs. It must be evaluated
before the approval check, so pausing for an approval-gated call does not
execute its auto-approved siblings (they are recorded as skipped instead).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from openrouter_agent import call_model, tool
from openrouter_agent.tool_types import Tool, UnsentToolResult
from tests._fixtures import MemoryStateAccessor, QueuedClient, function_call_item, make_response


class VArgs(BaseModel):
    v: str


class OkOutput(BaseModel):
    ok: bool


class Counter:
    """Counts executions; a class so the return value is explicit (see
    tests/unit/test_tool_execution_once.py)."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, params: Any, ctx: Any) -> Dict[str, Any]:
        self.count += 1
        return {"ok": True}


def _make_tools() -> Tuple[List[Tool], Counter, Counter]:
    auto_execute = Counter()
    gated_execute = Counter()
    auto = tool(name="auto", input_schema=VArgs, output_schema=OkOutput, execute=auto_execute)
    gated = tool(name="gated", input_schema=VArgs, output_schema=OkOutput, require_approval=True, execute=gated_execute)
    return [auto, gated], auto_execute, gated_execute


def _client() -> QueuedClient:
    return QueuedClient(
        [
            make_response(
                "resp_1",
                [
                    function_call_item("call_auto", "auto", '{"v":"x"}'),
                    function_call_item("call_gated", "gated", '{"v":"x"}'),
                ],
            )
        ]
    )


async def test_does_not_run_auto_sibling_of_gated_call_when_stop_when_already_met() -> None:
    tools, auto_execute, gated_execute = _make_tools()
    accessor = MemoryStateAccessor()
    client = _client()

    result = call_model(
        client,
        {
            "model": "test-model",
            "input": "go",
            "tools": tools,
            "state": accessor,
            "stop_when": lambda options: True,
            "allow_final_response": False,
        },
    )
    await result.get_response()

    assert auto_execute.count == 0
    assert gated_execute.count == 0
    assert len(client.requests) == 1

    saved = accessor.stored
    assert saved is not None
    assert saved.status == "awaiting_approval"
    assert [call.id for call in saved.pending_tool_calls or []] == ["call_gated"]
    # The skipped sibling still gets a matching output for the resume request.
    assert saved.unsent_tool_results == [
        UnsentToolResult(
            call_id="call_auto",
            name="auto",
            output=None,
            error="Tool execution skipped: stop condition met",
        )
    ]
    assert await result.requires_approval() is True


async def test_still_runs_auto_sibling_when_stop_when_is_not_met() -> None:
    tools, auto_execute, gated_execute = _make_tools()
    accessor = MemoryStateAccessor()
    client = _client()

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "go",
            "tools": tools,
            "state": accessor,
            "stop_when": lambda options: False,
            "allow_final_response": False,
        },
    ).get_response()

    assert auto_execute.count == 1
    assert gated_execute.count == 0
    assert len(client.requests) == 1
    saved = accessor.stored
    assert saved is not None
    assert saved.status == "awaiting_approval"
    assert [call.id for call in saved.pending_tool_calls or []] == ["call_gated"]
    assert [result.call_id for result in saved.unsent_tool_results or []] == ["call_auto"]
