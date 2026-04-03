"""Tests for concurrent tool execution in the tool orchestrator."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock

import anyio
import pytest
from pydantic import BaseModel

from openrouter_agent import (
    ToolContextStore,
    ToolEventBroadcaster,
    tool,
)
from openrouter_agent._tool_orchestrator import run_tool_loop
from openrouter_agent._types import ResponseStreamEvent, ToolWithExecute


class SlowInput(BaseModel):
    label: str


class SlowOutput(BaseModel):
    label: str


def _make_slow_tool(name: str) -> ToolWithExecute:
    """Create a tool that sleeps for 0.05s then returns."""

    async def _execute(params: SlowInput, ctx: Any) -> SlowOutput:
        await anyio.sleep(0.05)
        return SlowOutput(label=params.label)

    t = tool(
        name=name,
        description=f"Slow tool {name}",
        input_schema=SlowInput,
        output_schema=SlowOutput,
        execute=_execute,
    )
    assert isinstance(t, ToolWithExecute)
    return t


def _make_api_response(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a fake API response containing function_call items."""
    output: list[dict[str, Any]] = []
    for tc in tool_calls:
        output.append(
            {
                "type": "function_call",
                "id": tc["id"],
                "call_id": tc["id"],
                "name": tc["name"],
                "arguments": f'{{"label": "{tc["label"]}"}}',
            }
        )
    return {"output": output, "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}


def _make_client_returning(*responses: dict[str, Any]) -> Any:
    """Create a mock client that returns predefined responses in order."""
    call_iter = iter(responses)

    async def _send(**kwargs: Any) -> dict[str, Any]:
        return next(call_iter)

    client = AsyncMock()
    client.beta.responses.send_async = _send
    return client


@pytest.mark.anyio
async def test_concurrent_tool_execution_wall_clock() -> None:
    """Two tools each sleeping 0.05s should complete in under 0.1s total."""
    tool_a = _make_slow_tool("tool_a")
    tool_b = _make_slow_tool("tool_b")

    # First response triggers two tool calls; second response has no tool calls (stops loop).
    first_response = _make_api_response(
        [
            {"id": "call_1", "name": "tool_a", "label": "a"},
            {"id": "call_2", "name": "tool_b", "label": "b"},
        ]
    )
    second_response: dict[str, Any] = {
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "done"}]}
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "finish_reason": "stop",
    }

    client = _make_client_returning(first_response, second_response)
    broadcaster: ToolEventBroadcaster[ResponseStreamEvent] = ToolEventBroadcaster()
    context_store = ToolContextStore()

    start = time.monotonic()
    steps = await run_tool_loop(
        client=client,
        request_params={"model": "test", "input": "hi"},
        tools=[tool_a, tool_b],
        stop_conditions=[],
        context_store=context_store,
        broadcaster=broadcaster,
        max_steps=5,
    )
    elapsed = time.monotonic() - start

    # Both tools ran: we should have 2 steps (tool step + final text step).
    assert len(steps) == 2
    assert len(steps[0].tool_results) == 2
    assert steps[0].tool_results[0].tool_name == "tool_a"
    assert steps[0].tool_results[1].tool_name == "tool_b"

    # Concurrent execution: wall clock should be ~0.05s, not ~0.10s.
    assert elapsed < 0.1, f"Expected concurrent execution (<0.1s), but took {elapsed:.3f}s"
