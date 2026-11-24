"""Ported from upstream tests/unit/hooks-contract-fixes.test.ts, "blocking hooks
fail closed when the chain is aborted" (#160).

If a PreToolUse / UserPromptSubmit chain stops early because it was aborted,
the emit result reports `blocked=True`, so handlers that never ran (e.g. a
later deny guard) cannot be skipped into an approval.

Not ported: the fake-timer variant where a detached `async_timeout_ms` expiry
aborts the chain. It needs wall-clock interleaving between the timeout and a
slow handler; the abort path it reaches (`cancel_event`) is the one
`abort_inflight()` drives below.
"""

from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

from openrouter_agent import HookEntry, HookName, HooksManager, call_model, tool
from openrouter_agent.hooks_types import AsyncOutput
from tests._fixtures import QueuedClient, text_response, tool_call_response, recording


async def test_reports_pre_tool_use_as_blocked_when_abort_inflight_fires_mid_chain() -> None:
    manager = HooksManager()
    guard_calls: List[Any] = []
    manager.on(HookName.PreToolUse.value, HookEntry(handler=lambda _p, _c: manager.abort_inflight()))
    manager.on(
        HookName.PreToolUse.value,
        HookEntry(handler=recording(guard_calls, {"block": "policy forbids this call"})),
    )

    result = await manager.emit("PreToolUse", {"tool_name": "t", "tool_input": {}}, tool_name="t")

    assert guard_calls == []
    assert result.blocked is True


async def test_reports_user_prompt_submit_as_blocked_when_aborted_mid_chain() -> None:
    manager = HooksManager()
    manager.on(HookName.UserPromptSubmit.value, HookEntry(handler=lambda _p, _c: manager.abort_inflight()))
    manager.on(HookName.UserPromptSubmit.value, HookEntry(handler=lambda _p, _c: {"reject": "nope"}))

    result = await manager.emit("UserPromptSubmit", {"prompt": "hi"})

    assert result.blocked is True


async def test_still_runs_every_pre_tool_use_handler_when_detached_work_settles_in_time() -> None:
    manager = HooksManager()
    guard_calls: List[Any] = []
    done = asyncio.get_running_loop().create_future()
    done.set_result(None)
    manager.on(HookName.PreToolUse.value, HookEntry(handler=lambda _p, _c: AsyncOutput(work=done, async_timeout_ms=10)))
    manager.on(HookName.PreToolUse.value, HookEntry(handler=recording(guard_calls, None, lambda _p: 1)))

    result = await manager.emit("PreToolUse", {"tool_name": "t", "tool_input": {}}, tool_name="t")
    await manager.drain()

    assert guard_calls == [1]
    assert result.blocked is False


async def test_does_not_mark_non_blocking_hooks_as_blocked_when_aborted() -> None:
    manager = HooksManager()
    manager.on(HookName.PostToolUse.value, HookEntry(handler=lambda _p, _c: manager.abort_inflight()))
    manager.on(HookName.PostToolUse.value, HookEntry(handler=lambda _p, _c: None))

    result = await manager.emit(
        "PostToolUse",
        {"tool_name": "t", "tool_input": {}, "tool_output": None, "duration_ms": 1},
        tool_name="t",
    )

    assert result.blocked is False


async def test_an_aborted_pre_tool_use_chain_blocks_the_tool_call_in_the_loop() -> None:
    executed: List[Any] = []
    hooks = HooksManager()
    hooks.on(HookName.PreToolUse.value, HookEntry(handler=lambda _p, _c: hooks.abort_inflight()))
    # A later guard that never runs: the chain must not read as an approval.
    hooks.on(HookName.PreToolUse.value, HookEntry(handler=lambda _p, _c: None))
    danger = tool(name="danger", input_schema=dict, execute=recording(executed, {"ok": True}))
    client = QueuedClient([tool_call_response("r1", "danger", call_id="call_1"), text_response("r2", "done")])

    await call_model(client, {"model": "m", "input": "go", "tools": [danger], "hooks": hooks}).get_text()

    assert executed == []
    outputs = [
        item
        for item in client.requests[1]["input"]
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    assert [(o["call_id"], o["output"]) for o in outputs] == [("call_1", '{"error":"Blocked by PreToolUse hook"}')]


async def test_an_aborted_user_prompt_submit_chain_rejects_the_prompt() -> None:
    hooks = HooksManager()
    hooks.on(HookName.UserPromptSubmit.value, HookEntry(handler=lambda _p, _c: hooks.abort_inflight()))
    hooks.on(HookName.UserPromptSubmit.value, HookEntry(handler=lambda _p, _c: None))
    client = QueuedClient([])

    with pytest.raises(ValueError, match="Prompt rejected by hook"):
        await call_model(client, {"model": "m", "input": "hi", "hooks": hooks}).get_text()
    assert client.requests == []
