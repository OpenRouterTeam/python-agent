from __future__ import annotations

import asyncio

from openrouter_agent import (
    ToolContextStore,
    append_to_messages,
    build_tool_execute_context,
    create_initial_state,
    partition_tool_calls,
    tool,
    update_state,
)
from openrouter_agent.tool_event_broadcaster import ToolEventBroadcaster
from openrouter_agent.tool_types import ParsedToolCall


async def test_conversation_state_is_immutable_and_partitions_approval() -> None:
    state = create_initial_state("conv_test")
    updated = update_state(state, {"status": "complete"})
    assert state.status == "in_progress"
    assert updated.status == "complete"
    assert append_to_messages("hello", [{"role": "assistant", "content": "hi"}])[0]["role"] == "user"

    sensitive = tool(name="delete", input_schema=dict, execute=lambda params, ctx: {}, require_approval=True)
    safe = tool(name="read", input_schema=dict, execute=lambda params, ctx: {})
    split = await partition_tool_calls(
        [ParsedToolCall("1", "delete", {}), ParsedToolCall("2", "read", {})],
        [sensitive, safe],
        {"number_of_turns": 1},
    )
    assert [call.name for call in split["requires_approval"]] == ["delete"]
    assert [call.name for call in split["auto_execute"]] == ["read"]


def test_tool_context_local_and_shared_mutations_are_live() -> None:
    store = ToolContextStore({"lookup": {"token": "old"}, "shared": {"count": 1}})
    lookup = tool(name="lookup", input_schema=dict, execute=lambda params, ctx: {})
    ctx = build_tool_execute_context(lookup, {"number_of_turns": 1}, store)

    ctx["set_context"]({"token": "new"})
    ctx["set_shared_context"]({"count": 2})

    assert store.get_tool_context("lookup") == {"token": "new"}
    assert store.get_tool_context("shared") == {"count": 2}


async def test_tool_event_broadcaster_replays_to_multiple_consumers() -> None:
    broadcaster = ToolEventBroadcaster()
    first = broadcaster.create_consumer()
    broadcaster.push({"n": 1})
    broadcaster.push({"n": 2})
    second = broadcaster.create_consumer()
    broadcaster.complete()

    assert [item async for item in first] == [{"n": 1}, {"n": 2}]
    assert [item async for item in second] == [{"n": 1}, {"n": 2}]


async def test_tool_event_broadcaster_waits_for_future_events() -> None:
    broadcaster = ToolEventBroadcaster()

    async def consume():
        return [item async for item in broadcaster.create_consumer()]

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    broadcaster.push("later")
    broadcaster.complete()
    assert await task == ["later"]
