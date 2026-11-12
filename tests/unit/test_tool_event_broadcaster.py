"""Ports `packages/agent/tests/unit/tool-event-broadcaster.test.ts`.

Determinism: upstream's ``setTimeout`` gaps are replaced by explicit hand-offs
(an ``asyncio.Event`` per received item, or :func:`_park`, which yields to the
loop until the consuming task is blocked on its waiter). No wall-clock sleeps.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, TypeVar

import pytest

from openrouter_agent.tool_event_broadcaster import ToolEventBroadcaster

T = TypeVar("T")

_HANG_GUARD_SECONDS = 5


async def _collect(iterator: AsyncIterator[T]) -> List[T]:
    return [value async for value in iterator]


async def _park(task: "asyncio.Task[Any]") -> None:
    for _ in range(10):
        await asyncio.sleep(0)
    assert not task.done(), "consumer finished before it could be parked"


async def test_retains_full_replay_after_completion_by_default() -> None:
    broadcaster: ToolEventBroadcaster[int] = ToolEventBroadcaster()
    first = broadcaster.create_consumer()

    broadcaster.push(1)
    broadcaster.complete()
    assert await _collect(first) == [1]

    second = broadcaster.create_consumer()
    assert await _collect(second) == [1]


async def test_compacts_active_consumer_history_at_the_watermark() -> None:
    broadcaster: ToolEventBroadcaster[int] = ToolEventBroadcaster("active-consumers")
    first = broadcaster.create_consumer()

    broadcaster.push(1)
    broadcaster.push(2)
    broadcaster.complete()
    assert await first.__anext__() == 1
    second = broadcaster.create_consumer()
    assert await _collect(first) == [2]
    assert await _collect(second) == [2]


# -- single consumer -----------------------------------------------------------


async def test_delivers_events_to_a_single_consumer() -> None:
    broadcaster: ToolEventBroadcaster[int] = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    broadcaster.push(1)
    broadcaster.push(2)
    broadcaster.push(3)
    broadcaster.complete()

    assert await _collect(consumer) == [1, 2, 3]


async def test_handles_empty_stream() -> None:
    broadcaster: ToolEventBroadcaster[str] = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    broadcaster.complete()

    assert await _collect(consumer) == []


async def test_handles_consumer_cancellation_via_aclose() -> None:
    broadcaster: ToolEventBroadcaster[int] = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    broadcaster.push(1)
    broadcaster.push(2)

    assert await consumer.__anext__() == 1

    await consumer.aclose()

    with pytest.raises(StopAsyncIteration):
        await consumer.__anext__()


# -- multiple consumers --------------------------------------------------------


async def test_delivers_same_events_to_multiple_consumers() -> None:
    broadcaster: ToolEventBroadcaster[str] = ToolEventBroadcaster()
    consumer1 = broadcaster.create_consumer()
    consumer2 = broadcaster.create_consumer()

    broadcaster.push("a")
    broadcaster.push("b")
    broadcaster.complete()

    results1, results2 = await asyncio.gather(_collect(consumer1), _collect(consumer2))

    assert results1 == ["a", "b"]
    assert results2 == ["a", "b"]


async def test_allows_consumers_at_different_read_positions() -> None:
    broadcaster: ToolEventBroadcaster[int] = ToolEventBroadcaster()
    consumer1 = broadcaster.create_consumer()

    broadcaster.push(1)
    broadcaster.push(2)

    assert await consumer1.__anext__() == 1

    # Consumer 2 joins after events were pushed.
    consumer2 = broadcaster.create_consumer()

    broadcaster.push(3)
    broadcaster.complete()

    # Consumer 1 continues from position 1; consumer 2 replays from 0.
    assert await _collect(consumer1) == [2, 3]
    assert await _collect(consumer2) == [1, 2, 3]


# -- async waiting -------------------------------------------------------------


async def test_waits_for_events_when_consumer_is_ahead_of_buffer() -> None:
    broadcaster: ToolEventBroadcaster[int] = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    consume = asyncio.ensure_future(_collect(consumer))

    # Push only after the consumer is provably waiting.
    await _park(consume)
    broadcaster.push(1)
    await _park(consume)
    broadcaster.push(2)
    broadcaster.complete()

    assert await asyncio.wait_for(consume, _HANG_GUARD_SECONDS) == [1, 2]


async def test_handles_rapid_push_consume_interleaving() -> None:
    broadcaster: ToolEventBroadcaster[int] = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()
    received: List[int] = []
    arrived = [asyncio.Event() for _ in range(10)]

    async def consume() -> None:
        async for event in consumer:
            received.append(event)
            arrived[event].set()

    task = asyncio.ensure_future(consume())

    # Each push is handed off to the consumer before the next one.
    for index in range(10):
        broadcaster.push(index)
        await asyncio.wait_for(arrived[index].wait(), _HANG_GUARD_SECONDS)
    broadcaster.complete()

    await asyncio.wait_for(task, _HANG_GUARD_SECONDS)
    assert received == list(range(10))


# -- error handling ------------------------------------------------------------


async def test_propagates_errors_to_consumers_after_buffered_events() -> None:
    broadcaster: ToolEventBroadcaster[int] = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    broadcaster.push(1)
    broadcaster.complete(RuntimeError("Test error"))

    results: List[int] = []
    with pytest.raises(RuntimeError, match="^Test error$"):
        async for event in consumer:
            results.append(event)

    assert results == [1]


async def test_propagates_errors_to_waiting_consumers() -> None:
    broadcaster: ToolEventBroadcaster[int] = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    consume = asyncio.ensure_future(_collect(consumer))
    await _park(consume)
    broadcaster.complete(RuntimeError("Async error"))

    with pytest.raises(RuntimeError, match="Async error"):
        await asyncio.wait_for(consume, _HANG_GUARD_SECONDS)


async def test_error_alias_completes_with_error() -> None:
    broadcaster: ToolEventBroadcaster[int] = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()
    broadcaster.error(ValueError("bad"))
    assert broadcaster.is_complete is True
    with pytest.raises(ValueError, match="bad"):
        await consumer.__anext__()


# -- ignore after complete -----------------------------------------------------


async def test_ignores_pushes_after_complete() -> None:
    broadcaster: ToolEventBroadcaster[int] = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    broadcaster.push(1)
    broadcaster.complete()
    broadcaster.push(2)  # ignored

    assert await _collect(consumer) == [1]


# -- completion between iterations ---------------------------------------------


async def test_handles_completion_between_consumer_iterations() -> None:
    broadcaster: ToolEventBroadcaster[int] = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    broadcaster.push(1)
    assert await consumer.__anext__() == 1

    broadcaster.complete()

    with pytest.raises(StopAsyncIteration):
        await consumer.__anext__()


async def test_handles_completion_with_remaining_buffered_events() -> None:
    broadcaster: ToolEventBroadcaster[int] = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    broadcaster.push(1)
    broadcaster.push(2)
    broadcaster.push(3)

    assert await consumer.__anext__() == 1

    broadcaster.complete()

    assert await consumer.__anext__() == 2
    assert await consumer.__anext__() == 3
    with pytest.raises(StopAsyncIteration):
        await consumer.__anext__()


# -- typed events --------------------------------------------------------------


async def test_works_with_typed_tool_events() -> None:
    broadcaster: ToolEventBroadcaster[Dict[str, Any]] = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    broadcaster.push({"type": "delta", "content": "test"})
    broadcaster.push({"type": "preliminary_result", "toolCallId": "call_123", "result": {"progress": 50}})
    broadcaster.complete()

    assert await _collect(consumer) == [
        {"type": "delta", "content": "test"},
        {"type": "preliminary_result", "toolCallId": "call_123", "result": {"progress": 50}},
    ]


# -- lifecycle compaction (active-consumers) ------------------------------------


async def test_drops_backlog_once_every_consumer_departs_late_joiner_starts_at_watermark() -> None:
    broadcaster: ToolEventBroadcaster[int] = ToolEventBroadcaster("active-consumers")
    only = broadcaster.create_consumer()
    broadcaster.push(1)
    broadcaster.push(2)
    assert await only.__anext__() == 1
    assert await only.__anext__() == 2
    await only.aclose()

    # Nobody is attached: retained history is dropped, so a fresh consumer
    # starts at the watermark (retention would replay [1, 2] first).
    assert broadcaster.find_last_buffered(lambda _value: True) is None
    broadcaster.push(3)
    broadcaster.complete()
    late = broadcaster.create_consumer()
    assert await _collect(late) == [3]


async def test_backlog_survives_while_at_least_one_consumer_remains() -> None:
    broadcaster: ToolEventBroadcaster[int] = ToolEventBroadcaster("active-consumers")
    fast = broadcaster.create_consumer()
    slow = broadcaster.create_consumer()
    broadcaster.push(1)
    broadcaster.push(2)
    assert await fast.__anext__() == 1
    assert await fast.__anext__() == 2
    await fast.aclose()

    # The slow consumer still reads from position 0.
    assert await slow.__anext__() == 1
    assert await slow.__anext__() == 2
    broadcaster.complete()
    with pytest.raises(StopAsyncIteration):
        await slow.__anext__()


async def test_full_mode_retains_backlog_after_every_consumer_departs() -> None:
    broadcaster: ToolEventBroadcaster[int] = ToolEventBroadcaster()
    only = broadcaster.create_consumer()
    broadcaster.push(1)
    assert await only.__anext__() == 1
    await only.aclose()
    broadcaster.complete()

    assert await _collect(broadcaster.create_consumer()) == [1]


async def test_aclose_wakes_a_pending_next_instead_of_hanging() -> None:
    broadcaster: ToolEventBroadcaster[int] = ToolEventBroadcaster("active-consumers")
    consumer = broadcaster.create_consumer()
    broadcaster.push(1)
    assert await consumer.__anext__() == 1

    pending = asyncio.ensure_future(consumer.__anext__())
    await _park(pending)
    await consumer.aclose()
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(pending, _HANG_GUARD_SECONDS)


async def test_athrow_rejects_a_pending_next_with_the_same_error() -> None:
    broadcaster: ToolEventBroadcaster[int] = ToolEventBroadcaster("active-consumers")
    consumer = broadcaster.create_consumer()
    broadcaster.push(1)
    assert await consumer.__anext__() == 1

    pending = asyncio.ensure_future(consumer.__anext__())
    await _park(pending)
    with pytest.raises(RuntimeError, match="boom"):
        await consumer.athrow(RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await asyncio.wait_for(pending, _HANG_GUARD_SECONDS)
