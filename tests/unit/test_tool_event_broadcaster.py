"""Ports `packages/agent/tests/unit/tool-event-broadcaster.test.ts` (whole file).

Iterator mapping: upstream `next()` -> `anext()` (done -> `StopAsyncIteration`),
`return()` -> `aclose()`, `throw()` -> `athrow()`.

Determinism: upstream's `setTimeout` waits are replaced by `_park`, which yields
to the loop once and asserts the consumer task is still pending (provably
waiting), and by `asyncio.Event` hand-offs for interleaving.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

from openrouter_agent.tool_event_broadcaster import ToolEventBroadcaster


async def collect(iterator: AsyncIterator[Any]) -> List[Any]:
    return [item async for item in iterator]


async def _park(task: "asyncio.Task[Any]") -> None:
    await asyncio.sleep(0)
    assert not task.done(), "consumer should be parked waiting for events"


async def test_retains_full_replay_after_completion_by_default() -> None:
    broadcaster = ToolEventBroadcaster()
    first = broadcaster.create_consumer()

    broadcaster.push(1)
    broadcaster.complete()
    assert await collect(first) == [1]

    second = broadcaster.create_consumer()
    assert await collect(second) == [1]


async def test_compacts_active_consumer_history_at_the_watermark() -> None:
    broadcaster = ToolEventBroadcaster("active-consumers")
    first = broadcaster.create_consumer()

    broadcaster.push(1)
    broadcaster.push(2)
    broadcaster.complete()
    assert await anext(first) == 1
    second = broadcaster.create_consumer()
    assert await collect(first) == [2]
    assert await collect(second) == [2]


# -- single consumer ------------------------------------------------------------


async def test_single_consumer_delivers_events() -> None:
    broadcaster = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    broadcaster.push(1)
    broadcaster.push(2)
    broadcaster.push(3)
    broadcaster.complete()

    assert await collect(consumer) == [1, 2, 3]


async def test_single_consumer_handles_empty_stream() -> None:
    broadcaster = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    broadcaster.complete()

    assert await collect(consumer) == []


async def test_single_consumer_cancellation_via_aclose() -> None:
    broadcaster = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    broadcaster.push(1)
    broadcaster.push(2)

    assert await anext(consumer) == 1

    await consumer.aclose()

    with pytest.raises(StopAsyncIteration):
        await anext(consumer)


# -- multiple consumers ---------------------------------------------------------


async def test_multiple_consumers_receive_same_events() -> None:
    broadcaster = ToolEventBroadcaster()
    consumer1 = broadcaster.create_consumer()
    consumer2 = broadcaster.create_consumer()

    broadcaster.push("a")
    broadcaster.push("b")
    broadcaster.complete()

    results1, results2 = await asyncio.gather(collect(consumer1), collect(consumer2))

    assert results1 == ["a", "b"]
    assert results2 == ["a", "b"]


async def test_multiple_consumers_at_different_read_positions() -> None:
    broadcaster = ToolEventBroadcaster()
    consumer1 = broadcaster.create_consumer()

    broadcaster.push(1)
    broadcaster.push(2)

    assert await anext(consumer1) == 1

    # Consumer 2 joins after events pushed.
    consumer2 = broadcaster.create_consumer()

    broadcaster.push(3)
    broadcaster.complete()

    # Consumer 1 continues from position 1.
    assert await collect(consumer1) == [2, 3]
    # Consumer 2 gets all events from position 0 (full replay).
    assert await collect(consumer2) == [1, 2, 3]


# -- async waiting --------------------------------------------------------------


async def test_waits_for_events_when_consumer_is_ahead_of_buffer() -> None:
    broadcaster = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    consume = asyncio.ensure_future(collect(consumer))

    # Push events only after the consumer is provably waiting.
    await _park(consume)
    broadcaster.push(1)
    await _park(consume)
    broadcaster.push(2)
    broadcaster.complete()

    assert await consume == [1, 2]


async def test_rapid_push_consume_interleaving() -> None:
    broadcaster = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()
    received: List[int] = []
    got_one = asyncio.Event()

    async def consume() -> None:
        async for event in consumer:
            received.append(event)
            got_one.set()

    task = asyncio.ensure_future(consume())

    for i in range(10):
        broadcaster.push(i)
        await got_one.wait()
        got_one.clear()
        assert received == list(range(i + 1))
    broadcaster.complete()

    await task
    assert received == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


# -- error handling -------------------------------------------------------------


async def test_propagates_errors_to_consumers() -> None:
    broadcaster = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    broadcaster.push(1)
    broadcaster.complete(RuntimeError("Test error"))

    results: List[int] = []
    caught: Optional[BaseException] = None
    try:
        async for event in consumer:
            results.append(event)
    except RuntimeError as exc:
        caught = exc

    assert results == [1]
    assert caught is not None
    assert str(caught) == "Test error"


async def test_propagates_errors_to_waiting_consumers() -> None:
    broadcaster = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    consume = asyncio.ensure_future(collect(consumer))

    # Complete with error while the consumer is waiting.
    await _park(consume)
    broadcaster.complete(RuntimeError("Async error"))

    with pytest.raises(RuntimeError, match="Async error"):
        await consume


# -- ignore after complete ------------------------------------------------------


async def test_ignores_pushes_after_complete() -> None:
    broadcaster = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    broadcaster.push(1)
    broadcaster.complete()
    broadcaster.push(2)  # ignored

    assert await collect(consumer) == [1]


# -- completion between iterations ----------------------------------------------


async def test_completion_between_consumer_iterations() -> None:
    broadcaster = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    broadcaster.push(1)
    assert await anext(consumer) == 1

    broadcaster.complete()

    with pytest.raises(StopAsyncIteration):
        await anext(consumer)


async def test_completion_with_remaining_buffered_events() -> None:
    broadcaster = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    broadcaster.push(1)
    broadcaster.push(2)
    broadcaster.push(3)

    assert await anext(consumer) == 1

    broadcaster.complete()

    assert await anext(consumer) == 2
    assert await anext(consumer) == 3
    with pytest.raises(StopAsyncIteration):
        await anext(consumer)


# -- typed events ---------------------------------------------------------------


async def test_typed_tool_events() -> None:
    broadcaster = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    delta: Dict[str, Any] = {"type": "delta", "content": "test"}
    preliminary: Dict[str, Any] = {
        "type": "preliminary_result",
        "toolCallId": "call_123",
        "result": {"progress": 50},
    }
    broadcaster.push(delta)
    broadcaster.push(preliminary)
    broadcaster.complete()

    events = await collect(consumer)

    assert len(events) == 2
    assert events[0] == {"type": "delta", "content": "test"}
    assert events[1] == {
        "type": "preliminary_result",
        "toolCallId": "call_123",
        "result": {"progress": 50},
    }


# -- lifecycle compaction (active-consumers) ------------------------------------


async def test_active_drops_backlog_once_every_consumer_departs() -> None:
    broadcaster = ToolEventBroadcaster("active-consumers")
    only = broadcaster.create_consumer()
    broadcaster.push(1)
    broadcaster.push(2)
    assert await anext(only) == 1
    assert await anext(only) == 2
    await only.aclose()

    # Nobody is attached: retained history is dropped, so a fresh consumer
    # starts at the watermark (retention would replay [1, 2] first).
    broadcaster.push(3)
    broadcaster.complete()
    late = broadcaster.create_consumer()
    assert await collect(late) == [3]


async def test_active_backlog_survives_while_a_consumer_remains() -> None:
    broadcaster = ToolEventBroadcaster("active-consumers")
    fast = broadcaster.create_consumer()
    slow = broadcaster.create_consumer()
    broadcaster.push(1)
    broadcaster.push(2)
    assert await anext(fast) == 1
    assert await anext(fast) == 2
    await fast.aclose()

    # The slow consumer still reads from position 0.
    assert await anext(slow) == 1
    assert await anext(slow) == 2
    broadcaster.complete()
    with pytest.raises(StopAsyncIteration):
        await anext(slow)


async def test_active_aclose_wakes_a_pending_next_instead_of_hanging() -> None:
    broadcaster = ToolEventBroadcaster("active-consumers")
    consumer = broadcaster.create_consumer()
    broadcaster.push(1)
    assert await anext(consumer) == 1

    pending = asyncio.ensure_future(anext(consumer))
    await _park(pending)
    await consumer.aclose()
    with pytest.raises(StopAsyncIteration):
        await pending


async def test_active_athrow_rejects_a_pending_next_with_the_error() -> None:
    broadcaster = ToolEventBroadcaster("active-consumers")
    consumer = broadcaster.create_consumer()
    broadcaster.push(1)
    assert await anext(consumer) == 1

    pending = asyncio.ensure_future(anext(consumer))
    await _park(pending)
    with pytest.raises(ValueError, match="boom"):
        await consumer.athrow(ValueError("boom"))
    with pytest.raises(ValueError, match="boom"):
        await pending


async def test_active_cleanup_releases_buffer_when_completed_without_consumers() -> None:
    # Python-specific: upstream schedules cleanup via queueMicrotask after
    # complete(); here it is loop.call_soon. With no consumer ever attached the
    # buffer is released, so a consumer joining afterwards sees no history.
    broadcaster = ToolEventBroadcaster("active-consumers")
    broadcaster.push(1)
    broadcaster.complete()
    await asyncio.sleep(0)  # run the scheduled cleanup callback
    assert await collect(broadcaster.create_consumer()) == []


async def test_full_mode_cleanup_retains_buffer_when_completed_without_consumers() -> None:
    broadcaster = ToolEventBroadcaster()
    broadcaster.push(1)
    broadcaster.complete()
    await asyncio.sleep(0)
    assert await collect(broadcaster.create_consumer()) == [1]
