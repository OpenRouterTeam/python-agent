"""Ports `packages/agent/tests/unit/reusable-stream.test.ts`.

Contract tests for the replay-buffer memory/lifecycle edges:
- active-consumers mode must not pin O(stream) backlog once every consumer has
  departed
- full-replay mode (default) keeps replaying everything
- a pending next must never hang after aclose()/athrow()/cancel()

Iterator mapping: upstream `next()` -> `anext()` (done -> `StopAsyncIteration`),
`return()` -> `aclose()`, `throw()` -> `athrow()`.

Determinism: the controlled source is an `asyncio.Queue`; nothing races a clock.
Where a test needs a `next()` parked before acting, `_park` yields to the loop
once and asserts the task is still pending, i.e. provably waiting.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Generic, List, TypeVar

import pytest

from openrouter_agent.reusable_stream import ReusableReadableStream

T = TypeVar("T")
_CLOSE = object()


async def stream_of(values: List[Any]) -> AsyncIterator[Any]:
    for value in values:
        yield value


class ControlledStream(Generic[T]):
    """Source whose producer is manually controlled, for mid-stream assertions."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

    def push(self, value: T) -> None:
        self._queue.put_nowait(value)

    def close(self) -> None:
        self._queue.put_nowait(_CLOSE)

    def __aiter__(self) -> "ControlledStream[T]":
        return self

    async def __anext__(self) -> T:
        value = await self._queue.get()
        if value is _CLOSE:
            raise StopAsyncIteration
        return value  # type: ignore[no-any-return]


async def collect(iterator: AsyncIterator[Any]) -> List[Any]:
    return [item async for item in iterator]


async def _park(task: "asyncio.Task[Any]") -> None:
    await asyncio.sleep(0)
    assert not task.done(), "next() should be parked waiting for data"


async def test_retains_full_replay_for_late_consumers_by_default() -> None:
    stream = ReusableReadableStream(stream_of([1, 2]))
    assert await collect(stream.create_consumer()) == [1, 2]
    assert await collect(stream.create_consumer()) == [1, 2]


async def test_active_consumers_backlog_follows_slowest_remaining_consumer() -> None:
    source: ControlledStream[int] = ControlledStream()
    stream = ReusableReadableStream(source, stream_replay="active-consumers")
    fast = stream.create_consumer()
    slow = stream.create_consumer()

    source.push(1)
    source.push(2)
    assert await anext(fast) == 1
    assert await anext(fast) == 2
    await fast.aclose()

    # The slow consumer still reads from position 0 — trimming follows the
    # slowest attached consumer, never the fastest.
    assert await anext(slow) == 1
    assert await anext(slow) == 2
    source.close()
    with pytest.raises(StopAsyncIteration):
        await anext(slow)


async def test_active_consumers_backlog_dropped_once_every_consumer_departs() -> None:
    source: ControlledStream[int] = ControlledStream()
    stream = ReusableReadableStream(source, stream_replay="active-consumers")
    only = stream.create_consumer()
    source.push(1)
    source.push(2)
    assert await anext(only) == 1
    assert await anext(only) == 2
    await only.aclose()

    # Nobody is attached anymore: the retained backlog is dropped.
    assert stream.find_last_buffered(lambda _: True) is None

    # A late consumer starts at the watermark instead of replaying history.
    source.push(3)
    late = stream.create_consumer()
    assert await anext(late) == 3
    source.close()
    with pytest.raises(StopAsyncIteration):
        await anext(late)


async def test_active_consumers_fully_caught_up_consumer_leaves_nothing_for_later_joiners() -> None:
    stream = ReusableReadableStream(stream_of([1, 2]), stream_replay="active-consumers")
    assert await collect(stream.create_consumer()) == [1, 2]
    assert await collect(stream.create_consumer()) == []


async def test_aclose_wakes_a_pending_next_instead_of_hanging() -> None:
    source: ControlledStream[int] = ControlledStream()
    stream = ReusableReadableStream(source)
    consumer = stream.create_consumer()
    source.push(1)
    assert await anext(consumer) == 1

    pending = asyncio.ensure_future(anext(consumer))
    await _park(pending)
    await consumer.aclose()
    with pytest.raises(StopAsyncIteration):
        await pending
    source.close()


async def test_athrow_rejects_a_pending_next_with_the_error() -> None:
    source: ControlledStream[int] = ControlledStream()
    stream = ReusableReadableStream(source)
    consumer = stream.create_consumer()
    source.push(1)
    assert await anext(consumer) == 1

    pending = asyncio.ensure_future(anext(consumer))
    await _park(pending)
    with pytest.raises(ValueError, match="boom"):
        await consumer.athrow(ValueError("boom"))
    with pytest.raises(ValueError, match="boom"):
        await pending
    source.close()


async def test_athrow_normalizes_a_non_exception_value() -> None:
    # Upstream: `e instanceof Error ? e : new Error(String(e))`.
    stream = ReusableReadableStream(stream_of([1]))
    consumer = stream.create_consumer()
    with pytest.raises(Exception, match="not-an-error"):
        await consumer.athrow("not-an-error")


async def test_cancel_settles_waiters_and_leaves_no_buffered_backlog() -> None:
    source: ControlledStream[int] = ControlledStream()
    stream = ReusableReadableStream(source)
    consumer = stream.create_consumer()
    source.push(1)
    source.push(2)

    # Whether the parked next wakes with a buffered value or as done depends on
    # interleaving with the pump; the contract is only that it settles and
    # that no backlog survives.
    pending = asyncio.ensure_future(anext(consumer))
    results = await asyncio.gather(pending, stream.cancel(), return_exceptions=True)
    assert results[0] == 1 or isinstance(results[0], StopAsyncIteration)
    assert results[1] is None

    assert stream.find_last_buffered(lambda _: True) is None
    fresh = stream.create_consumer()
    with pytest.raises(StopAsyncIteration):
        await anext(fresh)
    # No source.close(): cancel() already terminated the source stream.


async def test_cancel_wakes_a_parked_consumer_as_done() -> None:
    # Python-specific: drives the cancel-while-pump-is-awaiting-source branch
    # deterministically (the pump task is parked on an empty queue).
    source: ControlledStream[int] = ControlledStream()
    stream = ReusableReadableStream(source)
    consumer = stream.create_consumer()
    pending = asyncio.ensure_future(anext(consumer))
    await _park(pending)

    await stream.cancel()

    with pytest.raises(StopAsyncIteration):
        await pending
    assert stream.is_complete is True


async def test_source_error_propagates_after_buffered_values() -> None:
    async def failing() -> AsyncIterator[int]:
        yield 1
        raise RuntimeError("source failed")

    stream = ReusableReadableStream(failing())
    received: List[int] = []
    with pytest.raises(RuntimeError, match="source failed"):
        async for item in stream.create_consumer():
            received.append(item)
    assert received == [1]
