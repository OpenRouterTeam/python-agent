"""Ports `packages/agent/tests/unit/reusable-stream.test.ts`.

Contract tests for the replay-buffer memory/lifecycle edges:

- active-consumers mode must not pin O(stream) backlog once every consumer has
  departed
- full-replay mode (default) keeps replaying everything
- a pending ``__anext__`` must never hang after ``aclose()`` / ``athrow()`` /
  ``cancel()``

Determinism: a pending read is parked with :func:`_park`, which yields to the
event loop until the reading task is blocked on its waiter. That is scheduler
ordering, not wall-clock timing. Upstream's "would hang" failure mode is caught
by ``asyncio.wait_for`` around the awaited result.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Generic, List, Optional, TypeVar

import pytest

from openrouter_agent.reusable_stream import ReusableReadableStream

T = TypeVar("T")

# Guard against a regression that re-introduces a hang; not a timing input.
_HANG_GUARD_SECONDS = 5


async def stream_of(values: List[T]) -> AsyncIterator[T]:
    for value in values:
        yield value


class ControlledSource(Generic[T]):
    """Source whose producer is driven by the test, for mid-stream assertions."""

    _CLOSE: Any = object()

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

    def push(self, value: T) -> None:
        self._queue.put_nowait(value)

    def close(self) -> None:
        self._queue.put_nowait(self._CLOSE)

    def __aiter__(self) -> "ControlledSource[T]":
        return self

    async def __anext__(self) -> T:
        value = await self._queue.get()
        if value is self._CLOSE:
            raise StopAsyncIteration
        return value  # type: ignore[no-any-return]


async def _park(task: "asyncio.Task[Any]") -> None:
    """Yield until ``task`` is suspended (blocked on its waiter) or finished."""
    for _ in range(10):
        await asyncio.sleep(0)
    assert not task.done(), "pending read resolved before it could be parked"


async def _collect(iterator: AsyncIterator[T]) -> List[T]:
    return [value async for value in iterator]


async def test_retains_full_replay_for_late_consumers_by_default() -> None:
    stream = ReusableReadableStream(stream_of([1, 2]))
    first = stream.create_consumer()
    assert await _collect(first) == [1, 2]

    second = stream.create_consumer()
    assert await _collect(second) == [1, 2]


async def test_active_consumers_backlog_follows_slowest_remaining_consumer() -> None:
    source: ControlledSource[int] = ControlledSource()
    stream = ReusableReadableStream(source, stream_replay="active-consumers")
    fast = stream.create_consumer()
    slow = stream.create_consumer()

    source.push(1)
    source.push(2)
    assert await fast.__anext__() == 1
    assert await fast.__anext__() == 2
    await fast.aclose()

    # The slow consumer still reads from position 0 — trimming follows the
    # slowest attached consumer, never the fastest.
    assert await slow.__anext__() == 1
    assert await slow.__anext__() == 2
    source.close()
    with pytest.raises(StopAsyncIteration):
        await slow.__anext__()


async def test_active_consumers_backlog_dropped_once_every_consumer_departs() -> None:
    source: ControlledSource[int] = ControlledSource()
    stream = ReusableReadableStream(source, stream_replay="active-consumers")
    only = stream.create_consumer()
    source.push(1)
    source.push(2)
    assert await only.__anext__() == 1
    assert await only.__anext__() == 2
    await only.aclose()

    # Nobody is attached anymore: the retained backlog is dropped.
    assert stream.find_last_buffered(lambda _value: True) is None

    # A late consumer starts at the watermark instead of replaying history.
    source.push(3)
    late = stream.create_consumer()
    assert await late.__anext__() == 3
    source.close()
    assert await _collect(late) == []


async def test_active_consumers_fully_caught_up_consumer_leaves_nothing_for_later_joiners() -> None:
    # Trimming follows consumption, not attachment time. Replay-after-detach is
    # the documented trade-off of active-consumers mode; "full" keeps it.
    stream = ReusableReadableStream(stream_of([1, 2]), stream_replay="active-consumers")
    only = stream.create_consumer()
    assert await _collect(only) == [1, 2]

    late = stream.create_consumer()
    assert await _collect(late) == []


async def test_aclose_wakes_a_pending_next_instead_of_hanging() -> None:
    source: ControlledSource[int] = ControlledSource()
    stream = ReusableReadableStream(source)
    consumer = stream.create_consumer()
    source.push(1)
    assert await consumer.__anext__() == 1

    pending = asyncio.ensure_future(consumer.__anext__())
    await _park(pending)
    await consumer.aclose()
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(pending, _HANG_GUARD_SECONDS)
    source.close()


async def test_athrow_rejects_a_pending_next_with_the_same_error() -> None:
    source: ControlledSource[int] = ControlledSource()
    stream = ReusableReadableStream(source)
    consumer = stream.create_consumer()
    source.push(1)
    assert await consumer.__anext__() == 1

    pending = asyncio.ensure_future(consumer.__anext__())
    await _park(pending)
    boom = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom") as thrown:
        await consumer.athrow(boom)
    assert thrown.value is boom
    with pytest.raises(RuntimeError, match="boom"):
        await asyncio.wait_for(pending, _HANG_GUARD_SECONDS)
    source.close()


async def test_athrow_normalizes_an_exception_type() -> None:
    source: ControlledSource[int] = ControlledSource()
    stream = ReusableReadableStream(source)
    consumer = stream.create_consumer()
    with pytest.raises(ValueError):
        await consumer.athrow(ValueError)
    # The consumer is detached afterwards.
    with pytest.raises(StopAsyncIteration):
        await consumer.__anext__()
    source.close()


async def test_cancel_settles_waiters_and_leaves_no_buffered_backlog() -> None:
    source: ControlledSource[int] = ControlledSource()
    stream = ReusableReadableStream(source)
    consumer = stream.create_consumer()
    source.push(1)
    source.push(2)

    # Whether the parked read wakes with a buffered value or as done depends on
    # interleaving with the pump; the contract is only that it settles and
    # that no backlog survives.
    pending = asyncio.ensure_future(consumer.__anext__())
    results = await asyncio.wait_for(
        asyncio.gather(pending, stream.cancel(), return_exceptions=True),
        _HANG_GUARD_SECONDS,
    )
    assert results[1] is None
    assert results[0] == 1 or isinstance(results[0], StopAsyncIteration)

    assert stream.find_last_buffered(lambda _value: True) is None
    assert stream.is_complete is True
    fresh = stream.create_consumer()
    assert await _collect(fresh) == []


async def test_cancel_closes_the_source_exactly_once() -> None:
    closed: List[str] = []

    class Source:
        def __aiter__(self) -> "Source":
            return self

        async def __anext__(self) -> int:
            await asyncio.Event().wait()  # never produces; cancel() must unblock
            raise AssertionError("unreachable")

        async def aclose(self) -> None:
            closed.append("closed")

    stream = ReusableReadableStream(Source())
    consumer = stream.create_consumer()
    pending = asyncio.ensure_future(consumer.__anext__())
    await _park(pending)

    await stream.cancel()
    await stream.cancel()

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(pending, _HANG_GUARD_SECONDS)
    assert closed == ["closed"]


async def test_source_error_propagates_after_buffered_values() -> None:
    async def failing() -> AsyncIterator[int]:
        yield 1
        raise RuntimeError("source failed")

    stream = ReusableReadableStream(failing())
    consumer = stream.create_consumer()
    received: List[int] = []
    with pytest.raises(RuntimeError, match="source failed"):
        async for value in consumer:
            received.append(value)
    assert received == [1]


async def test_find_last_buffered_scans_from_the_end() -> None:
    stream = ReusableReadableStream(stream_of([{"type": "a", "n": 1}, {"type": "b", "n": 2}, {"type": "a", "n": 3}]))
    assert len(await _collect(stream.create_consumer())) == 3

    found: Optional[dict] = stream.find_last_buffered(lambda item: item["type"] == "a")
    assert found == {"type": "a", "n": 3}
    assert stream.find_last_buffered(lambda item: item["type"] == "missing") is None


async def test_none_is_a_valid_stream_value() -> None:
    # The port uses a private sentinel for cleared slots (upstream uses
    # `undefined`), so `None` must survive replay and compaction.
    stream = ReusableReadableStream(stream_of([None, 1, None]), stream_replay="active-consumers")
    assert await _collect(stream.create_consumer()) == [None, 1, None]
