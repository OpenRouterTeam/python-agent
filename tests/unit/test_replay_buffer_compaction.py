"""Ports `packages/agent/tests/unit/replay-buffer-compaction.test.ts`."""

from __future__ import annotations

from typing import Any, AsyncIterator, List

from openrouter_agent.reusable_stream import BUFFER_COMPACTION_MIN_HEAD, ReusableReadableStream


async def source(values: List[int]) -> AsyncIterator[int]:
    for value in values:
        yield value


async def collect(iterator: AsyncIterator[Any]) -> List[Any]:
    return [item async for item in iterator]


class EndlessSource:
    """Upstream's `pull` source: produces 1, 2, 1, 2, ... until cancelled.

    Counts reads and `aclose()` calls (upstream `cancel()`), optionally failing
    cancellation.
    """

    def __init__(self, *, fail_cancel: bool = False) -> None:
        self.reads = 0
        self.cancellation_count = 0
        self._fail_cancel = fail_cancel

    def __aiter__(self) -> "EndlessSource":
        return self

    async def __anext__(self) -> int:
        self.reads += 1
        return 1 if self.reads % 2 == 1 else 2

    async def aclose(self) -> None:
        self.cancellation_count += 1
        if self._fail_cancel:
            raise RuntimeError("cleanup failed")


async def test_replays_complete_history_to_sequential_and_post_completion_consumers_by_default() -> None:
    stream = ReusableReadableStream(source([1, 2, 3]))
    assert await collect(stream.create_consumer()) == [1, 2, 3]
    assert await collect(stream.create_consumer()) == [1, 2, 3]


async def test_starts_new_active_consumer_consumers_at_the_current_watermark() -> None:
    stream = ReusableReadableStream(source([1, 2, 3]), stream_replay="active-consumers")
    first = stream.create_consumer()

    assert await anext(first) == 1
    second = stream.create_consumer()
    assert await collect(first) == [2, 3]
    assert await collect(second) == [2, 3]


async def test_continues_to_replay_active_consumer_events_through_repeated_compaction() -> None:
    values = list(range(2500))
    # Must exceed the compaction threshold more than once for this to exercise
    # the slice path repeatedly.
    assert len(values) > 2 * BUFFER_COMPACTION_MIN_HEAD
    stream = ReusableReadableStream(source(values), stream_replay="active-consumers")
    consumer = stream.create_consumer()

    assert await collect(consumer) == values
    assert await collect(stream.create_consumer()) == []


async def test_compaction_slice_preserves_unread_tail_past_the_threshold() -> None:
    # Python-specific coverage for the slice-compaction branch with a live
    # buffer tail: the source never suspends, so the pump buffers all 3000
    # values before the consumer reads; compaction then slices at head 1500
    # while 1500 unread values remain, and every value must still arrive in
    # order across the pause at 2000.
    values = list(range(3000))
    stream = ReusableReadableStream(source(values), stream_replay="active-consumers")
    slow = stream.create_consumer()
    seen: List[int] = []
    for _ in range(2000):
        seen.append(await anext(slow))
    seen.extend(await collect(slow))
    assert seen == values


async def test_stops_at_a_terminal_value_and_cancels_the_source_once() -> None:
    src = EndlessSource()
    stream = ReusableReadableStream(src, is_terminal_value=lambda value: value == 1)

    assert await collect(stream.create_consumer()) == [1]
    assert src.cancellation_count == 1
    # Upstream also counts `releaseLock()` == 1; Python has no reader lock, so
    # the analog asserted is that the pump read exactly once (stopped at the
    # terminal value rather than reading past it).
    assert src.reads == 1
    assert stream.is_complete is True

    # A later cancel() must not close the source a second time.
    await stream.cancel()
    assert src.cancellation_count == 1


async def test_retains_the_terminal_value_when_source_cancellation_fails() -> None:
    src = EndlessSource(fail_cancel=True)
    stream = ReusableReadableStream(src, is_terminal_value=lambda value: value == 1)

    assert await collect(stream.create_consumer()) == [1]
    assert src.cancellation_count == 1


async def test_treats_a_failed_terminal_value_as_the_final_buffered_event() -> None:
    stream = ReusableReadableStream(source([1, 2]), is_terminal_value=lambda value: value == 2)
    assert await collect(stream.create_consumer()) == [1, 2]


async def test_treats_an_incomplete_terminal_value_as_the_final_buffered_event() -> None:
    stream = ReusableReadableStream(source([1, 2]), is_terminal_value=lambda value: value == 2)
    assert await collect(stream.create_consumer()) == [1, 2]


async def test_on_value_observes_every_value_in_order_and_survives_compaction() -> None:
    # Python-specific: `onValue` is how ModelResult captures terminal events
    # that active-consumer compaction may have released.
    observed: List[int] = []
    stream = ReusableReadableStream(
        source([1, 2, 3]),
        stream_replay="active-consumers",
        on_value=observed.append,
        is_terminal_value=lambda value: value == 3,
    )
    assert await collect(stream.create_consumer()) == [1, 2, 3]
    assert observed == [1, 2, 3]
    assert stream.find_last_buffered(lambda value: value == 3) is None


async def test_find_last_buffered_scans_from_the_end_in_full_mode() -> None:
    stream = ReusableReadableStream(source([1, 2, 3, 4]))
    assert await collect(stream.create_consumer()) == [1, 2, 3, 4]
    assert stream.find_last_buffered(lambda value: value % 2 == 1) == 3
    assert stream.find_last_buffered(lambda value: value > 10) is None
