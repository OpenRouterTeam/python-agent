"""Ports `packages/agent/tests/unit/replay-buffer-compaction.test.ts`.

Replay policy of `ReusableReadableStream`: full replay by default, watermark
start for active-consumers mode, repeated compaction, and terminal-value
handling (``is_terminal_value`` stops the pump and closes the source once).
"""

from __future__ import annotations

from typing import AsyncIterator, List

from openrouter_agent.reusable_stream import BUFFER_COMPACTION_MIN_HEAD, ReusableReadableStream


async def source(values: List[int]) -> AsyncIterator[int]:
    for value in values:
        yield value


class EndlessSource:
    """Mirrors upstream's `pull()` source: produces 1, 2, 1, 2, ... forever.

    ``aclose`` is the Python analogue of ReadableStream ``cancel()``.
    """

    def __init__(self, *, fail_close: bool = False) -> None:
        self.close_count = 0
        self.produced: List[int] = []
        self._fail_close = fail_close

    def __aiter__(self) -> "EndlessSource":
        return self

    async def __anext__(self) -> int:
        value = 1 if len(self.produced) % 2 == 0 else 2
        self.produced.append(value)
        return value

    async def aclose(self) -> None:
        self.close_count += 1
        if self._fail_close:
            raise RuntimeError("cleanup failed")


async def _collect(iterator: AsyncIterator[int]) -> List[int]:
    return [value async for value in iterator]


async def test_replays_complete_history_to_sequential_and_post_completion_consumers_by_default() -> None:
    stream = ReusableReadableStream(source([1, 2, 3]))

    assert await _collect(stream.create_consumer()) == [1, 2, 3]
    assert await _collect(stream.create_consumer()) == [1, 2, 3]


async def test_starts_new_active_consumer_consumers_at_the_current_watermark() -> None:
    stream = ReusableReadableStream(source([1, 2, 3]), stream_replay="active-consumers")
    first = stream.create_consumer()

    assert await first.__anext__() == 1
    second = stream.create_consumer()
    assert await _collect(first) == [2, 3]
    assert await _collect(second) == [2, 3]


async def test_continues_to_replay_active_consumer_events_through_repeated_compaction() -> None:
    values = list(range(2500))
    # Large enough to cross the compaction threshold more than once.
    assert len(values) > 2 * BUFFER_COMPACTION_MIN_HEAD
    stream = ReusableReadableStream(source(values), stream_replay="active-consumers")
    consumer = stream.create_consumer()

    assert await _collect(consumer) == values
    late_consumer = stream.create_consumer()
    assert await _collect(late_consumer) == []


async def test_stops_at_a_terminal_value_and_closes_the_source_once() -> None:
    # Upstream additionally counts `releaseLock()` calls; Python async
    # iterators have no reader lock, so only the close count is ported.
    endless = EndlessSource()
    stream = ReusableReadableStream(endless, is_terminal_value=lambda value: value == 1)

    assert await _collect(stream.create_consumer()) == [1]
    assert endless.close_count == 1
    # The pump stopped at the terminal value instead of reading ahead.
    assert endless.produced == [1]
    assert stream.is_complete is True


async def test_retains_the_terminal_value_when_source_close_fails() -> None:
    endless = EndlessSource(fail_close=True)
    stream = ReusableReadableStream(endless, is_terminal_value=lambda value: value == 1)

    assert await _collect(stream.create_consumer()) == [1]
    assert endless.close_count == 1


async def test_treats_a_failed_terminal_value_as_the_final_buffered_event() -> None:
    stream = ReusableReadableStream(source([1, 2]), is_terminal_value=lambda value: value == 2)

    assert await _collect(stream.create_consumer()) == [1, 2]


async def test_treats_an_incomplete_terminal_value_as_the_final_buffered_event() -> None:
    # Upstream's two terminal cases (failed / incomplete response) share the
    # same shape at this layer; both are kept for 1:1 parity.
    stream = ReusableReadableStream(source([1, 2, 3]), is_terminal_value=lambda value: value == 2)

    assert await _collect(stream.create_consumer()) == [1, 2]


async def test_on_value_observes_every_buffered_value_once_in_order() -> None:
    seen: List[int] = []
    stream = ReusableReadableStream(
        source([1, 2, 3]),
        stream_replay="active-consumers",
        on_value=seen.append,
        is_terminal_value=lambda value: value == 2,
    )

    assert await _collect(stream.create_consumer()) == [1, 2]
    assert seen == [1, 2]
    # Captured via on_value even though active-consumers already trimmed it.
    assert stream.find_last_buffered(lambda _value: True) is None
