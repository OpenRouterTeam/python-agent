"""Port of upstream ``src/lib/tool-event-broadcaster.ts``.

A push-based event broadcaster that supports multiple concurrent consumers.
Similar to ``ReusableReadableStream`` but for push-based events from tool
execution. Each consumer gets its own position in the buffer. Full replay is
the default; ``"active-consumers"`` replay compacts consumed events.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from .reusable_stream import _CLEARED, ReplayConsumer, StreamReplay, _normalize_thrown, _ReplayBuffer, _settle


class ToolEventBroadcaster(_ReplayBuffer):
    def __init__(self, stream_replay: StreamReplay = "full") -> None:
        super().__init__(stream_replay)
        self._complete = False
        self._completion_error: Optional[BaseException] = None

    def push(self, event: Any) -> None:
        """Push a new event to all consumers. Ignored after ``complete()``.
        Events are buffered so late-joining consumers can catch up."""
        if self._complete:
            return
        self._buffer.append(event)
        self._notify_waiting_consumers()

    def complete(self, error: Optional[BaseException] = None) -> None:
        """Mark the broadcaster complete; optionally fail all consumers with
        ``error``. Schedules cleanup once consumers have observed completion."""
        self._complete = True
        self._completion_error = error
        self._notify_waiting_consumers()
        # Upstream: queueMicrotask(() => this.cleanup())
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._cleanup()
        else:
            loop.call_soon(self._cleanup)

    def error(self, error: BaseException) -> None:
        """Python convenience alias for ``complete(error)``."""
        self.complete(error)

    def _cleanup(self) -> None:
        if self._stream_replay == "active-consumers" and self._complete and not self._consumers:
            self._buffer = []
            self._buffer_head = 0

    def create_consumer(self) -> ReplayConsumer:
        """Create a consumer that independently iterates over events.
        Full-replay consumers start at position 0; active-consumer replay
        starts at the current trim watermark."""
        return ReplayConsumer(self, self._register_consumer())

    def __aiter__(self) -> ReplayConsumer:
        return self.create_consumer()

    async def _consumer_next(self, consumer_id: int) -> Any:
        while True:
            consumer = self._consumers.get(consumer_id)
            if consumer is None or consumer.cancelled:
                raise StopAsyncIteration

            index = self._index_of(consumer)
            if index < len(self._buffer):
                value = self._buffer[index]
                if value is _CLEARED:
                    raise RuntimeError("ToolEventBroadcaster buffer invariant violated: consumed slot was cleared")
                consumer.position += 1
                self._trim_consumed()
                return value

            if self._complete:
                self._consumers.pop(consumer_id, None)
                self._cleanup()
                if self._completion_error is not None:
                    raise self._completion_error
                raise StopAsyncIteration

            waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            consumer.waiter = waiter
            try:
                await waiter
            finally:
                if consumer.waiter is waiter:
                    consumer.waiter = None

    async def _consumer_return(self, consumer_id: int) -> None:
        if self._detach(consumer_id):
            self._cleanup()

    async def _consumer_throw(self, consumer_id: int, exc: Any) -> None:
        error = _normalize_thrown(exc)
        if self._detach(consumer_id, error):
            self._cleanup()
        raise error

    def _notify_waiting_consumers(self) -> None:
        for consumer in self._consumers.values():
            if consumer.waiter is not None:
                _settle(consumer.waiter, self._completion_error)
                consumer.waiter = None


__all__ = ["ToolEventBroadcaster"]
