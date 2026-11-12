"""Push-based multi-consumer event broadcaster.

Ports `packages/agent/src/lib/tool-event-broadcaster.ts`. Similar to
`ReusableReadableStream` but for push-based events from tool execution.

Each consumer gets its own position in the buffer. Full replay is the default;
``stream_replay="active-consumers"`` compacts events once every attached
consumer has read them.

Contract: ``push()`` after ``complete()`` is a silent no-op, so callers must
finish pushing before completing (see tests/unit/test_turn_end_race_condition.py).
"""

from __future__ import annotations

import asyncio
from typing import Optional, TypeVar

from .reusable_stream import ReplayConsumer, StreamReplay, _ReplayBuffer

T = TypeVar("T")


class ToolEventBroadcaster(_ReplayBuffer[T]):
    def __init__(self, stream_replay: StreamReplay = "full") -> None:
        super().__init__(stream_replay)
        self._complete = False
        self._error: Optional[BaseException] = None

    @property
    def is_complete(self) -> bool:
        return self._complete

    def push(self, event: T) -> None:
        """Push an event to all consumers. Ignored once complete."""
        if self._complete:
            return
        self._append(event)
        self._notify_waiting()

    def complete(self, error: Optional[BaseException] = None) -> None:
        """Mark the broadcaster complete, optionally signalling ``error`` to every
        consumer after it drains the buffered events. Idempotent."""
        if self._complete:
            return
        self._complete = True
        self._error = error
        self._notify_waiting()
        # Upstream schedules cleanup on a microtask so consumers created in
        # the same synchronous section still see the buffer.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.call_soon(self._cleanup)

    def error(self, error: BaseException) -> None:
        self.complete(error)

    def _cleanup(self) -> None:
        # Only release when complete and every consumer is done.
        if self._stream_replay == "active-consumers" and self._complete and not self._consumers:
            self._buffer = []
            self._buffer_head = 0

    def _after_detach(self) -> None:
        self._cleanup()

    def _terminal_state(self, consumer_id: int) -> bool:
        if not self._complete:
            return False
        self._consumers.pop(consumer_id, None)
        self._cleanup()
        if self._error is not None:
            raise self._error
        return True

    def __aiter__(self) -> ReplayConsumer[T]:
        return self.create_consumer()


__all__ = ["ToolEventBroadcaster"]
