"""Push-based event broadcaster with multi-consumer support."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Generic, TypeVar

import anyio

T = TypeVar("T")


class ToolEventBroadcaster(Generic[T]):
    """Push-based event broadcaster.

    Events are pushed by the tool execution loop and broadcast to all consumers.
    Late consumers replay from the internal buffer.
    """

    def __init__(self) -> None:
        self._buffer: list[T] = []
        self._done = False
        self._error: BaseException | None = None

    def push(self, event: T) -> None:
        """Push an event to all consumers."""
        if self._done:
            raise RuntimeError("Cannot push to completed broadcaster")
        self._buffer.append(event)

    def complete(self, error: BaseException | None = None) -> None:
        """Mark the broadcaster as complete."""
        self._done = True
        self._error = error

    def create_consumer(self) -> AsyncIterator[T]:
        """Create a new independent consumer."""
        return _BroadcastConsumer(self)

    @property
    def is_done(self) -> bool:
        return self._done


class _BroadcastConsumer(Generic[T]):
    """Independent consumer for the broadcaster."""

    def __init__(self, broadcaster: ToolEventBroadcaster[T]) -> None:
        self._broadcaster = broadcaster
        self._position = 0

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        while True:
            if self._position < len(self._broadcaster._buffer):
                item = self._broadcaster._buffer[self._position]
                self._position += 1
                return item

            if self._broadcaster._done:
                if self._broadcaster._error is not None:
                    raise self._broadcaster._error
                raise StopAsyncIteration

            await anyio.sleep(0.001)
