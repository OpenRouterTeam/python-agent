"""Multi-consumer async stream adapter using anyio."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Generic, TypeVar

import anyio
from anyio.abc import TaskGroup

T = TypeVar("T")


class ReusableAsyncStream(Generic[T]):
    """Wraps an async iterable source so multiple consumers can read independently.

    The source is pumped exactly once by a background task. Each consumer maintains
    its own position in a shared buffer and receives all events from the beginning.
    """

    def __init__(self, source: AsyncIterator[T]) -> None:
        self._source = source
        self._buffer: list[T] = []
        self._done = False
        self._error: BaseException | None = None
        self._event = anyio.Event()
        self._pump_started = False
        self._task_group: TaskGroup | None = None

    async def _pump(self) -> None:
        """Read all items from source into the shared buffer."""
        try:
            async for item in self._source:
                self._buffer.append(item)
                # Signal waiting consumers, then create a fresh event for the next cycle
                self._event.set()
                self._event = anyio.Event()
        except BaseException as e:
            self._error = e
        finally:
            self._done = True
            self._event.set()

    async def start(self, task_group: TaskGroup) -> None:
        """Start the background pump task."""
        if not self._pump_started:
            self._pump_started = True
            self._task_group = task_group
            task_group.start_soon(self._pump)

    def create_consumer(self) -> AsyncIterator[T]:
        """Create a new independent consumer that reads from the shared buffer."""
        return _Consumer(self)

    @property
    def is_done(self) -> bool:
        return self._done


class _Consumer(Generic[T]):
    """Independent consumer with its own read position in the shared buffer."""

    def __init__(self, stream: ReusableAsyncStream[T]) -> None:
        self._stream = stream
        self._position = 0

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        while True:
            if self._position < len(self._stream._buffer):
                item = self._stream._buffer[self._position]
                self._position += 1
                return item

            if self._stream._done:
                if self._stream._error is not None:
                    raise self._stream._error
                raise StopAsyncIteration

            await self._stream._event.wait()
