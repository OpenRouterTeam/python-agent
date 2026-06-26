from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, List, Optional

_SENTINEL = object()


class ToolEventBroadcaster:
    def __init__(self) -> None:
        self._buffer: List[Any] = []
        self._complete = False
        self._error: Optional[BaseException] = None
        self._condition = asyncio.Condition()

    def push(self, event: Any) -> None:
        if self._complete:
            return
        self._buffer.append(event)
        self._wake()

    def complete(self, error: Optional[BaseException] = None) -> None:
        if self._complete:
            return
        self._complete = True
        self._error = error
        self._wake()

    def error(self, error: BaseException) -> None:
        self.complete(error)

    def _wake(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._notify())

    async def _notify(self) -> None:
        async with self._condition:
            self._condition.notify_all()

    def create_consumer(self) -> AsyncIterator[Any]:
        async def gen() -> AsyncIterator[Any]:
            index = 0
            while True:
                async with self._condition:
                    while index >= len(self._buffer) and not self._complete:
                        await self._condition.wait()
                    if index < len(self._buffer):
                        item = self._buffer[index]
                        index += 1
                    elif self._error is not None:
                        raise self._error
                    else:
                        break
                yield item

        return gen()

    def __aiter__(self) -> AsyncIterator[Any]:
        return self.create_consumer()
