from __future__ import annotations

import asyncio
from typing import Any, AsyncIterable, AsyncIterator, List, Optional


class ReusableReadableStream:
    def __init__(self, source: AsyncIterable[Any]) -> None:
        self._source = source
        self._buffer: List[Any] = []
        self._complete = False
        self._error: Optional[BaseException] = None
        self._started = False
        self._condition = asyncio.Condition()

    def _ensure_started(self) -> None:
        if not self._started:
            self._started = True
            asyncio.create_task(self._pump())

    @property
    def is_complete(self) -> bool:
        """True once the source stream has been fully read into the buffer.
        A fresh consumer created after this point replays the retained
        buffer without waiting on the source."""
        return self._complete

    async def _pump(self) -> None:
        try:
            async for item in self._source:
                async with self._condition:
                    self._buffer.append(item)
                    self._condition.notify_all()
        except BaseException as exc:
            self._error = exc
        finally:
            async with self._condition:
                self._complete = True
                self._condition.notify_all()

    def create_consumer(self) -> AsyncIterator[Any]:
        self._ensure_started()

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
