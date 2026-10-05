"""Port of upstream ``src/lib/reusable-stream.ts``.

A reusable stream lets multiple consumers read the same source concurrently
while it is still streaming, without forcing consumers to wait for full
buffering.

- Multiple concurrent consumers with independent read positions
- New consumers can attach while streaming is active
- Full replay for delayed and sequential consumers by default
- Opt-in active-consumer replay compaction (``stream_replay="active-consumers"``)
  for bounded memory
- Each consumer reads at its own pace

Python mapping of upstream's iterator protocol: ``next()`` -> ``__anext__``
(``StopAsyncIteration`` instead of ``{done: true}``), ``return()`` ->
``aclose()``, ``throw(e)`` -> ``athrow(e)``. Reader cancellation maps to
cancelling the pump task and calling ``aclose()`` on the source iterator when
it has one.
"""

from __future__ import annotations

import asyncio
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
)

StreamReplay = Literal["full", "active-consumers"]

BUFFER_COMPACTION_MIN_HEAD = 1024

# Marks a buffer slot that has been released by compaction. A dedicated
# sentinel (rather than upstream's ``undefined``) so ``None`` stays a valid item.
_CLEARED: Any = object()


class _ConsumerState:
    __slots__ = ("position", "waiter", "cancelled")

    def __init__(self, position: int) -> None:
        self.position = position
        self.waiter: Optional[asyncio.Future[None]] = None
        self.cancelled = False


def _settle(waiter: Optional[asyncio.Future[None]], error: Optional[BaseException] = None) -> None:
    if waiter is None or waiter.done():
        return
    if error is not None:
        waiter.set_exception(error)
    else:
        waiter.set_result(None)


def _normalize_thrown(exc: Union[BaseException, Type[BaseException], Any]) -> BaseException:
    if isinstance(exc, BaseException):
        return exc
    if isinstance(exc, type) and issubclass(exc, BaseException):
        return exc()
    return Exception(str(exc))


class _ReplayBuffer:
    """Shared absolute-position replay buffer with amortized compaction.

    Consumer positions are absolute. Buffer index =
    ``buffer_head + position - trim_offset``.
    """

    def __init__(self, stream_replay: StreamReplay) -> None:
        self._stream_replay: StreamReplay = stream_replay
        self._buffer: List[Any] = []
        self._buffer_head = 0
        self._trim_offset = 0
        self._consumers: Dict[int, _ConsumerState] = {}
        self._next_consumer_id = 0

    def _register_consumer(self) -> int:
        consumer_id = self._next_consumer_id
        self._next_consumer_id += 1
        self._consumers[consumer_id] = _ConsumerState(self._trim_offset)
        return consumer_id

    def _index_of(self, consumer: _ConsumerState) -> int:
        return self._buffer_head + consumer.position - self._trim_offset

    def _trim_consumed(self) -> None:
        if self._stream_replay == "full":
            return
        if not self._consumers:
            self._drop_unread_backlog()
            return

        min_position = min(consumer.position for consumer in self._consumers.values())
        next_head = self._buffer_head + min_position - self._trim_offset
        if next_head <= self._buffer_head:
            return

        self._trim_offset = min_position
        if next_head == len(self._buffer):
            self._buffer = []
            self._buffer_head = 0
            return

        self._buffer[self._buffer_head : next_head] = [_CLEARED] * (next_head - self._buffer_head)
        if next_head >= BUFFER_COMPACTION_MIN_HEAD and next_head * 2 >= len(self._buffer):
            self._buffer = self._buffer[next_head:]
            self._buffer_head = 0
            return
        self._buffer_head = next_head

    def _drop_unread_backlog(self) -> None:
        """Drop the retained backlog once no consumer can ever read it again.

        Before the first consumer joins, the backlog IS the catch-up history
        late joiners replay — retain it. Once at least one consumer has existed
        and none remain, any future consumer starts at the watermark anyway.
        """
        if self._next_consumer_id == 0:
            return
        dropped = len(self._buffer) - self._buffer_head
        if dropped == 0:
            return
        self._trim_offset += dropped
        self._buffer = []
        self._buffer_head = 0

    def _detach(self, consumer_id: int, error: Optional[BaseException] = None) -> bool:
        """Shared ``return()`` / ``throw()`` teardown. Returns True if attached."""
        consumer = self._consumers.get(consumer_id)
        if consumer is None:
            return False
        consumer.cancelled = True
        # Wake (or reject) a pending next so it does not hang forever on a
        # consumer that has just been removed.
        _settle(consumer.waiter, error)
        consumer.waiter = None
        del self._consumers[consumer_id]
        self._trim_consumed()
        return True


class ReplayConsumer:
    """Independent async iterator over a replay buffer.

    ``aclose()`` mirrors upstream ``return()``, ``athrow()`` mirrors ``throw()``.
    Unlike JS ``for await``, a Python ``async for`` that ``break``s does not
    detach the consumer; call ``aclose()`` to release it.
    """

    def __init__(self, owner: Any, consumer_id: int) -> None:
        self._owner = owner
        self._id = consumer_id

    def __aiter__(self) -> "ReplayConsumer":
        return self

    async def __anext__(self) -> Any:
        return await self._owner._consumer_next(self._id)

    async def aclose(self) -> None:
        await self._owner._consumer_return(self._id)

    async def athrow(self, exc: Union[BaseException, Type[BaseException], Any]) -> Any:
        await self._owner._consumer_throw(self._id, exc)


class ReusableReadableStream(_ReplayBuffer):
    def __init__(
        self,
        source: AsyncIterable[Any],
        *,
        stream_replay: StreamReplay = "full",
        on_value: Optional[Callable[[Any], None]] = None,
        is_terminal_value: Optional[Callable[[Any], bool]] = None,
    ) -> None:
        super().__init__(stream_replay)
        self._source = source
        self._on_value = on_value
        self._is_terminal_value = is_terminal_value
        self._source_iterator: Optional[AsyncIterator[Any]] = None
        self._source_complete = False
        self._source_error: Optional[BaseException] = None
        self._pump_started = False
        self._pump_task: Optional[asyncio.Task[None]] = None
        # True while the pump holds the source (upstream: `sourceReader`).
        self._reader_active = False
        # True while the pump is closing the source after a terminal value.
        self._terminal_closing = False
        self._cancel_requested = False
        self._source_cancel_task: Optional[asyncio.Future[Any]] = None

    @property
    def is_complete(self) -> bool:
        """True once the source stream has been fully read into the buffer.
        A fresh consumer created after this point replays the retained
        buffer without waiting on the source."""
        return self._source_complete

    def find_last_buffered(self, predicate: Callable[[Any], bool]) -> Optional[Any]:
        """Synchronously scan the retained buffer from the end, returning the
        last item matching ``predicate`` (``None`` if none). Sees only what has
        been buffered so far."""
        for i in range(len(self._buffer) - 1, self._buffer_head - 1, -1):
            item = self._buffer[i]
            if item is not _CLEARED and predicate(item):
                return item
        return None

    def create_consumer(self) -> ReplayConsumer:
        """Create a consumer that independently iterates over the stream.
        Full-replay consumers start at position 0; active-consumer replay
        starts at the current trim watermark."""
        consumer_id = self._register_consumer()
        if not self._pump_started:
            self._start_pump()
        return ReplayConsumer(self, consumer_id)

    async def _consumer_next(self, consumer_id: int) -> Any:
        while True:
            consumer = self._consumers.get(consumer_id)
            if consumer is None or consumer.cancelled:
                raise StopAsyncIteration

            index = self._index_of(consumer)
            if index < len(self._buffer):
                value = self._buffer[index]
                if value is _CLEARED:
                    raise RuntimeError("ReusableReadableStream buffer invariant violated: consumed slot was cleared")
                consumer.position += 1
                self._trim_consumed()
                return value

            if self._source_complete:
                self._consumers.pop(consumer_id, None)
                raise StopAsyncIteration

            if self._source_error is not None:
                self._consumers.pop(consumer_id, None)
                raise self._source_error

            waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            consumer.waiter = waiter
            try:
                await waiter
            finally:
                if consumer.waiter is waiter:
                    consumer.waiter = None

    async def _consumer_return(self, consumer_id: int) -> None:
        self._detach(consumer_id)

    async def _consumer_throw(self, consumer_id: int, exc: Any) -> None:
        error = _normalize_thrown(exc)
        self._detach(consumer_id, error)
        raise error

    def _start_pump(self) -> None:
        if self._pump_started:
            return
        self._pump_started = True
        self._source_iterator = self._source.__aiter__()
        self._reader_active = True
        self._pump_task = asyncio.get_running_loop().create_task(self._pump(self._source_iterator))

    async def _pump(self, iterator: AsyncIterator[Any]) -> None:
        try:
            while True:
                try:
                    value = await iterator.__anext__()
                except StopAsyncIteration:
                    self._source_complete = True
                    self._notify_all_consumers()
                    break

                self._buffer.append(value)
                if self._on_value is not None:
                    self._on_value(value)
                self._notify_all_consumers()

                if self._is_terminal_value is not None and self._is_terminal_value(value):
                    self._source_complete = True
                    self._notify_all_consumers()
                    self._terminal_closing = True
                    try:
                        await self._cancel_source(iterator)
                    except Exception:
                        # The terminal event is authoritative; cancellation is cleanup only.
                        pass
                    break
        except asyncio.CancelledError as exc:
            if not self._cancel_requested:
                self._source_error = exc
                self._notify_all_consumers()
            else:
                # Upstream: reader.cancel() resolves the pending read as done.
                self._source_complete = True
                self._notify_all_consumers()
        except BaseException as exc:
            self._source_error = exc
            self._notify_all_consumers()
        finally:
            self._reader_active = False
            self._terminal_closing = False

    async def _cancel_source(self, iterator: AsyncIterator[Any]) -> None:
        """Close the source exactly once, sharing the in-flight close."""
        if self._source_cancel_task is not None:
            await asyncio.shield(self._source_cancel_task)
            return
        aclose = getattr(iterator, "aclose", None)
        if aclose is None:
            return
        # Run the first close inline (not as a scheduled task) so the source's
        # cancel hook starts synchronously, like upstream `reader.cancel()`;
        # concurrent callers share its outcome through the future.
        done: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._source_cancel_task = done
        try:
            await aclose()
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                done.cancel()
            else:
                done.set_exception(exc)
                # Mark retrieved: concurrent awaiters (if any) still observe it.
                done.exception()
            raise
        else:
            done.set_result(None)

    def _notify_all_consumers(self) -> None:
        for consumer in self._consumers.values():
            if consumer.waiter is not None:
                _settle(consumer.waiter, self._source_error)
                consumer.waiter = None

    async def cancel(self) -> None:
        """Cancel the source stream and all consumers."""
        for consumer in self._consumers.values():
            consumer.cancelled = True
            _settle(consumer.waiter)
            consumer.waiter = None
        self._consumers.clear()

        # Cancellation is terminal: every consumer is gone, so nobody can read
        # the buffered backlog anymore. Drop it instead of pinning it.
        self._drop_unread_backlog()

        if self._reader_active and self._source_iterator is not None:
            iterator = self._source_iterator
            task = self._pump_task
            if task is not None and not task.done() and not self._terminal_closing:
                # A running async generator cannot be aclose()d while the pump
                # is awaiting it; cancelling the pump is the Python analog of
                # reader.cancel() resolving the in-flight read.
                self._cancel_requested = True
                task.cancel()
                await asyncio.wait({task})
                if not self._source_complete and self._source_error is None:
                    # Task was cancelled before its first step.
                    self._source_complete = True
                    self._reader_active = False
            await self._cancel_source(iterator)

        # The pump may have landed one in-flight value before cancellation took
        # effect — sweep once more so the buffer is empty.
        self._drop_unread_backlog()


__all__ = [
    "BUFFER_COMPACTION_MIN_HEAD",
    "ReplayConsumer",
    "ReusableReadableStream",
    "StreamReplay",
]
