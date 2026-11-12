"""Replayable fan-out over a single async source.

Ports `packages/agent/src/lib/reusable-stream.ts`.

Key features:

- Multiple concurrent consumers with independent read positions
- New consumers can attach while streaming is active
- Full replay for delayed and sequential consumers by default
- Opt-in active-consumer replay compaction for bounded memory
  (``stream_replay="active-consumers"``)
- Each consumer can read at its own pace
"""

from __future__ import annotations

import asyncio
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
)

T = TypeVar("T")

StreamReplay = Literal["full", "active-consumers"]
"""Replay policy shared by `ReusableReadableStream` and `ToolEventBroadcaster`.

``"full"`` (default) retains every value so delayed and sequential consumers
replay the complete history. ``"active-consumers"`` trims values once every
attached consumer has read them; consumers created later start at the current
trim watermark instead of position 0.
"""

# Compact the backing list only once the consumed prefix is at least this long
# (and at least half the list), so trimming stays amortized O(1) per value.
BUFFER_COMPACTION_MIN_HEAD = 1024

# Marks a consumed slot that has been released. A dedicated sentinel (rather
# than upstream's `undefined`) because `None` is a legitimate stream value.
_CLEARED: Any = object()


class _ConsumerState:
    __slots__ = ("position", "waiter", "cancelled")

    def __init__(self, position: int) -> None:
        self.position = position
        self.waiter: Optional[asyncio.Future[None]] = None
        self.cancelled = False


class _ReplayBuffer(Generic[T]):
    """Buffer + consumer bookkeeping shared by the stream and the broadcaster.

    Consumer positions are absolute. Buffer index is
    ``buffer_head + position - trim_offset``.
    """

    _invariant_message = "replay buffer invariant violated: consumed slot was cleared"

    def __init__(self, stream_replay: StreamReplay = "full") -> None:
        self._stream_replay: StreamReplay = stream_replay
        self._buffer: List[Any] = []
        self._buffer_head = 0
        self._trim_offset = 0
        self._consumers: Dict[int, _ConsumerState] = {}
        self._next_consumer_id = 0

    @property
    def stream_replay(self) -> StreamReplay:
        return self._stream_replay

    # -- hooks the subclasses implement ------------------------------------

    def _on_consumer_attached(self) -> None:
        """Called after a consumer is registered (e.g. to start a pump)."""

    def _terminal_state(self, consumer_id: int) -> bool:
        """Return True (consumer done) or raise, when the consumer is caught up
        and the source has finished. Return False to wait for more data."""
        raise NotImplementedError

    # -- buffer management -------------------------------------------------

    def _append(self, value: T) -> None:
        self._buffer.append(value)

    def _buffer_index(self, state: _ConsumerState) -> int:
        return self._buffer_head + state.position - self._trim_offset

    def _has_data(self, state: _ConsumerState) -> bool:
        return self._buffer_index(state) < len(self._buffer)

    def find_last_buffered(self, predicate: Callable[[T], bool]) -> Optional[T]:
        """Synchronously scan the retained buffer from the end, returning the
        last item matching ``predicate``.

        Sees only what has been buffered (and retained) so far. Unlike a
        consumer replay this costs no per-item await.
        """
        for index in range(len(self._buffer) - 1, self._buffer_head - 1, -1):
            item = self._buffer[index]
            if item is not _CLEARED and predicate(item):
                return item
        return None

    def _trim_consumed(self) -> None:
        if self._stream_replay == "full":
            return
        if not self._consumers:
            self._drop_unread_backlog()
            return

        minimum = min(state.position for state in self._consumers.values())
        next_head = self._buffer_head + minimum - self._trim_offset
        if next_head <= self._buffer_head:
            return

        self._trim_offset = minimum
        if next_head == len(self._buffer):
            self._buffer = []
            self._buffer_head = 0
            return

        for index in range(self._buffer_head, next_head):
            self._buffer[index] = _CLEARED
        if next_head >= BUFFER_COMPACTION_MIN_HEAD and next_head * 2 >= len(self._buffer):
            self._buffer = self._buffer[next_head:]
            self._buffer_head = 0
            return
        self._buffer_head = next_head

    def _drop_unread_backlog(self) -> None:
        """Drop the retained backlog once no consumer can ever read it again.

        Before the first consumer joins, the backlog IS the catch-up history
        late joiners replay, so it is retained. Once at least one consumer has
        existed and none remain, any future consumer starts at the watermark
        anyway, so the retained values would stay pinned for nothing.
        """
        if self._next_consumer_id == 0:
            return
        dropped = len(self._buffer) - self._buffer_head
        if dropped == 0:
            return
        self._trim_offset += dropped
        self._buffer = []
        self._buffer_head = 0

    def _notify_waiting(self) -> None:
        for state in self._consumers.values():
            waiter = state.waiter
            state.waiter = None
            if waiter is not None and not waiter.done():
                waiter.set_result(None)

    # -- consumers ---------------------------------------------------------

    def create_consumer(self) -> "ReplayConsumer[T]":
        consumer_id = self._next_consumer_id
        self._next_consumer_id += 1
        self._consumers[consumer_id] = _ConsumerState(self._trim_offset)
        self._on_consumer_attached()
        return ReplayConsumer(self, consumer_id)

    def _detach(self, consumer_id: int, error: Optional[BaseException] = None) -> None:
        state = self._consumers.get(consumer_id)
        if state is None:
            return
        state.cancelled = True
        # Settle a pending __anext__ so it does not hang forever on a consumer
        # that has just been removed: it either re-runs and reports done, or
        # raises the same error passed to athrow().
        waiter = state.waiter
        state.waiter = None
        if waiter is not None and not waiter.done():
            if error is None:
                waiter.set_result(None)
            else:
                waiter.set_exception(error)
        del self._consumers[consumer_id]
        self._trim_consumed()
        self._after_detach()

    def _after_detach(self) -> None:
        """Called after a consumer is removed via aclose()/athrow()."""

    async def _next(self, consumer_id: int) -> T:
        while True:
            state = self._consumers.get(consumer_id)
            if state is None or state.cancelled:
                raise StopAsyncIteration

            index = self._buffer_index(state)
            if index < len(self._buffer):
                value = self._buffer[index]
                if value is _CLEARED:
                    raise RuntimeError(f"{type(self).__name__} buffer invariant violated: consumed slot was cleared")
                state.position += 1
                self._trim_consumed()
                return value

            if self._terminal_state(consumer_id):
                raise StopAsyncIteration

            waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            state.waiter = waiter
            try:
                await waiter
            finally:
                if state.waiter is waiter:
                    state.waiter = None


class ReplayConsumer(AsyncIterator[T]):
    """Independent read cursor over a replay buffer.

    Unlike a plain async generator, ``aclose()`` / ``athrow()`` may be called
    while another task is awaiting ``__anext__``: the pending read is settled
    (done, or the thrown error) instead of hanging.
    """

    def __init__(self, owner: _ReplayBuffer[T], consumer_id: int) -> None:
        self._owner = owner
        self._consumer_id = consumer_id

    def __aiter__(self) -> "ReplayConsumer[T]":
        return self

    async def __anext__(self) -> T:
        return await self._owner._next(self._consumer_id)

    async def aclose(self) -> None:
        self._owner._detach(self._consumer_id)

    async def athrow(self, error: Union[BaseException, type]) -> T:
        exc: BaseException
        if isinstance(error, BaseException):
            exc = error
        elif isinstance(error, type) and issubclass(error, BaseException):
            exc = error()
        else:
            exc = Exception(str(error))
        self._owner._detach(self._consumer_id, exc)
        raise exc


class ReusableReadableStream(_ReplayBuffer[T]):
    """Buffer a single async source and fan it out to many consumers.

    Options (keyword-only, mirroring upstream ``ReusableReadableStreamOptions``):

    - ``stream_replay``: ``"full"`` (default) or ``"active-consumers"``.
    - ``on_value``: called synchronously with each value as it is buffered.
    - ``is_terminal_value``: when it returns True for a value, that value is the
      final buffered event; the source is closed once and pumping stops.
    """

    def __init__(
        self,
        source: AsyncIterable[T],
        *,
        stream_replay: StreamReplay = "full",
        on_value: Optional[Callable[[T], None]] = None,
        is_terminal_value: Optional[Callable[[T], bool]] = None,
    ) -> None:
        super().__init__(stream_replay)
        self._source = source
        self._source_iterator: Optional[AsyncIterator[T]] = None
        self._complete = False
        self._error: Optional[BaseException] = None
        self._started = False
        self._pump_task: Optional[asyncio.Task[None]] = None
        self._source_closed = False
        self._on_value = on_value
        self._is_terminal_value = is_terminal_value

    @property
    def is_complete(self) -> bool:
        """True once the source stream has been fully read into the buffer.
        A fresh consumer created after this point replays the retained
        buffer without waiting on the source."""
        return self._complete

    def _on_consumer_attached(self) -> None:
        if not self._started:
            self._started = True
            self._pump_task = asyncio.get_running_loop().create_task(self._pump())

    def _terminal_state(self, consumer_id: int) -> bool:
        if self._error is not None:
            self._consumers.pop(consumer_id, None)
            raise self._error
        if self._complete:
            self._consumers.pop(consumer_id, None)
            return True
        return False

    async def _close_source(self) -> None:
        """Close the source iterator at most once (terminal value / cancel)."""
        if self._source_closed:
            return
        self._source_closed = True
        iterator = self._source_iterator
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()

    async def _pump(self) -> None:
        iterator = self._source.__aiter__()
        self._source_iterator = iterator
        try:
            while True:
                try:
                    value = await iterator.__anext__()
                except StopAsyncIteration:
                    break
                self._append(value)
                if self._on_value is not None:
                    self._on_value(value)
                self._notify_waiting()
                if self._is_terminal_value is not None and self._is_terminal_value(value):
                    self._complete = True
                    self._notify_waiting()
                    try:
                        await self._close_source()
                    except Exception:
                        # The terminal event is authoritative; closing is cleanup only.
                        pass
                    break
        except asyncio.CancelledError:
            # cancel() tore the pump down; consumers observe completion.
            pass
        except Exception as exc:
            self._error = exc
        finally:
            self._complete = True
            self._notify_waiting()

    async def cancel(self) -> None:
        """Cancel the source and every consumer, dropping the retained backlog."""
        for state in self._consumers.values():
            state.cancelled = True
            waiter = state.waiter
            state.waiter = None
            if waiter is not None and not waiter.done():
                waiter.set_result(None)
        self._consumers.clear()
        # Cancellation is terminal: nobody can read the backlog anymore.
        self._drop_unread_backlog()
        task = self._pump_task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
            # The pump swallows its own cancellation, so awaiting it only
            # raises if the caller of cancel() is itself cancelled.
            await task
        self._complete = True
        try:
            await self._close_source()
        except Exception:
            pass
        # The pump may have landed one in-flight value before it was
        # cancelled; sweep once more so the buffer is empty regardless.
        self._drop_unread_backlog()


__all__ = [
    "BUFFER_COMPACTION_MIN_HEAD",
    "ReplayConsumer",
    "ReusableReadableStream",
    "StreamReplay",
]
