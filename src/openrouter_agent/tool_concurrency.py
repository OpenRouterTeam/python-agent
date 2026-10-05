"""Minimal FIFO semaphore used to bound tool-execution concurrency.

Port of upstream `src/lib/tool-concurrency.ts`.

Three gates exist per run:

- the round gate (`tool_concurrency.round`) — bounds simultaneous tool
  executions within a round;
- per-tool gates (`max_concurrency` on a tool) — bound simultaneous
  executions of one tool across the run;
- the background pool (`tool_concurrency.background`) — bounds detached
  background-tool work that escaped the round barrier.

Waiters are released strictly FIFO so a burst of calls cannot starve an
earlier one, and a released slot is handed *directly* to the next waiter.

Python divergence (cancellation): an `acquire()` whose awaiting task is
cancelled leaves the queue cleanly — and if the slot had already been handed
to it, the slot is released again instead of leaking. Upstream has no
cancellable waits (a JS promise cannot be abandoned by its awaiter).
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Callable, Deque, List, Optional, Sequence

#: Release function returned by `Semaphore.acquire`. Idempotent.
SemaphoreRelease = Callable[[], None]


class Semaphore:
    """A counting semaphore with FIFO waiters and idempotent releases."""

    def __init__(self, limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"Semaphore limit must be a positive integer, got {limit}")
        self._available = limit
        self._waiters: Deque[asyncio.Future[SemaphoreRelease]] = deque()

    async def acquire(self) -> SemaphoreRelease:
        """Acquire one slot. Returns immediately when a slot is free,
        otherwise queues FIFO. The returned release function is idempotent —
        releasing twice does not free two slots."""
        if self._available > 0:
            self._available -= 1
            return self._make_release()
        future: asyncio.Future[SemaphoreRelease] = asyncio.get_running_loop().create_future()
        self._waiters.append(future)
        try:
            return await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                # The slot was handed to us just before the cancel landed:
                # pass it on rather than leaking it.
                future.result()()
            else:
                future.cancel()
                try:
                    self._waiters.remove(future)
                except ValueError:
                    pass
            raise

    def _make_release(self) -> SemaphoreRelease:
        released = False

        def release() -> None:
            nonlocal released
            if released:
                return
            released = True
            while self._waiters:
                nxt = self._waiters.popleft()
                if nxt.done():
                    continue  # abandoned waiter
                # Hand the slot directly to the next waiter (no churn).
                nxt.set_result(self._make_release())
                return
            self._available += 1

        return release


async def acquire_all(gates: Sequence[Optional[Semaphore]]) -> SemaphoreRelease:
    """Acquire multiple gates in the given (fixed) order and return a single
    release that frees them in reverse order. `None` entries are skipped.

    Callers MUST pass gates in a globally consistent order (round gate before
    per-tool gate) — fixed ordering is what makes multi-gate acquisition
    deadlock-free. If acquisition is interrupted (cancellation), gates
    already held are released before the exception propagates.
    """
    releases: List[SemaphoreRelease] = []
    try:
        for gate in gates:
            if gate is not None:
                releases.append(await gate.acquire())
    except BaseException:
        for release in reversed(releases):
            release()
        raise

    def release_all() -> None:
        for release in reversed(releases):
            release()

    return release_all


__all__ = ["Semaphore", "SemaphoreRelease", "acquire_all"]
