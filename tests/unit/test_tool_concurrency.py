"""Focused tests for `tool_concurrency.py` (upstream `src/lib/tool-concurrency.ts`).

Upstream has no dedicated unit test file for this module (it is exercised via
the `call_model` concurrency tests); these pin the primitive's contract:
FIFO handoff, idempotent release, fixed-order multi-gate acquire/release, and
the Python-only cancellation hygiene.
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from openrouter_agent.tool_concurrency import Semaphore, acquire_all


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_limit_must_be_a_positive_integer(limit: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        Semaphore(limit)  # type: ignore[arg-type]


async def test_waiters_are_released_strictly_fifo() -> None:
    sem = Semaphore(1)
    first = await sem.acquire()
    order: List[int] = []

    async def waiter(i: int) -> None:
        release = await sem.acquire()
        order.append(i)
        release()

    tasks = [asyncio.ensure_future(waiter(i)) for i in range(4)]
    await asyncio.sleep(0)  # every waiter is queued
    assert order == []
    first()
    await asyncio.gather(*tasks)
    assert order == [0, 1, 2, 3]


async def test_release_is_idempotent() -> None:
    sem = Semaphore(1)
    release = await sem.acquire()
    release()
    release()  # must not free a second slot
    a = await sem.acquire()
    blocked = asyncio.ensure_future(sem.acquire())
    await asyncio.sleep(0)
    assert blocked.done() is False
    a()
    b = await blocked
    b()


async def test_limit_bounds_concurrent_holders() -> None:
    sem = Semaphore(2)
    active = 0
    peak = 0
    gates = [asyncio.Event() for _ in range(5)]

    async def body(i: int) -> None:
        nonlocal active, peak
        release = await sem.acquire()
        active += 1
        peak = max(peak, active)
        await gates[i].wait()
        active -= 1
        release()

    tasks = [asyncio.ensure_future(body(i)) for i in range(5)]
    await asyncio.sleep(0)
    assert active == 2
    for gate in gates:
        gate.set()
    await asyncio.gather(*tasks)
    assert peak == 2
    assert active == 0


async def test_cancelled_waiter_leaves_the_queue_without_leaking_a_slot() -> None:
    sem = Semaphore(1)
    held = await sem.acquire()
    cancelled = asyncio.ensure_future(sem.acquire())
    nxt = asyncio.ensure_future(sem.acquire())
    await asyncio.sleep(0)
    cancelled.cancel()
    await asyncio.gather(cancelled, return_exceptions=True)
    held()
    release = await nxt  # slot skipped the cancelled waiter
    release()
    again = await sem.acquire()
    again()


async def test_slot_handed_to_a_waiter_cancelled_before_resuming_is_returned() -> None:
    sem = Semaphore(1)
    held = await sem.acquire()
    waiter = asyncio.ensure_future(sem.acquire())
    await asyncio.sleep(0)
    held()  # hands the slot to `waiter` (resolves its future)
    waiter.cancel()  # ...which is cancelled before it resumes
    await asyncio.gather(waiter, return_exceptions=True)
    assert waiter.cancelled() is True
    release = await asyncio.wait_for(sem.acquire(), timeout=5)
    release()


async def test_acquire_all_skips_none_and_releases_in_reverse_order() -> None:
    round_gate = Semaphore(1)
    tool_gate = Semaphore(1)
    release = await acquire_all([round_gate, None, tool_gate])
    blocked_round = asyncio.ensure_future(round_gate.acquire())
    blocked_tool = asyncio.ensure_future(tool_gate.acquire())
    await asyncio.sleep(0)
    assert (blocked_round.done(), blocked_tool.done()) == (False, False)
    release()
    (await blocked_round)()
    (await blocked_tool)()


async def test_acquire_all_releases_held_gates_when_interrupted() -> None:
    a = Semaphore(1)
    b = Semaphore(1)
    hold_b = await b.acquire()
    pending = asyncio.ensure_future(acquire_all([a, b]))
    await asyncio.sleep(0)  # holds `a`, waits on `b`
    pending.cancel()
    await asyncio.gather(pending, return_exceptions=True)
    # `a` was released on the way out.
    release_a = await asyncio.wait_for(a.acquire(), timeout=5)
    release_a()
    hold_b()
