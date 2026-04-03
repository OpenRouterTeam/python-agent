"""Tests for ToolEventBroadcaster."""

from collections.abc import AsyncIterator

import anyio
import pytest

from openrouter_agent import ToolEventBroadcaster


@pytest.mark.anyio
async def test_broadcaster_single_consumer():
    b: ToolEventBroadcaster[str] = ToolEventBroadcaster()
    b.push("event1")
    b.push("event2")
    b.complete()

    consumer = b.create_consumer()
    collected = []
    async for event in consumer:
        collected.append(event)

    assert collected == ["event1", "event2"]


@pytest.mark.anyio
async def test_broadcaster_multi_consumer():
    b: ToolEventBroadcaster[int] = ToolEventBroadcaster()
    b.push(1)
    b.push(2)
    b.push(3)
    b.complete()

    c1 = b.create_consumer()
    c2 = b.create_consumer()

    r1 = [item async for item in c1]
    r2 = [item async for item in c2]

    assert r1 == [1, 2, 3]
    assert r2 == [1, 2, 3]


@pytest.mark.anyio
async def test_broadcaster_error_propagation():
    b: ToolEventBroadcaster[str] = ToolEventBroadcaster()
    b.push("ok")
    b.complete(error=ValueError("test error"))

    consumer = b.create_consumer()
    collected = []
    with pytest.raises(ValueError, match="test error"):
        async for event in consumer:
            collected.append(event)

    assert collected == ["ok"]


def test_broadcaster_push_after_complete_raises():
    b: ToolEventBroadcaster[str] = ToolEventBroadcaster()
    b.complete()
    with pytest.raises(RuntimeError):
        b.push("too late")


@pytest.mark.anyio
async def test_broadcaster_concurrent_push_and_consume():
    """Verify consumers wake via event signaling, not spin-polling."""
    b: ToolEventBroadcaster[int] = ToolEventBroadcaster()
    consumer = b.create_consumer()
    collected: list[int] = []

    async def producer() -> None:
        for i in range(5):
            await anyio.sleep(0.01)
            b.push(i)
        b.complete()

    async def consume() -> None:
        async for event in consumer:
            collected.append(event)

    async with anyio.create_task_group() as tg:
        tg.start_soon(producer)
        tg.start_soon(consume)

    assert collected == [0, 1, 2, 3, 4]


@pytest.mark.anyio
async def test_broadcaster_multiple_concurrent_consumers():
    """Multiple consumers receiving events concurrently."""
    b: ToolEventBroadcaster[int] = ToolEventBroadcaster()
    c1 = b.create_consumer()
    c2 = b.create_consumer()
    r1: list[int] = []
    r2: list[int] = []

    async def producer() -> None:
        for i in range(3):
            await anyio.sleep(0.01)
            b.push(i)
        b.complete()

    async with anyio.create_task_group() as tg:
        tg.start_soon(producer)

        async def consume(consumer: AsyncIterator[int], results: list[int]) -> None:
            async for event in consumer:
                results.append(event)

        tg.start_soon(consume, c1, r1)
        tg.start_soon(consume, c2, r2)

    assert r1 == [0, 1, 2]
    assert r2 == [0, 1, 2]
