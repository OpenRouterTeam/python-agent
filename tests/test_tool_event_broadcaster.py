"""Tests for ToolEventBroadcaster."""

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
