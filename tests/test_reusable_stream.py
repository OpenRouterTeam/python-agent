"""Tests for ReusableAsyncStream multi-consumer adapter."""

import anyio
import pytest

from openrouter_agent import ReusableAsyncStream


async def _aiter_list(items: list) -> None:
    for item in items:
        yield item


@pytest.mark.anyio
async def test_single_consumer():
    source = _aiter_list([1, 2, 3])
    stream = ReusableAsyncStream(source)

    async with anyio.create_task_group() as tg:
        await stream.start(tg)
        consumer = stream.create_consumer()
        collected = []
        async for item in consumer:
            collected.append(item)

    assert collected == [1, 2, 3]


@pytest.mark.anyio
async def test_multiple_consumers():
    source = _aiter_list(["a", "b", "c"])
    stream = ReusableAsyncStream(source)

    async with anyio.create_task_group() as tg:
        await stream.start(tg)
        c1 = stream.create_consumer()
        c2 = stream.create_consumer()

        result1 = []
        result2 = []
        async for item in c1:
            result1.append(item)
        async for item in c2:
            result2.append(item)

    assert result1 == ["a", "b", "c"]
    assert result2 == ["a", "b", "c"]


@pytest.mark.anyio
async def test_late_consumer():
    source = _aiter_list([10, 20, 30])
    stream = ReusableAsyncStream(source)

    async with anyio.create_task_group() as tg:
        await stream.start(tg)

        # First consumer reads everything
        c1 = stream.create_consumer()
        r1 = []
        async for item in c1:
            r1.append(item)

        # Late consumer should still get all items from buffer
        c2 = stream.create_consumer()
        r2 = []
        async for item in c2:
            r2.append(item)

    assert r1 == [10, 20, 30]
    assert r2 == [10, 20, 30]
