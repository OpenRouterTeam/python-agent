"""turn.end must never be silently dropped.

Ports `packages/agent/tests/unit/turn-end-race-condition.test.ts`.

Upstream's bug: `startTurnBroadcasterExecution()` called `broadcaster.complete()`
without awaiting the pipe that was still draining the response stream, so the
`turn.end` pushed afterwards was silently discarded. The root cause is a
*contract* on `ToolEventBroadcaster`, not a defect in it: `push()` after
`complete()` is a no-op, so every caller must finish pushing before completing.

Two layers are covered, because either alone is insufficient:

1. The broadcaster contract (`push` after `complete` drops) and the two calling
   patterns — fire-and-forget vs. await-then-complete. These pin the mechanism so
   a future refactor toward the buggy shape fails here.
2. The **production** invariant through `call_model`. This port does not use
   upstream's fire-and-forget pipe — `ModelResult._run` appends turn events
   sequentially (`model_result.py:648-672`) — so the mechanism tests alone would
   pass even if the real loop stopped emitting `turn.end`. Layer 2 is what
   actually guards shipped behavior.

Determinism: upstream races a 20ms stream tick against a 5ms executor. Ported as
timing, it would flip under CI load. Here the stream is gated on an
`asyncio.Event` so ordering is explicit, not hoped for. Async primitives are
constructed inside the test body: `asyncio.Condition()` binds the running loop
eagerly on 3.9 and lazily on 3.13, so module-scope construction is a
version-specific landmine.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List

from openrouter_agent import call_model, tool
from openrouter_agent.reusable_stream import ReusableReadableStream
from openrouter_agent.tool_event_broadcaster import ToolEventBroadcaster
from tests._fixtures import QueuedClient, text_response, tool_call_response


async def _gated_stream(events: List[Dict[str, Any]], gate: asyncio.Event) -> AsyncIterator[Dict[str, Any]]:
    """Yield nothing until `gate` is set, so the pipe is provably mid-drain."""
    await gate.wait()
    for event in events:
        yield event


# -- Layer 1: the broadcaster contract ------------------------------------------


async def test_broadcaster_silently_drops_events_pushed_after_complete() -> None:
    """Characterization: this is the mechanism behind the dropped turn.end."""
    broadcaster = ToolEventBroadcaster()
    consumer = broadcaster.create_consumer()

    broadcaster.push({"type": "turn.start"})
    broadcaster.push({"type": "event1"})
    broadcaster.complete()
    broadcaster.push({"type": "turn.end"})  # silently dropped

    collected = [event async for event in consumer]

    assert [event["type"] for event in collected] == ["turn.start", "event1"]
    assert [event for event in collected if event["type"] == "turn.end"] == []


async def test_buggy_pattern_drops_turn_end_when_complete_precedes_pipe() -> None:
    """Fire-and-forget pipe + early complete() loses turn.end. Documents the bug."""
    broadcaster = ToolEventBroadcaster()
    gate = asyncio.Event()
    stream = ReusableReadableStream(
        _gated_stream([{"type": "response.output_text.delta"}, {"type": "response.completed"}], gate)
    )

    async def pipe() -> None:
        broadcaster.push({"type": "turn.start", "turnNumber": 0})
        async for event in stream.create_consumer():
            broadcaster.push(event)
        broadcaster.push({"type": "turn.end", "turnNumber": 0})

    pipe_task = asyncio.create_task(pipe())
    consumer = broadcaster.create_consumer()

    # Let the pipe push turn.start and block on the gated stream. The pipe is now
    # unambiguously unfinished — no sleep required.
    await asyncio.sleep(0)

    # BUG: complete() without awaiting pipe_task.
    broadcaster.complete()

    collected = [event async for event in consumer]

    gate.set()
    await pipe_task

    assert len([e for e in collected if e["type"] == "turn.start"]) == 1
    assert len([e for e in collected if e["type"] == "turn.end"]) == 0, (
        "turn.end should be dropped by the buggy pattern; if this now survives, "
        "the broadcaster's push-after-complete contract changed"
    )


async def test_fixed_pattern_preserves_turn_end_when_pipe_is_awaited() -> None:
    """Awaiting the pipe before complete() keeps turn.end, ordered, with turnNumber."""
    broadcaster = ToolEventBroadcaster()
    gate = asyncio.Event()
    stream = ReusableReadableStream(
        _gated_stream([{"type": "response.output_text.delta"}, {"type": "response.completed"}], gate)
    )

    async def pipe() -> None:
        broadcaster.push({"type": "turn.start", "turnNumber": 0})
        async for event in stream.create_consumer():
            broadcaster.push(event)
        broadcaster.push({"type": "turn.end", "turnNumber": 0})

    pipe_task = asyncio.create_task(pipe())
    consumer = broadcaster.create_consumer()

    async def execute_then_complete() -> None:
        gate.set()
        await pipe_task  # THE FIX: let turn.end land before completing
        broadcaster.complete()

    execution = asyncio.create_task(execute_then_complete())
    collected = [event async for event in consumer]
    await execution

    types = [event["type"] for event in collected]
    assert types.count("turn.start") == 1
    assert types.count("turn.end") == 1
    assert types.index("turn.start") < types.index("turn.end")

    # camelCase matches upstream; this port emits `turn_number` alongside it
    # (model_result.py:651-652).
    turn_start = next(event for event in collected if event["type"] == "turn.start")
    turn_end = next(event for event in collected if event["type"] == "turn.end")
    assert turn_start["turnNumber"] == 0
    assert turn_end["turnNumber"] == 0


# -- Layer 2: the production invariant ------------------------------------------


async def test_call_model_emits_exactly_one_paired_turn_end_per_turn() -> None:
    """Every turn.start is matched by exactly one turn.end, in order, across turns.

    A tool round plus a final turn means two of each. Asserting count and order —
    not mere membership — is the point: a membership check passes even if turn.end
    is emitted twice, never, or before turn.start.
    """
    client = QueuedClient([tool_call_response("r1", "echo"), text_response("r2", "done")])
    echo = tool(name="echo", input_schema=dict, output_schema=dict, execute=lambda params, ctx: {"ok": True})

    result = call_model(client, {"model": "test-model", "input": "go", "tools": [echo]})
    turn_events = [
        event async for event in result.get_full_responses_stream() if str(event["type"]).startswith("turn.")
    ]

    assert [event["type"] for event in turn_events] == ["turn.start", "turn.end", "turn.start", "turn.end"]
    assert [event["turnNumber"] for event in turn_events] == [0, 0, 1, 1]
