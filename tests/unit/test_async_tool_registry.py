"""Ports `packages/agent/tests/unit/async-tool-registry.test.ts`.

Divergences from upstream's test mechanics (not its assertions):

- Upstream uses vitest fake timers (`vi.advanceTimersByTimeAsync`). This port
  has no fake clock; the timeout case instead uses a 1 ms deadline and gates
  on the controller's own `asyncio.Event` (`controller.wait()`), so the test
  waits for exactly the event it asserts rather than racing a sleep.
- `AbortController` -> `CancellationController` (`tool_task.py` docstring
  documents the mapping): `signal.aborted` -> `cancelled`, `signal.reason`
  -> `reason`.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from openrouter_agent.async_tool_registry import AsyncToolRegistry
from openrouter_agent.tool_task import CancellationController, ToolTask


def make_task(call_id: str, controller: Optional[CancellationController] = None) -> ToolTask:
    return ToolTask(
        task_id=f"task_{call_id}",
        call_id=call_id,
        tool_name="render_video",
        mode="background",
        controller=controller,
    )


# --- AsyncToolRegistry — timeout settlement ---------------------------------


async def test_deadline_expiry_aborts_the_running_body() -> None:
    """Controller captured before settle clears it."""
    registry = AsyncToolRegistry()
    controller = CancellationController()
    task = make_task("call_t1", controller)

    # Work that only settles when its cancellation fires — the shape of a
    # cooperative long-running body.
    async def work() -> Any:
        await controller.wait()
        assert controller.reason is not None
        raise controller.reason

    registry.track_background(task, work(), timeout_ms=1)
    await asyncio.wait_for(controller.wait(), timeout=5)

    # The body's signal MUST have fired — a timed-out task is told to stop.
    assert controller.cancelled is True
    assert "timed out after 1ms" in str(controller.reason)

    settled = registry.take_settled()
    assert len(settled) == 1
    assert settled[0].call_id == "call_t1"
    assert settled[0].status == "timed_out"
    # Leak guard still holds after settlement.
    assert task.controller is None


async def test_work_settling_before_the_deadline_wins_the_race() -> None:
    registry = AsyncToolRegistry()
    controller = CancellationController()
    task = make_task("call_t2", controller)

    done: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    done.set_result({"url": "https://done"})
    registry.track_background(task, done, timeout_ms=5)
    await asyncio.sleep(0)  # let the done-callback land

    settled = registry.take_settled()
    assert len(settled) == 1
    assert settled[0].call_id == "call_t2"
    assert settled[0].status == "completed"
    assert settled[0].result == {"url": "https://done"}

    # Past the (cleared) deadline: asyncio runs timers in deadline order, so a
    # surviving 5 ms timer would fire before this 20 ms sleep resumes.
    await asyncio.sleep(0.02)
    assert controller.cancelled is False
    assert registry.take_settled() == []


# --- AsyncToolRegistry — grace-window visibility (register/untrack) ---------


def test_registered_task_is_reachable_by_steer_and_cancel() -> None:
    registry = AsyncToolRegistry()
    controller = CancellationController()
    task = make_task("call_g1", controller)
    registry.register(task)

    # Visible to snapshots and lookups during the grace window.
    assert registry.get_task(task.task_id) is task
    assert len(registry.snapshot()) == 1

    # Steering queues into the task inbox instead of returning False.
    assert registry.send_to_task(task.task_id, "go faster") is True
    received: List[Any] = []
    task.on_message(received.append)
    assert received == ["go faster"]

    # Cancel aborts the controller (the grace race observes the rejection).
    assert registry.cancel_task(task.task_id, "changed my mind") is True
    assert controller.cancelled is True


def test_untrack_removes_the_task_and_any_queued_settlement() -> None:
    registry = AsyncToolRegistry()
    controller = CancellationController()
    task = make_task("call_g2", controller)
    registry.register(task)
    registry.cancel_task(task.task_id, "racing in-window settle")

    # In-window settle path: the sync output already reports the outcome —
    # no envelope may remain queued.
    registry.untrack("call_g2")
    assert registry.take_settled() == []
    assert registry.get_task(task.task_id) is None
    assert registry.has_tasks() is False


# --- ToolTask — tail_logs bounds --------------------------------------------


def test_tail_logs_zero_returns_no_entries() -> None:
    task = make_task("call_t3")
    task.append_log("one")
    task.append_log("two")
    assert task.tail_logs(0) == []
    assert task.tail_logs(-3) == []
    assert len(task.tail_logs(1.9)) == 1
    assert len(task.tail_logs(2)) == 2


# --- Additional registry behaviors (no upstream counterpart) -----------------


async def test_settlement_is_first_writer_wins_and_queued_exactly_once() -> None:
    registry = AsyncToolRegistry()
    controller = CancellationController()
    task = make_task("call_fw", controller)
    gate = asyncio.Event()

    async def work() -> Any:
        await gate.wait()
        return {"late": True}

    body = asyncio.ensure_future(work())
    registry.track_background(task, body)
    assert registry.has_in_flight() is True
    assert registry.cancel_task(task.task_id, "stop") is True
    assert registry.cancel_task(task.task_id, "again") is False  # already settled
    gate.set()
    # The registry's done-callback was registered before this await's wakeup,
    # so it has run by the time `await body` returns.
    assert await body == {"late": True}

    settled = registry.take_settled()
    assert [(s.call_id, s.status, s.error) for s in settled] == [("call_fw", "cancelled", "stop")]
    assert task.status == "cancelled"
    assert task.result is None  # the body's late completion was a no-op
    assert task.last_log is not None and task.last_log.kind == "system"
    assert registry.has_in_flight() is False


async def test_failed_vs_cancelled_classification_and_input_passthrough() -> None:
    registry = AsyncToolRegistry()
    plain = ToolTask(task_id="t_f", call_id="c_f", tool_name="x", mode="background", input={"q": 1})

    async def boom() -> Any:
        raise RuntimeError("kaput")

    registry.track_background(plain, boom())
    await registry.drain(5_000)
    settled = registry.take_settled()
    assert len(settled) == 1
    assert settled[0].status == "failed"
    assert settled[0].error == "kaput"
    assert settled[0].input == {"q": 1}
    assert settled[0].duration_ms >= 0


async def test_drain_waits_for_every_in_flight_task_in_settle_order() -> None:
    registry = AsyncToolRegistry()
    gate_a, gate_b = asyncio.Event(), asyncio.Event()

    async def run(gate: asyncio.Event, value: str) -> str:
        await gate.wait()
        return value

    body_a = asyncio.ensure_future(run(gate_a, "A"))
    body_b = asyncio.ensure_future(run(gate_b, "B"))
    registry.track_background(make_task("a"), body_a)
    registry.track_background(make_task("b"), body_b)
    drain = asyncio.ensure_future(registry.drain(5_000))
    await asyncio.sleep(0)
    assert drain.done() is False
    gate_b.set()
    await body_b
    assert registry.has_in_flight() is True
    assert drain.done() is False  # 'a' still in flight
    gate_a.set()
    assert await drain is True
    assert [s.result for s in registry.take_settled()] == ["B", "A"]


async def test_drain_times_out_when_work_never_settles() -> None:
    registry = AsyncToolRegistry()
    never = asyncio.Event()
    controller = CancellationController()
    body = asyncio.ensure_future(never.wait())
    controller.attach_task(body)  # cancel() also asyncio-cancels the body
    registry.track_background(make_task("stuck", controller), body)
    assert await registry.drain(1) is False
    assert registry.has_in_flight() is True
    registry.abort_all("Run ended (on_run_end: cancel)")
    await asyncio.gather(body, return_exceptions=True)
    assert body.cancelled() is True
    settled = registry.take_settled()
    assert [(s.status, s.error) for s in settled] == [("cancelled", "Run ended (on_run_end: cancel)")]
    assert await registry.drain(1) is True


async def test_deferred_tasks_are_not_in_flight_and_reject_steering() -> None:
    registry = AsyncToolRegistry()
    task = registry.track_deferred(call_id="c_d", task_id="ticket_1", name="legal_review", poll_after_ms=60_000)
    assert task.mode == "defer"
    assert registry.has_in_flight() is False
    assert registry.has_tasks() is True
    try:
        registry.send_to_task("ticket_1", "hurry")
    except RuntimeError as error:
        assert "deferred" in str(error)
    else:  # pragma: no cover
        raise AssertionError("send_to_task must raise for a deferred task")
    # abort_all leaves deferred tasks alone; cancel_task is local-only but works.
    registry.abort_all()
    assert task.status == "working"
    assert registry.cancel_task("ticket_1") is True
    assert [(s.status, s.error) for s in registry.take_settled()] == [("cancelled", "Task ticket_1 cancelled")]
    snap = registry.snapshot()
    assert len(snap) == 1
    assert (snap[0].mode, snap[0].status, snap[0].poll_after_ms) == ("defer", "cancelled", 60_000)


def test_mark_working_as_orphaned_and_snapshot_last_log() -> None:
    registry = AsyncToolRegistry()
    task = make_task("c_o")
    registry.register(task)
    task.append_log({"step": "x" * 300})
    orphaned = registry.mark_working_as_orphaned()
    assert len(orphaned) == 1
    assert orphaned[0].orphaned is True
    assert orphaned[0].last_log is not None
    assert len(orphaned[0].last_log.text) == 201  # 200 chars + ellipsis
    assert orphaned[0].last_log.text.endswith("…")
    assert task.orphaned is True
    assert registry.snapshot()[0].orphaned is True


def test_generate_task_id_is_unique_and_prefixed() -> None:
    registry = AsyncToolRegistry()
    ids = {registry.generate_task_id() for _ in range(5)}
    assert len(ids) == 5
    assert all(i.startswith("task_") for i in ids)
