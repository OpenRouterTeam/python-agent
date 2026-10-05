"""Per-run registry of async tool tasks (background, deferred, agent).

Port of upstream `src/lib/async-tool-registry.ts`.

Owns the in-process half of async tool support:

- background/agent tasks: tracks the in-flight work + `ToolTask` (logs,
  inbox, transcript source), queues settled outcomes for turn-boundary
  harvesting, supports drain / cancel / abort-all at run end;
- deferred tasks: tracks identity only (the work lives outside the process);
  the durable copy is mirrored to `ConversationState.pending_async_tools` by
  the engine.

The registry knows nothing about the tool loop or concurrency pools — the
engine (`ModelResult`) owns semaphores and decides when to harvest and how to
deliver. Settlement is first-writer-wins: a task settles exactly once (cancel
racing completion is safe), and settled tasks drop their controller
references so they cannot retain the run graph.

Timing uses the running asyncio loop (`loop.call_later` for task deadlines,
`asyncio.wait` for drain), so every method that starts a timer or awaits
(`track_background`, `drain`) must run inside an event loop.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Dict, List, Optional

from typing_extensions import Literal, TypeAlias

from .tool_task import TaskLogKind, ToolTask
from .tool_types import PendingAsyncTool, PendingAsyncToolLastLog

SettledStatus: TypeAlias = Literal["completed", "failed", "cancelled", "timed_out"]


@dataclass(frozen=True)
class SettledToolTask:
    """Outcome of a settled async tool task, harvested by the engine at turn
    boundaries (`flush_async_tool_deliveries`) or during end-of-run drain."""

    call_id: str
    task_id: str
    name: str
    status: SettledStatus
    #: Wall-clock ms from task start to settlement — feeds PostToolUse.
    duration_ms: int
    #: Meaningful when status is 'completed' (may legitimately be None).
    result: Any = None
    #: Present otherwise.
    error: Optional[str] = None
    #: The call's arguments, when tracked — feeds PostToolUse at settle.
    input: Optional[Dict[str, Any]] = None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _fmt_ms(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _error_message(error: BaseException) -> str:
    message = str(error)
    return message if message else type(error).__name__


def render_last_log(data: Any) -> str:
    """Render a log entry's data to the small persisted `last_log.text` form."""
    if isinstance(data, str):
        text = data
    else:
        try:
            text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(data)
    return f"{text[:200]}…" if len(text) > 200 else text


def _last_log_of(task: ToolTask) -> Optional[PendingAsyncToolLastLog]:
    last = task.last_log
    if last is None:
        return None
    return PendingAsyncToolLastLog(at=last.at, text=render_last_log(last.data))


class AsyncToolRegistry:
    """Per-run registry of async tool tasks. See module docstring."""

    def __init__(self) -> None:
        # Keyed by call id (insertion-ordered, like upstream's Map).
        self._tasks: Dict[str, ToolTask] = {}
        #: Settled outcomes not yet harvested by the engine.
        self._settled_queue: List[SettledToolTask] = []
        self._task_counter = 0
        #: Futures waiting on "any task settled" (drain support).
        self._settle_waiters: List[asyncio.Future[bool]] = []
        #: Background work futures, held so they are not garbage-collected.
        self._work: Dict[str, asyncio.Future[Any]] = {}

    def generate_task_id(self) -> str:
        """Generate a task id for background/agent tasks (deferred bring their own)."""
        self._task_counter += 1
        return f"task_{uuid.uuid4()}"

    def register(self, task: ToolTask) -> None:
        """Register a task so it is visible to steering / cancel / snapshots.
        Used for background tasks DURING their grace window."""
        self._tasks[task.call_id] = task

    def untrack(self, call_id: str) -> None:
        """Remove a task registered via `register` whose work settled inside
        the grace window, along with any settlement queued for it meanwhile
        (e.g. a `cancel_task` racing the in-window completion)."""
        self._tasks.pop(call_id, None)
        self._settled_queue = [s for s in self._settled_queue if s.call_id != call_id]

    def track_background(
        self,
        task: ToolTask,
        work: Awaitable[Any],
        *,
        timeout_ms: Optional[float] = None,
    ) -> None:
        """Track an already-started background/agent task.

        `work` is the tool's in-flight run (output-validated) — a coroutine,
        `asyncio.Task` or future. When `timeout_ms` is set the task is raced
        against it, so a body that ignores its cancellation still settles as
        `timed_out` instead of hanging the drain. The deadline is anchored at
        `task.started_at`, so `timeout_ms` bounds the task's TOTAL runtime.
        """
        self._tasks[task.call_id] = task
        loop = asyncio.get_running_loop()
        future = asyncio.ensure_future(work)
        self._work[task.call_id] = future

        timer: Optional[asyncio.TimerHandle] = None
        if timeout_ms is not None and timeout_ms > 0:
            remaining_ms = max(0.0, timeout_ms - (_now_ms() - task.started_at))

            def on_timeout() -> None:
                timeout_error = TimeoutError(f'Tool "{task.tool_name}" task timed out after {_fmt_ms(timeout_ms)}ms')
                # Capture the controller BEFORE settling — settle() drops the
                # reference as a leak guard, and the running body must still
                # be told to stop.
                controller = task.controller
                self._settle(task.call_id, "timed_out", error=str(timeout_error))
                if controller is not None:
                    controller.cancel(timeout_error)

            timer = loop.call_later(remaining_ms / 1000, on_timeout)

        def on_done(done: asyncio.Future[Any]) -> None:
            if timer is not None:
                timer.cancel()
            if self._work.get(task.call_id) is done:
                del self._work[task.call_id]
            controller = task.controller
            if done.cancelled():
                reason = controller.reason if controller is not None else None
                self._settle(
                    task.call_id,
                    "cancelled",
                    error=_error_message(reason) if reason is not None else "cancelled",
                )
                return
            error = done.exception()
            if error is not None:
                cancelled = controller.cancelled if controller is not None else False
                self._settle(task.call_id, "cancelled" if cancelled else "failed", error=_error_message(error))
                return
            self._settle(task.call_id, "completed", result=done.result())

        future.add_done_callback(on_done)

    def track_deferred(
        self,
        *,
        call_id: str,
        task_id: str,
        name: str,
        expires_at: Optional[int] = None,
        poll_after_ms: Optional[int] = None,
    ) -> ToolTask:
        """Track a deferred task (identity only; work lives outside the process)."""
        task = ToolTask(
            task_id=task_id,
            call_id=call_id,
            tool_name=name,
            mode="defer",
            expires_at=expires_at,
            poll_after_ms=poll_after_ms,
        )
        self._tasks[call_id] = task
        return task

    def _settle(
        self,
        call_id: str,
        status: SettledStatus,
        *,
        result: Any = None,
        error: Optional[str] = None,
    ) -> None:
        task = self._tasks.get(call_id)
        if task is None or task.status != "working":
            return  # first writer wins
        task.status = "completed" if status == "completed" else "cancelled" if status == "cancelled" else "failed"
        task.settled_at = _now_ms()
        if status == "completed":
            task.result = result
        else:
            task.error = error
            task.append_log(error, "system")
        # Leak guard: a settled task must not retain the run graph.
        task.controller = None
        self._settled_queue.append(
            SettledToolTask(
                call_id=call_id,
                task_id=task.task_id,
                name=task.tool_name,
                status=status,
                duration_ms=task.settled_at - task.started_at,
                result=result if status == "completed" else None,
                error=None if status == "completed" else error,
                input=task.input,
            )
        )
        waiters, self._settle_waiters = self._settle_waiters, []
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(True)

    def take_settled(self) -> List[SettledToolTask]:
        """Harvest (and clear) settled outcomes queued since the last call."""
        settled, self._settled_queue = self._settled_queue, []
        return settled

    def has_in_flight(self) -> bool:
        """True when any background/agent task is still in flight."""
        return any(t.mode != "defer" and t.status == "working" for t in self._tasks.values())

    def has_unharvested_settled(self) -> bool:
        """True when any settled outcome awaits harvesting."""
        return len(self._settled_queue) > 0

    def has_tasks(self) -> bool:
        """True when any task (any mode) has been tracked this run."""
        return len(self._tasks) > 0

    def get_task(self, task_id: str) -> Optional[ToolTask]:
        """Find the live task for a task id."""
        for task in self._tasks.values():
            if task.task_id == task_id:
                return task
        return None

    def find_by_call_id(self, call_id: str) -> Optional[ToolTask]:
        """Find the live task for a call id."""
        return self._tasks.get(call_id)

    def list_tasks(self) -> List[ToolTask]:
        """All live tasks."""
        return list(self._tasks.values())

    def append_log(self, call_id: str, data: Any, kind: Optional[TaskLogKind] = None) -> None:
        """Append a log entry to a task by call id (engine log sink)."""
        task = self._tasks.get(call_id)
        if task is not None:
            if kind is None:
                task.append_log(data)
            else:
                task.append_log(data, kind)

    def mark_working_as_orphaned(self) -> List[PendingAsyncTool]:
        """Mark every still-working background/agent task as orphaned (detach mode)."""
        orphaned: List[PendingAsyncTool] = []
        for task in self._tasks.values():
            if task.mode != "defer" and task.status == "working":
                task.orphaned = True
                orphaned.append(
                    PendingAsyncTool(
                        call_id=task.call_id,
                        task_id=task.task_id,
                        name=task.tool_name,
                        mode=task.mode,
                        status=task.status,
                        started_at=task.started_at,
                        orphaned=True,
                        last_log=_last_log_of(task),
                    )
                )
        return orphaned

    def snapshot(self) -> List[PendingAsyncTool]:
        """Snapshot of every tracked task (for `get_async_tasks()` / persistence)."""
        return [
            PendingAsyncTool(
                call_id=t.call_id,
                task_id=t.task_id,
                name=t.tool_name,
                mode=t.mode,
                status=t.status,
                started_at=t.started_at,
                expires_at=t.expires_at,
                poll_after_ms=t.poll_after_ms,
                orphaned=True if t.orphaned is True else None,
                last_log=_last_log_of(t),
            )
            for t in self._tasks.values()
        ]

    def send_to_task(self, task_id: str, message: Any) -> bool:
        """Send a steering message to a working task's run body.

        Deferred tasks are owned by an external system — steering them raises.
        Returns True when a working in-process task received the message.
        """
        task = self.get_task(task_id)
        if task is None or task.status != "working":
            return False
        if task.mode == "defer":
            raise RuntimeError(
                f'Task "{task_id}" is deferred — its work runs in an external system; '
                "deliver steering messages there instead."
            )
        task.send(message)
        return True

    def cancel_task(self, task_id: str, reason: Optional[str] = None) -> bool:
        """Cancel one task by task id. Settles it as 'cancelled' immediately
        (first-writer-wins makes the racing body's own settlement a no-op) and
        cancels the controller so cooperative bodies stop working.

        Deferred tasks: cancellation is LOCAL-ONLY — the external system is
        not notified. Returns True when a working task was cancelled.
        """
        task = self.get_task(task_id)
        if task is None or task.status != "working":
            return False
        controller = task.controller
        message = reason if reason is not None else f"Task {task_id} cancelled"
        self._settle(task.call_id, "cancelled", error=message)
        if controller is not None:
            controller.cancel(Exception(message))
        return True

    def abort_all(self, reason: Optional[str] = None) -> None:
        """Cancel every in-flight background/agent task (run abort /
        `on_run_end: 'cancel'`)."""
        message = reason if reason is not None else "Run aborted"
        for task in list(self._tasks.values()):
            if task.mode != "defer" and task.status == "working":
                controller = task.controller
                self._settle(task.call_id, "cancelled", error=message)
                if controller is not None:
                    controller.cancel(Exception(message))

    async def drain(self, timeout_ms: float) -> bool:
        """Wait until every in-flight task settles or `timeout_ms` elapses,
        whichever comes first. Returns True when everything settled."""
        deadline = _now_ms() + timeout_ms
        while self.has_in_flight():
            remaining = deadline - _now_ms()
            if remaining <= 0:
                return False
            settled = await self._wait_for_settle(remaining)
            if not settled:
                return not self.has_in_flight()
        return True

    async def _wait_for_settle(self, timeout_ms: float) -> bool:
        """True on the next settle event, or False after `timeout_ms`."""
        waiter: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._settle_waiters.append(waiter)
        try:
            done, _ = await asyncio.wait({waiter}, timeout=timeout_ms / 1000)
        finally:
            if not waiter.done():
                waiter.cancel()
                if waiter in self._settle_waiters:
                    self._settle_waiters.remove(waiter)
        return waiter in done


__all__ = ["AsyncToolRegistry", "SettledToolTask", "render_last_log"]
