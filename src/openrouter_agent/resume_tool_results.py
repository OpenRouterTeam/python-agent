"""Port of upstream `inner-loop/resume-tool-results.ts`.

Deliver results for pending async tool tasks (started by deferred tools, or
background tasks orphaned across a run boundary) into a persisted
conversation -- typically from a different process than the one that started
them (a webhook handler, a queue worker). Prefer the typed
``.resolve()`` / ``.fail()`` / ``.cancel()`` methods on the deferred tool
itself (`DeferredTool`).

SECURITY: this injects a value the model treats as a tool result.
Authenticate the webhook/caller BEFORE invoking it, and pass ``tools`` so
outputs are validated against the owning tool's ``output_schema``.

CONCURRENCY: this is a read-modify-write over the state accessor; serialize
calls per conversation (or batch completions into one ``results`` list).

HOOKS: PostToolUse / PostToolUseFailure do NOT fire here -- hooks are
run-scoped and this entry point runs outside any run.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ._utils import dump, maybe_await, validate_schema
from .conversation_state import append_to_messages, update_state
from .tool_task import TASK_RESULT_BOUNDARY
from .tool_types import PendingAsyncTool, get_tool_function, is_client_tool, is_unified_tool


class ToolTaskAlreadySettledError(Exception):
    """Raised when the target task was already settled -- the at-most-once
    guard against double resolution and replayed webhooks. Opt out per call
    with ``if_settled="ignore"``."""

    def __init__(self, task_id: str, call_id: str) -> None:
        super().__init__(f'Tool task "{task_id}" (call {call_id}) has already been settled')
        self.name = "ToolTaskAlreadySettledError"
        self.task_id = task_id
        self.call_id = call_id


#: One task resolution: ``{"task_id" | "call_id": ..., "output": ...}`` or
#: ``{..., "error": str, "status"?: "failed" | "cancelled" | "expired"}``.
ResumeToolResultEntry = Mapping[str, Any]
#: Run configuration for continuing immediately (any `call_model` request
#: field except ``state`` / ``input`` / approval decisions).
ResumeRunConfig = Mapping[str, Any]
#: The ``tool_task_result`` envelope injected as a user-role message.
ToolTaskResultEnvelope = Dict[str, Any]


def build_task_result_message(envelope: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the user-role message carrying a task-result envelope."""
    return {
        "role": "user",
        "content": f"{TASK_RESULT_BOUNDARY}\n{json.dumps(envelope, separators=(',', ':'), default=dump)}",
    }


def _resolve_pending_task(
    entry: ResumeToolResultEntry,
    pending: Sequence[PendingAsyncTool],
    settled_ids: set,
    if_settled: Optional[str],
) -> Optional[PendingAsyncTool]:
    task_id = entry.get("task_id")
    call_id = entry.get("call_id")
    task = next(
        (t for t in pending if (t.task_id == task_id if task_id is not None else t.call_id == call_id)),
        None,
    )
    if task is None:
        if call_id is not None and call_id in settled_ids:
            if if_settled == "ignore":
                return None
            raise ToolTaskAlreadySettledError(task_id or call_id, call_id)
        raise LookupError(f'resume_tool_results: no pending async tool task found for "{task_id or call_id}"')
    if task.call_id in settled_ids or task.status != "working":
        if if_settled == "ignore":
            return None
        raise ToolTaskAlreadySettledError(task.task_id, task.call_id)
    return task


def _build_resume_envelope(
    entry: ResumeToolResultEntry, task: PendingAsyncTool, tools: Optional[Sequence[Any]]
) -> ToolTaskResultEnvelope:
    if "output" in entry and entry.get("error") is None:
        tool = next(
            (t for t in (tools or []) if is_client_tool(t) and get_tool_function(t).get("name") == task.name),
            None,
        )
        if tools is not None and tool is None:
            raise ValueError(
                f'resume_tool_results: task "{task.task_id}" belongs to tool "{task.name}", which is not in the '
                "supplied tools list — its output cannot be validated. Include the tool, or omit `tools` to skip "
                "validation explicitly."
            )
        output = entry.get("output")
        if tool is not None and is_unified_tool(tool) and get_tool_function(tool).get("output_schema") is not None:
            output = dump(validate_schema(get_tool_function(tool)["output_schema"], output))
        return {
            "type": "tool_task_result",
            "tool": task.name,
            "task_id": task.task_id,
            "call_id": task.call_id,
            "status": "completed",
            "result": output,
        }
    return {
        "type": "tool_task_result",
        "tool": task.name,
        "task_id": task.task_id,
        "call_id": task.call_id,
        "status": entry.get("status") or "failed",
        "error": entry.get("error") or "Task failed",
    }


async def resume_tool_results(client: Any, request: Mapping[str, Any], options: Any = None) -> Any:
    """Record async task results on persisted state and optionally continue.

    ``request`` keys: ``state`` (accessor, required), ``results`` (entries),
    ``tools`` (validation), ``if_settled`` (``"throw"`` default |
    ``"ignore"``), ``run`` (continue immediately with this config), and
    ``expect_tool_name`` (ownership guard set by the typed tool methods).

    Returns the continued `ModelResult` with ``run`` config, else ``None``
    (also ``None`` WITHOUT running when every entry was skipped under
    ``if_settled="ignore"`` -- a replayed webhook never triggers a duplicate
    continuation).
    """
    accessor = request["state"]
    state = await maybe_await(accessor.load())
    if state is None:
        raise LookupError("resume_tool_results: no conversation state found for the given state accessor")

    pending: List[PendingAsyncTool] = list(state.pending_async_tools or [])
    settled_ids = set(state.settled_async_call_ids or [])
    tools = request.get("tools")
    expect_tool_name = request.get("expect_tool_name")

    envelopes: List[Dict[str, Any]] = []
    settled_now: Dict[str, str] = {}
    for entry in request.get("results") or []:
        task = _resolve_pending_task(entry, pending, settled_ids, request.get("if_settled"))
        if task is None:
            continue
        if expect_tool_name is not None and task.name != expect_tool_name:
            raise ValueError(
                f'resume_tool_results: task "{task.task_id}" belongs to tool "{task.name}", not '
                f'"{expect_tool_name}" — use the owning tool\'s completion methods (or the untyped '
                "resume_tool_results entry point)."
            )
        envelope = _build_resume_envelope(entry, task, tools)
        envelopes.append(build_task_result_message(envelope))
        status = envelope["status"]
        settled_now[task.call_id] = status if status in ("completed", "cancelled") else "failed"
        settled_ids.add(task.call_id)

    if not envelopes:
        return None

    next_pending = [
        dataclasses.replace(t, status=settled_now[t.call_id]) if t.call_id in settled_now else t for t in pending
    ]
    still_working = any(t.status == "working" and t.orphaned is not True for t in next_pending)
    run = request.get("run")
    status_owned = state.status in ("awaiting_async_tools", "in_progress", None) or (
        state.status == "complete" and run is not None
    )
    updates: Dict[str, Any] = {
        "messages": append_to_messages(state.messages, envelopes),
        "settled_async_call_ids": [*(state.settled_async_call_ids or []), *settled_now.keys()],
        "pending_async_tools": next_pending,
    }
    if status_owned:
        updates["status"] = "awaiting_async_tools" if still_working else "in_progress"
    await maybe_await(accessor.save(update_state(state, updates)))

    if run is None:
        return None

    from .call_model import call_model

    continued: Dict[str, Any] = {
        **dict(run),
        "input": [],
        "tools": run.get("tools") if run.get("tools") is not None else tools,
        "state": accessor,
    }
    return call_model(client, continued, options)
