"""The universal model-facing `task` tool: schema, registration, and dispatch.

Port of upstream `src/lib/tool-check.ts`, plus the engine-side dispatch that
upstream keeps as private `ModelResult` methods (`answerTaskToolCall`,
`taskToolCancel`, `taskToolSteer`, `taskToolCheck`, `buildTaskHandle`,
`taskToolActive`) and the module-level `taskToolResultIfSettled`
(`model-result.ts:4497-4735`, `:7643-7664`). They live here as standalone
functions so the Python engine can call them without re-deriving the
model-facing outputs.

ONE static tool (`task`) handles every running task — addressed by
`task_id` — while the *implementations* stay tool-resident (each tool's
`check` config): no per-tool wire-schema growth, tool-specific behavior
preserved.

Casing: model-facing dict keys are snake_case (`task_id`, `tool_name`,
`elapsed_ms`, ...) per the port contract. The task tool's input accepts
`task_id` and, for robustness, upstream's `taskId` too.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Literal

from ._utils import get_field, maybe_await, schema_to_json_schema, validate_schema
from .tool_task import TaskLogEntry, ToolTask, ToolTaskMode, ToolTaskStatus
from .tool_types import (
    PendingAsyncTool,
    ToolType,
    get_tool_function,
    is_client_tool,
    is_long_running_tool,
    is_unified_tool,
)

if TYPE_CHECKING:
    from .async_tool_registry import AsyncToolRegistry

#: Reserved name of the single, universal task-interaction tool.
TASK_TOOL_NAME = "task"

#: Default character budget for the `transcript` (and `logs`) check view
#: (upstream `asyncTools.maxTranscriptChars` default, `model-result.ts:4661`).
DEFAULT_MAX_TRANSCRIPT_CHARS = 20_000

TaskToolAction = Literal["check", "steer", "result", "cancel"]
TaskToolView = Literal["status", "logs", "transcript"]


class TaskToolInputSchema(BaseModel):
    """Actions the universal task tool supports (upstream Zod `TaskToolInputSchema`)."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    task_id: str = Field(
        validation_alias=AliasChoices("task_id", "taskId"),
        description="Task id from a pending tool output.",
    )
    action: Optional[TaskToolAction] = Field(
        default=None,
        description=(
            "'check' (default): progress views. 'steer': send guidance to the running task. "
            "'result': the final result if settled, else current status. 'cancel': stop the task."
        ),
    )
    view: Optional[TaskToolView] = Field(
        default=None,
        description=(
            "For action=check: 'status' (default) is a one-line state; 'logs' returns recent "
            "progress entries; 'transcript' returns full detail."
        ),
    )
    tail: Optional[int] = Field(
        default=None,
        gt=0,
        le=200,
        description="For view=logs: how many recent entries. Default 20.",
    )
    message: Optional[str] = Field(default=None, description="For action=steer: the guidance to send.")
    reason: Optional[str] = Field(default=None, description="For action=cancel: why.")
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extra parameters for the tool's custom check handler, when it declares one.",
    )

    @field_validator("tail", mode="before")
    @classmethod
    def _integer_tail(cls, value: Any) -> Any:
        # Zod `z.number().int()`: numbers only (no numeric strings / bools),
        # integral floats are integers.
        if value is None:
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("tail must be an integer")
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError("tail must be an integer")
            return int(value)
        return value


#: Upstream `TaskToolInput` (the inferred input type).
TaskToolInput = TaskToolInputSchema

_TASK_TOOL_DESCRIPTION = (
    "Interact with a long-running task started by another tool: check progress (status, recent logs, "
    "or the full transcript), steer it with a message, fetch its result, or cancel it. "
    "Task ids come from pending tool outputs."
)

_task_tool_parameters: Optional[Dict[str, Any]] = None


def _strip_optional_nulls(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Make pydantic's `Optional[X]` properties look like Zod `.optional()`
    (not-required, no `null` branch), and drop pydantic's `default: null`."""
    props = schema.get("properties")
    if not isinstance(props, dict):
        return schema
    cleaned: Dict[str, Any] = {}
    for key, prop in props.items():
        if isinstance(prop, dict):
            prop = dict(prop)
            any_of = prop.get("anyOf")
            if isinstance(any_of, list):
                non_null = [branch for branch in any_of if branch != {"type": "null"}]
                if len(non_null) == 1 and isinstance(non_null[0], dict):
                    prop.pop("anyOf")
                    prop = {**non_null[0], **prop}
            if prop.get("default", 0) is None:
                prop.pop("default")
            prop.pop("title", None)
        cleaned[key] = prop
    out = dict(schema)
    out["properties"] = cleaned
    out.pop("title", None)
    # The model docstring is not part of the wire schema (Zod emits none).
    out.pop("description", None)
    return out


def build_task_tool_api_definition(
    convert: Optional[Callable[[Any], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """The wire definition for the universal task tool. Static — identical
    regardless of how many long-running tools are registered, so the schema
    conversion runs once per process (memoized on first call)."""
    global _task_tool_parameters
    if _task_tool_parameters is None:
        converter = convert or schema_to_json_schema
        _task_tool_parameters = _strip_optional_nulls(converter(TaskToolInputSchema))
    return {
        "type": "function",
        "name": TASK_TOOL_NAME,
        "description": _TASK_TOOL_DESCRIPTION,
        "strict": None,
        "parameters": _task_tool_parameters,
    }


def has_task_tool_name_collision(tools: Sequence[Mapping[str, Any]]) -> bool:
    """True when a user tool already claims the reserved task-tool name.

    Shared by registration (`needs_task_tool`) and the engine's interception
    guard (`task_tool_active`) so a collision disables BOTH.
    """
    return any(
        isinstance(t, Mapping) and "function" in t and get_tool_function(t).get("name") == TASK_TOOL_NAME for t in tools
    )


def needs_task_tool(tools: Sequence[Mapping[str, Any]]) -> bool:
    """True when the tool list warrants registering the task tool: at least
    one long-running-capable tool, and no user tool already claiming the name.
    Warns (only when something was actually disabled) on a name collision."""
    if has_task_tool_name_collision(tools):
        if any(is_long_running_tool(t) for t in tools):
            warnings.warn(
                f'[AsyncTools] a user tool is named "{TASK_TOOL_NAME}" — the built-in task tool is disabled; '
                "models cannot check on long-running tasks.",
                stacklevel=2,
            )
        return False
    return any(is_long_running_tool(t) for t in tools)


def task_tool_active(tools: Sequence[Mapping[str, Any]], *, checkins: bool = True) -> bool:
    """True when the built-in task tool is active for a run (upstream private
    `ModelResult.taskToolActive`): mirrors `needs_task_tool` without warning,
    and is off when `async_tools.checkins` is False. When False, calls named
    "task" must NOT be intercepted — they belong to the user's tool."""
    if checkins is False:
        return False
    if has_task_tool_name_collision(tools):
        return False
    return any(is_long_running_tool(t) for t in tools)


def build_task_tool_stub() -> Dict[str, Any]:
    """Minimal tool stub for the engine's bookkeeping around task-tool
    answers (result events, output shaping). Never executed as a user tool."""
    return {
        "type": ToolType.Function.value,
        "function": {"name": TASK_TOOL_NAME, "input_schema": TaskToolInputSchema, "execute": False},
    }


class ResolvedCheckConfig(NamedTuple):
    schema: Any
    execute: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Any]]


def resolve_check_config(check: Any) -> ResolvedCheckConfig:
    """Resolve a tool's `check` config (`True` or `{"schema", "execute"}`)
    to `(schema, execute)` with defaults."""
    if check is None or check is True or check is False:
        return ResolvedCheckConfig(schema=None, execute=None)
    return ResolvedCheckConfig(schema=get_field(check, "schema"), execute=get_field(check, "execute"))


class ToolTaskHandle(Protocol):
    """Narrow façade over a running task, handed to check `execute` handlers
    via `turn_context["task"]` (upstream `ToolTaskHandle`)."""

    @property
    def task_id(self) -> str: ...
    @property
    def tool_name(self) -> str: ...
    @property
    def mode(self) -> ToolTaskMode: ...
    def status(self) -> ToolTaskStatus: ...
    def status_view(self) -> Dict[str, Any]: ...
    def tail_logs(self, n: float) -> List[TaskLogEntry]: ...
    def transcript(self, max_chars: Optional[int] = None) -> str: ...
    def send(self, message: Any) -> None: ...
    def cancel(self, reason: Optional[str] = None) -> bool: ...


class _TaskHandle:
    def __init__(self, task: ToolTask, registry: Optional["AsyncToolRegistry"], max_transcript_chars: int) -> None:
        self._task = task
        self._registry = registry
        self._max_transcript_chars = max_transcript_chars

    @property
    def task_id(self) -> str:
        return self._task.task_id

    @property
    def tool_name(self) -> str:
        return self._task.tool_name

    @property
    def mode(self) -> ToolTaskMode:
        return self._task.mode

    def status(self) -> ToolTaskStatus:
        return self._task.status

    def status_view(self) -> Dict[str, Any]:
        return self._task.to_status_view()

    def tail_logs(self, n: float) -> List[TaskLogEntry]:
        return self._task.tail_logs(n)

    def transcript(self, max_chars: Optional[int] = None) -> str:
        return self._task.render_transcript(max_chars if max_chars is not None else self._max_transcript_chars)

    def send(self, message: Any) -> None:
        self._task.send(message)

    def cancel(self, reason: Optional[str] = None) -> bool:
        return self._registry.cancel_task(self._task.task_id, reason) if self._registry is not None else False


def build_task_handle(
    task: ToolTask,
    registry: Optional["AsyncToolRegistry"],
    max_transcript_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS,
) -> ToolTaskHandle:
    """Build the `ToolTaskHandle` façade for check `execute` handlers."""
    return _TaskHandle(task, registry, max_transcript_chars)


def _data_size(data: Any) -> int:
    if isinstance(data, str):
        return len(data)
    try:
        return len(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    except (TypeError, ValueError):
        return 0


def _data_text(data: Any) -> str:
    if isinstance(data, str):
        return data
    try:
        return json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return str(data)


TaskInputLike = Union[TaskToolInputSchema, Mapping[str, Any]]


def default_check_result(
    input: TaskInputLike,
    turn_context: Mapping[str, Any],
    *,
    max_transcript_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS,
) -> Any:
    """The SDK default check handler: status / logs / transcript views."""
    task: Optional[ToolTaskHandle] = turn_context.get("task")
    if task is None:
        return {"error": "unknown_task", "hint": "No task context available for this check call."}
    view = get_field(input, "view") or "status"
    status_view = task.status_view()

    if view == "logs":
        tail = get_field(input, "tail")
        entries = task.tail_logs(tail if tail is not None else 20)
        # Same character budget as the transcript view; newest entries win —
        # drop from the OLDEST end when over budget.
        budget = max_transcript_chars
        first_kept = len(entries)
        for i in range(len(entries) - 1, -1, -1):
            size = _data_size(entries[i].data)
            if size > budget:
                break
            budget -= size
            first_kept = i
        kept = [entry.to_dict() for entry in entries[first_kept:]]
        # Never answer with NOTHING when progress exists.
        if not kept and entries:
            newest = entries[-1]
            body = _data_text(newest.data)
            kept = [{**newest.to_dict(), "data": f"{body[: max(0, max_transcript_chars - 12)]}…[truncated]"}]
        result: Dict[str, Any] = {**status_view, "logs": kept}
        if len(kept) < len(entries):
            result["note"] = (
                f"Truncated to the {len(kept)} most recent entries (character budget); "
                "use view: 'transcript' or a smaller tail for more."
            )
        return result
    if view == "transcript":
        return {**status_view, "transcript": task.transcript(max_transcript_chars)}
    return status_view


def _now_ms() -> int:
    return int(time.time() * 1000)


def persisted_task_check_result(input: TaskInputLike, pending: PendingAsyncTool) -> Any:
    """Answer a task-tool call against a PERSISTED task (deferred /
    cross-process, after a restart — no live registry entry). Only
    status-grade data survives: identity, timing, `last_log`."""
    base: Dict[str, Any] = {
        "task_id": pending.task_id,
        "tool_name": pending.name,
        "mode": pending.mode,
        "status": pending.status,
        "started_at": pending.started_at,
        "elapsed_ms": _now_ms() - pending.started_at,
    }
    if pending.last_log is not None:
        base["last_log"] = pending.last_log.text
    if pending.poll_after_ms is not None:
        base["poll_after_ms"] = pending.poll_after_ms
    if pending.expires_at is not None:
        base["expires_at"] = pending.expires_at
    if pending.orphaned is True:
        base["orphaned"] = True
        base["note"] = "This task was detached; its result will not be delivered."

    # View-specific explanations must not displace the orphaned warning.
    def with_note(view_note: str) -> str:
        return f"{base['note']} {view_note}" if "note" in base else view_note

    view = get_field(input, "view") or "status"
    if view == "logs":
        logs = [{"at": pending.last_log.at, "data": pending.last_log.text}] if pending.last_log is not None else []
        return {**base, "logs": logs, "note": with_note("Full logs are not retained across processes.")}
    if view == "transcript":
        return {
            **base,
            "transcript": "",
            "note": with_note(
                "No transcript available — this task is owned by an external system."
                if pending.mode == "defer"
                else "No transcript available — the task ran in a previous process."
            ),
        }
    return base


# ---------------------------------------------------------------------------
# Dispatch (upstream private ModelResult methods, model-result.ts:4510-4735)
# ---------------------------------------------------------------------------


def task_tool_result_if_settled(input: TaskToolInputSchema, live_task: Optional[ToolTask]) -> Optional[Dict[str, Any]]:
    """`action: 'result'`: the final result if settled, else None."""
    if live_task is not None and live_task.status == "completed":
        return {"task_id": input.task_id, "status": "completed", "result": live_task.result}
    if live_task is not None and live_task.status != "working":
        out: Dict[str, Any] = {"task_id": input.task_id, "status": live_task.status}
        if live_task.error is not None:
            out["error"] = live_task.error
        return out
    return None


def task_tool_cancel(
    input: TaskToolInputSchema,
    live_task: Optional[ToolTask],
    registry: Optional["AsyncToolRegistry"],
) -> Dict[str, Any]:
    """`action: 'cancel'`."""
    cancelled = (
        registry.cancel_task(input.task_id, input.reason) if live_task is not None and registry is not None else False
    )
    if cancelled:
        return {"task_id": input.task_id, "status": "cancelled"}
    return {
        "task_id": input.task_id,
        "error": "not_cancellable",
        "hint": "The task has already settled."
        if live_task is not None
        else "This task is owned by an external system — cancel it there (or via the tool’s .cancel() method).",
    }


def task_tool_steer(input: TaskToolInputSchema, live_task: Optional[ToolTask]) -> Tuple[Any, Optional[Exception]]:
    """`action: 'steer'`. Returns `(result, error)`."""
    if not isinstance(input.message, str) or len(input.message) == 0:
        return None, ValueError("action 'steer' requires a non-empty `message`")
    if live_task is None or live_task.mode == "defer":
        return {
            "task_id": input.task_id,
            "error": "not_steerable",
            "hint": "This task runs in an external system — steer it there.",
        }, None
    live_task.send(input.message)
    return {"task_id": input.task_id, "steered": True}, None


def _find_tool(tools: Sequence[Mapping[str, Any]], name: str) -> Optional[Mapping[str, Any]]:
    for t in tools:
        if is_client_tool(t) and get_tool_function(t).get("name") == name:
            return t
    return None


def _to_params_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Mapping):
        return dict(value)
    return value  # type: ignore[no-any-return]


async def task_tool_check(
    input: TaskToolInputSchema,
    live_task: Optional[ToolTask],
    persisted: Optional[PendingAsyncTool],
    *,
    tools: Sequence[Mapping[str, Any]],
    registry: Optional["AsyncToolRegistry"],
    number_of_turns: int,
    max_transcript_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS,
) -> Any:
    """`action: 'check'` (and the status-view fallback for `result` on an
    unsettled task): the owning tool's `check["execute"]` when declared, else
    the SDK default views — live-task-backed in-process, persisted-state-
    backed post-restart."""
    owning_name = live_task.tool_name if live_task is not None else (persisted.name if persisted is not None else "")
    owning_tool = _find_tool(tools, owning_name)
    config = get_tool_function(owning_tool).get("check") if owning_tool and is_unified_tool(owning_tool) else None
    schema, execute = resolve_check_config(config)

    # Custom params are validated against check.schema when both are present.
    custom_params: Dict[str, Any] = dict(input.params) if input.params is not None else {}
    if schema is not None and input.params is not None:
        custom_params = _to_params_dict(validate_schema(schema, input.params))

    if live_task is not None:
        handle = build_task_handle(live_task, registry, max_transcript_chars)
        check_ctx: Dict[str, Any] = {
            "number_of_turns": number_of_turns,
            "tool_call_status": live_task.status,
            "accumulated_yielded_events": live_task.accumulated_yielded_events,
            "task": handle,
        }
        if execute is not None:
            return await maybe_await(execute(custom_params, check_ctx))
        return default_check_result(input, check_ctx, max_transcript_chars=max_transcript_chars)

    if persisted is None:  # pragma: no cover - callers resolve unknown ids first
        return {"error": "unknown_task", "task_id": input.task_id}
    # Cross-process / post-restart: a custom check still runs with a
    # state-backed context (no `task` handle); fall back to the persisted
    # status view when it answers None.
    persisted_ctx: Dict[str, Any] = {
        "number_of_turns": number_of_turns,
        "tool_call_status": persisted.status,
        "accumulated_yielded_events": [persisted.last_log.text] if persisted.last_log is not None else [],
    }
    custom_result = await maybe_await(execute(custom_params, persisted_ctx)) if execute is not None else None
    return custom_result if custom_result is not None else persisted_task_check_result(input, persisted)


@dataclass(frozen=True)
class TaskToolAnswer:
    """Outcome of `answer_task_tool_call`. Exactly one of `result` / `error`
    is meaningful; the engine renders `error` as `{"error": str(error)}`."""

    result: Any = None
    error: Optional[Exception] = None


UNKNOWN_TASK_HINT = (
    "No task with this id exists in this conversation. It may belong to another conversation, "
    "or its record was dropped."
)


async def answer_task_tool_call(
    arguments: Any,
    *,
    registry: Optional["AsyncToolRegistry"],
    tools: Sequence[Mapping[str, Any]],
    pending_async_tools: Optional[Sequence[PendingAsyncTool]] = None,
    number_of_turns: int = 1,
    max_transcript_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS,
) -> TaskToolAnswer:
    """Answer a call to the universal `task` tool (upstream private
    `ModelResult.answerTaskToolCall`).

    Resolves the task id to its live registry task (or persisted
    `pending_async_tools` entry, post-restart) and its OWNING tool, then
    dispatches by action: `check` (default), `steer`, `result`, `cancel`.
    """
    try:
        input = TaskToolInputSchema.model_validate(arguments if arguments is not None else {})
    except Exception as error:
        return TaskToolAnswer(error=error)

    live_task = registry.get_task(input.task_id) if registry is not None else None
    persisted = next((t for t in (pending_async_tools or []) if t.task_id == input.task_id), None)
    if live_task is None and persisted is None:
        return TaskToolAnswer(result={"error": "unknown_task", "task_id": input.task_id, "hint": UNKNOWN_TASK_HINT})

    try:
        action = input.action or "check"
        if action == "cancel":
            return TaskToolAnswer(result=task_tool_cancel(input, live_task, registry))
        if action == "steer":
            result, error = task_tool_steer(input, live_task)
            return TaskToolAnswer(result=result, error=error)
        if action == "result":
            settled = task_tool_result_if_settled(input, live_task)
            if settled is not None:
                return TaskToolAnswer(result=settled)
        return TaskToolAnswer(
            result=await task_tool_check(
                input,
                live_task,
                persisted,
                tools=tools,
                registry=registry,
                number_of_turns=number_of_turns,
                max_transcript_chars=max_transcript_chars,
            )
        )
    except Exception as error:
        return TaskToolAnswer(error=error)


__all__ = [
    "DEFAULT_MAX_TRANSCRIPT_CHARS",
    "TASK_TOOL_NAME",
    "UNKNOWN_TASK_HINT",
    "ResolvedCheckConfig",
    "TaskToolAnswer",
    "TaskToolInput",
    "TaskToolInputSchema",
    "ToolTaskHandle",
    "answer_task_tool_call",
    "build_task_handle",
    "build_task_tool_api_definition",
    "build_task_tool_stub",
    "default_check_result",
    "has_task_tool_name_collision",
    "needs_task_tool",
    "persisted_task_check_result",
    "resolve_check_config",
    "task_tool_active",
    "task_tool_cancel",
    "task_tool_check",
    "task_tool_result_if_settled",
    "task_tool_steer",
]
