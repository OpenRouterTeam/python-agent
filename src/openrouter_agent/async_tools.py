"""The async-tool subsystem's façade (port of upstream `src/lib/async-tools.ts`).

Task lifecycle, the model-facing `task` tool, the registry of in-flight
tasks, and the concurrency primitives that bound them. Re-exports only — no
logic. Anything needing a subset should import the specific module directly.
"""

from __future__ import annotations

from .async_tool_registry import AsyncToolRegistry, SettledToolTask
from .tool_check import (
    TASK_TOOL_NAME,
    TaskToolInput,
    TaskToolInputSchema,
    build_task_tool_stub,
    default_check_result,
    has_task_tool_name_collision,
    persisted_task_check_result,
    resolve_check_config,
)
from .tool_concurrency import Semaphore, acquire_all
from .tool_task import TASK_RESULT_BOUNDARY, CancellationController, ToolTask, ToolTaskMode

#: Upstream `export type { Semaphore as ToolSemaphore }`.
ToolSemaphore = Semaphore

__all__ = [
    "TASK_RESULT_BOUNDARY",
    "TASK_TOOL_NAME",
    "AsyncToolRegistry",
    "CancellationController",
    "Semaphore",
    "SettledToolTask",
    "TaskToolInput",
    "TaskToolInputSchema",
    "ToolSemaphore",
    "ToolTask",
    "ToolTaskMode",
    "acquire_all",
    "build_task_tool_stub",
    "default_check_result",
    "has_task_tool_name_collision",
    "persisted_task_check_result",
    "resolve_check_config",
]
