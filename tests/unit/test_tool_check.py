"""Ports `packages/agent/tests/unit/tool-check.test.ts`.

Upstream drives most cases through `callModel` with a background `tool()`
(`lifecycle: 'background'`, `run`, `graceMs`). The Python `tool()` /
`ModelResult` do not yet support unified `run` tools (that integration is
being ported separately), so:

Ported 1:1 and running
  - `convertToolsToAPIFormat never augments per-tool schemas`
  - `needsTaskTool: true with a long-running tool, false without, false on
    name collision` (minus the `tool()` reserved-name half — see skips)
  - `a user tool named 'task' executes — the engine must not intercept on
    collision` (holds on the current engine, which never intercepts)

Ported with a faithful body, skipped pending ModelResult async-tool integration
  - `tool() rejects the reserved name` (the `tool()` half of the needsTaskTool case)
  - `callModel appends the single task tool to the API tool list when warranted`
  - `asyncTools.checkins: false suppresses the task tool`

Not ported as `call_model` tests (they need a mock transport that computes
turn N's response from turn N-1's request — reading the `task_id` the
placeholder advertised — which `QueuedClient` cannot express without a
bespoke fake client; listed for the integration run):
  - `task-tool dispatch` › default status view / logs view (tail respected) /
    transcript view / unknown taskId / custom check.execute + steer /
    check calls exempt from doom-loop detection / `asyncTools.checkins: false`
    restores the no-polling placeholder / post-restart deferred check answers
    from persisted state.

The dispatch invariants of those cases (exact output payloads) are instead
asserted below against `answer_task_tool_call`, the port of the engine's
private `answerTaskToolCall` the integration will call. Model-facing keys are
snake_case per the port contract (`task_id`, `log_count`, `last_log`,
`poll_after_ms`, ...), not upstream's camelCase.
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, ValidationError

from openrouter_agent import call_model, tool
from openrouter_agent.async_tool_registry import AsyncToolRegistry
from openrouter_agent.tool_check import (
    TASK_TOOL_NAME,
    TaskToolInputSchema,
    answer_task_tool_call,
    build_task_tool_api_definition,
    build_task_tool_stub,
    default_check_result,
    has_task_tool_name_collision,
    needs_task_tool,
    persisted_task_check_result,
    resolve_check_config,
    task_tool_active,
)
from openrouter_agent.tool_executor import convert_tools_to_api_format
from openrouter_agent.tool_task import CancellationController, ToolTask
from openrouter_agent.tool_types import (
    PendingAsyncTool,
    PendingAsyncToolLastLog,
    is_agent_tool,
    is_deferred_handle,
    is_long_running_tool,
    is_unified_tool,
)
from tests._fixtures import QueuedClient, text_response, tool_call_response

PENDING = "pending ModelResult async-tool integration"


class EmptyInput(BaseModel):
    pass


class QInput(BaseModel):
    q: str


class OkOutput(BaseModel):
    ok: bool


async def _ok_run(params: Any, ctx: Any = None) -> Dict[str, Any]:
    return {"ok": True}


def unified_tool(
    name: str,
    *,
    lifecycle: str = "background",
    input_schema: Any = EmptyInput,
    check: Any = None,
    **extra: Any,
) -> Dict[str, Any]:
    """A unified `run` tool dict, as upstream `tool({ lifecycle, run })` builds
    (`tool.ts:977-1003`): `function` carries `lifecycle` and `run`."""
    fn: Dict[str, Any] = {
        "lifecycle": lifecycle,
        "name": name,
        "input_schema": input_schema,
        "output_schema": OkOutput,
        "run": _ok_run,
    }
    if check is not None:
        fn["check"] = check
    fn.update(extra)
    return {"type": "function", "function": fn}


def sync_tool(name: str = "sync") -> Dict[str, Any]:
    return unified_tool(name, lifecycle="sync")


# ---------------------------------------------------------------------------
# universal task tool registration
# ---------------------------------------------------------------------------


def test_convert_tools_to_api_format_never_augments_per_tool_schemas() -> None:
    api = convert_tools_to_api_format([unified_tool("lr", input_schema=QInput)])
    # The tool's own schema is untouched — no anyOf, no task_id.
    assert api[0]["parameters"].get("anyOf") is None
    assert list(api[0]["parameters"].get("properties", {}).keys()) == ["q"]


def test_needs_task_tool_true_false_and_collision() -> None:
    long_running = unified_tool("lr")
    sync = sync_tool()

    assert needs_task_tool([long_running, sync]) is True
    assert needs_task_tool([sync]) is False

    # A dynamically-built tool list can bypass tool(); the built-in is
    # suppressed with a warning instead of silently colliding.
    collider = {"type": "function", "function": {"name": "task", "input_schema": EmptyInput, "execute": _ok_run}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert needs_task_tool([long_running, collider]) is False
    assert len(caught) == 1
    assert 'a user tool is named "task"' in str(caught[0].message)

    # No warning when nothing was disabled (no long-running tool present).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert needs_task_tool([sync, collider]) is False
    assert caught == []


@pytest.mark.skip(reason=PENDING)
def test_tool_rejects_the_reserved_task_name() -> None:
    with pytest.raises(ValueError, match="reserved"):
        tool(name="task", input_schema=EmptyInput, execute=_ok_run)


class TaskIdInput(BaseModel):
    task_id: str


async def test_user_tool_named_task_executes_and_is_not_intercepted() -> None:
    calls: List[Any] = []

    async def user_task_execute(params: Any, ctx: Any = None) -> Dict[str, Any]:
        calls.append(params)
        return {"fromUserTool": True}

    collider = {
        "type": "function",
        "function": {"name": "task", "input_schema": TaskIdInput, "execute": user_task_execute},
    }
    long_running = unified_tool("lr", graceMs=1_000)
    client = QueuedClient(
        [
            tool_call_response("resp_1", "task", call_id="call_1", arguments='{"task_id":"t-1"}'),
            text_response("resp_2", "done"),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        await call_model(
            client, {"model": "test-model", "input": "use the task tool", "tools": [collider, long_running]}
        ).get_text()

    # The user's tool ran exactly once; the engine did not answer the call itself.
    assert len(calls) == 1
    assert len(client.requests) == 2
    outputs = [
        item
        for item in client.requests[1]["input"]
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    # Snake_case `call_id`: `ModelResult._send` normalizes at the transport
    # boundary (`model_result.py:148-156`); upstream asserts `callId`.
    matching = [o for o in outputs if o.get("call_id") == "call_1"]
    assert len(matching) == 1
    assert "fromUserTool" in matching[0]["output"]


@pytest.mark.skip(reason=PENDING)
async def test_call_model_appends_the_single_task_tool_when_warranted() -> None:
    client = QueuedClient([text_response("resp_1", "hi")])
    await call_model(client, {"model": "test-model", "input": "hello", "tools": [unified_tool("lr")]}).get_text()
    assert [t.get("name") for t in client.requests[0]["tools"]] == ["lr", "task"]


@pytest.mark.skip(reason=PENDING)
async def test_async_tools_checkins_false_suppresses_the_task_tool() -> None:
    client = QueuedClient([text_response("resp_1", "hi")])
    await call_model(
        client,
        {
            "model": "test-model",
            "input": "hello",
            "tools": [unified_tool("lr")],
            "async_tools": {"checkins": False},
        },
    ).get_text()
    assert [t.get("name") for t in client.requests[0]["tools"]] == ["lr"]


def test_task_tool_active_mirrors_needs_task_tool_and_checkins() -> None:
    lr = unified_tool("lr")
    collider = {"type": "function", "function": {"name": "task", "execute": _ok_run}}
    assert task_tool_active([lr]) is True
    assert task_tool_active([lr], checkins=False) is False
    assert task_tool_active([lr, collider]) is False
    assert task_tool_active([sync_tool()]) is False
    assert has_task_tool_name_collision([collider]) is True
    assert has_task_tool_name_collision([lr]) is False


def test_task_tool_api_definition_is_static_and_memoized() -> None:
    first = build_task_tool_api_definition()
    second = build_task_tool_api_definition()
    assert first["type"] == "function"
    assert first["name"] == TASK_TOOL_NAME == "task"
    assert first["strict"] is None
    assert first["parameters"] is second["parameters"]
    params = first["parameters"]
    assert params["required"] == ["task_id"]
    assert list(params["properties"].keys()) == ["task_id", "action", "view", "tail", "message", "reason", "params"]
    assert params["properties"]["action"]["enum"] == ["check", "steer", "result", "cancel"]
    assert params["properties"]["view"]["enum"] == ["status", "logs", "transcript"]
    # Zod `.optional()` shape: no null branch.
    assert "anyOf" not in params["properties"]["tail"]
    assert params["properties"]["tail"]["type"] == "integer"


def test_task_tool_stub_and_check_config_resolution() -> None:
    stub = build_task_tool_stub()
    assert stub["function"]["name"] == "task"
    assert stub["function"]["execute"] is False
    assert resolve_check_config(None) == (None, None)
    assert resolve_check_config(True) == (None, None)
    resolved = resolve_check_config({"schema": QInput, "execute": _ok_run})
    assert resolved.schema is QInput
    assert resolved.execute is _ok_run


def test_task_tool_input_schema_validation() -> None:
    assert TaskToolInputSchema.model_validate({"taskId": "t"}).task_id == "t"  # upstream casing accepted
    assert TaskToolInputSchema.model_validate({"task_id": "t", "tail": 2.0}).tail == 2
    for bad in ({}, {"task_id": "t", "tail": 0}, {"task_id": "t", "tail": 201}, {"task_id": "t", "tail": 1.5}):
        with pytest.raises(ValidationError):
            TaskToolInputSchema.model_validate(bad)
    with pytest.raises(ValidationError):
        TaskToolInputSchema.model_validate({"task_id": "t", "action": "explode"})


def test_tool_type_predicates() -> None:
    assert is_unified_tool(unified_tool("x")) is True
    assert is_unified_tool({"type": "function", "function": {"name": "x", "execute": _ok_run}}) is False
    assert is_unified_tool({"_brand": "server-tool", "config": {}}) is False
    assert is_long_running_tool(unified_tool("x", lifecycle="deferred")) is True
    assert is_long_running_tool(sync_tool()) is False
    assert is_agent_tool(unified_tool("x", kind="agent")) is True
    assert is_agent_tool(unified_tool("x")) is False
    assert is_deferred_handle({"__deferred": True, "task_id": "t"}) is True
    assert is_deferred_handle({"__deferred": True, "taskId": "t"}) is True
    assert is_deferred_handle({"__deferred": "yes", "task_id": "t"}) is False
    assert is_deferred_handle({"__deferred": True}) is False
    assert is_deferred_handle(None) is False


# ---------------------------------------------------------------------------
# task-tool dispatch (invariants of upstream's driveCheck cases)
# ---------------------------------------------------------------------------


def observable_task(registry: AsyncToolRegistry) -> ToolTask:
    """The state upstream's `makeObservableTool` reaches at check time: a
    working background `render_video` task that yielded two progress steps."""
    task = ToolTask(
        task_id=registry.generate_task_id(),
        call_id="call_start",
        tool_name="render_video",
        mode="background",
        controller=CancellationController(),
    )
    registry.register(task)
    task.append_log({"step": "downloading assets"})
    task.append_log({"step": "rendering frames"})
    return task


def render_video_tool(check: Any = None) -> Dict[str, Any]:
    return unified_tool("render_video", check=check)


async def answer(
    registry: Optional[AsyncToolRegistry],
    args: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]] = None,
    pending: Optional[List[PendingAsyncTool]] = None,
) -> Any:
    out = await answer_task_tool_call(
        args,
        registry=registry,
        tools=tools if tools is not None else [render_video_tool()],
        pending_async_tools=pending,
        number_of_turns=2,
    )
    assert out.error is None, out.error
    # Model-facing outputs must be JSON-serializable.
    json.dumps(out.result)
    return out.result


async def test_default_status_view_working_task_with_log_count_and_last_log() -> None:
    registry = AsyncToolRegistry()
    task = observable_task(registry)
    result = await answer(registry, {"task_id": task.task_id})
    assert result["status"] == "working"
    assert result["tool_name"] == "render_video"
    assert result["mode"] == "background"
    assert result["log_count"] == 2
    assert result["last_log"] == {"step": "rendering frames"}
    assert isinstance(result["elapsed_ms"], int)


async def test_logs_view_returns_the_yielded_entries_tail_respected() -> None:
    registry = AsyncToolRegistry()
    task = observable_task(registry)
    result = await answer(registry, {"task_id": task.task_id, "view": "logs", "tail": 1})
    assert len(result["logs"]) == 1
    assert result["logs"][0]["data"] == {"step": "rendering frames"}
    assert result["logs"][0]["seq"] == 2
    assert "note" not in result


async def test_transcript_view_renders_the_log_entries_as_text() -> None:
    registry = AsyncToolRegistry()
    task = observable_task(registry)
    result = await answer(registry, {"task_id": task.task_id, "view": "transcript"})
    transcript = result["transcript"]
    assert "downloading assets" in transcript
    assert "rendering frames" in transcript
    assert transcript.index("downloading assets") < transcript.index("rendering frames")


async def test_unknown_task_id_yields_an_error_result_not_a_new_task() -> None:
    registry = AsyncToolRegistry()
    observable_task(registry)
    result = await answer(registry, {"task_id": "task_nope"})
    assert result["error"] == "unknown_task"
    assert result["task_id"] == "task_nope"
    assert len(registry.list_tasks()) == 1


class FocusParams(BaseModel):
    focus: Optional[str] = None


async def test_custom_check_execute_receives_status_and_events_and_can_steer() -> None:
    steered: List[Any] = []
    seen_ctx: List[Dict[str, Any]] = []

    async def check_execute(params: Dict[str, Any], turn_context: Dict[str, Any]) -> Dict[str, Any]:
        seen_ctx.append(turn_context)
        if isinstance(params.get("focus"), str):
            turn_context["task"].send(params["focus"])
        return {
            "state": turn_context["tool_call_status"],
            "seen": len(turn_context.get("accumulated_yielded_events") or []),
        }

    registry = AsyncToolRegistry()
    task = ToolTask(task_id="task_s", call_id="call_s", tool_name="steerable", mode="background")
    registry.register(task)
    task.on_message(steered.append)
    task.append_log("starting", "text")
    steerable = unified_tool("steerable", check={"schema": FocusParams, "execute": check_execute})

    result = await answer(registry, {"task_id": "task_s", "params": {"focus": "focus on pricing"}}, tools=[steerable])
    assert steered == ["focus on pricing"]
    assert result == {"state": "working", "seen": 1}
    assert len(seen_ctx) == 1
    assert seen_ctx[0]["number_of_turns"] == 2


async def test_custom_check_params_are_validated_against_check_schema() -> None:
    class StrictParams(BaseModel):
        focus: str

    calls: List[Any] = []
    registry = AsyncToolRegistry()
    registry.register(ToolTask(task_id="task_v", call_id="call_v", tool_name="v", mode="background"))
    tool_v = unified_tool("v", check={"schema": StrictParams, "execute": lambda p, c: calls.append(p)})
    out = await answer_task_tool_call({"task_id": "task_v", "params": {"focus": 3}}, registry=registry, tools=[tool_v])
    assert isinstance(out.error, ValidationError)
    assert calls == []


async def test_post_restart_deferred_check_answers_from_persisted_state() -> None:
    pending = PendingAsyncTool(
        call_id="call_d",
        task_id="ticket_c1",
        name="legal_review",
        mode="defer",
        status="working",
        started_at=0,
        poll_after_ms=60_000,
    )
    legal = unified_tool("legal_review", lifecycle="deferred")
    # Fresh process: empty registry, only persisted state survives.
    result = await answer(
        AsyncToolRegistry(), {"task_id": "ticket_c1", "view": "transcript"}, tools=[legal], pending=[pending]
    )
    assert result["status"] == "working"
    assert result["mode"] == "defer"
    assert result["poll_after_ms"] == 60_000
    assert result["transcript"] == ""
    assert "external system" in result["note"]


# --- additional dispatch actions (steer / result / cancel) -------------------


async def test_steer_action_delivers_to_inbox() -> None:
    registry = AsyncToolRegistry()
    task = observable_task(registry)
    received: List[Any] = []
    task.on_message(received.append)
    result = await answer(registry, {"task_id": task.task_id, "action": "steer", "message": "faster"})
    assert result == {"task_id": task.task_id, "steered": True}
    assert received == ["faster"]

    missing = await answer_task_tool_call(
        {"task_id": task.task_id, "action": "steer"}, registry=registry, tools=[render_video_tool()]
    )
    assert str(missing.error) == "action 'steer' requires a non-empty `message`"
    assert received == ["faster"]


async def test_steer_on_deferred_task_is_not_steerable() -> None:
    registry = AsyncToolRegistry()
    registry.track_deferred(call_id="c", task_id="ticket", name="legal_review")
    result = await answer(registry, {"task_id": "ticket", "action": "steer", "message": "x"})
    assert result["error"] == "not_steerable"


async def test_result_action_returns_status_until_settled_then_result() -> None:
    registry = AsyncToolRegistry()
    task = observable_task(registry)
    unsettled = await answer(registry, {"task_id": task.task_id, "action": "result"})
    assert unsettled["status"] == "working"
    assert unsettled["log_count"] == 2

    task.status = "completed"
    task.result = {"url": "https://cdn/final.mp4"}
    settled = await answer(registry, {"task_id": task.task_id, "action": "result"})
    assert settled == {"task_id": task.task_id, "status": "completed", "result": {"url": "https://cdn/final.mp4"}}


async def test_cancel_action_cancels_once_then_reports_not_cancellable() -> None:
    registry = AsyncToolRegistry()
    task = observable_task(registry)
    controller = task.controller
    assert controller is not None
    first = await answer(registry, {"task_id": task.task_id, "action": "cancel", "reason": "user asked"})
    assert first == {"task_id": task.task_id, "status": "cancelled"}
    assert controller.cancelled is True
    assert [(s.status, s.error) for s in registry.take_settled()] == [("cancelled", "user asked")]

    second = await answer(registry, {"task_id": task.task_id, "action": "cancel"})
    assert second["error"] == "not_cancellable"
    assert second["hint"] == "The task has already settled."

    failed_result = await answer(registry, {"task_id": task.task_id, "action": "result"})
    assert failed_result == {"task_id": task.task_id, "status": "cancelled", "error": "user asked"}


async def test_cancel_on_persisted_only_task_is_not_cancellable() -> None:
    pending = PendingAsyncTool(
        call_id="c", task_id="ticket", name="legal_review", mode="defer", status="working", started_at=0
    )
    result = await answer(None, {"task_id": "ticket", "action": "cancel"}, pending=[pending])
    assert result["error"] == "not_cancellable"
    assert "external system" in result["hint"]


async def test_invalid_arguments_yield_an_error() -> None:
    out = await answer_task_tool_call({"view": "logs"}, registry=AsyncToolRegistry(), tools=[])
    assert isinstance(out.error, ValidationError)
    assert out.result is None


async def test_check_handler_exception_becomes_error() -> None:
    def boom(params: Any, ctx: Any) -> Any:
        raise RuntimeError("check exploded")

    registry = AsyncToolRegistry()
    registry.register(ToolTask(task_id="t", call_id="c", tool_name="b", mode="background"))
    out = await answer_task_tool_call(
        {"task_id": "t"}, registry=registry, tools=[unified_tool("b", check={"execute": boom})]
    )
    assert str(out.error) == "check exploded"


async def test_persisted_custom_check_none_falls_back_to_persisted_view() -> None:
    pending = PendingAsyncTool(
        call_id="c",
        task_id="ticket",
        name="legal_review",
        mode="defer",
        status="working",
        started_at=0,
        last_log=PendingAsyncToolLastLog(at=5, text="halfway"),
    )
    contexts: List[Dict[str, Any]] = []
    legal = unified_tool("legal_review", lifecycle="deferred", check={"execute": lambda p, c: contexts.append(c)})
    result = await answer(None, {"task_id": "ticket"}, tools=[legal], pending=[pending])
    assert result["last_log"] == "halfway"
    assert len(contexts) == 1
    assert "task" not in contexts[0]
    assert contexts[0]["accumulated_yielded_events"] == ["halfway"]


# --- default / persisted check-result helpers --------------------------------


def test_default_check_result_without_task_context() -> None:
    assert default_check_result({}, {"number_of_turns": 1}) == {
        "error": "unknown_task",
        "hint": "No task context available for this check call.",
    }


async def test_logs_view_character_budget_keeps_newest_and_never_answers_nothing() -> None:
    registry = AsyncToolRegistry()
    task = ToolTask(task_id="t", call_id="c", tool_name="x", mode="background")
    registry.register(task)
    for i in range(3):
        task.append_log(f"{i}" * 10, "text")
    out = await answer_task_tool_call(
        {"task_id": "t", "view": "logs"}, registry=registry, tools=[unified_tool("x")], max_transcript_chars=25
    )
    assert [entry["data"] for entry in out.result["logs"]] == ["1" * 10, "2" * 10]
    assert out.result["note"].startswith("Truncated to the 2 most recent entries")

    tiny = await answer_task_tool_call(
        {"task_id": "t", "view": "logs"}, registry=registry, tools=[unified_tool("x")], max_transcript_chars=15
    )
    assert [entry["data"] for entry in tiny.result["logs"]] == ["2" * 10]

    over = await answer_task_tool_call(
        {"task_id": "t", "view": "logs"}, registry=registry, tools=[unified_tool("x")], max_transcript_chars=9
    )
    # Even the newest entry alone is over budget: it is returned truncated to
    # `max_chars - 12` (here nothing) plus the marker, never an empty list.
    assert [entry["data"] for entry in over.result["logs"]] == ["…[truncated]"]
    assert over.result["logs"][0]["seq"] == 3


def test_persisted_check_orphaned_note_survives_view_notes() -> None:
    pending = PendingAsyncTool(
        call_id="c",
        task_id="t",
        name="render",
        mode="background",
        status="working",
        started_at=0,
        orphaned=True,
        last_log=PendingAsyncToolLastLog(at=7, text="frame 9"),
    )
    status = persisted_task_check_result({}, pending)
    assert status["orphaned"] is True
    assert status["note"] == "This task was detached; its result will not be delivered."
    logs = persisted_task_check_result({"view": "logs"}, pending)
    assert logs["logs"] == [{"at": 7, "data": "frame 9"}]
    assert logs["note"] == (
        "This task was detached; its result will not be delivered. Full logs are not retained across processes."
    )
    transcript = persisted_task_check_result({"view": "transcript"}, pending)
    assert transcript["note"].endswith("No transcript available — the task ran in a previous process.")


def test_async_tools_barrel_re_exports_the_same_objects() -> None:
    from openrouter_agent import async_tool_registry, async_tools, tool_check, tool_concurrency, tool_task

    assert async_tools.AsyncToolRegistry is async_tool_registry.AsyncToolRegistry
    assert async_tools.TaskToolInputSchema is tool_check.TaskToolInputSchema
    assert async_tools.TASK_TOOL_NAME == tool_check.TASK_TOOL_NAME
    assert async_tools.ToolSemaphore is async_tools.Semaphore is tool_concurrency.Semaphore
    assert async_tools.acquire_all is tool_concurrency.acquire_all
    assert async_tools.ToolTask is tool_task.ToolTask
    assert async_tools.TASK_RESULT_BOUNDARY == tool_task.TASK_RESULT_BOUNDARY
    assert set(async_tools.__all__) >= {
        "build_task_tool_stub",
        "default_check_result",
        "has_task_tool_name_collision",
        "persisted_task_check_result",
        "resolve_check_config",
    }
