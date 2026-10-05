from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .async_params import CLIENT_ONLY_FIELDS, strip_tool_set_snapshot_metadata
from .hooks_resolve import resolve_hooks
from .model_result import ModelResult
from .tool_check import build_task_tool_api_definition, needs_task_tool
from .tool_executor import convert_tools_to_api_format
from .tool_types import get_tool_function, is_server_tool


def call_model(client: Any, request: Mapping[str, Any], options: Optional[Mapping[str, Any]] = None) -> ModelResult:
    """Start an agent run against the Responses API (upstream `callModel`).

    Client-only options (stripped before the request is sent): ``tools``,
    ``active_tools``, ``stop_when``, ``state``, ``require_approval``,
    ``approve_tool_calls``, ``reject_tool_calls``, ``context``,
    ``shared_context_schema``, ``on_turn_start``, ``on_turn_end``,
    ``stream_replay``, ``allow_final_response``, ``strict_final_response``,
    ``hooks``, ``doom_loop``, ``signal``, ``tool_timeout_ms``,
    ``tool_concurrency``, ``async_tools``.
    """
    tools = request.get("tools")
    active_tools = request.get("active_tools")

    # Narrow tools to the active subset before API conversion and before they
    # are registered for execution, so the model cannot call filtered tools
    # and the executor carries no orphaned definitions. Server tools always
    # pass; unknown names are ignored.
    if active_tools is not None and tools is not None:
        active = set(active_tools)
        tools = [t for t in tools if is_server_tool(t) or get_tool_function(t).get("name") in active]
    # A filtered-to-empty (or explicitly empty) list collapses to no tools so
    # the outbound `tools` key is omitted entirely (providers reject `[]`).
    filtered_tools = list(tools) if tools else None

    final_request: Dict[str, Any] = dict(request)
    strip_tool_set_snapshot_metadata(final_request)
    for key in CLIENT_ONLY_FIELDS:
        final_request.pop(key, None)
    final_request.pop("tools", None)

    async_tools = request.get("async_tools") or {}
    if filtered_tools is not None:
        api_tools = convert_tools_to_api_format(filtered_tools)
        # Append the single universal `task` tool when any long-running tool
        # is registered (and check-ins aren't disabled): ONE static wire
        # definition for check/steer/result/cancel across every async tool.
        if async_tools.get("checkins") is not False and needs_task_tool(filtered_tools):
            api_tools.append(build_task_tool_api_definition())
        final_request["tools"] = api_tools

    headers = dict((options or {}).get("headers", {})) if options else {}
    headers["x-openrouter-callmodel"] = "true"
    response_options = dict(options or {})
    response_options["headers"] = headers
    engine: Dict[str, Any] = {
        "client": client,
        "request": final_request,
        "options": response_options,
        "tools": filtered_tools,
    }
    for key in (
        "stop_when",
        "state",
        "require_approval",
        "approve_tool_calls",
        "reject_tool_calls",
        "context",
        "shared_context_schema",
        "on_turn_start",
        "on_turn_end",
        "stream_replay",
        "allow_final_response",
        "strict_final_response",
        "doom_loop",
        "signal",
        "tool_timeout_ms",
        "tool_concurrency",
        "async_tools",
    ):
        engine[key] = request.get(key)
    engine["hooks"] = resolve_hooks(request.get("hooks"))
    return ModelResult(engine)
