from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .hooks_resolve import resolve_hooks
from .model_result import ModelResult
from .tool_executor import convert_tools_to_api_format


def call_model(client: Any, request: Mapping[str, Any], options: Optional[Mapping[str, Any]] = None) -> ModelResult:
    from .tool_set import strip_tool_set_snapshot_metadata
    from .tool_types import is_server_tool

    tools = request.get("tools")
    active_tools = request.get("active_tools")
    # Narrow tools to the active subset (if provided) before API conversion
    # and before they are registered for execution, so the model cannot call
    # filtered tools and the executor does not carry orphaned definitions.
    # Server tools always pass the filter.
    if active_tools is not None and tools is not None:
        active_set = set(active_tools)
        tools = [t for t in tools if is_server_tool(t) or t.get("function", {}).get("name") in active_set]
    # Collapse a filtered-to-empty (or explicitly empty) tools list to None so
    # the outbound `tools` key is omitted entirely instead of sending
    # `tools: []` -- several providers reject an empty array outright.
    if tools is not None and len(tools) == 0:
        tools = None
    final_request: Dict[Any, Any] = dict(request)
    strip_tool_set_snapshot_metadata(final_request)
    for key in (
        "tools",
        "active_tools",
        "stream_replay",
        "stop_when",
        "state",
        "require_approval",
        "approve_tool_calls",
        "reject_tool_calls",
        "context",
        "shared_context_schema",
        "on_turn_start",
        "on_turn_end",
        "allow_final_response",
        "strict_final_response",
        "hooks",
        "doom_loop",
    ):
        final_request.pop(key, None)
    if tools is not None:
        final_request["tools"] = convert_tools_to_api_format(tools)
    headers = dict((options or {}).get("headers", {})) if options else {}
    headers["x-openrouter-callmodel"] = "true"
    response_options = dict(options or {})
    response_options["headers"] = headers
    return ModelResult(
        {
            "client": client,
            "request": final_request,
            "options": response_options,
            "tools": tools,
            "stop_when": request.get("stop_when"),
            "state": request.get("state"),
            "require_approval": request.get("require_approval"),
            "approve_tool_calls": request.get("approve_tool_calls"),
            "reject_tool_calls": request.get("reject_tool_calls"),
            "context": request.get("context"),
            "shared_context_schema": request.get("shared_context_schema"),
            "on_turn_start": request.get("on_turn_start"),
            "on_turn_end": request.get("on_turn_end"),
            "allow_final_response": request.get("allow_final_response"),
            "strict_final_response": request.get("strict_final_response"),
            "hooks": resolve_hooks(request.get("hooks")),
            "doom_loop": request.get("doom_loop"),
        }
    )
