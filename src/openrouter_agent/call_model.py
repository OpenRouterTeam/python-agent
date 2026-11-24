from __future__ import annotations

from typing import Any, Mapping, Optional

from .hooks_resolve import resolve_hooks
from .model_result import ModelResult
from .tool_executor import convert_tools_to_api_format
from .tool_types import get_tool_function, is_server_tool


#: Upstream `callModel` options whose features this port does not implement
#: yet. Refused instead of being forwarded to the Responses API, where they
#: would be sent as unknown request fields or silently ignored.
UNSUPPORTED_CALL_MODEL_OPTIONS = (
    "doom_loop",
    "signal",
    "tool_timeout_ms",
    "tool_concurrency",
    "async_tools",
    "stream_replay",
)


def call_model(client: Any, request: Mapping[str, Any], options: Optional[Mapping[str, Any]] = None) -> ModelResult:
    unsupported = [key for key in UNSUPPORTED_CALL_MODEL_OPTIONS if request.get(key) is not None]
    if unsupported:
        raise NotImplementedError(
            f"call_model option(s) not supported by this Python port yet: {', '.join(unsupported)}"
        )
    tools = request.get("tools")
    # Narrow tools to the active subset (if provided) before API conversion and
    # before they are registered for execution, so the model cannot call
    # filtered tools and the executor does not carry orphaned definitions.
    # Server tools are always kept.
    active_tools = request.get("active_tools")
    if active_tools is not None and tools is not None:
        active_set = set(active_tools)
        tools = [t for t in tools if is_server_tool(t) or get_tool_function(t).get("name") in active_set]
    # A filtered-to-empty (or explicitly empty) tools list omits the outbound
    # `tools` key entirely instead of sending `tools: []`, which several
    # providers reject; the engine treats it as its no-tools state.
    if tools is not None and len(tools) == 0:
        tools = None
    final_request = dict(request)
    for key in (
        "tools",
        "active_tools",
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
        }
    )
