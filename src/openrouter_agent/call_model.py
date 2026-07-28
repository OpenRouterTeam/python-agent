from __future__ import annotations

from typing import Any, Mapping, Optional

from .hooks_resolve import resolve_hooks
from .model_result import ModelResult
from .tool_executor import convert_tools_to_api_format


def call_model(client: Any, request: Mapping[str, Any], options: Optional[Mapping[str, Any]] = None) -> ModelResult:
    tools = request.get("tools")
    final_request = dict(request)
    for key in (
        "tools",
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
