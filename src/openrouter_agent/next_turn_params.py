from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from ._utils import maybe_await
from .tool_types import ParsedToolCall, Tool, get_tool_function, is_client_tool

NEXT_TURN_KEYS = ("input", "model", "models", "temperature", "max_output_tokens", "top_p", "top_k", "instructions")


def build_next_turn_params_context(request: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: request.get(key) for key in NEXT_TURN_KEYS if key in request}


async def execute_next_turn_params_functions(
    tool_calls: Sequence[ParsedToolCall], tools: Sequence[Tool], request: Mapping[str, Any]
) -> Dict[str, Any]:
    context = build_next_turn_params_context(request)
    computed: Dict[str, Any] = {}
    for call in tool_calls:
        matching = next(
            (tool for tool in tools if is_client_tool(tool) and get_tool_function(tool).get("name") == call.name), None
        )
        if not matching:
            continue
        fns = get_tool_function(matching).get("next_turn_params") or {}
        for key, fn in fns.items():
            computed[key] = await maybe_await(fn(call.arguments, context))
            context[key] = computed[key]
    return computed


def apply_next_turn_params_to_request(request: Mapping[str, Any], params: Mapping[str, Any]) -> Dict[str, Any]:
    updated = dict(request)
    for key, value in params.items():
        if value is None:
            updated.pop(key, None)
        else:
            updated[key] = value
    return updated
