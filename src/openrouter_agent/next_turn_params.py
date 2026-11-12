from __future__ import annotations

import warnings
from typing import Any, Dict, Mapping, Sequence

from ._utils import maybe_await
from .tool_types import ParsedToolCall, Tool, get_tool_function, is_client_tool

NEXT_TURN_KEYS = (
    "input",
    "tool_choice",
    "model",
    "models",
    "temperature",
    "max_output_tokens",
    "top_p",
    "top_k",
    "instructions",
)


def build_next_turn_params_context(request: Mapping[str, Any]) -> Dict[str, Any]:
    """Current parameter values handed to `next_turn_params` functions, with
    upstream's defaults for absent fields. `tool_choice` and `top_k` are
    passed through only when present (upstream leaves them `undefined`)."""
    context: Dict[str, Any] = {
        "input": request.get("input") if request.get("input") is not None else [],
        "model": request.get("model") or "",
        "models": request.get("models") or [],
        "temperature": request.get("temperature"),
        "max_output_tokens": request.get("max_output_tokens"),
        "top_p": request.get("top_p"),
        "instructions": request.get("instructions"),
    }
    for optional_key in ("tool_choice", "top_k"):
        if request.get(optional_key) is not None:
            context[optional_key] = request[optional_key]
    return context


def _describe_type(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "object"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "object"


async def execute_next_turn_params_functions(
    tool_calls: Sequence[ParsedToolCall], tools: Sequence[Tool], request: Mapping[str, Any]
) -> Dict[str, Any]:
    """Execute `next_turn_params` functions for all called tools, in tools
    order, composing values when several tools modify the same parameter
    (later tools see earlier tools' results)."""
    context = build_next_turn_params_context(request)
    computed: Dict[str, Any] = {}
    for matching in tools:
        if not is_client_tool(matching):
            continue
        fn_def = get_tool_function(matching)
        next_params = fn_def.get("next_turn_params")
        if not next_params:
            continue
        name = fn_def.get("name")
        for call in [c for c in tool_calls if c.name == name]:
            if not isinstance(call.arguments, Mapping):
                raise TypeError(
                    f"Tool call arguments for {name} must be an object, got {_describe_type(call.arguments)}"
                )
            for key, fn in next_params.items():
                if not callable(fn):
                    continue
                if key not in NEXT_TURN_KEYS:
                    warnings.warn(
                        f'Invalid next_turn_params key "{key}" in tool "{name}". '
                        f"Valid keys: {', '.join(NEXT_TURN_KEYS)}",
                        stacklevel=2,
                    )
                    continue
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
