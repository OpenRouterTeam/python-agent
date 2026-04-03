"""Dynamic parameter computation per turn based on tool nextTurnParams."""

from __future__ import annotations

import inspect
from typing import Any

from ._tool_executor import find_tool_by_name
from ._types import NextTurnParamsContext, ParsedToolCall, Tool


def build_next_turn_params_context(
    request: dict[str, Any],
) -> NextTurnParamsContext:
    """Build the context object for next-turn parameter functions."""
    return NextTurnParamsContext(
        input=request.get("input", ""),
        model=request.get("model", ""),
        models=request.get("models", []),
        temperature=request.get("temperature"),
        max_output_tokens=request.get("max_output_tokens"),
        top_p=request.get("top_p"),
        top_k=request.get("top_k"),
        instructions=request.get("instructions"),
    )


async def execute_next_turn_params_functions(
    tool_calls: list[ParsedToolCall],
    tools: list[Tool],
    current_request: dict[str, Any],
) -> dict[str, Any]:
    """Execute nextTurnParams functions from all called tools.

    Returns a dict of parameter overrides to apply to the next request.
    """
    overrides: dict[str, Any] = {}
    context = build_next_turn_params_context(current_request)

    for tc in tool_calls:
        t = find_tool_by_name(tools, tc.name)
        if t is None or t.function.next_turn_params is None:
            continue

        for param_name, fn in t.function.next_turn_params.items():
            result = fn(tc.arguments, context)
            if inspect.isawaitable(result):
                result = await result
            overrides[param_name] = result

    return overrides


def apply_next_turn_params_to_request(
    request: dict[str, Any],
    computed_params: dict[str, Any],
) -> dict[str, Any]:
    """Apply computed next-turn parameters to the request."""
    updated = dict(request)
    for key, value in computed_params.items():
        if value is not None:
            updated[key] = value
    return updated
