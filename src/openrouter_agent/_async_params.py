"""Async parameter resolution for CallModelInput fields."""

from __future__ import annotations

import inspect
from typing import Any

from ._types import TurnContext

# Fields that are SDK-only and never sent to the API
CLIENT_ONLY_FIELDS = frozenset({
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
    "stream",
})


async def resolve_field(
    value: Any, context: TurnContext, field_name: str
) -> Any:
    """Resolve a single field that may be static or a callable."""
    if not callable(value):
        return value
    try:
        result = value(context)
        if inspect.isawaitable(result):
            result = await result
        return result
    except Exception as e:
        raise RuntimeError(f"Error resolving async field '{field_name}': {e}") from e


async def resolve_async_functions(
    input_dict: dict[str, Any], context: TurnContext
) -> dict[str, Any]:
    """Resolve all async/callable fields in a CallModelInput dict.

    Client-only fields are excluded from resolution (they're handled by the SDK).
    """
    resolved: dict[str, Any] = {}
    for key, value in input_dict.items():
        if key in CLIENT_ONLY_FIELDS:
            resolved[key] = value
        else:
            resolved[key] = await resolve_field(value, context, key)
    return resolved


def has_async_functions(input_dict: dict[str, Any]) -> bool:
    """Check if any non-client-only fields are callables."""
    for key, value in input_dict.items():
        if key not in CLIENT_ONLY_FIELDS and callable(value):
            return True
    return False
