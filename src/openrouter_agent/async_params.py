from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from typing_extensions import TypedDict

from ._utils import maybe_await


class CallModelInput(TypedDict, total=False):
    model: str
    input: Any
    tools: Sequence[Any]
    stop_when: Any
    state: Any
    require_approval: Any
    approve_tool_calls: Sequence[str]
    reject_tool_calls: Sequence[str]
    context: Mapping[str, Any]
    shared_context_schema: Any
    allow_final_response: Any
    strict_final_response: bool
    hooks: Any


CallModelInputWithState = CallModelInput


class ResolvedCallModelInput(TypedDict, total=False):
    model: str
    input: Any
    tools: Sequence[Mapping[str, Any]]
    stream: bool


_EXCLUDED = {
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
}


def has_async_functions(request: Mapping[str, Any]) -> bool:
    return any(callable(value) for key, value in request.items() if key not in _EXCLUDED)


async def resolve_async_functions(request: Mapping[str, Any], turn_context: Mapping[str, Any]) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    for key, value in request.items():
        if key in _EXCLUDED:
            continue
        if callable(value):
            resolved[key] = await maybe_await(value(turn_context))
        else:
            resolved[key] = value
    return resolved
