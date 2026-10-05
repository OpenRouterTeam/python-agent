from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Sequence

from typing_extensions import Final, TypedDict

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
    active_tools: Sequence[str]
    stream_replay: str
    doom_loop: Any
    signal: Any
    tool_timeout_ms: float
    tool_concurrency: Any
    async_tools: Mapping[str, Any]


CallModelInputWithState = CallModelInput


class ResolvedCallModelInput(TypedDict, total=False):
    model: str
    input: Any
    tools: Sequence[Mapping[str, Any]]
    stream: bool


#: Client-only request fields: handled by the engine, never sent to the API
#: (upstream `clientOnlyFields`).
CLIENT_ONLY_FIELDS = frozenset(
    {
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
        "hooks",
        "doom_loop",
        "signal",
        "tool_timeout_ms",
        "tool_concurrency",
        "async_tools",
        "active_tools",
    }
)

_EXCLUDED = {
    "stream_replay",
    "doom_loop",
    "signal",
    "tool_timeout_ms",
    "tool_concurrency",
    "async_tools",
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
}


def has_async_functions(request: Mapping[str, Any]) -> bool:
    return any(callable(value) for key, value in request.items() if key not in _EXCLUDED)


async def resolve_async_functions(request: Mapping[str, Any], turn_context: Mapping[str, Any]) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    copied = dict(request)
    strip_tool_set_snapshot_metadata(copied)
    for key, value in copied.items():
        if key in _EXCLUDED:
            continue
        if callable(value):
            resolved[key] = await maybe_await(value(turn_context))
        else:
            resolved[key] = value
    return resolved


#: Marker identifying dicts produced by `openrouter_agent.tool_set`
#: (`ToolSet.resolve` / `infer_tools` / `resolve_situation`). Upstream uses
#: `Symbol.for('@openrouter/agent/tool-set/snapshot')`; Python has no symbols, so
#: a reserved string key stands in. Like the symbol, it survives dict spreading
#: (`{**snapshot, "model": ...}`), which is what lets `call_model` strip the
#: snapshot's metadata without reserving otherwise legitimate request keys.
TOOL_SET_SNAPSHOT: Final = "__openrouter_tool_set_snapshot__"

_TOOL_SET_SNAPSHOT_METADATA_KEYS = ("enabled", "disabled", "status_by_tool", "call_model")


def strip_tool_set_snapshot_metadata(request: MutableMapping[str, Any]) -> None:
    """Remove tool-set metadata in place, only from marked snapshots or their spreads.

    Identically named keys on ordinary (unmarked) requests are preserved.
    """
    if request.get(TOOL_SET_SNAPSHOT) is not True:
        return
    for key in _TOOL_SET_SNAPSHOT_METADATA_KEYS:
        request.pop(key, None)
    request.pop(TOOL_SET_SNAPSHOT, None)
