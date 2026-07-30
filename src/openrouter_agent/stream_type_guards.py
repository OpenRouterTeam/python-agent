"""Stream item type guards.

Ported for 1:1 module parity with upstream `lib/stream-type-guards.ts`. Upstream
routes `model-result`, `tool-executor`, and `stream-transformers` through these
predicates; this port inlines the equivalent checks in `stream_transformers.py`,
so nothing here is currently reachable.

Kept deliberately: the porting contract (`.upstreamer/upstreamer.md`) requires one
Python module per upstream lib module, so deleting this would be a parity
regression that the next sync re-creates. Excluded from the coverage floor in
`pyproject.toml` rather than deleted. Do not re-litigate.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


def has_type_property(value: Any, expected: Optional[str] = None) -> bool:
    has_type = isinstance(value, Mapping) and isinstance(value.get("type"), str)
    return has_type if expected is None else has_type and value.get("type") == expected


def is_output_text_delta_event(event: Mapping[str, Any]) -> bool:
    return event.get("type") in {"response.output_text.delta", "response.content_part.delta"}


def is_reasoning_delta_event(event: Mapping[str, Any]) -> bool:
    return event.get("type") in {"response.reasoning_text.delta", "response.reasoning.delta"}


def is_response_completed_event(event: Mapping[str, Any]) -> bool:
    return event.get("type") == "response.completed"


def is_response_failed_event(event: Mapping[str, Any]) -> bool:
    return event.get("type") == "response.failed"


def is_response_incomplete_event(event: Mapping[str, Any]) -> bool:
    return event.get("type") == "response.incomplete"


def is_function_call_item(item: Mapping[str, Any]) -> bool:
    return item.get("type") == "function_call"


def is_function_call_output_item(item: Mapping[str, Any]) -> bool:
    return item.get("type") == "function_call_output"


def is_server_tool_result_item(item: Mapping[str, Any]) -> bool:
    return item.get("type") not in {"message", "reasoning", "function_call", "function_call_output"}
