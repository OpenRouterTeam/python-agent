from __future__ import annotations

import dataclasses
import json
import time
import uuid
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ._utils import dump, json_dumps, maybe_await
from .tool_types import (
    ConversationState,
    ParsedToolCall,
    PartialResponse,
    Tool,
    UnsentToolResult,
    get_tool_function,
    is_client_tool,
)
from .turn_context import normalize_input_to_array

#: Currently supported ConversationState serialization version.
CONVERSATION_STATE_VERSION = 1


class UnsupportedStateVersionError(Exception):
    """Raised by `deserialize_conversation_state` when a state blob's
    `version` is not supported by this SDK build."""

    def __init__(self, found: int, supported: Sequence[int] = (CONVERSATION_STATE_VERSION,)) -> None:
        supported_list = list(supported)
        super().__init__(
            f"Unsupported ConversationState version {found}; supported version(s): "
            f"{', '.join(str(v) for v in supported_list)}"
        )
        self.name = "UnsupportedStateVersionError"
        self.found = found
        self.supported = supported_list


class InvalidStateError(Exception):
    """Raised by `deserialize_conversation_state` when given JSON that is not
    a well-formed ConversationState (missing/wrong required fields, or
    invalid JSON)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.name = "InvalidStateError"


def _now_ms() -> int:
    return int(time.time() * 1000)


def generate_conversation_id() -> str:
    return f"conv_{uuid.uuid4()}"


def create_initial_state(id: Optional[str] = None) -> ConversationState:
    now = _now_ms()
    return ConversationState(
        id=id or generate_conversation_id(),
        messages=[],
        status="in_progress",
        created_at=now,
        updated_at=now,
        version=CONVERSATION_STATE_VERSION,
    )


def update_state(state: ConversationState, updates: Mapping[str, Any]) -> ConversationState:
    normalized: Dict[str, Any] = {}
    for key, value in updates.items():
        if key in ("id", "created_at", "version"):
            continue
        normalized[key] = value
    return replace(state, **normalized, updated_at=_now_ms())


def _describe_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def serialize_conversation_state(state: ConversationState) -> str:
    """Serialize a `ConversationState` to a stable JSON string for durable
    storage.

    Guarantees the `version` field is present (injects `CONVERSATION_STATE_VERSION`
    when the input state lacks it). Treat the returned JSON as **opaque**:
    consumers should round-trip via these helpers rather than introspecting
    item shapes.

    Note: the StateAccessor load/save contract is unchanged -- these helpers
    are opt-in for callers that need a durable, versioned wire format.
    """
    payload = dataclasses.asdict(state)
    payload["version"] = state.version if state.version is not None else CONVERSATION_STATE_VERSION
    # State built from live responses can hold SDK pydantic items (e.g.
    # OutputFunctionCallItem), which dataclasses.asdict passes through
    # untouched; dump() them to plain dicts so the wire format stays JSON.
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=dump)


def deserialize_conversation_state(raw_json: str) -> ConversationState:
    """Parse and validate a previously serialized `ConversationState`.

    Accepts version-less legacy blobs and states with `version: 1`,
    normalizing both to `version: 1`. Raises `UnsupportedStateVersionError`
    for any other version. Raises `InvalidStateError` for malformed JSON or
    missing required fields (`id`, `messages`, `status`, `created_at`,
    `updated_at`).
    """
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as error:
        raise InvalidStateError(f"Invalid ConversationState JSON: {error}") from error

    if not isinstance(parsed, dict):
        raise InvalidStateError("ConversationState must be a JSON object")

    # Version check runs before structural validation: a future-version blob
    # may have a different shape, and it must fail with
    # UnsupportedStateVersionError rather than a misleading InvalidStateError.
    version = parsed.get("version")
    if version is not None and version != CONVERSATION_STATE_VERSION:
        if not isinstance(version, int) or isinstance(version, bool):
            raise InvalidStateError(
                f'ConversationState field "version" must be a number when present (got {_describe_type(version)})'
            )
        raise UnsupportedStateVersionError(version, [CONVERSATION_STATE_VERSION])

    if not isinstance(parsed.get("id"), str):
        raise InvalidStateError(
            f'ConversationState missing or invalid field "id" (expected string, got {_describe_type(parsed.get("id"))})'
        )
    if not isinstance(parsed.get("messages"), list):
        raise InvalidStateError(
            'ConversationState missing or invalid field "messages" '
            f"(expected array, got {_describe_type(parsed.get('messages'))})"
        )
    if not isinstance(parsed.get("status"), str):
        raise InvalidStateError(
            f'ConversationState missing or invalid field "status" '
            f"(expected string, got {_describe_type(parsed.get('status'))})"
        )
    if not isinstance(parsed.get("created_at"), (int, float)) or isinstance(parsed.get("created_at"), bool):
        raise InvalidStateError(
            'ConversationState missing or invalid field "created_at" '
            f"(expected number, got {_describe_type(parsed.get('created_at'))})"
        )
    if not isinstance(parsed.get("updated_at"), (int, float)) or isinstance(parsed.get("updated_at"), bool):
        raise InvalidStateError(
            'ConversationState missing or invalid field "updated_at" '
            f"(expected number, got {_describe_type(parsed.get('updated_at'))})"
        )

    pending_tool_calls = None
    if parsed.get("pending_tool_calls") is not None:
        pending_tool_calls = [ParsedToolCall(**item) for item in parsed["pending_tool_calls"]]

    unsent_tool_results = None
    if parsed.get("unsent_tool_results") is not None:
        unsent_tool_results = [UnsentToolResult(**item) for item in parsed["unsent_tool_results"]]

    partial_response = None
    if parsed.get("partial_response") is not None:
        partial_response = PartialResponse(**parsed["partial_response"])

    return ConversationState(
        id=parsed["id"],
        messages=parsed["messages"],
        status=parsed["status"],
        created_at=parsed["created_at"],
        updated_at=parsed["updated_at"],
        previous_response_id=parsed.get("previous_response_id"),
        pending_tool_calls=pending_tool_calls,
        unsent_tool_results=unsent_tool_results,
        partial_response=partial_response,
        interrupted_by=parsed.get("interrupted_by"),
        version=CONVERSATION_STATE_VERSION,
    )


def append_to_messages(current: Any, new_items: Sequence[Any]) -> List[Any]:
    return [*normalize_input_to_array(current), *list(new_items)]


async def tool_requires_approval(
    tool_call: ParsedToolCall,
    tools: Sequence[Tool],
    context: Mapping[str, Any],
    call_level_check: Any = None,
) -> bool:
    if call_level_check is not None:
        return bool(await maybe_await(call_level_check(tool_call, context)))
    matching = next(
        (
            candidate
            for candidate in tools
            if is_client_tool(candidate) and get_tool_function(candidate).get("name") == tool_call.name
        ),
        None,
    )
    if not matching:
        return False
    requirement = get_tool_function(matching).get("require_approval")
    if callable(requirement):
        return bool(await maybe_await(requirement(tool_call.arguments, context)))
    return bool(requirement)


async def partition_tool_calls(
    tool_calls: Sequence[ParsedToolCall],
    tools: Sequence[Tool],
    context: Mapping[str, Any],
    call_level_check: Any = None,
) -> Dict[str, List[ParsedToolCall]]:
    requires_approval: List[ParsedToolCall] = []
    auto_execute: List[ParsedToolCall] = []
    for call in tool_calls:
        if await tool_requires_approval(call, tools, context, call_level_check):
            requires_approval.append(call)
        else:
            auto_execute.append(call)
    return {"requires_approval": requires_approval, "auto_execute": auto_execute}


def create_unsent_result(call_id: str, name: str, output: Any) -> UnsentToolResult:
    return UnsentToolResult(call_id=call_id, name=name, output=output)


def create_rejected_result(call_id: str, name: str, reason: Optional[str] = None) -> UnsentToolResult:
    return UnsentToolResult(call_id=call_id, name=name, output=None, error=reason or "Tool call rejected by user")


def is_content_array(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(item, Mapping) and item.get("type") in {"input_text", "input_image", "input_file"}
            for item in value
        )
    )


def unsent_results_to_api_format(results: Sequence[UnsentToolResult]) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for result in results:
        if result.error:
            output: Any = json_dumps({"error": result.error})
        elif is_content_array(result.output):
            output = result.output
        else:
            output = json_dumps(result.output)
        formatted.append(
            {
                "type": "function_call_output",
                "id": f"output_{result.call_id}",
                "callId": result.call_id,
                "output": output,
            }
        )
    return formatted


def extract_text_from_response(response: Any) -> str:
    from .stream_transformers import extract_text_from_response as _extract

    return _extract(response)


def extract_tool_calls_from_response(response: Any) -> List[ParsedToolCall]:
    from .stream_transformers import extract_tool_calls_from_response as _extract

    return _extract(response)
