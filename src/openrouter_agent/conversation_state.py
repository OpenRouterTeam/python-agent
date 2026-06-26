from __future__ import annotations

import time
import uuid
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ._utils import json_dumps, maybe_await
from .tool_types import ConversationState, ParsedToolCall, Tool, UnsentToolResult, get_tool_function, is_client_tool
from .turn_context import normalize_input_to_array


def _now_ms() -> int:
    return int(time.time() * 1000)


def generate_conversation_id() -> str:
    return f"conv_{uuid.uuid4()}"


def create_initial_state(id: Optional[str] = None) -> ConversationState:
    now = _now_ms()
    return ConversationState(
        id=id or generate_conversation_id(), messages=[], status="in_progress", created_at=now, updated_at=now
    )


def update_state(state: ConversationState, updates: Mapping[str, Any]) -> ConversationState:
    normalized: Dict[str, Any] = {}
    for key, value in updates.items():
        normalized[key] = value
    return replace(state, **normalized, updated_at=_now_ms())


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
