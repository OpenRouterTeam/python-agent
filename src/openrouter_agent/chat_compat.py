from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence

from ._utils import get_field
from .stream_transformers import extract_responses_message_from_response


def _content_to_string(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, separators=(",", ":"))


def from_chat_messages(messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            output.append(
                {
                    "type": "function_call_output",
                    "callId": msg.get("tool_call_id") or msg.get("toolCallId"),
                    "output": _content_to_string(msg.get("content")),
                }
            )
            continue
        if role == "assistant":
            content = _content_to_string(msg.get("content"))
            tool_calls = msg.get("tool_calls") or msg.get("toolCalls") or []
            # Skip the message item only when there is no content AND tool
            # calls replace it. A content-less assistant message with no tool
            # calls still round-trips as an empty message (upstream #11/#41).
            if content or not tool_calls:
                output.append({"type": "message", "role": role, "content": content})
            # One function_call item per tool call. Chat-format arguments are
            # already a JSON string, so they are forwarded as-is.
            for tc in tool_calls:
                fn = get_field(tc, "function", {}) or {}
                tc_id = get_field(tc, "id")
                output.append(
                    {
                        "type": "function_call",
                        "callId": tc_id,
                        "id": tc_id,
                        "name": get_field(fn, "name"),
                        "arguments": get_field(fn, "arguments", ""),
                        "status": "completed",
                    }
                )
            continue
        output.append({"type": "message", "role": role, "content": _content_to_string(msg.get("content"))})
    return output


def to_chat_message(response: Any) -> Dict[str, Any]:
    message = extract_responses_message_from_response(response)
    text_parts = []
    for content in get_field(message, "content", []) or []:
        if get_field(content, "type") in {"output_text", "text"}:
            text_parts.append(str(get_field(content, "text", "")))
    result: Dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) if text_parts else None}
    tool_calls = []
    for item in getattr(response, "output", None) or (
        response.get("output", []) if isinstance(response, Mapping) else []
    ):
        if get_field(item, "type") == "function_call":
            tool_calls.append(
                {
                    "id": get_field(item, "callId", get_field(item, "id")),
                    "type": "function",
                    "function": {"name": get_field(item, "name"), "arguments": get_field(item, "arguments", "{}")},
                }
            )
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result
