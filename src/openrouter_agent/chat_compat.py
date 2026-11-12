from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence

from ._utils import get_field
from .stream_transformers import extract_responses_message_from_response

ChatMessageInputItem = Dict[str, Any]
"""An OpenResponses input item emitted by `from_chat_messages`: a message item
(user / system / assistant / developer), a ``function_call`` item, or a
``function_call_output`` item."""


def _content_to_string(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, separators=(",", ":"))


def _tool_call_arguments(arguments: Any) -> str:
    # Chat-format arguments are already a JSON string and are forwarded as-is
    # (never re-stringified). A pre-parsed mapping is serialized once.
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, separators=(",", ":"))


def from_chat_messages(messages: Sequence[Mapping[str, Any]]) -> List[ChatMessageInputItem]:
    output: List[ChatMessageInputItem] = []
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

        content = _content_to_string(msg.get("content"))
        if role == "assistant":
            tool_calls = list(msg.get("tool_calls") or msg.get("toolCalls") or [])

            # Skip the message item only when there is no content AND there are
            # tool calls to emit in its place. A content-less assistant message
            # with no tool calls still round-trips as an empty message.
            if content or not tool_calls:
                output.append({"type": "message", "role": role, "content": content})

            # One function_call item per tool call.
            for tool_call in tool_calls:
                call_id = get_field(tool_call, "id")
                function = get_field(tool_call, "function", {}) or {}
                output.append(
                    {
                        "type": "function_call",
                        "callId": call_id,
                        "id": call_id,
                        "name": get_field(function, "name"),
                        "arguments": _tool_call_arguments(get_field(function, "arguments")),
                        "status": "completed",
                    }
                )
            continue

        # System, user, developer messages
        output.append({"type": "message", "role": role, "content": content})
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
