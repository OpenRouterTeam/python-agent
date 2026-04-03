"""Chat format (OpenAI-style) <-> OpenResponses format conversion."""

from __future__ import annotations

import json
from typing import Any


def from_chat_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OpenAI Chat-format messages to OpenResponses input format.

    Chat format uses role + content/tool_calls/tool_call_id.
    OpenResponses format uses typed items.
    """
    items: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")

        if role in ("user", "system", "developer"):
            items.append({
                "role": role,
                "content": msg.get("content", ""),
            })

        elif role == "assistant":
            content = msg.get("content")
            tool_calls = msg.get("tool_calls", [])

            if content:
                items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                })

            for tc in tool_calls:
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                items.append({
                    "type": "function_call",
                    "id": tc.get("id", ""),
                    "call_id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "arguments": args if isinstance(args, str) else json.dumps(args),
                })

        elif role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": msg.get("content", ""),
            })

    return items


def to_chat_message(response: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenResponses result to a Chat-format assistant message."""
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    output = response.get("output", [])

    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "")

            if item_type == "message":
                inner_content = item.get("content", [])
                if isinstance(inner_content, list):
                    for block in inner_content:
                        if isinstance(block, dict) and block.get("type") == "output_text":
                            content_parts.append(block.get("text", ""))
                elif isinstance(inner_content, str):
                    content_parts.append(inner_content)

            elif item_type == "output_text":
                content_parts.append(item.get("text", ""))

            elif item_type == "function_call":
                args = item.get("arguments", {})
                tool_calls.append({
                    "id": item.get("call_id", item.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": args if isinstance(args, str) else json.dumps(args),
                    },
                })

    result: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts) if content_parts else None,
    }
    if tool_calls:
        result["tool_calls"] = tool_calls

    return result
