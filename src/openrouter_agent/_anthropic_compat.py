"""Claude/Anthropic format <-> OpenResponses format conversion."""

from __future__ import annotations

import json
from typing import Any

from ._types import CLAUDE_CONTENT_BLOCK_TYPE


def from_claude_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Claude-format messages to OpenResponses input format.

    Claude format uses role + content blocks with types like "text", "tool_use", "tool_result".
    OpenResponses format uses typed items like "message", "function_call", "function_call_output".
    """
    items: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            if isinstance(content, str):
                items.append({
                    "role": "user",
                    "content": content,
                })
            elif isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "")
                        if block_type == CLAUDE_CONTENT_BLOCK_TYPE["Text"]:
                            text_parts.append(block.get("text", ""))
                        elif block_type == CLAUDE_CONTENT_BLOCK_TYPE["ToolResult"]:
                            items.append({
                                "type": "function_call_output",
                                "call_id": block.get("tool_use_id", ""),
                                "output": _serialize_content(block.get("content", "")),
                            })
                if text_parts:
                    items.append({
                        "role": "user",
                        "content": "".join(text_parts),
                    })

        elif role == "assistant":
            if isinstance(content, str):
                items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                })
            elif isinstance(content, list):
                output_items: list[dict[str, Any]] = []
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "")
                        if block_type == CLAUDE_CONTENT_BLOCK_TYPE["Text"]:
                            output_items.append({
                                "type": "output_text",
                                "text": block.get("text", ""),
                            })
                        elif block_type == CLAUDE_CONTENT_BLOCK_TYPE["ToolUse"]:
                            items.append({
                                "type": "function_call",
                                "id": block.get("id", ""),
                                "call_id": block.get("id", ""),
                                "name": block.get("name", ""),
                                "arguments": _serialize_content(block.get("input", {})),
                            })
                if output_items:
                    items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": output_items,
                    })

        elif role == "system":
            items.append({
                "role": "system",
                "content": content if isinstance(content, str) else str(content),
            })

        elif role == "developer":
            items.append({
                "role": "developer",
                "content": content if isinstance(content, str) else str(content),
            })

        elif role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_use_id", ""),
                "output": _serialize_content(content),
            })

    return items


def to_claude_message(response: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenResponses result to a Claude-format assistant message."""
    content_blocks: list[dict[str, Any]] = []
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
                            content_blocks.append({
                                "type": "text",
                                "text": block.get("text", ""),
                            })
                elif isinstance(inner_content, str):
                    content_blocks.append({
                        "type": "text",
                        "text": inner_content,
                    })

            elif item_type == "output_text":
                content_blocks.append({
                    "type": "text",
                    "text": item.get("text", ""),
                })

            elif item_type == "function_call":
                args = item.get("arguments", {})
                content_blocks.append({
                    "type": "tool_use",
                    "id": item.get("call_id", item.get("id", "")),
                    "name": item.get("name", ""),
                    "input": json.loads(args) if isinstance(args, str) else args,
                })

    return {
        "role": "assistant",
        "content": content_blocks,
    }


def _serialize_content(content: Any) -> str:
    """Serialize content to string for API format."""
    if isinstance(content, str):
        return content
    return json.dumps(content)
