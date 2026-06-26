from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from ._utils import get_field, json_dumps, json_loads_maybe


def _unsupported(original_type: str, data: Any, reason: str) -> Dict[str, Any]:
    return {"original_type": original_type, "data": data, "reason": reason}


def from_claude_messages(messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if isinstance(content, str):
            output.append({"type": "message", "role": role, "content": content})
            continue
        parts: List[Dict[str, Any]] = []

        def flush_parts() -> None:
            if parts:
                output.append({"type": "message", "role": role, "content": list(parts)})
                parts.clear()

        for block in content or []:
            typ = block.get("type")
            if typ == "text":
                parts.append({"type": "input_text", "text": block.get("text", "")})
            elif typ == "image":
                source = block.get("source", {})
                url = source.get("url") or source.get("data") or ""
                parts.append({"type": "input_image", "image_url": url, "detail": "auto"})
            elif typ == "tool_use":
                flush_parts()
                output.append(
                    {
                        "type": "function_call",
                        "id": block.get("id"),
                        "callId": block.get("id"),
                        "name": block.get("name"),
                        "arguments": json_dumps(block.get("input", {})),
                    }
                )
            elif typ == "tool_result":
                flush_parts()
                output.append(
                    {
                        "type": "function_call_output",
                        "callId": block.get("tool_use_id"),
                        "output": block.get("content", ""),
                    }
                )
            else:
                parts.append(
                    {
                        "type": "unsupported_content",
                        "unsupported_content": [
                            _unsupported(str(typ or "unknown"), dict(block), "Claude block has no Responses equivalent")
                        ],
                    }
                )
        flush_parts()
        for item in message.get("unsupported_content", []) or []:
            if isinstance(item, Mapping) and "data" in item:
                data = item["data"]
                output.append(
                    dict(data)
                    if isinstance(data, Mapping)
                    else {"type": item.get("original_type", "unknown"), "data": data}
                )
    return output


def to_claude_message(response: Any) -> Dict[str, Any]:
    blocks: List[Dict[str, Any]] = []
    unsupported: List[Dict[str, Any]] = []
    stop_reason = "end_turn"
    for item in get_field(response, "output", []) or []:
        typ = get_field(item, "type")
        if typ == "message":
            for content in get_field(item, "content", []) or []:
                ctyp = get_field(content, "type")
                if ctyp in {"output_text", "text"}:
                    blocks.append({"type": "text", "text": str(get_field(content, "text", ""))})
                elif ctyp == "unsupported_content":
                    for entry in get_field(content, "unsupported_content", []) or []:
                        if isinstance(entry, Mapping) and {"original_type", "data", "reason"}.issubset(entry.keys()):
                            unsupported.append(dict(entry))
                        else:
                            unsupported.append(
                                _unsupported(
                                    str(get_field(entry, "type", "unknown")),
                                    entry,
                                    "Unsupported message content preserved",
                                )
                            )
                elif ctyp == "refusal":
                    unsupported.append(
                        _unsupported("refusal", content, "Claude does not have a native refusal content type")
                    )
        elif typ == "function_call":
            stop_reason = "tool_use"
            blocks.append(
                {
                    "type": "tool_use",
                    "id": get_field(item, "callId", get_field(item, "id")),
                    "name": get_field(item, "name"),
                    "input": json_loads_maybe(get_field(item, "arguments", {})),
                }
            )
        elif typ == "reasoning":
            blocks.append({"type": "thinking", "thinking": get_field(item, "summary", get_field(item, "text", ""))})
            encrypted = get_field(item, "encrypted_content", get_field(item, "encryptedContent", None))
            if encrypted:
                unsupported.append(
                    _unsupported("reasoning_encrypted", item, "Encrypted reasoning content preserved for round-trip")
                )
        elif typ:
            unsupported.append(
                _unsupported(str(typ), item, "Claude format cannot represent this Responses output item")
            )
    usage = get_field(response, "usage", {}) or {}
    result = {
        "id": get_field(response, "id", "response"),
        "type": "message",
        "role": "assistant",
        "model": get_field(response, "model", "unknown"),
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": get_field(response, "stop_sequence", get_field(response, "stopSequence", None)),
        "usage": {
            "input_tokens": get_field(usage, "input_tokens", get_field(usage, "inputTokens", 0)),
            "output_tokens": get_field(usage, "output_tokens", get_field(usage, "outputTokens", 0)),
            "cache_creation_input_tokens": get_field(usage, "cache_creation_input_tokens", 0),
            "cache_read_input_tokens": get_field(usage, "cache_read_input_tokens", 0),
        },
    }
    if unsupported:
        result["unsupported_content"] = unsupported
    return result
