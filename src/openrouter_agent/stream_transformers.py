from __future__ import annotations

import json
from typing import Any, AsyncIterable, AsyncIterator, Dict, List, Mapping, Optional

from ._utils import get_field, json_loads_maybe
from .tool_types import ParsedToolCall

StreamableOutputItem = Dict[str, Any]
ItemInProgress = Dict[str, Any]
stream_termination_events = {"response.completed", "response.failed", "response.incomplete"}
streamTerminationEvents = stream_termination_events
itemsStreamHandlers: Dict[str, Any] = {}


def _output_items(response: Any) -> List[Any]:
    output = get_field(response, "output", [])
    if output is None:
        return []
    return output if isinstance(output, list) else [output]


def extract_text_from_response(response: Any) -> str:
    parts: List[str] = []
    for item in _output_items(response):
        if get_field(item, "type") != "message":
            continue
        for content in get_field(item, "content", []) or []:
            if get_field(content, "type") in {"output_text", "text"}:
                parts.append(str(get_field(content, "text", "")))
    return "".join(parts)


def extract_responses_message_from_response(response: Any) -> Any:
    for item in _output_items(response):
        if get_field(item, "type") == "message":
            return item
    raise ValueError("Response does not contain a message output item")


def extract_tool_calls_from_response(response: Any) -> List[ParsedToolCall]:
    calls: List[ParsedToolCall] = []
    for item in _output_items(response):
        if get_field(item, "type") != "function_call":
            continue
        raw_args = get_field(item, "arguments", {})
        args = json_loads_maybe(raw_args)
        calls.append(
            ParsedToolCall(
                id=str(get_field(item, "callId", get_field(item, "call_id", get_field(item, "id", "")))),
                name=str(get_field(item, "name", "")),
                arguments=args,
            )
        )
    return calls


async def consume_stream_for_completion(stream: AsyncIterable[Any]) -> Any:
    completed: Optional[Any] = None
    async for event in stream:
        typ = get_field(event, "type")
        if typ in {"response.completed", "response.incomplete"}:
            completed = get_field(event, "response")
        if typ == "response.failed":
            raise RuntimeError(str(get_field(event, "message", "Response failed")))
    if completed is None:
        raise RuntimeError("Stream ended without response.completed")
    return completed


async def extract_text_deltas(events: AsyncIterable[Any]) -> AsyncIterator[str]:
    async for event in events:
        if get_field(event, "type") == "response.output_text.delta":
            yield str(get_field(event, "delta", ""))


async def extract_reasoning_deltas(events: AsyncIterable[Any]) -> AsyncIterator[str]:
    async for event in events:
        if get_field(event, "type") == "response.reasoning_text.delta":
            yield str(get_field(event, "delta", ""))


async def extract_tool_deltas(events: AsyncIterable[Any]) -> AsyncIterator[Dict[str, Any]]:
    async for event in events:
        if get_field(event, "type") == "response.function_call_arguments.delta":
            yield {"type": "delta", "content": str(get_field(event, "delta", ""))}


async def build_tool_call_stream(events: AsyncIterable[Any]) -> AsyncIterator[ParsedToolCall]:
    names: Dict[str, str] = {}
    call_ids: Dict[str, str] = {}
    buffers: Dict[str, List[str]] = {}
    async for event in events:
        typ = get_field(event, "type")
        if typ == "response.output_item.added":
            item = get_field(event, "item", {})
            if get_field(item, "type") == "function_call":
                item_id = str(get_field(item, "id", get_field(item, "callId", "")))
                names[item_id] = str(get_field(item, "name", ""))
                call_ids[item_id] = str(get_field(item, "callId", item_id))
                buffers.setdefault(item_id, [])
        elif typ == "response.function_call_arguments.delta":
            item_id = str(get_field(event, "itemId", get_field(event, "item_id", get_field(event, "callId", ""))))
            buffers.setdefault(item_id, []).append(str(get_field(event, "delta", "")))
        elif typ == "response.function_call_arguments.done":
            item_id = str(get_field(event, "itemId", get_field(event, "item_id", get_field(event, "callId", ""))))
            raw = get_field(event, "arguments", "".join(buffers.get(item_id, [])))
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                args = raw
            yield ParsedToolCall(
                id=call_ids.get(item_id, item_id),
                name=names.get(item_id, str(get_field(event, "name", ""))),
                arguments=args,
            )


async def build_items_stream(events: AsyncIterable[Any]) -> AsyncIterator[Any]:
    async for event in events:
        if get_field(event, "type") in {"response.output_item.done", "tool.call_output"}:
            yield get_field(event, "item", get_field(event, "output", event))


async def build_responses_message_stream(events: AsyncIterable[Any]) -> AsyncIterator[Any]:
    async for event in events:
        if (
            get_field(event, "type") == "response.output_item.done"
            and get_field(get_field(event, "item", {}), "type") == "message"
        ):
            yield get_field(event, "item")


def extract_unsupported_content(value: Any) -> List[Any]:
    found: List[Any] = []

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            if "unsupported_content" in node:
                content = node["unsupported_content"]
                found.extend(content if isinstance(content, list) else [content])
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return found


def has_unsupported_content(value: Any) -> bool:
    return bool(extract_unsupported_content(value))


def get_unsupported_content_summary(value: Any) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for item in extract_unsupported_content(value):
        typ = str(get_field(item, "original_type", get_field(item, "type", "unknown")))
        summary[typ] = summary.get(typ, 0) + 1
    return summary


def convert_to_claude_message(response: Any) -> Dict[str, Any]:
    from .anthropic_compat import to_claude_message

    return to_claude_message(response)
