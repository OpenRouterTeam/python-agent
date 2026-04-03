"""Stream extraction and transformation utilities."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ._types import (
    ResponseStreamEvent,
    ToolStreamEvent,
    is_tool_preliminary_result_event,
)


async def extract_text_deltas(
    stream: AsyncIterator[ResponseStreamEvent],
) -> AsyncIterator[str]:
    """Extract text content deltas from a response stream."""
    async for event in stream:
        if isinstance(event, dict):
            # Handle OpenRouter streaming events
            event_type = event.get("type", "")
            if event_type == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    yield delta
            elif event_type == "response.content_part.delta":
                delta = event.get("delta", {})
                if isinstance(delta, dict):
                    text = delta.get("text", "")
                    if text:
                        yield text


async def extract_reasoning_deltas(
    stream: AsyncIterator[ResponseStreamEvent],
) -> AsyncIterator[str]:
    """Extract reasoning content deltas from a response stream."""
    async for event in stream:
        if isinstance(event, dict):
            event_type = event.get("type", "")
            if event_type == "response.reasoning.delta":
                delta = event.get("delta", "")
                if delta:
                    yield delta


async def extract_tool_stream_events(
    stream: AsyncIterator[ResponseStreamEvent],
) -> AsyncIterator[ToolStreamEvent]:
    """Extract tool-related events from a response stream."""
    async for event in stream:
        if isinstance(event, dict):
            event_type = event.get("type", "")
            if event_type == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    yield ToolStreamEvent(type="delta", content=delta)

        if is_tool_preliminary_result_event(event):
            if hasattr(event, "tool_call_id"):
                yield ToolStreamEvent(
                    type="preliminary_result",
                    tool_call_id=event.tool_call_id,
                    result=event.result,
                )
            elif isinstance(event, dict):
                yield ToolStreamEvent(
                    type="preliminary_result",
                    tool_call_id=event.get("tool_call_id", ""),
                    result=event.get("result"),
                )


def extract_unsupported_content(
    message: dict[str, Any], original_type: str = ""
) -> list[dict[str, Any]]:
    """Extract unsupported content blocks from a message."""
    unsupported: list[dict[str, Any]] = []
    output = message.get("output", [])
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict) and item.get("type") not in (
                "message",
                "function_call",
                "function_call_output",
            ):
                unsupported.append({
                    "type": item.get("type", "unknown"),
                    "original_type": original_type,
                })
    return unsupported


def has_unsupported_content(message: dict[str, Any]) -> bool:
    """Check if a message contains unsupported content."""
    return len(extract_unsupported_content(message, "")) > 0


def get_unsupported_content_summary(message: dict[str, Any]) -> dict[str, int]:
    """Get a summary count of unsupported content types."""
    items = extract_unsupported_content(message, "")
    summary: dict[str, int] = {}
    for item in items:
        t = item.get("type", "unknown")
        summary[t] = summary.get(t, 0) + 1
    return summary
