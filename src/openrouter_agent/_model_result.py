"""ModelResult - response wrapper with multiple concurrent consumption patterns."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ._stream_transformers import (
    extract_reasoning_deltas,
    extract_text_deltas,
    extract_tool_stream_events,
)
from ._tool_event_broadcaster import ToolEventBroadcaster
from ._types import (
    ConversationState,
    ParsedToolCall,
    ResponseStreamEvent,
    StepResult,
    ToolStreamEvent,
)


class ModelResult:
    """Response wrapper with multiple concurrent consumption patterns.

    A single call_model() invocation returns a ModelResult that can be consumed
    in multiple ways concurrently. The same underlying data feeds all consumers.
    """

    def __init__(
        self,
        steps: list[StepResult],
        broadcaster: ToolEventBroadcaster[ResponseStreamEvent],
        state: ConversationState | None = None,
    ) -> None:
        self._steps = steps
        self._broadcaster = broadcaster
        self._state = state

    # -----------------------------------------------------------------------
    # Text consumption
    # -----------------------------------------------------------------------

    def get_text(self) -> str:
        """Get the full text response (concatenated across all steps)."""
        return "".join(step.text for step in self._steps)

    async def get_text_stream(self) -> AsyncIterator[str]:
        """Stream text deltas as they arrive."""
        consumer = self._broadcaster.create_consumer()
        async for delta in extract_text_deltas(consumer):
            yield delta

    # -----------------------------------------------------------------------
    # Reasoning consumption
    # -----------------------------------------------------------------------

    async def get_reasoning_stream(self) -> AsyncIterator[str]:
        """Stream reasoning deltas."""
        consumer = self._broadcaster.create_consumer()
        async for delta in extract_reasoning_deltas(consumer):
            yield delta

    # -----------------------------------------------------------------------
    # Full response
    # -----------------------------------------------------------------------

    def get_response(self) -> dict[str, Any]:
        """Get the final API response."""
        if self._steps:
            return self._steps[-1].response
        return {}

    async def get_full_responses_stream(self) -> AsyncIterator[ResponseStreamEvent]:
        """Stream all response events (API events + tool events + turn events)."""
        consumer = self._broadcaster.create_consumer()
        async for event in consumer:
            yield event

    # -----------------------------------------------------------------------
    # Items stream
    # -----------------------------------------------------------------------

    async def get_items_stream(self) -> AsyncIterator[dict[str, Any]]:
        """Stream output items as they arrive."""
        consumer = self._broadcaster.create_consumer()
        async for event in consumer:
            if isinstance(event, dict):
                output = event.get("output", [])
                if isinstance(output, list):
                    for item in output:
                        yield item

    # -----------------------------------------------------------------------
    # Message stream
    # -----------------------------------------------------------------------

    async def get_new_messages_stream(self) -> AsyncIterator[dict[str, Any]]:
        """Stream cumulative message snapshots."""
        for step in self._steps:
            response = step.response
            if response:
                yield {
                    "role": "assistant",
                    "content": step.text,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments,
                            },
                        }
                        for tc in step.tool_calls
                    ]
                    if step.tool_calls
                    else None,
                }

    # -----------------------------------------------------------------------
    # Tool consumption
    # -----------------------------------------------------------------------

    def get_tool_calls(self) -> list[ParsedToolCall]:
        """Get all tool calls across all steps."""
        calls: list[ParsedToolCall] = []
        for step in self._steps:
            calls.extend(step.tool_calls)
        return calls

    async def get_tool_calls_stream(self) -> AsyncIterator[ParsedToolCall]:
        """Stream tool calls as they are parsed."""
        for step in self._steps:
            for tc in step.tool_calls:
                yield tc

    async def get_tool_stream(self) -> AsyncIterator[ToolStreamEvent]:
        """Stream tool-related events (deltas + preliminary results)."""
        consumer = self._broadcaster.create_consumer()
        async for event in extract_tool_stream_events(consumer):
            yield event

    # -----------------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------------

    def get_pending_tool_calls(self) -> list[ParsedToolCall]:
        """Get tool calls that are pending approval."""
        if self._state and self._state.pending_tool_calls:
            return list(self._state.pending_tool_calls)
        return []

    def get_state(self) -> ConversationState | None:
        """Get the current conversation state."""
        return self._state

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def steps(self) -> list[StepResult]:
        """All step results from the execution."""
        return self._steps
