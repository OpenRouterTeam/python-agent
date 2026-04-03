"""Conversation state management, approval workflow, and persistence."""

from __future__ import annotations

import inspect
import json
import time
import uuid
from typing import Any

from ._tool_executor import find_tool_by_name
from ._types import (
    ConversationState,
    ConversationStatus,
    ParsedToolCall,
    Tool,
    TurnContext,
    UnsentToolResult,
    is_manual_tool,
)


def generate_conversation_id() -> str:
    """Generate a unique conversation ID."""
    return str(uuid.uuid4())


def create_initial_state(
    conversation_id: str | None = None,
) -> ConversationState:
    """Create a fresh conversation state."""
    now = time.time()
    return ConversationState(
        id=conversation_id or generate_conversation_id(),
        messages=[],
        status=ConversationStatus.IN_PROGRESS,
        created_at=now,
        updated_at=now,
    )


def update_state(
    state: ConversationState,
    updates: dict[str, Any],
) -> ConversationState:
    """Create an updated copy of the conversation state."""
    return state.model_copy(update={**updates, "updated_at": time.time()})


def append_to_messages(
    current: list[dict[str, Any]],
    new_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append new message items to existing messages."""
    return list(current) + list(new_items)


async def tool_requires_approval(
    tool_call: ParsedToolCall,
    tools: list[Tool],
    context: TurnContext,
    call_level_check: Any | None = None,
) -> bool:
    """Check if a specific tool call requires approval."""
    if call_level_check is not None:
        result = call_level_check(tool_call, context)
        if inspect.isawaitable(result):
            result = await result
        if result:
            return True

    t = find_tool_by_name(tools, tool_call.name)
    if t is None:
        return False

    approval = t.function.require_approval
    if isinstance(approval, bool):
        return approval

    if callable(approval):
        result = approval(tool_call.arguments, context)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)

    return False


async def partition_tool_calls(
    tool_calls: list[ParsedToolCall],
    tools: list[Tool],
    context: TurnContext,
    call_level_check: Any | None = None,
) -> tuple[list[ParsedToolCall], list[ParsedToolCall]]:
    """Partition tool calls into those requiring approval and those that can auto-execute.

    Returns:
        (requires_approval, auto_execute) tuple.
    """
    requires_approval: list[ParsedToolCall] = []
    auto_execute: list[ParsedToolCall] = []

    for tc in tool_calls:
        t = find_tool_by_name(tools, tc.name)
        if t is not None and is_manual_tool(t):
            requires_approval.append(tc)
            continue

        needs_approval = await tool_requires_approval(
            tc, tools, context, call_level_check
        )
        if needs_approval:
            requires_approval.append(tc)
        else:
            auto_execute.append(tc)

    return requires_approval, auto_execute


def create_unsent_result(
    call_id: str,
    name: str,
    output: Any,
) -> UnsentToolResult:
    """Create an unsent tool result for deferred submission."""
    return UnsentToolResult(
        call_id=call_id,
        name=name,
        output=output,
    )


def create_rejected_result(
    call_id: str,
    name: str,
    reason: str | None = None,
) -> UnsentToolResult:
    """Create a rejected tool result."""
    return UnsentToolResult(
        call_id=call_id,
        name=name,
        output=None,
        error=reason or "Tool call rejected by user",
    )


def unsent_results_to_api_format(
    results: list[UnsentToolResult],
) -> list[dict[str, Any]]:
    """Convert unsent tool results to API input format."""
    items: list[dict[str, Any]] = []
    for result in results:
        if result.error:
            output = json.dumps({"error": result.error})
        elif result.output is not None and not isinstance(result.output, str):
            output = json.dumps(result.output)
        elif result.output is None:
            output = json.dumps(None)
        else:
            output = result.output

        items.append({
            "type": "function_call_output",
            "call_id": result.call_id,
            "output": output,
        })
    return items
