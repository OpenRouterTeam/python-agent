"""call_model - primary entry point for the OpenRouter Agent SDK."""

from __future__ import annotations

from typing import Any

from ._model_result import ModelResult
from ._stop_conditions import step_count_is
from ._tool_context import ToolContextStore, resolve_context
from ._tool_event_broadcaster import ToolEventBroadcaster
from ._tool_orchestrator import DEFAULT_MAX_STEPS, run_tool_loop
from ._types import (
    OpenRouterClient,
    ResponseStreamEvent,
    StopCondition,
    Tool,
)


async def call_model(
    client: OpenRouterClient,
    request: dict[str, Any],
    options: dict[str, Any] | None = None,
) -> ModelResult:
    """Call a model with optional tools, stop conditions, and state management.

    Args:
        client: An OpenRouter client instance.
        request: A dict containing the request parameters. Supports all
            OpenRouter Responses API parameters plus SDK-specific fields:
            - tools: List of Tool definitions
            - stop_when: StopCondition or list of StopConditions
            - context: Initial tool context data
            - require_approval: Function to check if a tool call needs approval
            - on_turn_start: Callback invoked at the start of each turn
            - on_turn_end: Callback invoked at the end of each turn
            - state: StateAccessor for conversation persistence
            - approve_tool_calls: List of tool call IDs to approve
            - reject_tool_calls: List of tool call IDs to reject
        options: Additional request options (headers, timeout, etc.)

    Returns:
        A ModelResult that can be consumed in multiple ways concurrently.
    """
    # Extract SDK-specific fields
    tools: list[Tool] = request.get("tools", [])
    stop_when = request.get("stop_when")
    context_input = request.get("context")
    require_approval = request.get("require_approval")
    on_turn_start = request.get("on_turn_start")
    on_turn_end = request.get("on_turn_end")

    # Normalize stop conditions
    stop_conditions: list[StopCondition] = []
    if stop_when is not None:
        if isinstance(stop_when, list):
            stop_conditions = list(stop_when)
        else:
            stop_conditions = [stop_when]

    # Always add a default max steps guard
    if not stop_conditions:
        stop_conditions.append(step_count_is(DEFAULT_MAX_STEPS))

    # Initialize context store
    initial_contexts: dict[str, dict[str, Any]] | None = None
    if context_input is not None:
        from ._turn_context import build_turn_context
        tc = build_turn_context()
        initial_contexts = await resolve_context(context_input, tc)

    context_store = ToolContextStore(initial_contexts)

    # Create broadcaster for streaming events
    broadcaster: ToolEventBroadcaster[ResponseStreamEvent] = ToolEventBroadcaster()

    # Add custom header
    if options is None:
        options = {}
    headers = options.get("headers", {})
    headers["x-openrouter-callmodel"] = "true"
    options["headers"] = headers

    # Run the tool loop
    steps = await run_tool_loop(
        client=client,
        request_params=request,
        tools=tools,
        stop_conditions=stop_conditions,
        context_store=context_store,
        broadcaster=broadcaster,
        on_turn_start=on_turn_start,
        on_turn_end=on_turn_end,
        require_approval=require_approval,
    )

    return ModelResult(
        steps=steps,
        broadcaster=broadcaster,
    )
