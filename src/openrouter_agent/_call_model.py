"""call_model - primary entry point for the OpenRouter Agent SDK."""

from __future__ import annotations

from typing import Any

from ._conversation_state import (
    create_initial_state,
    create_rejected_result,
    unsent_results_to_api_format,
    update_state,
)
from ._model_result import ModelResult
from ._stop_conditions import step_count_is
from ._tool_context import ToolContextStore, resolve_context
from ._tool_event_broadcaster import ToolEventBroadcaster
from ._tool_orchestrator import DEFAULT_MAX_STEPS, run_tool_loop
from ._turn_context import normalize_input_to_array
from ._types import (
    ConversationState,
    ConversationStatus,
    ResponseStreamEvent,
    StateAccessor,
    StopCondition,
    Tool,
)


async def call_model(
    client: Any,
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
    state: StateAccessor | None = request.get("state")
    approve_tool_call_ids: set[str] = set(request.get("approve_tool_calls", []))
    reject_tool_call_ids: set[str] = set(request.get("reject_tool_calls", []))

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

    # Load conversation state if a StateAccessor is provided
    loaded_state: ConversationState | None = None
    if state is not None:
        loaded_state = await state.load()
        if loaded_state is None:
            loaded_state = create_initial_state()

    # Handle approval/rejection of pending tool calls from a previous turn
    if (
        loaded_state is not None
        and loaded_state.pending_tool_calls
        and (approve_tool_call_ids or reject_tool_call_ids)
    ):
        unsent_items: list[dict[str, Any]] = []
        remaining_pending = []

        for tc in loaded_state.pending_tool_calls:
            if tc.id in reject_tool_call_ids:
                rejected = create_rejected_result(tc.id, tc.name)
                unsent_items.extend(
                    unsent_results_to_api_format([rejected])
                )
            elif tc.id in approve_tool_call_ids:
                pass  # Approved calls proceed to the tool loop for execution
            else:
                remaining_pending.append(tc)

        # Append rejected-result items to the request input so the model
        # sees the rejection before continuing.
        if unsent_items:
            current_input = normalize_input_to_array(request.get("input", []))
            current_input.extend(unsent_items)
            request = {**request, "input": current_input}

        # Update loaded state: clear pending if all handled
        loaded_state = update_state(loaded_state, {
            "pending_tool_calls": remaining_pending or None,
            "status": ConversationStatus.IN_PROGRESS,
        })

    # Initialize context store
    initial_contexts: dict[str, dict[str, Any]] | None = None
    if context_input is not None:
        from ._turn_context import build_turn_context
        turn_ctx = build_turn_context()
        initial_contexts = await resolve_context(context_input, turn_ctx)

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
    steps, final_state = await run_tool_loop(
        client=client,
        request_params=request,
        tools=tools,
        stop_conditions=stop_conditions,
        context_store=context_store,
        broadcaster=broadcaster,
        on_turn_start=on_turn_start,
        on_turn_end=on_turn_end,
        require_approval=require_approval,
        state=loaded_state,
    )

    # Save conversation state if a StateAccessor is provided
    if state is not None and final_state is not None:
        await state.save(final_state)

    return ModelResult(
        steps=steps,
        broadcaster=broadcaster,
        state=final_state,
    )
