"""Multi-step tool execution orchestration loop."""

from __future__ import annotations

import json
import time
from typing import Any

from ._async_params import CLIENT_ONLY_FIELDS, resolve_async_functions
from ._next_turn_params import (
    apply_next_turn_params_to_request,
    execute_next_turn_params_functions,
)
from ._stop_conditions import is_stop_condition_met
from ._tool_context import ToolContextStore
from ._tool_event_broadcaster import ToolEventBroadcaster
from ._tool_executor import execute_tool, tool_to_api_format
from ._turn_context import build_turn_context
from ._types import (
    FunctionCallItem,
    ParsedToolCall,
    ResponseStreamEvent,
    StepResult,
    StopCondition,
    Tool,
    ToolCallOutputEvent,
    ToolExecutionResult,
    ToolPreliminaryResultEvent,
    ToolResultEvent,
    TurnEndEvent,
    TurnStartEvent,
    Usage,
)

DEFAULT_MAX_STEPS = 5


def _build_tool_index(tools: list[Tool]) -> dict[str, Tool]:
    """Build a name -> tool lookup dict for O(1) access."""
    return {t.function.name: t for t in tools}


def _serialize_tool_output(result: ToolExecutionResult) -> str:
    """Serialize a tool execution result to a string for API consumption."""
    if result.error:
        return json.dumps({"error": result.error})
    if result.result is not None:
        if isinstance(result.result, str):
            return result.result
        return json.dumps(result.result)
    return json.dumps(None)


def _extract_tool_calls_from_response(response: dict[str, Any]) -> list[ParsedToolCall]:
    """Extract parsed tool calls from an API response."""
    tool_calls: list[ParsedToolCall] = []
    output = response.get("output", [])
    if not isinstance(output, list):
        return tool_calls

    for item in output:
        if isinstance(item, dict) and item.get("type") == "function_call":
            try:
                args = item.get("arguments", "{}")
                if isinstance(args, str):
                    parsed_args = json.loads(args)
                else:
                    parsed_args = args
                tool_calls.append(ParsedToolCall(
                    id=item.get("id", item.get("call_id", "")),
                    name=item.get("name", ""),
                    arguments=parsed_args,
                ))
            except json.JSONDecodeError:
                tool_calls.append(ParsedToolCall(
                    id=item.get("id", item.get("call_id", "")),
                    name=item.get("name", ""),
                    arguments={},
                ))
    return tool_calls


def _extract_text_from_response(response: dict[str, Any]) -> str:
    """Extract text content from an API response (single pass)."""
    output = response.get("output", [])
    if not isinstance(output, list):
        return ""

    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            content = item.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        texts.append(block.get("text", ""))
            elif isinstance(content, str):
                texts.append(content)
        elif item_type == "output_text":
            texts.append(item.get("text", ""))
    return "".join(texts)


def _extract_usage_from_response(response: dict[str, Any]) -> Usage | None:
    """Extract usage info from an API response."""
    usage = response.get("usage")
    if isinstance(usage, dict):
        return Usage(
            prompt_tokens=usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            completion_tokens=usage.get("output_tokens", usage.get("completion_tokens", 0)),
            total_tokens=usage.get("total_tokens", 0),
        )
    return None


def _build_api_request(
    resolved_params: dict[str, Any],
    tools: list[Tool],
) -> dict[str, Any]:
    """Build the API request dict from resolved params and tools."""
    request: dict[str, Any] = {}
    for key, value in resolved_params.items():
        if key not in CLIENT_ONLY_FIELDS:
            request[key] = value

    if tools:
        request["tools"] = [tool_to_api_format(t).model_dump(exclude_none=True) for t in tools]

    return request


def _tool_results_to_input_items(results: list[ToolExecutionResult]) -> list[dict[str, Any]]:
    """Convert tool execution results to API input items."""
    return [
        {
            "type": "function_call_output",
            "call_id": result.tool_call_id,
            "output": _serialize_tool_output(result),
        }
        for result in results
    ]


async def run_tool_loop(
    client: Any,
    request_params: dict[str, Any],
    tools: list[Tool],
    stop_conditions: list[StopCondition],
    context_store: ToolContextStore,
    broadcaster: ToolEventBroadcaster[ResponseStreamEvent],
    max_steps: int = DEFAULT_MAX_STEPS,
    on_turn_start: Any | None = None,
    on_turn_end: Any | None = None,
    require_approval: Any | None = None,
    stream: bool = True,
) -> list[StepResult]:
    """Run the multi-step tool execution loop."""
    steps: list[StepResult] = []
    current_request = dict(request_params)
    tool_index = _build_tool_index(tools)

    def _on_preliminary(call_id: str, data: Any, ts: float) -> None:
        broadcaster.push(ToolPreliminaryResultEvent(
            type="tool.preliminary_result",
            tool_call_id=call_id,
            result=data,
            timestamp=ts,
        ))

    for step_num in range(max_steps):
        turn_context = build_turn_context(number_of_turns=step_num)

        resolved = await resolve_async_functions(current_request, turn_context)
        api_request = _build_api_request(resolved, tools)

        broadcaster.push(TurnStartEvent(
            type="turn.start",
            turn_number=step_num,
            timestamp=time.time(),
        ))

        if on_turn_start:
            await on_turn_start(turn_context)

        response = await _call_api(
            client, api_request, stream=stream, broadcaster=broadcaster
        )
        if not stream:
            # In non-streaming mode the full response was not pushed yet.
            broadcaster.push(response)

        broadcaster.push(TurnEndEvent(
            type="turn.end",
            turn_number=step_num,
            timestamp=time.time(),
        ))

        if on_turn_end:
            await on_turn_end(turn_context, response)

        text = _extract_text_from_response(response)
        tool_calls = _extract_tool_calls_from_response(response)
        usage = _extract_usage_from_response(response)
        step_type = "initial" if step_num == 0 else "continue"

        if not tool_calls:
            step = StepResult(
                step_type=step_type,
                text=text,
                tool_calls=[],
                tool_results=[],
                response=response,
                usage=usage,
                finish_reason=response.get("finish_reason", "stop"),
            )
            steps.append(step)
            break

        tool_results: list[ToolExecutionResult] = []
        for tc in tool_calls:
            t = tool_index.get(tc.name)
            if t is None:
                tool_results.append(ToolExecutionResult(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    error=f"Tool '{tc.name}' not found",
                ))
                continue

            tool_turn_context = build_turn_context(
                number_of_turns=step_num,
                tool_call=FunctionCallItem(
                    id=tc.id, name=tc.name, arguments=json.dumps(tc.arguments)
                ),
                turn_request=api_request,
            )

            result = await execute_tool(
                t, tc, tool_turn_context, context_store, _on_preliminary
            )
            tool_results.append(result)

            broadcaster.push(ToolResultEvent(
                type="tool.result",
                tool_call_id=result.tool_call_id,
                result=result.result,
                timestamp=time.time(),
                preliminary_results=result.preliminary_results,
            ))

            broadcaster.push(ToolCallOutputEvent(
                type="tool.call_output",
                output={
                    "type": "function_call_output",
                    "call_id": result.tool_call_id,
                    "output": _serialize_tool_output(result),
                },
                timestamp=time.time(),
            ))

        step = StepResult(
            step_type=step_type,
            text=text,
            tool_calls=tool_calls,
            tool_results=tool_results,
            response=response,
            usage=usage,
            finish_reason=response.get("finish_reason"),
        )
        steps.append(step)

        if await is_stop_condition_met(stop_conditions, steps):
            break

        next_params = await execute_next_turn_params_functions(
            tool_calls, tools, current_request
        )

        tool_result_items = _tool_results_to_input_items(tool_results)
        current_input = current_request.get("input", [])
        if isinstance(current_input, str):
            current_input = [{"role": "user", "content": current_input}]
        else:
            current_input = list(current_input)  # avoid mutating caller's list

        output_items = response.get("output", [])
        if isinstance(output_items, list):
            current_input.extend(output_items)
        current_input.extend(tool_result_items)

        current_request["input"] = current_input
        current_request = apply_next_turn_params_to_request(
            current_request, next_params
        )

    broadcaster.complete()
    return steps


async def _call_api(
    client: Any,
    request: dict[str, Any],
    *,
    stream: bool = True,
    broadcaster: ToolEventBroadcaster[ResponseStreamEvent] | None = None,
) -> dict[str, Any]:
    """Call the OpenRouter API via the client.

    When *stream* is ``True`` (the default), the response is consumed as SSE
    events.  Each event is pushed to *broadcaster* so downstream consumers
    receive real-time deltas.  The final accumulated response dict is returned.

    When *stream* is ``False``, the API is called in non-streaming mode and the
    response dict is returned directly (legacy behaviour).
    """
    try:
        if not stream:
            response = await client.beta.responses.send_async(
                stream=False,
                **request,
            )
            return _response_to_dict(response)

        # --- streaming path ---
        sse_stream = await client.beta.responses.send_async(
            stream=True,
            **request,
        )
        return await _accumulate_stream(sse_stream, broadcaster)
    except Exception as e:
        raise RuntimeError(f"API call failed: {e}") from e


def _response_to_dict(response: Any) -> dict[str, Any]:
    """Normalize a non-streaming response into a plain dict."""
    if hasattr(response, "model_dump"):
        result: dict[str, Any] = response.model_dump()
        return result
    if hasattr(response, "to_dict"):
        result2: dict[str, Any] = response.to_dict()
        return result2
    if isinstance(response, dict):
        return response
    return {"output": [], "id": getattr(response, "id", "")}


async def _accumulate_stream(
    sse_stream: Any,
    broadcaster: ToolEventBroadcaster[ResponseStreamEvent] | None,
) -> dict[str, Any]:
    """Iterate over SSE events, push each to *broadcaster*, and accumulate
    the final response dict that ``run_tool_loop`` expects.
    """
    accumulated: dict[str, Any] = {"output": [], "id": ""}
    output_items: dict[int, dict[str, Any]] = {}  # index -> partial item

    async for event in sse_stream:
        # Normalise event to a dict
        if hasattr(event, "model_dump"):
            event_dict: dict[str, Any] = event.model_dump()
        elif hasattr(event, "to_dict"):
            event_dict = event.to_dict()
        elif isinstance(event, dict):
            event_dict = event
        else:
            event_dict = {"type": getattr(event, "type", "unknown")}

        # Broadcast every SSE event to consumers
        if broadcaster is not None:
            broadcaster.push(event_dict)

        event_type = event_dict.get("type", "")

        # Accumulate response-level metadata
        if event_type == "response.created":
            response_data = event_dict.get("response", {})
            if isinstance(response_data, dict):
                accumulated["id"] = response_data.get("id", accumulated["id"])

        elif event_type == "response.completed":
            response_data = event_dict.get("response", {})
            if isinstance(response_data, dict):
                # The completed event carries the full response; prefer it.
                return response_data

        # Accumulate output items
        elif event_type == "response.output_item.added":
            idx = event_dict.get("output_index", len(output_items))
            item = event_dict.get("item", {})
            if isinstance(item, dict):
                output_items[idx] = dict(item)

        elif event_type == "response.output_item.done":
            idx = event_dict.get("output_index", -1)
            item = event_dict.get("item", {})
            if isinstance(item, dict) and idx >= 0:
                output_items[idx] = dict(item)

        # Accumulate text deltas
        elif event_type == "response.output_text.delta":
            idx = event_dict.get("output_index", 0)
            _ensure_text_item(output_items, idx)
            delta = event_dict.get("delta", "")
            _append_text_delta(output_items[idx], delta)

        # Accumulate function-call argument deltas
        elif event_type == "response.function_call_arguments.delta":
            idx = event_dict.get("output_index", 0)
            item = output_items.setdefault(idx, {"type": "function_call", "arguments": ""})
            item["arguments"] = item.get("arguments", "") + event_dict.get("delta", "")

        # Usage
        elif event_type == "response.usage":
            accumulated["usage"] = event_dict.get("usage", {})

    # Fallback: assemble from accumulated items
    accumulated["output"] = [output_items[k] for k in sorted(output_items)]
    return accumulated


def _ensure_text_item(items: dict[int, dict[str, Any]], idx: int) -> None:
    """Make sure *items[idx]* is a text-bearing output item."""
    if idx not in items:
        items[idx] = {
            "type": "message",
            "content": [{"type": "output_text", "text": ""}],
        }


def _append_text_delta(item: dict[str, Any], delta: str) -> None:
    """Append *delta* to the text content of *item*."""
    content = item.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "output_text":
                block["text"] = block.get("text", "") + delta
                return
        item["content"].append({"type": "output_text", "text": delta})
    elif isinstance(content, str):
        item["content"] = content + delta
    else:
        item["content"] = [{"type": "output_text", "text": delta}]
