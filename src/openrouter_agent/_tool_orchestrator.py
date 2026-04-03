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
    APIError,
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

        response = await _call_api(client, api_request)
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


async def _call_api(client: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Call the OpenRouter API via the client."""
    try:
        response = await client.beta.responses.send_async(
            stream=False,
            **request,
        )
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if hasattr(response, "to_dict"):
            return response.to_dict()
        if isinstance(response, dict):
            return response
        return {"output": [], "id": getattr(response, "id", "")}
    except Exception as e:
        raise APIError(f"API call failed: {e}") from e
