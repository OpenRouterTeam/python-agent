"""Tool execution logic with Pydantic validation and JSON Schema conversion."""

from __future__ import annotations

import inspect
import json
import logging
import time
import traceback
from typing import Any

from pydantic import BaseModel, ValidationError

from ._result import Err, Ok, Result
from ._tool_context import ToolContextStore, build_tool_execute_context
from ._types import (
    APITool,
    ParsedToolCall,
    Tool,
    ToolExecutionResult,
    ToolWithExecute,
    ToolWithGenerator,
    TurnContext,
    is_generator_tool,
    is_manual_tool,
    is_regular_execute_tool,
)

logger = logging.getLogger(__name__)


def tool_to_api_format(t: Tool) -> APITool:
    """Convert a Tool to the wire format for the OpenRouter API."""
    json_schema = t.function.input_schema.model_json_schema()
    return APITool(
        type="function",
        name=t.function.name,
        description=t.function.description,
        strict=None,
        parameters=json_schema,
    )


def parse_tool_call_arguments(
    raw_arguments: str, input_schema: type[BaseModel]
) -> Result[BaseModel, str]:
    """Parse and validate tool call arguments against the input schema."""
    try:
        args_dict = json.loads(raw_arguments)
    except json.JSONDecodeError as e:
        return Err(f"Invalid JSON in tool arguments: {e}")

    try:
        validated = input_schema.model_validate(args_dict)
        return Ok(validated)
    except ValidationError as e:
        return Err(f"Validation error: {e}")


def _validate_input(
    t: Tool, tool_call: ParsedToolCall
) -> ToolExecutionResult | BaseModel:
    """Validate tool call arguments. Returns parsed model or error result."""
    try:
        return t.function.input_schema.model_validate(tool_call.arguments)
    except ValidationError as e:
        return ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            error=f"Input validation error: {e}",
        )


def _to_result_dict(value: Any) -> Any:
    """Convert a value to a JSON-serializable dict if it's a BaseModel."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    return value


async def execute_regular_tool(
    t: ToolWithExecute,
    tool_call: ParsedToolCall,
    turn_context: TurnContext,
    store: ToolContextStore,
) -> ToolExecutionResult:
    """Execute a regular (non-generator) tool."""
    parsed = _validate_input(t, tool_call)
    if isinstance(parsed, ToolExecutionResult):
        return parsed

    context = build_tool_execute_context(turn_context, store, t.function.name)

    try:
        result = t.function.execute(parsed, context)
        if inspect.isawaitable(result):
            result = await result

        if t.function.output_schema is not None:
            try:
                t.function.output_schema.model_validate(
                    result if isinstance(result, dict) else result
                )
            except ValidationError as e:
                logger.warning("Output validation failed for tool '%s': %s", t.function.name, e)

        return ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=_to_result_dict(result),
        )
    except Exception as e:
        return ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )


async def execute_generator_tool(
    t: ToolWithGenerator,
    tool_call: ParsedToolCall,
    turn_context: TurnContext,
    store: ToolContextStore,
    on_preliminary_result: Any | None = None,
) -> ToolExecutionResult:
    """Execute a generator tool, collecting preliminary results."""
    parsed = _validate_input(t, tool_call)
    if isinstance(parsed, ToolExecutionResult):
        return parsed

    context = build_tool_execute_context(turn_context, store, t.function.name)
    preliminary_results: list[Any] = []
    final_result: Any = None

    def _emit_event(value: Any) -> None:
        event_data = _to_result_dict(value)
        preliminary_results.append(event_data)
        if on_preliminary_result:
            on_preliminary_result(tool_call.id, event_data, time.time())

    try:
        gen = t.function.execute(parsed, context)
        async for value in gen:
            if isinstance(value, t.function.output_schema):
                final_result = value
            elif isinstance(value, t.function.event_schema):
                _emit_event(value)
            else:
                # Fallback: try output validation for untyped values (e.g. dicts)
                try:
                    val_dict = _to_result_dict(value)
                    t.function.output_schema.model_validate(
                        val_dict if isinstance(val_dict, dict) else val_dict
                    )
                    final_result = value
                    logger.warning(
                        "Generator yielded ambiguous value for tool '%s' -- treated as output",
                        t.function.name,
                    )
                except ValidationError:
                    _emit_event(value)

        return ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=_to_result_dict(final_result),
            preliminary_results=preliminary_results if preliminary_results else None,
        )
    except Exception as e:
        return ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )


async def execute_tool(
    t: Tool,
    tool_call: ParsedToolCall,
    turn_context: TurnContext,
    store: ToolContextStore,
    on_preliminary_result: Any | None = None,
) -> ToolExecutionResult:
    """Execute any tool type, dispatching to the appropriate executor."""
    if is_manual_tool(t):
        return ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            error="Manual tool cannot be auto-executed",
        )

    if is_generator_tool(t):
        assert isinstance(t, ToolWithGenerator)
        return await execute_generator_tool(
            t, tool_call, turn_context, store, on_preliminary_result
        )

    if is_regular_execute_tool(t):
        assert isinstance(t, ToolWithExecute)
        return await execute_regular_tool(t, tool_call, turn_context, store)

    return ToolExecutionResult(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        error=f"Unknown tool type for tool '{tool_call.name}'",
    )


def find_tool_by_name(tools: list[Tool], name: str) -> Tool | None:
    """Find a tool by name in a list of tools."""
    for t in tools:
        if t.function.name == name:
            return t
    return None
