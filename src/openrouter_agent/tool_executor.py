from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ._utils import is_content_array, json_dumps, json_loads_maybe, maybe_await, schema_to_json_schema, validate_schema
from .tool_context import ToolContextStore, build_tool_execute_context
from .tool_types import (
    ParsedToolCall,
    Tool,
    get_tool_function,
    is_client_tool,
    is_generator_tool,
    is_hitl_tool,
    is_mcp_tool,
    is_server_tool,
)


def convert_zod_to_json_schema(schema: Any) -> Dict[str, Any]:
    return schema_to_json_schema(schema)


def sanitize_json_schema(schema: Any) -> Any:
    from ._utils import sanitize_json_schema as _sanitize

    return _sanitize(schema)


def _try_validate(schema: Any, value: Any) -> bool:
    if schema is None:
        return False
    try:
        validate_schema(schema, value)
    except Exception:
        return False
    return True


def convert_tools_to_api_format(tools: Sequence[Tool]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for item in tools:
        if is_server_tool(item):
            config = item.get("config", {})
            converted.append(dict(config) if isinstance(config, Mapping) else {})
            continue
        fn = get_tool_function(item)
        wire_schema = fn.get("wire_input_schema")
        api: Dict[str, Any] = {
            "type": "function",
            "name": fn.get("name"),
            "parameters": sanitize_json_schema(dict(wire_schema))
            if wire_schema is not None
            else schema_to_json_schema(fn.get("input_schema")),
        }
        if fn.get("description") is not None:
            api["description"] = fn["description"]
        if fn.get("strict") is not None:
            api["strict"] = fn["strict"]
        converted.append(api)
    return converted


async def execute_regular_tool(
    tool: Tool, tool_call: ParsedToolCall, context: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    fn = get_tool_function(tool)
    source = "mcp" if is_mcp_tool(tool) else "client"
    try:
        args = validate_schema(fn.get("input_schema"), tool_call.arguments)
        result = await maybe_await(fn["execute"](args, _with_tool_call(context, tool_call)))
        if fn.get("output_schema") is not None:
            result = validate_schema(fn.get("output_schema"), result)
        return {"tool_call_id": tool_call.id, "tool_name": tool_call.name, "source": source, "result": result}
    except Exception as exc:
        return {
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "source": source,
            "result": None,
            "error": exc,
        }


async def execute_generator_tool(
    tool: Tool,
    tool_call: ParsedToolCall,
    context: Optional[Mapping[str, Any]] = None,
    on_preliminary_result: Optional[Callable[[str, Any], Any]] = None,
) -> Dict[str, Any]:
    fn = get_tool_function(tool)
    source = "mcp" if is_mcp_tool(tool) else "client"
    # Upstream (DEV-817) stopped accumulating every yield into a per-call
    # `preliminary_results` array: yields are broadcast live through
    # `on_preliminary_result` only, and the terminal result no longer carries
    # the yield history.
    try:
        args = validate_schema(fn.get("input_schema"), tool_call.arguments)
        produced = fn["execute"](args, _with_tool_call(context, tool_call))
        if not hasattr(produced, "__aiter__"):
            produced = await maybe_await(produced)
        final = None
        has_final = False
        last_value = None
        has_value = False
        broad_overlapping_schemas = fn.get("event_schema") is dict and fn.get("output_schema") is dict
        pending_broad_event = None
        async for value in produced:
            if broad_overlapping_schemas:
                if pending_broad_event is not None:
                    if on_preliminary_result is not None:
                        await maybe_await(on_preliminary_result(tool_call.id, pending_broad_event))
                pending_broad_event = value
                has_value = True
                last_value = value
                continue
            has_value = True
            last_value = value
            matches_output = _try_validate(fn.get("output_schema"), value)
            matches_event = _try_validate(fn.get("event_schema"), value)
            if matches_output and (not matches_event or fn.get("event_schema") is dict) and not has_final:
                final = validate_schema(fn.get("output_schema"), value)
                has_final = True
                continue
            if fn.get("event_schema") is not None:
                value = validate_schema(fn.get("event_schema"), value)
            if on_preliminary_result is not None:
                await maybe_await(on_preliminary_result(tool_call.id, value))
        if broad_overlapping_schemas and pending_broad_event is not None:
            final = validate_schema(fn.get("output_schema"), pending_broad_event)
            has_final = True
        if not has_final:
            if not has_value:
                raise ValueError(f'Generator tool "{tool_call.name}" completed without yielding a final output')
            final = validate_schema(fn.get("output_schema"), last_value)
        return {
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "source": source,
            "result": final,
        }
    except Exception as exc:
        return {
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "source": source,
            "result": None,
            "error": exc,
        }


async def execute_hitl_tool(
    tool: Tool, tool_call: ParsedToolCall, context: Optional[Mapping[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    fn = get_tool_function(tool)
    source = "mcp" if is_mcp_tool(tool) else "client"
    try:
        args = validate_schema(fn.get("input_schema"), tool_call.arguments)
        result = await maybe_await(fn["on_tool_called"](args, _with_tool_call(context, tool_call)))
        if result is None:
            return None
        result = validate_schema(fn.get("output_schema"), result)
        return {"tool_call_id": tool_call.id, "tool_name": tool_call.name, "source": source, "result": result}
    except Exception as exc:
        return {
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "source": source,
            "result": None,
            "error": exc,
        }


def _with_tool_call(context: Optional[Mapping[str, Any]], tool_call: ParsedToolCall) -> Dict[str, Any]:
    """Thread the executed call into the execute context. `tool_call` is part
    of the tool-facing contract: a caller-provided one wins; otherwise the
    executed call fills the gap as a wire-shaped function_call item. Every
    executor entry point applies this (upstream `buildExecuteCtx`)."""
    merged = dict(context or {})
    if not merged.get("tool_call"):
        merged["tool_call"] = to_function_call_item(tool_call)
    return merged


def to_function_call_item(tool_call: ParsedToolCall) -> Dict[str, Any]:
    """Convert an executor-shaped `ParsedToolCall` back into the wire-shaped
    function_call item the execute context declares. Arguments are always
    re-serialized (semantically equal to the wire arguments, not guaranteed
    byte-identical)."""
    arguments = tool_call.arguments
    return {
        "type": "function_call",
        "id": tool_call.id,
        "callId": tool_call.id,
        "name": tool_call.name,
        "arguments": json_dumps(arguments) if arguments is not None else "{}",
    }


async def execute_tool(
    tool: Tool,
    tool_call: ParsedToolCall,
    turn_context: Optional[Mapping[str, Any]] = None,
    on_preliminary_result: Optional[Callable[[str, Any], Any]] = None,
    context_store: Optional[ToolContextStore] = None,
    shared_context_schema: Any = None,
) -> Optional[Dict[str, Any]]:
    if not is_client_tool(tool):
        return None
    context = build_tool_execute_context(tool, turn_context, context_store, shared_context_schema)
    if is_hitl_tool(tool):
        return await execute_hitl_tool(tool, tool_call, context)
    if is_generator_tool(tool):
        return await execute_generator_tool(tool, tool_call, context, on_preliminary_result)
    if callable(get_tool_function(tool).get("execute")):
        return await execute_regular_tool(tool, tool_call, context)
    return None


async def apply_on_response_received_hooks(
    input_items: Any,
    tools: Optional[Sequence[Tool]],
    turn_context: Optional[Mapping[str, Any]] = None,
    context_store: Optional[ToolContextStore] = None,
    shared_context_schema: Any = None,
) -> Any:
    if not tools:
        return input_items
    items = input_items if isinstance(input_items, list) else [input_items]
    call_names: Dict[str, str] = {}
    for item in items:
        if isinstance(item, Mapping) and item.get("type") == "function_call":
            call_names[str(item.get("callId") or item.get("call_id") or item.get("id"))] = str(item.get("name", ""))
    rewritten: List[Any] = []
    changed = False
    for item in items:
        if not (isinstance(item, Mapping) and item.get("type") == "function_call_output"):
            rewritten.append(item)
            continue
        call_id = str(item.get("callId") or item.get("call_id") or "")
        name = call_names.get(call_id)
        tool = next(
            (
                candidate
                for candidate in tools
                if is_client_tool(candidate) and get_tool_function(candidate).get("name") == name
            ),
            None,
        )
        if tool is None:
            rewritten.append(item)
            continue
        hook = get_tool_function(tool).get("on_response_received")
        ctx = build_tool_execute_context(tool, turn_context, context_store, shared_context_schema)
        new_item = dict(item)
        raw = json_loads_maybe(new_item.get("output"))
        if not hook:
            try:
                validate_schema(get_tool_function(tool).get("output_schema"), raw)
                rewritten.append(item)
                continue
            except Exception as exc:
                new_item["output"] = json_dumps({"error": str(exc), "originalOutput": raw})
                changed = True
                rewritten.append(new_item)
                continue
        try:
            transformed = await maybe_await(hook(raw, ctx))
            transformed = validate_schema(get_tool_function(tool).get("output_schema"), transformed)
            new_item["output"] = transformed if is_content_array(transformed) else json_dumps(transformed)
        except Exception as exc:
            new_item["output"] = json_dumps({"error": str(exc), "originalOutput": raw})
        changed = True
        rewritten.append(new_item)
    if not isinstance(input_items, list):
        return rewritten[0]
    return rewritten if changed else input_items
