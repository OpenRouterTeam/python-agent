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


def to_function_call_item(tool_call: ParsedToolCall) -> Dict[str, Any]:
    """Convert an executor-shaped `ParsedToolCall` (whose `id` is the wire
    `call_id` and whose `arguments` are already parsed) back into the
    wire-shaped `function_call` item. `arguments` is always re-serialized, so
    it is semantically equal to the wire arguments but not guaranteed
    byte-identical."""
    try:
        arguments = json_dumps(tool_call.arguments)
    except (TypeError, ValueError):
        arguments = "{}"
    return {
        "type": "function_call",
        "id": tool_call.id,
        "callId": tool_call.id,
        "name": tool_call.name,
        "arguments": arguments,
    }


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
        # A caller-owned (manual tool) or server-supplied (MCP) JSON Schema is
        # sent verbatim -- sanitized of `~`-prefixed keys, as a copy -- instead
        # of one regenerated from `input_schema`.
        if wire_schema is not None:
            parameters = sanitize_json_schema(wire_schema)
        else:
            parameters = schema_to_json_schema(fn.get("input_schema"))
        api: Dict[str, Any] = {
            "type": "function",
            "name": fn.get("name"),
            # Passed through instead of hardcoded: `None` (null) leaves the
            # provider default, `True`/`False` are sent as declared.
            "strict": fn.get("strict"),
            "parameters": parameters,
        }
        if fn.get("description") is not None:
            api["description"] = fn["description"]
        converted.append(api)
    return converted


async def execute_regular_tool(
    tool: Tool, tool_call: ParsedToolCall, context: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    fn = get_tool_function(tool)
    source = "mcp" if is_mcp_tool(tool) else "client"
    try:
        args = validate_schema(fn.get("input_schema"), tool_call.arguments)
        result = await maybe_await(fn["execute"](args, context))
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
    # Yields are broadcast live via `on_preliminary_result`; they are not
    # accumulated onto the result (upstream 1.0.0 dropped
    # `preliminaryResults` to stop per-call memory growth).
    try:
        args = validate_schema(fn.get("input_schema"), tool_call.arguments)
        produced = fn["execute"](args, context)
        if not hasattr(produced, "__aiter__"):
            produced = await maybe_await(produced)
        final = None
        has_final = False
        last_value = None
        has_value = False
        broad_event_schema = fn.get("event_schema") is dict
        pending_output = None
        has_pending_output = False

        async def emit_preliminary(event: Any) -> None:
            if on_preliminary_result is not None:
                await maybe_await(on_preliminary_result(tool_call.id, event))

        async for value in produced:
            has_value = True
            last_value = value
            if broad_event_schema:
                # A dict event schema accepts every yield, so only the last yield can be the output.
                if has_pending_output:
                    await emit_preliminary(pending_output)
                    has_pending_output = False
                if _try_validate(fn.get("output_schema"), value):
                    pending_output = value
                    has_pending_output = True
                else:
                    await emit_preliminary(value)
                continue
            matches_output = _try_validate(fn.get("output_schema"), value)
            matches_event = _try_validate(fn.get("event_schema"), value)
            if matches_output and not matches_event and not has_final:
                final = validate_schema(fn.get("output_schema"), value)
                has_final = True
                continue
            if fn.get("event_schema") is not None:
                value = validate_schema(fn.get("event_schema"), value)
            await emit_preliminary(value)
        if has_pending_output:
            final = validate_schema(fn.get("output_schema"), pending_output)
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
        result = await maybe_await(fn["on_tool_called"](args, context))
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
    # `tool_call` is part of the tool-facing context contract. A caller-provided
    # one wins; otherwise thread the executed call, converted back to the
    # wire-shaped `function_call` item, so every execution path carries it.
    if turn_context is None or turn_context.get("tool_call") is None:
        turn_context = {**dict(turn_context or {}), "tool_call": to_function_call_item(tool_call)}
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
