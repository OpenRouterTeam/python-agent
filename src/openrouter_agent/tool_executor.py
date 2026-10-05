from __future__ import annotations

import asyncio
import copy
import inspect
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence

from ._utils import (
    dump,
    is_content_array,
    json_dumps,
    json_loads_maybe,
    maybe_await,
    schema_to_json_schema,
    validate_schema,
)
from .tool_context import ToolContextStore, build_tool_execute_context, build_tool_run_context
from .tool_types import (
    ParsedToolCall,
    Tool,
    get_tool_function,
    is_client_tool,
    is_deferred_handle,
    is_generator_tool,
    is_hitl_tool,
    is_manual_tool,
    is_mcp_tool,
    is_server_tool,
    is_unified_tool,
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
    """Convert tools to Responses API format. Server tools pass their config
    through untouched; client tools are packaged into the function-call
    shape. ``strict`` is passed through (``None`` when undeclared) and a
    manual tool's caller-owned ``wire_input_schema`` replaces the
    schema-derived parameters (sanitized copy; the caller's dict is never
    mutated). Wire definitions are never augmented per tool -- the single
    universal ``task`` tool is appended by `call_model` when warranted."""
    converted: List[Dict[str, Any]] = []
    for item in tools:
        if is_server_tool(item):
            config = item.get("config", {})
            converted.append(dict(config) if isinstance(config, Mapping) else {})
            continue
        fn = get_tool_function(item)
        wire_schema = fn.get("wire_input_schema")
        if is_manual_tool(item) and wire_schema is not None:
            parameters = sanitize_json_schema(copy.deepcopy(wire_schema))
        else:
            parameters = schema_to_json_schema(fn.get("input_schema"))
        api: Dict[str, Any] = {
            "type": "function",
            "name": fn.get("name"),
            "parameters": parameters,
            "strict": fn.get("strict"),
        }
        if fn.get("description") is not None:
            api["description"] = fn["description"]
        converted.append(api)
    return converted


def _to_function_call_item(tool_call: ParsedToolCall) -> Dict[str, Any]:
    """Convert an executor-shaped ParsedToolCall back to a wire-shaped
    function_call item (upstream `toFunctionCallItem`). Arguments are always
    re-serialized (semantically equal to the wire arguments, not guaranteed
    byte-identical)."""
    try:
        arguments = json.dumps(tool_call.arguments, separators=(",", ":"), default=dump)
    except (TypeError, ValueError):
        arguments = "{}"
    return {
        "type": "function_call",
        "id": tool_call.id,
        "callId": tool_call.id,
        "name": tool_call.name,
        "arguments": arguments,
    }


def _build_execute_ctx(
    tool: Tool,
    tool_call: Optional[ParsedToolCall],
    turn_context: Optional[Mapping[str, Any]],
    context_store: Optional[ToolContextStore] = None,
    shared_context_schema: Any = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Thread the executed call into the execute context (upstream
    `buildExecuteCtx`, #91): a caller-provided ``tool_call`` on the turn
    context wins; otherwise the executed call is converted to a wire-shaped
    function_call item, so ``on_tool_called`` / ``execute`` see
    ``ctx["tool_call"]`` on every path."""
    base = dict(turn_context or {})
    if base.get("tool_call") is None and tool_call is not None:
        base["tool_call"] = _to_function_call_item(tool_call)
    return build_tool_execute_context(tool, base, context_store, shared_context_schema, extras)


async def execute_regular_tool(
    tool: Tool, tool_call: ParsedToolCall, context: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Execute a regular (non-generator) tool. ``context`` is the fully built
    execute context (see `execute_tool`)."""
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
    """Execute a legacy generator tool. Yields are broadcast live through
    ``on_preliminary_result`` and are NOT retained on the terminal result
    (upstream drop-preliminary-results): long-running generators no longer
    accumulate their whole yield history in memory."""
    fn = get_tool_function(tool)
    source = "mcp" if is_mcp_tool(tool) else "client"
    try:
        args = validate_schema(fn.get("input_schema"), tool_call.arguments)
        produced = fn["execute"](args, context)
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


@dataclass
class AsyncToolInvocation:
    """Tagged result for unified tools whose work escapes the synchronous
    round (upstream `AsyncToolInvocation`).

    - ``async_mode="background"``: ``run()`` executes the tool's body (input
      already validated, context built) and resolves with the
      output-validated final value. The engine decides when to invoke it and
      whether it settles in-round (grace window) or later.
    - ``async_mode="defer"``: the run returned a durable task handle; the
      conversation pauses until the task is resolved externally.
    """

    async_mode: str
    run: Optional[Callable[[], Awaitable[Any]]] = None
    ack: Any = None
    grace_ms: float = 250
    task_id: Optional[str] = None
    poll_after_ms: Optional[int] = None
    expires_at: Optional[int] = None


def is_async_tool_invocation(value: Any) -> bool:
    return isinstance(value, AsyncToolInvocation)


def _resolve_ack(ack: Any, validated_input: Any) -> Any:
    if ack is None:
        return None
    if callable(ack):
        return ack(validated_input)
    return ack


async def _run_unified_tool(
    fn: Mapping[str, Any], validated_input: Any, run_context: Any, tool_name: str, on_yield: Callable[[Any], Any]
) -> Any:
    """Drive a unified tool's ``run`` to completion.

    Generator runs: Python async generators cannot ``return`` a value
    (idiomatic divergence #3), so the FINAL yielded value is the result and
    every earlier yield is a log/event entry (validated against
    ``event_schema`` when declared). Plain coroutine runs: the return value is
    the result. A ``ctx["defer"]()`` handle is only legal for
    ``lifecycle="deferred"`` tools."""
    returned = fn["run"](validated_input, run_context)
    result: Any
    if hasattr(returned, "__aiter__"):
        has_value = False
        previous: Any = None
        async for value in returned:
            if has_value:
                event = validate_schema(fn.get("event_schema"), previous) if fn.get("event_schema") else previous
                await maybe_await(on_yield(event))
            previous = value
            has_value = True
        result = previous if has_value else None
    else:
        result = await maybe_await(returned)
    if is_deferred_handle(result):
        if fn.get("lifecycle") != "deferred":
            raise RuntimeError(
                f"Tool \"{tool_name}\": run() returned a DeferredHandle but lifecycle is "
                f"'{fn.get('lifecycle')}'. Only lifecycle: 'deferred' tools may defer."
            )
        return result
    return validate_schema(fn.get("output_schema"), result) if fn.get("output_schema") is not None else result


async def prepare_unified_invocation(
    tool: Tool,
    tool_call: ParsedToolCall,
    turn_context: Optional[Mapping[str, Any]] = None,
    on_preliminary_result: Optional[Callable[[str, Any], Any]] = None,
    context_store: Optional[ToolContextStore] = None,
    shared_context_schema: Any = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Prepare a unified ``run`` tool invocation (upstream
    `prepareUnifiedInvocation`). Validates input eagerly, then per lifecycle:
    ``sync`` runs inline (a plain execution result); ``background`` returns a
    background `AsyncToolInvocation` thunk; ``deferred`` awaits the run -- a
    plain value is the fast path, a defer handle becomes a ``defer``
    invocation."""
    fn = get_tool_function(tool)
    source = "mcp" if is_mcp_tool(tool) else "client"
    try:
        validated_input = validate_schema(fn.get("input_schema"), tool_call.arguments)
    except Exception as exc:
        return {"tool_call_id": tool_call.id, "tool_name": tool_call.name, "source": source, "result": None, "error": exc}

    run_extras = dict((extras or {}).get("run_extras") or {})
    engine_log = run_extras.get("log")

    async def on_yield(event: Any) -> None:
        if engine_log is not None:
            engine_log(event)
        if on_preliminary_result is not None:
            await maybe_await(on_preliminary_result(tool_call.id, event))

    def ctx_log(entry: Any) -> None:
        # Bare strings are human-readable progress notes and skip
        # event_schema validation; structured entries are validated.
        validated = (
            validate_schema(fn.get("event_schema"), entry)
            if fn.get("event_schema") is not None and not isinstance(entry, str)
            else entry
        )
        if engine_log is not None:
            engine_log(validated)
        if on_preliminary_result is not None:
            outcome = on_preliminary_result(tool_call.id, validated)
            if inspect.isawaitable(outcome):
                asyncio.ensure_future(outcome)

    run_extras["log"] = ctx_log
    run_context = build_tool_run_context(
        tool,
        _with_tool_call(turn_context, tool_call),
        context_store,
        shared_context_schema,
        {**dict(extras or {}), "run_extras": run_extras},
    )

    async def invoke_run() -> Any:
        return await _run_unified_tool(fn, validated_input, run_context, str(tool_call.name), on_yield)

    if fn.get("lifecycle") == "background":

        async def background_run() -> Any:
            result = await invoke_run()
            if is_deferred_handle(result):
                raise RuntimeError(f'Tool "{tool_call.name}": background run cannot defer')
            return result

        return AsyncToolInvocation(
            async_mode="background",
            run=background_run,
            ack=_resolve_ack(fn.get("ack"), validated_input),
            grace_ms=fn.get("grace_ms") if fn.get("grace_ms") is not None else 250,
        )

    try:
        result = await invoke_run()
        if is_deferred_handle(result):
            task_id = result.get("task_id", result.get("taskId"))
            if not isinstance(task_id, str) or len(task_id) == 0 or len(task_id) > 256:
                raise ValueError(f'Tool "{tool_call.name}": ctx.defer() taskId must be 1-256 characters')
            ack = result.get("ack") if result.get("ack") is not None else _resolve_ack(fn.get("ack"), validated_input)
            poll_after_ms = (
                result.get("poll_after_ms") if result.get("poll_after_ms") is not None else fn.get("poll_after_ms")
            )
            return AsyncToolInvocation(
                async_mode="defer",
                task_id=task_id,
                ack=ack,
                poll_after_ms=poll_after_ms,
                expires_at=result.get("expires_at"),
            )
        return {"tool_call_id": tool_call.id, "tool_name": tool_call.name, "source": source, "result": result}
    except Exception as exc:
        return {"tool_call_id": tool_call.id, "tool_name": tool_call.name, "source": source, "result": None, "error": exc}


def _with_tool_call(turn_context: Optional[Mapping[str, Any]], tool_call: ParsedToolCall) -> Dict[str, Any]:
    base = dict(turn_context or {})
    if base.get("tool_call") is None:
        base["tool_call"] = _to_function_call_item(tool_call)
    return base


async def execute_tool(
    tool: Tool,
    tool_call: ParsedToolCall,
    turn_context: Optional[Mapping[str, Any]] = None,
    on_preliminary_result: Optional[Callable[[str, Any], Any]] = None,
    context_store: Optional[ToolContextStore] = None,
    shared_context_schema: Any = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Execute a tool call, dispatching on its shape (HITL, unified ``run``,
    generator, regular). Returns ``None`` for a HITL pause (or a non-client /
    non-executable tool), an `AsyncToolInvocation` for background/deferred
    unified tools, otherwise an execution-result dict."""
    if not is_client_tool(tool):
        return None
    if is_hitl_tool(tool):
        context = _build_execute_ctx(tool, tool_call, turn_context, context_store, shared_context_schema, extras)
        return await execute_hitl_tool(tool, tool_call, context)
    if is_unified_tool(tool):
        return await prepare_unified_invocation(
            tool, tool_call, turn_context, on_preliminary_result, context_store, shared_context_schema, extras
        )
    context = _build_execute_ctx(tool, tool_call, turn_context, context_store, shared_context_schema, extras)
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
