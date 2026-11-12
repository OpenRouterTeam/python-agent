from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .tool_types import SHARED_CONTEXT_KEY, ToolType


def tool(
    *,
    name: str,
    input_schema: Any = None,
    description: Optional[str] = None,
    execute: Any = None,
    output_schema: Any = None,
    event_schema: Any = None,
    context_schema: Any = None,
    next_turn_params: Any = None,
    require_approval: Any = None,
    on_tool_called: Any = None,
    on_response_received: Any = None,
    to_model_output: Any = None,
    strict: Optional[bool] = None,
    wire_input_schema: Optional[Mapping[str, Any]] = None,
    loop_key: Any = None,
) -> Dict[str, Any]:
    """Create a client tool.

    `strict` is passed through to the wire function-tool definition so
    providers can enforce structured-outputs-style argument adherence.
    `wire_input_schema` is a caller-owned JSON Schema serialized to the model
    verbatim (sanitized) in place of the one generated from `input_schema`,
    which keeps validating the call. `loop_key` declares the call identity
    used by doom-loop detection: a callable over the validated arguments
    (returning `None` exempts the call) or `False` (statically exempt).
    """
    if name == SHARED_CONTEXT_KEY:
        raise ValueError('Tool name "shared" is reserved for shared context. Choose a different name.')
    fn: Dict[str, Any] = {"name": name, "input_schema": input_schema}
    if description is not None:
        fn["description"] = description
    if output_schema is not None:
        fn["output_schema"] = output_schema
    if event_schema is not None:
        fn["event_schema"] = event_schema
    if context_schema is not None:
        fn["context_schema"] = context_schema
    if next_turn_params is not None:
        fn["next_turn_params"] = next_turn_params
    if require_approval is not None:
        fn["require_approval"] = require_approval
    if to_model_output is not None:
        fn["to_model_output"] = to_model_output
    if strict is not None:
        fn["strict"] = strict
    if wire_input_schema is not None:
        fn["wire_input_schema"] = wire_input_schema
    if loop_key is not None:
        fn["loop_key"] = loop_key
    if on_tool_called is not None:
        if output_schema is None:
            raise ValueError(f'HITL tool "{name}" must declare an output_schema.')
        fn["on_tool_called"] = on_tool_called
        if on_response_received is not None:
            fn["on_response_received"] = on_response_received
    elif execute is not False:
        if execute is None:
            raise ValueError(f'Tool "{name}" must provide execute, execute=False, or on_tool_called.')
        fn["execute"] = execute
    return {"type": ToolType.Function.value, "function": fn}


def server_tool(config: Mapping[str, Any], options: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Create an OpenRouter server-executed tool.

    Each server tool carries a stable tool-set `id` (default
    `server:{config["type"]}`) so activation APIs (`create_tool_set`) can
    address it. Override it with `options={"id": ...}`.
    """
    explicit_id = (options or {}).get("id")
    if explicit_id == "":
        raise ValueError("Server tool ID must not be empty")
    tool_id = explicit_id if explicit_id is not None else f"server:{config.get('type')}"
    return {"_brand": "server-tool", "config": dict(config), "id": tool_id}


def mark_mcp(tool_to_mark: Dict[str, Any], options: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Add the additive MCP brand to an already-built client tool (see
    `is_mcp_tool`). Non-mutating: returns a shallow copy carrying `_mcp`, so
    the tool's runtime behavior and wire shape are unchanged -- only the
    `is_mcp_tool` check now identifies it as MCP-originated. Used by
    `@openrouter/mcp`-equivalent integrations to mark wrapped remote tools.
    """
    marked = {**tool_to_mark, "_mcp": True}
    # `options["loop_key"]` attaches a doom-loop identity to the wrapped tool;
    # `options["wire_input_schema"]` is the server's own JSON Schema, sent to
    # the model in place of one regenerated from `input_schema`.
    if options and isinstance(marked.get("function"), Mapping):
        fn = dict(marked["function"])
        if options.get("loop_key") is not None:
            fn["loop_key"] = options["loop_key"]
        if options.get("wire_input_schema") is not None:
            fn["wire_input_schema"] = options["wire_input_schema"]
        marked["function"] = fn
    return marked
