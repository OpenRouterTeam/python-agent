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
) -> Dict[str, Any]:
    """Build a client tool.

    `strict` is serialized onto the wire tool definition so providers can
    enforce structured-outputs-style schema adherence on tool-call arguments
    (OpenAI-style strict mode requires every object property in `required`;
    use `Optional[...]` without a default for conceptually optional values).
    `None` leaves the provider default.

    `wire_input_schema` (manual tools only, `execute=False`) is a caller-owned
    JSON Schema serialized to the model instead of one generated from
    `input_schema`, which still describes the call's arguments.
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
        if execute is not False or on_tool_called is not None:
            # Upstream restricts this at the type level (`wireInputSchema?:
            # never` on executable configs); Python enforces it at runtime.
            raise ValueError(f'Tool "{name}": wire_input_schema is only supported on manual tools (execute=False).')
        fn["wire_input_schema"] = wire_input_schema
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


def server_tool(config: Dict[str, Any]) -> Dict[str, Any]:
    return {"_brand": "server-tool", "config": dict(config)}


def mark_mcp(
    tool_to_mark: Dict[str, Any],
    *,
    annotations: Optional[Mapping[str, Any]] = None,
    wire_input_schema: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Add the additive MCP brand to an already-built client tool (see
    `is_mcp_tool`). Non-mutating: returns a shallow copy carrying `_mcp`, so
    the tool's runtime behavior is unchanged -- only the `is_mcp_tool` check
    now identifies it as MCP-originated. Used by `@openrouter/mcp`-equivalent
    integrations to mark wrapped remote tools.

    `annotations` records the server's `tools/list` annotations
    (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`,
    `title`) on the branded tool so a consumer can read them back.
    `wire_input_schema` is the server's own JSON Schema for the tool's
    arguments; it is serialized to the model in place of one regenerated from
    `input_schema`, which keeps validating the call.
    """
    marked: Dict[str, Any] = {**tool_to_mark, "_mcp": True}
    if annotations is not None:
        marked["annotations"] = annotations
    # A non-object root is not a usable tool schema; the generated object
    # schema is serialized instead.
    if (
        wire_input_schema is not None
        and wire_input_schema.get("type") == "object"
        and isinstance(marked.get("function"), Mapping)
    ):
        marked["function"] = {**marked["function"], "wire_input_schema": wire_input_schema}
    return marked
