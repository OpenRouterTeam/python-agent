from __future__ import annotations

from typing import Any, Dict, Optional

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
) -> Dict[str, Any]:
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
