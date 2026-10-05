"""Port of upstream `lib/next-turn-params.ts`.

Tools may declare ``next_turn_params``: a mapping of request-parameter names to
functions ``(params, context) -> value`` that compute the parameter for the
NEXT model turn after the tool runs. Functions compose: later tools (in
``tools`` order) see modifications made by earlier ones.

Valid keys are the snake_case mapping of upstream's: ``input``,
``tool_choice`` (upstream ``toolChoice``), ``model``, ``models``,
``temperature``, ``max_output_tokens``, ``top_p``, ``top_k``,
``instructions``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Mapping, Sequence

from ._utils import maybe_await
from .tool_types import ParsedToolCall, Tool, get_tool_function, is_client_tool

NEXT_TURN_KEYS = (
    "input",
    "tool_choice",
    "model",
    "models",
    "temperature",
    "max_output_tokens",
    "top_p",
    "top_k",
    "instructions",
)

_VALID_KEYS = frozenset(NEXT_TURN_KEYS)


def build_next_turn_params_context(request: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract the fields a `next_turn_params` function may read/modify, with
    upstream's defaults for missing fields (``input`` -> ``[]``, ``model`` ->
    ``""``, ``models`` -> ``[]``, numeric/instructions -> ``None``)."""
    input_value = request.get("input")
    model = request.get("model")
    models = request.get("models")
    return {
        "input": input_value if input_value is not None else [],
        "tool_choice": request.get("tool_choice"),
        "model": model if model is not None else "",
        "models": models if models is not None else [],
        "temperature": request.get("temperature"),
        "max_output_tokens": request.get("max_output_tokens"),
        "top_p": request.get("top_p"),
        "top_k": request.get("top_k"),
        "instructions": request.get("instructions"),
    }


async def execute_next_turn_params_functions(
    tool_calls: Sequence[ParsedToolCall], tools: Sequence[Tool], request: Mapping[str, Any]
) -> Dict[str, Any]:
    """Execute `next_turn_params` functions for every called tool, in
    ``tools`` order, composing results. Raises when a call's arguments are not
    an object."""
    working_context = build_next_turn_params_context(request)
    computed: Dict[str, Any] = {}
    for tool in tools:
        if not is_client_tool(tool):
            continue
        fn = get_tool_function(tool)
        next_params = fn.get("next_turn_params")
        if not next_params:
            continue
        tool_name = fn.get("name")
        for call in (c for c in tool_calls if c.name == tool_name):
            if not isinstance(call.arguments, Mapping):
                type_str = "array" if isinstance(call.arguments, list) else type(call.arguments).__name__
                raise TypeError(f"Tool call arguments for {tool_name} must be an object, got {type_str}")
            for key, param_fn in next_params.items():
                if not callable(param_fn):
                    continue
                if key not in _VALID_KEYS:
                    warnings.warn(
                        f'Invalid next_turn_params key "{key}" in tool "{tool_name}". '
                        "Valid keys: input, tool_choice, model, models, temperature, "
                        "max_output_tokens, top_p, top_k, instructions",
                        stacklevel=2,
                    )
                    continue
                value = await maybe_await(param_fn(call.arguments, working_context))
                computed[key] = value
                working_context[key] = value
    return computed


def apply_next_turn_params_to_request(request: Mapping[str, Any], params: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a new request with the computed params applied. ``None`` values
    clear the field (upstream maps ``null`` to ``undefined``)."""
    updated = dict(request)
    for key, value in params.items():
        if value is None:
            updated.pop(key, None)
        else:
            updated[key] = value
    return updated
