"""Tool factory - creates typed tool definitions from configuration."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, overload

from pydantic import BaseModel

from ._types import (
    ManualTool,
    ManualToolFunction,
    Tool,
    ToolFunctionWithExecute,
    ToolFunctionWithGenerator,
    ToolType,
    ToolWithExecute,
    ToolWithGenerator,
)


@overload
def tool(
    *,
    name: str,
    description: str | None = None,
    input_schema: type[BaseModel],
    event_schema: type[BaseModel],
    output_schema: type[BaseModel],
    execute: Callable[..., AsyncGenerator[Any, None]],
    context_schema: type[BaseModel] | None = None,
    next_turn_params: dict[str, Callable[..., Any]] | None = None,
    require_approval: bool | Callable[..., bool | Awaitable[bool]] = False,
) -> ToolWithGenerator: ...


@overload
def tool(
    *,
    name: str,
    description: str | None = None,
    input_schema: type[BaseModel],
    output_schema: type[BaseModel] | None = None,
    context_schema: type[BaseModel] | None = None,
    next_turn_params: dict[str, Callable[..., Any]] | None = None,
    require_approval: bool | Callable[..., bool | Awaitable[bool]] = False,
) -> ManualTool: ...


@overload
def tool(
    *,
    name: str,
    description: str | None = None,
    input_schema: type[BaseModel],
    execute: Callable[..., Any],
    output_schema: type[BaseModel] | None = None,
    context_schema: type[BaseModel] | None = None,
    next_turn_params: dict[str, Callable[..., Any]] | None = None,
    require_approval: bool | Callable[..., bool | Awaitable[bool]] = False,
) -> ToolWithExecute: ...


def tool(
    *,
    name: str,
    description: str | None = None,
    input_schema: type[BaseModel],
    execute: Callable[..., Any] | None = None,
    output_schema: type[BaseModel] | None = None,
    event_schema: type[BaseModel] | None = None,
    context_schema: type[BaseModel] | None = None,
    next_turn_params: dict[str, Callable[..., Any]] | None = None,
    require_approval: bool | Callable[..., bool | Awaitable[bool]] = False,
) -> Tool:
    """Create a typed tool definition.

    - With execute + event_schema + output_schema: creates a generator tool
    - With execute (no event_schema): creates a regular execute tool
    - Without execute: creates a manual tool (human-in-the-loop)
    """
    base_kwargs: dict[str, Any] = {
        "name": name,
        "description": description,
        "input_schema": input_schema,
        "context_schema": context_schema,
        "next_turn_params": next_turn_params,
        "require_approval": require_approval,
    }

    if execute is not None and event_schema is not None and output_schema is not None:
        func = ToolFunctionWithGenerator(
            **base_kwargs,
            event_schema=event_schema,
            output_schema=output_schema,
            execute=execute,
        )
        return ToolWithGenerator(type=ToolType.FUNCTION, function=func)

    if execute is not None:
        func_exec = ToolFunctionWithExecute(
            **base_kwargs,
            output_schema=output_schema,
            execute=execute,
        )
        return ToolWithExecute(type=ToolType.FUNCTION, function=func_exec)

    func_manual = ManualToolFunction(
        **base_kwargs,
        output_schema=output_schema,
    )
    return ManualTool(type=ToolType.FUNCTION, function=func_manual)
