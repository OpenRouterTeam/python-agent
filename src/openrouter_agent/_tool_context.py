"""Tool context store and context building utilities."""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from typing import Any

from ._types import (
    SHARED_CONTEXT_KEY,
    ToolExecuteContext,
    TurnContext,
)


class ToolContextStore:
    """Mutable context store for tool execution. Persists across turns."""

    def __init__(
        self, initial_contexts: dict[str, dict[str, Any]] | None = None
    ) -> None:
        self._contexts: dict[str, dict[str, Any]] = dict(initial_contexts or {})
        self._listeners: list[Callable[[], None]] = []

    def subscribe(self, listener: Callable[[], None]) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def _notify(self) -> None:
        for listener in self._listeners:
            listener()

    def get_snapshot(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self._contexts)

    def get_tool_context(self, tool_name: str) -> dict[str, Any]:
        return dict(self._contexts.get(tool_name, {}))

    def set_tool_context(self, tool_name: str, values: dict[str, Any]) -> None:
        self._contexts[tool_name] = dict(values)
        self._notify()

    def merge_tool_context(self, tool_name: str, partial: dict[str, Any]) -> None:
        current = self._contexts.get(tool_name, {})
        current.update(partial)
        self._contexts[tool_name] = current
        self._notify()


def build_tool_execute_context(
    turn_context: TurnContext,
    store: ToolContextStore,
    tool_name: str,
) -> ToolExecuteContext[Any, Any]:
    """Build the context object passed to tool execute functions."""
    local = store.get_tool_context(tool_name)
    shared = store.get_tool_context(SHARED_CONTEXT_KEY)

    def _set_context(partial: dict[str, Any]) -> None:
        store.merge_tool_context(tool_name, partial)

    def _set_shared_context(partial: dict[str, Any]) -> None:
        store.merge_tool_context(SHARED_CONTEXT_KEY, partial)

    return ToolExecuteContext(
        tool_call=turn_context.tool_call,
        number_of_turns=turn_context.number_of_turns,
        turn_request=turn_context.turn_request,
        tool_name=tool_name,
        local=local,
        shared=shared,
        set_context_fn=_set_context,
        set_shared_context_fn=_set_shared_context,
    )


async def resolve_context(
    context_input: Any | Callable[[TurnContext], Any] | Callable[[TurnContext], Awaitable[Any]],
    turn_context: TurnContext,
) -> Any:
    """Resolve a ContextInput value (static, sync callable, or async callable)."""
    if callable(context_input):
        result = context_input(turn_context)
        if hasattr(result, "__await__"):
            return await result
        return result
    return context_input
