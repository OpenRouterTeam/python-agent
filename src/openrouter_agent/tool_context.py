from __future__ import annotations

import copy
from collections.abc import Mapping as MappingABC
from typing import Any, Callable, Dict, List, Mapping, Optional

from ._utils import maybe_await, validate_schema
from .tool_types import SHARED_CONTEXT_KEY, ContextInput as ContextInput


class ToolContextStore:
    def __init__(self, initial: Optional[Mapping[str, Any]] = None) -> None:
        self._contexts: Dict[str, Any] = copy.deepcopy(dict(initial or {}))
        self._subscribers: List[Callable[[Dict[str, Any]], Any]] = []

    def subscribe(self, callback: Callable[[Dict[str, Any]], Any]) -> Callable[[], None]:
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return unsubscribe

    def _notify(self) -> None:
        snapshot = self.get_snapshot()
        for callback in list(self._subscribers):
            callback(snapshot)

    def get_snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self._contexts)

    def get_tool_context(self, tool_name: str) -> Dict[str, Any]:
        value = self._contexts.get(tool_name, {})
        return copy.deepcopy(value if isinstance(value, dict) else {})

    def set_tool_context(self, tool_name: str, context: Mapping[str, Any]) -> None:
        self._contexts[tool_name] = copy.deepcopy(dict(context))
        self._notify()

    def merge_tool_context(self, tool_name: str, partial: Mapping[str, Any]) -> None:
        current = self.get_tool_context(tool_name)
        current.update(dict(partial))
        self.set_tool_context(tool_name, current)


def _filter_schema_keys(schema: Any, value: Mapping[str, Any]) -> Dict[str, Any]:
    if schema is None:
        return dict(value)
    fields = getattr(schema, "model_fields", None)
    if fields:
        return {key: val for key, val in value.items() if key in fields}
    return dict(value)


def _parse_partial_against_schema(
    schema: Any, partial: Mapping[str, Any], existing: Mapping[str, Any]
) -> Dict[str, Any]:
    """Filter a partial context update to the schema's known keys and return
    the schema's *parsed* value for each updated key, so field transforms,
    defaults and coercions apply to the stored value. The merged result
    (existing values plus the update) is validated as a whole first, so
    model-level validators are enforced before anything is stored. Raises on
    invalid input; the caller leaves the stored context unchanged."""
    filtered = _filter_schema_keys(schema, partial)
    if schema is None:
        return filtered
    validated = validate_schema(schema, {**dict(existing), **filtered})
    if hasattr(validated, "model_dump"):
        dumped = validated.model_dump()
        return {key: dumped[key] for key in filtered if key in dumped}
    if isinstance(validated, Mapping):
        return {key: validated[key] for key in filtered if key in validated}
    return filtered


class LiveContext(MappingABC):
    def __init__(self, getter: Callable[[], Dict[str, Any]]) -> None:
        self._getter = getter

    def __getitem__(self, key: str) -> Any:
        return self._getter()[key]

    def __iter__(self):
        return iter(self._getter())

    def __len__(self) -> int:
        return len(self._getter())

    def __repr__(self) -> str:
        return repr(self._getter())

    def __eq__(self, other: Any) -> bool:
        return self._getter() == other


def extract_tool_context(tool: Mapping[str, Any], context_map: Mapping[str, Any]) -> Dict[str, Any]:
    fn = tool.get("function", {})
    schema = fn.get("context_schema")
    raw = context_map.get(fn.get("name"), {})
    if raw is None:
        raw = {}
    validated = validate_schema(schema, raw)
    if hasattr(validated, "model_dump"):
        return validated.model_dump()
    return dict(validated or {})


async def resolve_context(context: Any, turn_context: Mapping[str, Any]) -> Dict[str, Any]:
    if context is None:
        return {}
    if callable(context):
        resolved = await maybe_await(context(turn_context))
        return dict(resolved or {})
    return dict(context)


def build_tool_execute_context(
    tool: Mapping[str, Any],
    turn_context: Optional[Mapping[str, Any]] = None,
    store: Optional[ToolContextStore] = None,
    shared_context_schema: Any = None,
) -> Dict[str, Any]:
    fn = tool.get("function", {})
    name = str(fn.get("name", ""))
    base = dict(turn_context or {})
    if store is None:
        store = ToolContextStore({})

    def local() -> Dict[str, Any]:
        return store.get_tool_context(name)

    def shared() -> Dict[str, Any]:
        return store.get_tool_context(SHARED_CONTEXT_KEY)

    def set_context(partial: Mapping[str, Any]) -> None:
        parsed = _parse_partial_against_schema(fn.get("context_schema"), partial, store.get_tool_context(name))
        store.merge_tool_context(name, parsed)

    def set_shared_context(partial: Mapping[str, Any]) -> None:
        parsed = _parse_partial_against_schema(
            shared_context_schema, partial, store.get_tool_context(SHARED_CONTEXT_KEY)
        )
        store.merge_tool_context(SHARED_CONTEXT_KEY, parsed)

    base.update(
        {
            "local": LiveContext(local),
            "shared": LiveContext(shared),
            "set_context": set_context,
            "set_shared_context": set_shared_context,
        }
    )
    return base
