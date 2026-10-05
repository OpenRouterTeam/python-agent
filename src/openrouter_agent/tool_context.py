from __future__ import annotations

import copy
from collections.abc import Mapping as MappingABC
from typing import Any, Callable, Dict, List, Mapping, Optional

from typing_extensions import TypedDict

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


class ToolExecutionExtras(TypedDict, total=False):
    """Per-execution extras threaded into the tool execute context (upstream
    `ToolExecutionExtras`): the per-call cancellation signal, the call id, the
    conversation id, the parent run's client (agent tools start child runs
    with it), and -- for unified ``run`` tools -- the async-task affordances
    (``run_extras``: ``defer``, ``log``, ``on_message``, ``task_id`` getter,
    ``task_transcript`` slot)."""

    signal: Any
    call_id: str
    conversation_id: str
    client: Any
    run_extras: Any


def _never_cancel_signal() -> Any:
    """A fresh never-firing signal per context (upstream `neverAbortSignal`):
    keeps ``ctx["signal"]`` always present so tool bodies can use it
    unconditionally."""
    from .tool_task import CancellationController

    return CancellationController()


def build_tool_execute_context(
    tool: Mapping[str, Any],
    turn_context: Optional[Mapping[str, Any]] = None,
    store: Optional[ToolContextStore] = None,
    shared_context_schema: Any = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    fn = tool.get("function", {})
    name = str(fn.get("name", ""))
    base = dict(turn_context or {})
    extras = extras or {}
    base["signal"] = extras.get("signal") if extras.get("signal") is not None else _never_cancel_signal()
    if extras.get("call_id") is not None:
        base["call_id"] = extras["call_id"]
    if extras.get("conversation_id") is not None:
        base["conversation_id"] = extras["conversation_id"]
    if store is None:
        store = ToolContextStore({})

    def local() -> Dict[str, Any]:
        return store.get_tool_context(name)

    def shared() -> Dict[str, Any]:
        return store.get_tool_context(SHARED_CONTEXT_KEY)

    def set_context(partial: Mapping[str, Any]) -> None:
        filtered = _filter_schema_keys(fn.get("context_schema"), partial)
        store.merge_tool_context(name, filtered)

    def set_shared_context(partial: Mapping[str, Any]) -> None:
        filtered = _filter_schema_keys(shared_context_schema, partial)
        store.merge_tool_context(SHARED_CONTEXT_KEY, filtered)

    base.update(
        {
            "local": LiveContext(local),
            "shared": LiveContext(shared),
            "set_context": set_context,
            "set_shared_context": set_shared_context,
        }
    )
    return base


class _RunContext(dict):  # type: ignore[type-arg]
    """Execute context for unified ``run`` tools. ``task_id`` is a LIVE
    lookup: the ToolTask (and its id) is created only after the call escapes
    the round, after this context is built (upstream defines a getter)."""

    _task_id_getter: Optional[Callable[[], Any]] = None

    def __getitem__(self, key: Any) -> Any:
        if key == "task_id" and self._task_id_getter is not None:
            return self._task_id_getter()
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        if key == "task_id" and self._task_id_getter is not None:
            value = self._task_id_getter()
            return default if value is None else value
        return super().get(key, default)


def build_tool_run_context(
    tool: Mapping[str, Any],
    turn_context: Optional[Mapping[str, Any]] = None,
    store: Optional[ToolContextStore] = None,
    shared_context_schema: Any = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the context for a unified ``run`` tool: the base execute context
    plus ``defer`` / ``log`` / ``on_message`` / ``task_id`` / ``client``
    (upstream `buildToolRunContext`)."""
    fn = tool.get("function", {})
    name = str(fn.get("name", ""))
    base = build_tool_execute_context(tool, turn_context, store, shared_context_schema, extras)
    run_extras: Mapping[str, Any] = (extras or {}).get("run_extras") or {}

    def _no_defer(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"Tool \"{name}\": ctx.defer() is only available on lifecycle: 'deferred' tools")

    ctx = _RunContext(base)
    ctx["defer"] = run_extras.get("defer") or _no_defer
    ctx["log"] = run_extras.get("log") or (lambda _entry: None)
    ctx["on_message"] = run_extras.get("on_message") or (lambda _handler: None)
    if run_extras.get("task_transcript") is not None:
        ctx["task_transcript"] = run_extras["task_transcript"]
    if (extras or {}).get("client") is not None:
        ctx["client"] = (extras or {})["client"]
    getter = run_extras.get("task_id_getter")
    if callable(getter):
        ctx._task_id_getter = getter
        ctx["task_id"] = None
    return ctx
