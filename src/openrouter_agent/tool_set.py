"""Stateful tool activation sets.

Port of upstream ``src/lib/tool-set.ts`` plus the runtime parts of
``src/lib/tool-set-types.ts``. A `ToolSet` holds an ordered list of tools and a
per-tool activation directive (static on/off, or a predicate over
``{"state", "context"}``), plus optional named *situations* that overlay the base
directives. Resolving produces an exhaustive snapshot whose ``call_model`` field
(``{"tools": [...], "active_tools": [...]}``) is meant to be spread into a
`call_model` request.

Upstream's compile-time partition tracking (``Partition``, ``InferEnabledIds``,
``FilterToolsByIds``, ...) has no Python equivalent and is dropped (contract
divergence #4). Runtime behavior is faithful.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from typing_extensions import Literal, TypeAlias, TypedDict

from .async_params import TOOL_SET_SNAPSHOT
from .tool import get_server_tool_id
from .tool_types import ConversationState, get_tool_function, is_server_tool

__all__ = [
    "TOOL_SET_SNAPSHOT",
    "ActivationInput",
    "ActivationPredicate",
    "ActivationMode",
    "CallModelToolsInput",
    "InferredToolsSnapshot",
    "ResolvedToolSnapshot",
    "SituationConditionalRule",
    "SituationConfig",
    "StatusByToolMap",
    "StatusReason",
    "ToolDirective",
    "ToolSet",
    "ToolStatusEntry",
    "create_tool_set",
    "get_tool_id",
]

# ─── runtime types (from tool-set-types.ts) ─────────────────────────────────


class ActivationInput(TypedDict, total=False):
    """Input passed verbatim to every activation predicate."""

    state: ConversationState
    context: Mapping[str, Any]


#: Predicate deciding activation. Only a return value that ``is True`` counts
#: as true (upstream compares ``=== true``).
ActivationPredicate: TypeAlias = Callable[[ActivationInput], Any]

#: Mode literals keep upstream's values (they are data, not identifiers).
ActivationMode: TypeAlias = Literal["activateWhen", "deactivateWhen"]


class _SituationRuleObject(TypedDict, total=False):
    mode: ActivationMode
    predicate: ActivationPredicate


SituationConditionalRule: TypeAlias = Union[ActivationPredicate, _SituationRuleObject]


class SituationConfig(TypedDict, total=False):
    """One named situation. IDs it does not mention keep the base directive."""

    enabled: Sequence[str]
    disabled: Sequence[str]
    #: Map of tool ID → predicate (``activateWhen``) or ``{"mode", "predicate"}``.
    #: ``None`` entries are ignored.
    conditional: Mapping[str, Optional[SituationConditionalRule]]


StatusReason: TypeAlias = Literal["default", "activate", "deactivate", "activateWhen", "deactivateWhen", "situation"]
ToolDirective: TypeAlias = Literal["activate", "deactivate", "activateWhen", "deactivateWhen"]


class _ToolStatusEntryRequired(TypedDict):
    enabled: bool
    reason: StatusReason


class ToolStatusEntry(_ToolStatusEntryRequired, total=False):
    #: The directive that produced this status (absent for ``reason="default"``).
    directive: ToolDirective
    #: Present (always ``True``) when a predicate was evaluated.
    predicate: bool


StatusByToolMap: TypeAlias = Dict[str, ToolStatusEntry]


class CallModelToolsInput(TypedDict):
    """The spread-safe subset of a snapshot: exactly what `call_model` accepts."""

    tools: List[Any]
    active_tools: List[str]


# Functional syntax so the marker key (`TOOL_SET_SNAPSHOT`) can be declared.
ResolvedToolSnapshot = TypedDict(
    "ResolvedToolSnapshot",
    {
        "tools": List[Any],
        "active_tools": List[str],
        "call_model": CallModelToolsInput,
        "enabled": List[str],
        "disabled": List[str],
        "status_by_tool": StatusByToolMap,
        "__openrouter_tool_set_snapshot__": Literal[True],
    },
)

InferredToolsSnapshot = TypedDict(
    "InferredToolsSnapshot",
    {
        "tools": List[Any],
        "active_tools": List[str],
        "enabled": List[str],
        "disabled": List[str],
        "status_by_tool": StatusByToolMap,
        "__openrouter_tool_set_snapshot__": Literal[True],
    },
)

# ─── internals ──────────────────────────────────────────────────────────────

_StaticSource = Literal["default", "activate", "deactivate", "situation"]


class _Entry:
    """One activation directive (upstream ``ActivationEntry``)."""

    __slots__ = ("kind", "active", "predicate", "source")

    def __init__(
        self,
        kind: Literal["static", "activateWhen", "deactivateWhen"],
        source: str,
        *,
        active: bool = True,
        predicate: Optional[ActivationPredicate] = None,
    ) -> None:
        self.kind = kind
        self.source = source
        self.active = active
        self.predicate = predicate


class _Situation:
    __slots__ = ("enabled", "disabled", "conditional")

    def __init__(
        self,
        enabled: List[str],
        disabled: List[str],
        conditional: List[Tuple[str, ActivationMode, ActivationPredicate]],
    ) -> None:
        self.enabled = enabled
        self.disabled = disabled
        self.conditional = conditional


class _Index:
    __slots__ = ("ordered_tools", "ordered_ids", "tool_by_id")

    def __init__(self, tools: Iterable[Mapping[str, Any]]) -> None:
        self.ordered_tools: List[Mapping[str, Any]] = list(tools)
        self.ordered_ids: List[str] = []
        self.tool_by_id: Dict[str, Mapping[str, Any]] = {}
        for item in self.ordered_tools:
            tool_id = get_tool_id(item)
            if tool_id in self.tool_by_id:
                raise ValueError(f'Duplicate tool ID: "{tool_id}"')
            self.tool_by_id[tool_id] = item
            self.ordered_ids.append(tool_id)


def get_tool_id(tool: Mapping[str, Any]) -> str:
    """Tool-set ID: a client tool's function name, or a server tool's stable ID."""
    if is_server_tool(tool):
        return get_server_tool_id(tool)
    return str(get_tool_function(tool).get("name"))


def _to_id_list(names: Union[str, Sequence[str]]) -> List[str]:
    return [names] if isinstance(names, str) else list(names)


def _predicate_is_true(predicate: ActivationPredicate, activation_input: ActivationInput) -> bool:
    result = predicate(activation_input)
    if inspect.iscoroutine(result):
        # Predicates are synchronous upstream; an awaitable is never `=== true`.
        # Close it so it does not surface as a "never awaited" warning.
        result.close()
        return False
    return result is True


def _normalize_conditional_rule(situation: str, tool_id: str, rule: Any) -> Tuple[ActivationMode, ActivationPredicate]:
    if callable(rule):
        return "activateWhen", rule
    mode = rule.get("mode") if isinstance(rule, Mapping) else None
    if (
        not isinstance(rule, Mapping)
        or "predicate" not in rule
        or not callable(rule["predicate"])
        or (mode is not None and mode not in ("activateWhen", "deactivateWhen"))
    ):
        raise ValueError(
            f'Situation "{situation}": conditional rule for tool "{tool_id}" '
            "must be a function or { mode, predicate } object"
        )
    return (mode if mode is not None else "activateWhen"), rule["predicate"]


# ─── ToolSet ────────────────────────────────────────────────────────────────


class ToolSet:
    """Immutable-by-default stateful set of tools.

    Each tool is enabled, disabled, or conditional (predicate-driven); named
    situations overlay that base partition. Mutators (`activate`, `deactivate`,
    `activate_when`, `deactivate_when`, `define_situations`) return a new
    instance unless the set was created with ``mutable=True``, in which case
    they mutate in place and return ``self``. Last call wins per tool.
    """

    __slots__ = ("_index", "_activation", "_situations", "_mutable")

    def __init__(
        self,
        _index: _Index,
        _activation: Dict[str, _Entry],
        _situations: Dict[str, _Situation],
        _mutable: bool,
    ) -> None:
        # Internal constructor. Prefer `create_tool_set` / `ToolSet.create`.
        self._index = _index
        self._activation = _activation
        self._situations = _situations
        self._mutable = _mutable

    @classmethod
    def create(cls, *, tools: Iterable[Mapping[str, Any]], mutable: bool = False) -> "ToolSet":
        return cls(_Index(tools), {}, {}, bool(mutable))

    @property
    def tools(self) -> List[Any]:
        """All tools in construction order, regardless of activation state."""
        return list(self._index.ordered_tools)

    @property
    def mutable(self) -> bool:
        return self._mutable

    def _assert_known(self, tool_id: str) -> None:
        if tool_id not in self._index.tool_by_id:
            raise ValueError(f'Unknown tool: "{tool_id}"')

    def _with_partition_mutation(self, mutate: Callable[[Dict[str, _Entry]], None]) -> "ToolSet":
        if self._mutable:
            mutate(self._activation)
            return self
        next_activation = dict(self._activation)
        mutate(next_activation)
        return ToolSet(self._index, next_activation, self._situations, False)

    def _set_static(self, names: Union[str, Sequence[str]], active: bool) -> "ToolSet":
        ids = _to_id_list(names)
        for tool_id in ids:
            self._assert_known(tool_id)
        source = "activate" if active else "deactivate"

        def mutate(activation: Dict[str, _Entry]) -> None:
            for tool_id in ids:
                activation[tool_id] = _Entry("static", source, active=active)

        return self._with_partition_mutation(mutate)

    def activate(self, names: Union[str, Sequence[str]]) -> "ToolSet":
        """Statically enable one ID or a list of IDs."""
        return self._set_static(names, True)

    def deactivate(self, names: Union[str, Sequence[str]]) -> "ToolSet":
        """Statically disable one ID or a list of IDs."""
        return self._set_static(names, False)

    def _set_conditional(
        self,
        kind: ActivationMode,
        name_or_map: Union[str, Mapping[str, Optional[ActivationPredicate]]],
        predicate: Optional[ActivationPredicate],
    ) -> "ToolSet":
        entries = self._normalize_predicate_arg(name_or_map, predicate)

        def mutate(activation: Dict[str, _Entry]) -> None:
            for tool_id, pred in entries:
                activation[tool_id] = _Entry(kind, kind, predicate=pred)

        return self._with_partition_mutation(mutate)

    def activate_when(
        self,
        name_or_map: Union[str, Mapping[str, Optional[ActivationPredicate]]],
        predicate: Optional[ActivationPredicate] = None,
    ) -> "ToolSet":
        """Enable a tool only when its predicate returns ``True``.

        Accepts ``(name, predicate)`` or a ``{name: predicate}`` map.
        """
        return self._set_conditional("activateWhen", name_or_map, predicate)

    def deactivate_when(
        self,
        name_or_map: Union[str, Mapping[str, Optional[ActivationPredicate]]],
        predicate: Optional[ActivationPredicate] = None,
    ) -> "ToolSet":
        """Disable a tool when its predicate returns ``True`` (active otherwise)."""
        return self._set_conditional("deactivateWhen", name_or_map, predicate)

    def _normalize_predicate_arg(
        self,
        name_or_map: Union[str, Mapping[str, Optional[ActivationPredicate]]],
        predicate: Optional[ActivationPredicate],
    ) -> List[Tuple[str, ActivationPredicate]]:
        if isinstance(name_or_map, str):
            if not predicate:
                raise ValueError("activate_when/deactivate_when requires a predicate when called with a name")
            self._assert_known(name_or_map)
            return [(name_or_map, predicate)]
        if not isinstance(name_or_map, Mapping):
            raise ValueError("activate_when/deactivate_when requires a name+predicate or predicate map")
        entries = [(tool_id, pred) for tool_id, pred in name_or_map.items() if callable(pred)]
        for tool_id, _ in entries:
            self._assert_known(tool_id)
        return entries

    def define_situations(self, situations: Mapping[str, SituationConfig]) -> "ToolSet":
        """Register named situations, replacing any previously defined ones.

        Each situation overlays the base partition; IDs it does not mention keep
        whatever the base set declares. An ID may appear at most once across a
        situation's ``enabled`` / ``disabled`` / ``conditional``.
        """
        next_situations: Dict[str, _Situation] = {}
        for name, config in situations.items():
            enabled = list(config.get("enabled") or [])
            disabled = list(config.get("disabled") or [])
            conditional_entries = [
                (tool_id, rule) for tool_id, rule in (config.get("conditional") or {}).items() if rule is not None
            ]

            seen: set = set()

            def record(tool_id: str, situation_name: str = name) -> None:
                self._assert_known(tool_id)
                if tool_id in seen:
                    raise ValueError(
                        f'Situation "{situation_name}" lists tool "{tool_id}" more than once '
                        "(across enabled/disabled/conditional)"
                    )
                seen.add(tool_id)

            for tool_id in enabled:
                record(tool_id)
            for tool_id in disabled:
                record(tool_id)
            for tool_id, _ in conditional_entries:
                record(tool_id)

            conditional: List[Tuple[str, ActivationMode, ActivationPredicate]] = []
            for tool_id, rule in conditional_entries:
                mode, pred = _normalize_conditional_rule(name, tool_id, rule)
                conditional.append((tool_id, mode, pred))
            next_situations[name] = _Situation(enabled, disabled, conditional)

        if self._mutable:
            self._situations.clear()
            self._situations.update(next_situations)
            return self
        return ToolSet(self._index, dict(self._activation), next_situations, False)

    def resolve(self, input: Optional[ActivationInput] = None) -> ResolvedToolSnapshot:
        """Resolve against the base partition (no situation overlay)."""
        return self._resolve_with_activation(self._activation, input)

    def infer_tools(self, input: Optional[ActivationInput] = None) -> InferredToolsSnapshot:
        """Back-compat alias of `resolve` (without the ``call_model`` field).

        Only ``tools`` and ``active_tools`` are valid `call_model` input. The
        result carries the `TOOL_SET_SNAPSHOT` marker so `call_model` can strip
        the remaining metadata when the whole dict is spread in.
        """
        snapshot = self.resolve(input)
        return {
            "tools": list(snapshot["tools"]),
            "active_tools": list(snapshot["active_tools"]),
            "enabled": snapshot["enabled"],
            "disabled": snapshot["disabled"],
            "status_by_tool": snapshot["status_by_tool"],
            TOOL_SET_SNAPSHOT: True,
        }

    def resolve_situation(self, name: str, input: Optional[ActivationInput] = None) -> ResolvedToolSnapshot:
        """Resolve a previously defined named situation."""
        situation = self._situations.get(name)
        if situation is None:
            raise ValueError(f'Unknown situation: "{name}"')
        activation = dict(self._activation)
        for tool_id in situation.enabled:
            activation[tool_id] = _Entry("static", "situation", active=True)
        for tool_id in situation.disabled:
            activation[tool_id] = _Entry("static", "situation", active=False)
        for tool_id, mode, pred in situation.conditional:
            activation[tool_id] = _Entry(mode, "situation", predicate=pred)
        return self._resolve_with_activation(activation, input)

    def _resolve_with_activation(
        self, activation: Mapping[str, _Entry], activation_input: Optional[ActivationInput]
    ) -> ResolvedToolSnapshot:
        resolved_input: ActivationInput = activation_input if activation_input is not None else {}
        tools: List[Any] = []
        active_tools: List[str] = []
        enabled: List[str] = []
        disabled: List[str] = []
        status_by_tool: StatusByToolMap = {}

        for tool_id in self._index.ordered_ids:
            item = self._index.tool_by_id[tool_id]
            entry = activation.get(tool_id)
            active = self._evaluate(entry, resolved_input)
            status_by_tool[tool_id] = self._to_status_entry(active, entry)
            if active:
                tools.append(item)
                enabled.append(tool_id)
                if not is_server_tool(item):
                    active_tools.append(tool_id)
            else:
                disabled.append(tool_id)

        return {
            "tools": tools,
            "active_tools": active_tools,
            "call_model": {"tools": tools, "active_tools": active_tools},
            "enabled": enabled,
            "disabled": disabled,
            "status_by_tool": status_by_tool,
            TOOL_SET_SNAPSHOT: True,
        }

    @staticmethod
    def _evaluate(entry: Optional[_Entry], activation_input: ActivationInput) -> bool:
        if entry is None:
            return True
        if entry.kind == "static":
            return entry.active
        assert entry.predicate is not None
        if entry.kind == "activateWhen":
            return _predicate_is_true(entry.predicate, activation_input)
        return not _predicate_is_true(entry.predicate, activation_input)

    @staticmethod
    def _to_status_entry(active: bool, entry: Optional[_Entry]) -> ToolStatusEntry:
        if entry is None:
            return {"enabled": active, "reason": "default"}
        if entry.kind == "static":
            directive: ToolDirective = "activate" if entry.active else "deactivate"
            reason: StatusReason
            if entry.source == "situation":
                reason = "situation"
            elif entry.source == "default":
                reason = "default"
            else:
                reason = directive
            return {"enabled": active, "reason": reason, "directive": directive}
        kind: ActivationMode = "activateWhen" if entry.kind == "activateWhen" else "deactivateWhen"
        return {
            "enabled": active,
            "reason": "situation" if entry.source == "situation" else kind,
            "directive": kind,
            "predicate": True,
        }

    def clone(self, *, mutable: Optional[bool] = None) -> "ToolSet":
        """Copy state into a fresh, independent instance.

        ``mutable=None`` inherits the source's mode.
        """
        next_mutable = self._mutable if mutable is None else bool(mutable)
        return ToolSet(self._index, dict(self._activation), dict(self._situations), next_mutable)

    def __repr__(self) -> str:
        return f"ToolSet(ids={self._index.ordered_ids!r}, mutable={self._mutable})"


def create_tool_set(*, tools: Iterable[Mapping[str, Any]], mutable: bool = False) -> ToolSet:
    """Construct a `ToolSet`. Raises ``ValueError`` on duplicate tool IDs."""
    return ToolSet.create(tools=tools, mutable=mutable)
