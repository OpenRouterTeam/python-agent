"""Declarative, state- and context-aware tool activation.

Port of upstream ``packages/agent/src/lib/tool-set.ts`` and
``packages/agent/src/lib/tool-set-types.ts`` (``@openrouter/agent/tool-set``).

Third-party notice (mirrors upstream ``packages/agent/THIRD_PARTY_NOTICES.md``):
``@openrouter/agent/tool-set`` -- and therefore this module -- is adapted from
``ai-tool-set`` v1.0.0 (https://github.com/zirkelc/ai-tool-set/tree/v1.0.0),
which is licensed under the MIT License:

    MIT License

    Copyright (c) 2024 Chris

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Usage::

    ts = create_tool_set(tools=[search, write, server_tool({"type": "openrouter:datetime"})])
    ts = ts.deactivate("write").activate_when("write", lambda i: (i.get("context") or {}).get("admin") is True)
    resolved = ts.resolve({"context": {"admin": True}})
    result = call_model(client, {"model": "...", "input": "...", **resolved.call_model})

Divergences from upstream (see the contract's Idiomatic Divergences):

- The TypeScript three-way compile-time partition (``Partition``,
  ``FilterToolsByIds``, ``Infer*Ids``, ``WidenedPartition`` ...) is type-level
  only and is not reproduced (divergence 4). Runtime behavior -- including
  mutable-vs-immutable aliasing, last-call-wins, situation overlays and the
  exhaustive ``status_by_tool`` snapshot -- is ported exactly.
- ``resolve()`` / ``resolve_situation()`` return a frozen
  :class:`ResolvedToolSnapshot` dataclass. Its ``call_model`` attribute is the
  spread-safe ``{"tools": ..., "active_tools": ...}`` mapping. ``infer_tools()``
  returns a plain ``dict`` carrying the :data:`TOOL_SET_SNAPSHOT` sentinel key
  (the Python analogue of upstream's ``Symbol.for(...)`` marker, which survives
  ``{**snapshot}``), so :func:`strip_tool_set_snapshot_metadata` can remove its
  metadata when the whole dict is spread into a ``call_model`` request.
- String literals that are *data* (status ``reason``, ``directive``, situation
  rule ``mode``: ``"activateWhen"`` / ``"deactivateWhen"``) keep upstream's
  values verbatim, like event type strings elsewhere in this port.
- Upstream ``serverTool(config, { id })`` stamps a stable ``id`` on every server
  tool. This port's ``server_tool`` has no ``id`` option yet, so
  :func:`get_server_tool_id` applies the same rule: an explicit non-empty
  ``id`` key on the tool wins, else ``f"server:{config['type']}"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

from typing_extensions import TypeAlias, TypedDict

from .tool_types import ConversationState, get_tool_function, is_server_tool


class _ToolSetSnapshotMarker:
    """Sentinel type for :data:`TOOL_SET_SNAPSHOT` (upstream ``Symbol.for``)."""

    _instance: Optional[_ToolSetSnapshotMarker] = None

    def __new__(cls) -> _ToolSetSnapshotMarker:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "TOOL_SET_SNAPSHOT"

    def __reduce__(self) -> str:
        return "TOOL_SET_SNAPSHOT"


#: Identifies dicts produced by the tool-set module (upstream ``TOOL_SET_SNAPSHOT``
#: in ``async-params.ts``). A non-string dict key, so it survives ``{**snapshot}``
#: but can never collide with a request field name.
TOOL_SET_SNAPSHOT: Any = _ToolSetSnapshotMarker()

_TOOL_SET_SNAPSHOT_METADATA_KEYS = ("enabled", "disabled", "status_by_tool", "call_model")


def strip_tool_set_snapshot_metadata(request: Dict[Any, Any]) -> None:
    """Remove tool-set metadata only from marked snapshots or their spreads.

    Port of upstream ``stripToolSetSnapshotMetadata`` (``async-params.ts``).
    Mutates ``request`` in place. Unmarked requests are left untouched, so
    identically named user fields survive.
    """
    if request.get(TOOL_SET_SNAPSHOT) is not True:
        return
    for key in _TOOL_SET_SNAPSHOT_METADATA_KEYS:
        request.pop(key, None)
    request.pop(TOOL_SET_SNAPSHOT, None)


# ─── public types ────────────────────────────────────────────────────────────


class ActivationInput(TypedDict, total=False):
    """Input handed to every activation predicate."""

    state: ConversationState
    context: Mapping[str, Any]


ActivationPredicate: TypeAlias = Callable[[ActivationInput], Any]

ConditionalMode = Literal["activateWhen", "deactivateWhen"]

StatusReason = Literal["default", "activate", "deactivate", "activateWhen", "deactivateWhen", "situation"]

Directive = Literal["activate", "deactivate", "activateWhen", "deactivateWhen"]


class SituationConditionalRuleObject(TypedDict, total=False):
    mode: ConditionalMode
    predicate: ActivationPredicate


SituationConditionalRule: TypeAlias = Union[ActivationPredicate, SituationConditionalRuleObject]


class SituationConfig(TypedDict, total=False):
    """Declarative fixed partition overlay for one named situation.

    Ids not listed keep the base ToolSet partition after the overlay.
    ``conditional`` rules default to ``activateWhen`` mode; use
    ``{"mode": "deactivateWhen", "predicate": fn}`` for the reverse default.
    """

    enabled: Sequence[str]
    disabled: Sequence[str]
    conditional: Mapping[str, Optional[SituationConditionalRule]]


class ToolStatusEntry(TypedDict, total=False):
    #: Whether the tool resolved active.
    enabled: bool
    reason: StatusReason
    #: The last applicable directive before predicates ran. Absent when the
    #: tool is still at its construction default.
    directive: Directive
    #: True when the final state depended on evaluating a runtime predicate.
    predicate: bool


@dataclass(frozen=True)
class ResolvedToolSnapshot:
    """What :meth:`ToolSet.resolve` / :meth:`ToolSet.resolve_situation` return."""

    #: Active tools only, construction order preserved.
    tools: List[Any]
    #: Active client tool names only (``call_model`` ``active_tools`` wire
    #: format). Server tool ids are omitted.
    active_tools: List[str]
    #: Spread-safe ``call_model`` input: exactly ``{"tools", "active_tools"}``.
    call_model: Dict[str, Any]
    #: Ids that resolved active (client + server).
    enabled: List[str]
    #: Ids that resolved inactive (client + server).
    disabled: List[str]
    #: Exhaustive id -> status entry; every known id is present.
    status_by_tool: Dict[str, ToolStatusEntry] = field(default_factory=dict)


# ─── internals ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _ActivationEntry:
    kind: Literal["static", "activateWhen", "deactivateWhen"]
    source: Literal["default", "activate", "deactivate", "activateWhen", "deactivateWhen", "situation"]
    active: bool = False
    predicate: Optional[ActivationPredicate] = None


@dataclass(frozen=True)
class _SituationRuntime:
    enabled: Tuple[str, ...]
    disabled: Tuple[str, ...]
    conditional: Tuple[Tuple[str, ConditionalMode, ActivationPredicate], ...]


@dataclass(frozen=True)
class _IndexedTools:
    ordered_tools: Tuple[Any, ...]
    ordered_ids: Tuple[str, ...]
    tool_by_id: Dict[str, Any]
    client_names: frozenset[str]
    server_ids: frozenset[str]


def get_server_tool_id(tool: Mapping[str, Any]) -> str:
    """Stable tool-set id of a server tool (upstream ``defaultServerId``).

    An explicit non-empty string ``id`` wins; otherwise ``server:{config.type}``.
    The prefix guarantees it can never collide with a client function name.
    """
    explicit = tool.get("id")
    if isinstance(explicit, str) and len(explicit) > 0:
        return explicit
    config = tool.get("config")
    tool_type = config.get("type") if isinstance(config, Mapping) else None
    return f"server:{tool_type}"


def get_tool_id(tool: Mapping[str, Any]) -> str:
    """Addressable tool-set id: server id for server tools, else ``function.name``."""
    if is_server_tool(tool):
        return get_server_tool_id(tool)
    return str(get_tool_function(tool).get("name"))


def _index_tools(tools: Sequence[Any]) -> _IndexedTools:
    tool_by_id: Dict[str, Any] = {}
    ordered_ids: List[str] = []
    client_names: set[str] = set()
    server_ids: set[str] = set()
    for t in tools:
        tid = get_tool_id(t)
        if tid in tool_by_id:
            raise ValueError(f'Duplicate tool ID: "{tid}"')
        tool_by_id[tid] = t
        ordered_ids.append(tid)
        if is_server_tool(t):
            server_ids.add(tid)
        else:
            client_names.add(tid)
    return _IndexedTools(
        ordered_tools=tuple(tools),
        ordered_ids=tuple(ordered_ids),
        tool_by_id=tool_by_id,
        client_names=frozenset(client_names),
        server_ids=frozenset(server_ids),
    )


def _to_id_list(names: Union[str, Sequence[str]]) -> List[str]:
    return [names] if isinstance(names, str) else list(names)


def _normalize_conditional_rule(situation: str, tool_id: str, rule: Any) -> Tuple[ConditionalMode, ActivationPredicate]:
    if callable(rule):
        return "activateWhen", rule
    mode = rule.get("mode") if isinstance(rule, Mapping) else None
    if (
        not isinstance(rule, Mapping)
        or not callable(rule.get("predicate"))
        or (mode is not None and mode not in ("activateWhen", "deactivateWhen"))
    ):
        raise ValueError(
            f'Situation "{situation}": conditional rule for tool "{tool_id}" '
            "must be a function or { mode, predicate } object"
        )
    return (mode or "activateWhen"), rule["predicate"]


class ToolSet:
    """Immutable-by-default stateful set of tools.

    Each tool id is enabled (default / :meth:`activate`), disabled
    (:meth:`deactivate`) or conditional (:meth:`activate_when` /
    :meth:`deactivate_when`); the most recent directive per id wins. Named
    situations (:meth:`define_situations`) overlay that base partition.

    Immutable instances return a new ``ToolSet`` from every mutator; mutable
    instances (``mutable=True``) mutate in place and return ``self``.
    """

    __slots__ = ("_index", "_activation", "_situations", "_mutable")

    def __init__(
        self,
        index: _IndexedTools,
        activation: Dict[str, _ActivationEntry],
        situations: Dict[str, _SituationRuntime],
        mutable: bool,
    ) -> None:
        self._index = index
        self._activation = activation
        self._situations = situations
        self._mutable = mutable

    @classmethod
    def create(cls, *, tools: Sequence[Any], mutable: bool = False) -> ToolSet:
        """Internal factory. Prefer :func:`create_tool_set`."""
        return cls(_index_tools(tools), {}, {}, bool(mutable))

    @property
    def tools(self) -> Tuple[Any, ...]:
        """All tools in construction order, regardless of activation state."""
        return self._index.ordered_tools

    @property
    def mutable(self) -> bool:
        return self._mutable

    def _assert_known(self, tool_id: str) -> None:
        if tool_id not in self._index.tool_by_id:
            raise ValueError(f'Unknown tool: "{tool_id}"')

    def _with_partition_mutation(self, mutate: Callable[[Dict[str, _ActivationEntry]], None]) -> ToolSet:
        if self._mutable:
            mutate(self._activation)
            return self
        next_activation = dict(self._activation)
        mutate(next_activation)
        return ToolSet(self._index, next_activation, self._situations, False)

    def activate(self, names: Union[str, Sequence[str]]) -> ToolSet:
        ids = _to_id_list(names)
        for n in ids:
            self._assert_known(n)

        def mutate(activation: Dict[str, _ActivationEntry]) -> None:
            for n in ids:
                activation[n] = _ActivationEntry(kind="static", active=True, source="activate")

        return self._with_partition_mutation(mutate)

    def deactivate(self, names: Union[str, Sequence[str]]) -> ToolSet:
        ids = _to_id_list(names)
        for n in ids:
            self._assert_known(n)

        def mutate(activation: Dict[str, _ActivationEntry]) -> None:
            for n in ids:
                activation[n] = _ActivationEntry(kind="static", active=False, source="deactivate")

        return self._with_partition_mutation(mutate)

    def activate_when(
        self,
        name_or_map: Union[str, Mapping[str, Optional[ActivationPredicate]]],
        predicate: Optional[ActivationPredicate] = None,
    ) -> ToolSet:
        """Inactive unless ``predicate(input) is True`` at resolve time.

        Accepts ``(name, predicate)`` or a ``{name: predicate}`` map.
        """
        entries = self._normalize_predicate_arg(name_or_map, predicate)

        def mutate(activation: Dict[str, _ActivationEntry]) -> None:
            for n, p in entries:
                activation[n] = _ActivationEntry(kind="activateWhen", predicate=p, source="activateWhen")

        return self._with_partition_mutation(mutate)

    def deactivate_when(
        self,
        name_or_map: Union[str, Mapping[str, Optional[ActivationPredicate]]],
        predicate: Optional[ActivationPredicate] = None,
    ) -> ToolSet:
        """Active unless ``predicate(input) is True`` at resolve time.

        Accepts ``(name, predicate)`` or a ``{name: predicate}`` map.
        """
        entries = self._normalize_predicate_arg(name_or_map, predicate)

        def mutate(activation: Dict[str, _ActivationEntry]) -> None:
            for n, p in entries:
                activation[n] = _ActivationEntry(kind="deactivateWhen", predicate=p, source="deactivateWhen")

        return self._with_partition_mutation(mutate)

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
        entries = [(n, p) for n, p in name_or_map.items() if callable(p)]
        for n, _ in entries:
            self._assert_known(n)
        return entries

    def define_situations(self, situations: Mapping[str, SituationConfig]) -> ToolSet:
        """Register named declarative situations.

        Each situation overlays the base partition -- ids it does not mention
        keep whatever the base set declares. Replaces any previously defined
        situations (last-call-wins at the registry level).
        """
        next_situations: Dict[str, _SituationRuntime] = {}
        for name, config in situations.items():
            enabled = list(config.get("enabled") or [])
            disabled = list(config.get("disabled") or [])
            conditional_entries = [
                (tid, rule) for tid, rule in (config.get("conditional") or {}).items() if rule is not None
            ]

            seen: set[str] = set()

            def record(tool_id: str) -> None:
                self._assert_known(tool_id)
                if tool_id in seen:
                    raise ValueError(
                        f'Situation "{name}" lists tool "{tool_id}" more than once (across enabled/disabled/conditional)'
                    )
                seen.add(tool_id)

            for tid in enabled:
                record(tid)
            for tid in disabled:
                record(tid)
            for tid, _ in conditional_entries:
                record(tid)

            conditional: List[Tuple[str, ConditionalMode, ActivationPredicate]] = []
            for tid, rule in conditional_entries:
                mode, pred = _normalize_conditional_rule(name, tid, rule)
                conditional.append((tid, mode, pred))

            next_situations[name] = _SituationRuntime(
                enabled=tuple(enabled), disabled=tuple(disabled), conditional=tuple(conditional)
            )

        if self._mutable:
            self._situations.clear()
            self._situations.update(next_situations)
            return self
        return ToolSet(self._index, dict(self._activation), next_situations, False)

    def resolve(self, input: Optional[ActivationInput] = None) -> ResolvedToolSnapshot:
        """Resolve against the base partition (no situation overlay)."""
        return self._resolve_with_activation(self._activation, input)

    def infer_tools(self, input: Optional[ActivationInput] = None) -> Dict[Any, Any]:
        """Back-compat alias for :meth:`resolve`. Prefer ``resolve`` for new code.

        Returns a ``dict`` with ``tools``, ``active_tools``, ``enabled``,
        ``disabled``, ``status_by_tool`` and the :data:`TOOL_SET_SNAPSHOT`
        marker. Only ``tools`` and ``active_tools`` are meant to reach
        ``call_model``; the marker lets :func:`strip_tool_set_snapshot_metadata`
        drop the rest when the whole dict is spread into a request.
        """
        snapshot = self.resolve(input)
        return {
            "tools": list(snapshot.tools),
            "active_tools": list(snapshot.active_tools),
            "enabled": snapshot.enabled,
            "disabled": snapshot.disabled,
            "status_by_tool": snapshot.status_by_tool,
            TOOL_SET_SNAPSHOT: True,
        }

    def resolve_situation(self, name: str, input: Optional[ActivationInput] = None) -> ResolvedToolSnapshot:
        """Resolve a previously defined named situation."""
        situation = self._situations.get(name)
        if situation is None:
            raise ValueError(f'Unknown situation: "{name}"')

        activation = dict(self._activation)
        for tid in situation.enabled:
            activation[tid] = _ActivationEntry(kind="static", active=True, source="situation")
        for tid in situation.disabled:
            activation[tid] = _ActivationEntry(kind="static", active=False, source="situation")
        for tid, mode, pred in situation.conditional:
            activation[tid] = _ActivationEntry(kind=mode, predicate=pred, source="situation")
        return self._resolve_with_activation(activation, input)

    def _resolve_with_activation(
        self, activation: Mapping[str, _ActivationEntry], input: Optional[ActivationInput]
    ) -> ResolvedToolSnapshot:
        resolved_input: ActivationInput = input if input is not None else {}
        tools: List[Any] = []
        active_tools: List[str] = []
        enabled: List[str] = []
        disabled: List[str] = []
        status_by_tool: Dict[str, ToolStatusEntry] = {}

        for tid in self._index.ordered_ids:
            tool = self._index.tool_by_id.get(tid)
            if tool is None:
                continue
            entry = activation.get(tid)
            active = self._evaluate(entry, resolved_input)
            status_by_tool[tid] = self._to_status_entry(active, entry)
            if active:
                tools.append(tool)
                enabled.append(tid)
                if not is_server_tool(tool):
                    active_tools.append(tid)
            else:
                disabled.append(tid)

        return ResolvedToolSnapshot(
            tools=tools,
            active_tools=active_tools,
            call_model={"tools": tools, "active_tools": active_tools},
            enabled=enabled,
            disabled=disabled,
            status_by_tool=status_by_tool,
        )

    @staticmethod
    def _evaluate(entry: Optional[_ActivationEntry], input: ActivationInput) -> bool:
        if entry is None:
            return True
        if entry.kind == "static":
            return entry.active
        assert entry.predicate is not None
        if entry.kind == "activateWhen":
            return entry.predicate(input) is True
        return entry.predicate(input) is not True

    @staticmethod
    def _to_status_entry(active: bool, entry: Optional[_ActivationEntry]) -> ToolStatusEntry:
        if entry is None:
            return {"enabled": active, "reason": "default"}
        if entry.kind == "static":
            directive: Directive = "activate" if entry.active else "deactivate"
            reason: StatusReason
            if entry.source == "situation":
                reason = "situation"
            elif entry.source == "default":
                reason = "default"
            else:
                reason = directive
            return {"enabled": active, "reason": reason, "directive": directive}
        cond_directive: Directive = entry.kind
        cond_reason: StatusReason = "situation" if entry.source == "situation" else cond_directive
        return {"enabled": active, "reason": cond_reason, "directive": cond_directive, "predicate": True}

    def clone(self, *, mutable: Optional[bool] = None) -> ToolSet:
        """Copy state into a fresh, independent instance.

        ``mutable=None`` inherits the source's mode.
        """
        next_mutable = self._mutable if mutable is None else bool(mutable)
        return ToolSet(self._index, dict(self._activation), dict(self._situations), next_mutable)


def create_tool_set(tools: Sequence[Any], *, mutable: bool = False) -> ToolSet:
    """Construct a :class:`ToolSet` (upstream ``createToolSet``).

    Every tool starts enabled. Raises ``ValueError`` on duplicate tool ids.
    """
    return ToolSet.create(tools=tools, mutable=mutable)


__all__ = [
    "TOOL_SET_SNAPSHOT",
    "ActivationInput",
    "ActivationPredicate",
    "ConditionalMode",
    "Directive",
    "ResolvedToolSnapshot",
    "SituationConditionalRule",
    "SituationConditionalRuleObject",
    "SituationConfig",
    "StatusReason",
    "ToolSet",
    "ToolStatusEntry",
    "create_tool_set",
    "get_server_tool_id",
    "get_tool_id",
    "strip_tool_set_snapshot_metadata",
]
