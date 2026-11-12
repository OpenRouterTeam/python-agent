"""Ports tests/unit/tool-context.test.ts (delta + the ToolContextStore,
buildToolExecuteContext and resolveContext sections).

Delta at this sync: upstream added two `buildToolExecuteContext` cases about the
per-context never-abort `signal`. This port has no `signal` on the execute
context -- cancellation uses Python-native mechanisms (contract, Idiomatic
Divergences #5) -- so both are skipped as out of scope. The other delta hunk
(`tool<SharedCtx>()({...})` curried generic) is type-level only.

Signature note: upstream `buildToolExecuteContext(turnContext, store, toolName,
contextSchema, sharedSchema)`; Python
`build_tool_execute_context(tool, turn_context, store, shared_context_schema)`
reads name and `context_schema` from the tool definition.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from pydantic import BaseModel, ValidationError

from openrouter_agent import ToolContextStore, tool
from openrouter_agent.tool_context import build_tool_execute_context, resolve_context

TURN_CONTEXT: Dict[str, Any] = {"number_of_turns": 1}


class ApiKey(BaseModel):
    apiKey: str


class Count(BaseModel):
    count: float


class Empty(BaseModel):
    pass


def _tool(name: str, context_schema: Any = None) -> Dict[str, Any]:
    return tool(name=name, input_schema=Empty, context_schema=context_schema, execute=lambda params, ctx: None)


# -- ToolContextStore ------------------------------------------------------------


def test_store_initializes_with_provided_values() -> None:
    store = ToolContextStore({"get_weather": {"apiKey": "sk-123"}, "db_query": {"dbUrl": "postgres://..."}})

    assert store.get_snapshot() == {"get_weather": {"apiKey": "sk-123"}, "db_query": {"dbUrl": "postgres://..."}}


def test_get_snapshot_returns_a_copy() -> None:
    store = ToolContextStore({"my_tool": {"key": "value"}})

    first = store.get_snapshot()
    second = store.get_snapshot()

    assert first == second
    assert first is not second


def test_get_tool_context_by_name() -> None:
    store = ToolContextStore({"get_weather": {"apiKey": "sk-123"}})

    assert store.get_tool_context("get_weather") == {"apiKey": "sk-123"}
    assert store.get_tool_context("nonexistent") == {}


def test_set_tool_context_notifies_listeners_once() -> None:
    store = ToolContextStore({"get_weather": {"apiKey": "old"}})
    snapshots: List[Dict[str, Any]] = []
    store.subscribe(snapshots.append)

    store.set_tool_context("get_weather", {"apiKey": "new"})

    assert store.get_tool_context("get_weather") == {"apiKey": "new"}
    assert snapshots == [{"get_weather": {"apiKey": "new"}}]


def test_merge_tool_context_merges_partial_values_and_notifies_once() -> None:
    store = ToolContextStore({"my_tool": {"a": 1, "b": 2}})
    snapshots: List[Dict[str, Any]] = []
    store.subscribe(snapshots.append)

    store.merge_tool_context("my_tool", {"b": 3, "c": 4})

    assert store.get_tool_context("my_tool") == {"a": 1, "b": 3, "c": 4}
    assert len(snapshots) == 1


def test_store_supports_unsubscribing() -> None:
    store = ToolContextStore({})
    snapshots: List[Dict[str, Any]] = []
    unsubscribe = store.subscribe(snapshots.append)

    store.set_tool_context("my_tool", {"key": "value1"})
    assert len(snapshots) == 1

    unsubscribe()
    store.set_tool_context("my_tool", {"key": "value2"})
    assert len(snapshots) == 1


def test_get_tool_context_returns_a_defensive_copy() -> None:
    store = ToolContextStore({"my_tool": {"key": "value"}})

    returned = store.get_tool_context("my_tool")
    returned["key"] = "mutated"

    assert store.get_tool_context("my_tool") == {"key": "value"}


# -- buildToolExecuteContext -----------------------------------------------------


@pytest.mark.skip(reason="Out of scope: cancellation signals (no `signal` on the Python execute context).")
def test_gives_each_context_its_own_never_abort_signal() -> None:
    raise AssertionError("not ported")


@pytest.mark.skip(reason="Out of scope: cancellation signals (no `signal` on the Python execute context).")
def test_prefers_a_caller_supplied_signal_over_a_fresh_one() -> None:
    raise AssertionError("not ported")


def test_returns_flat_context_with_local_shared_and_set_context() -> None:
    store = ToolContextStore({"get_weather": {"apiKey": "sk-123"}})

    ctx = build_tool_execute_context(_tool("get_weather", ApiKey), TURN_CONTEXT, store)

    # Turn-context fields are merged in.
    assert ctx["number_of_turns"] == 1
    assert ctx["local"]["apiKey"] == "sk-123"
    assert dict(ctx["shared"]) == {}
    assert callable(ctx["set_context"])
    assert callable(ctx["set_shared_context"])


def test_set_context_mutates_the_store() -> None:
    store = ToolContextStore({"get_weather": {"apiKey": "old"}})
    ctx = build_tool_execute_context(_tool("get_weather", ApiKey), TURN_CONTEXT, store)

    ctx["set_context"]({"apiKey": "new"})

    assert store.get_tool_context("get_weather") == {"apiKey": "new"}
    # `local` is a live view.
    assert ctx["local"]["apiKey"] == "new"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "PORT GAP: upstream setContext validates each kept key against the contextSchema shape and "
        "throws on mismatch (tool-context.ts:113-133 validatePartialAgainstSchema, z4.parse at :129). "
        "Python build_tool_execute_context's set_context (tool_context.py "
        "_filter_schema_keys) only filters keys by model_fields and never validates values."
    ),
)
def test_set_context_validates_values_against_the_schema() -> None:
    store = ToolContextStore({"my_tool": {"count": 0}})
    ctx = build_tool_execute_context(_tool("my_tool", Count), TURN_CONTEXT, store)

    with pytest.raises(ValidationError):
        ctx["set_context"]({"count": "not a number"})


def test_set_context_filters_out_keys_not_in_the_schema() -> None:
    store = ToolContextStore({"my_tool": {"apiKey": "old", "extra": "extra"}})
    ctx = build_tool_execute_context(_tool("my_tool", ApiKey), TURN_CONTEXT, store)

    ctx["set_context"]({"apiKey": "new", "notInSchema": "ignored"})

    assert store.get_tool_context("my_tool") == {"apiKey": "new", "extra": "extra"}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "PORT GAP: upstream setContext returns early (no-op) when the tool has no contextSchema "
        "(tool-context.ts:237-240 `if (!store || !schema) return`; same for setSharedContext). "
        "Python _filter_schema_keys (tool_context.py) returns the partial unfiltered when the "
        "schema is None, so set_context merges it into the store."
    ),
)
def test_set_context_is_a_no_op_when_the_schema_is_undefined() -> None:
    store = ToolContextStore({"my_tool": {"key": "value"}})
    ctx = build_tool_execute_context(_tool("my_tool"), TURN_CONTEXT, store)

    ctx["set_context"]({"key": "new"})

    assert store.get_tool_context("my_tool") == {"key": "value"}


def test_returns_empty_context_when_the_store_is_undefined() -> None:
    ctx = build_tool_execute_context(_tool("my_tool"), TURN_CONTEXT, None)

    assert ctx["number_of_turns"] == 1
    assert dict(ctx["local"]) == {}


def test_set_context_notifies_store_listeners_once() -> None:
    store = ToolContextStore({"get_weather": {"apiKey": "old"}})
    ctx = build_tool_execute_context(_tool("get_weather", ApiKey), TURN_CONTEXT, store)
    snapshots: List[Dict[str, Any]] = []
    store.subscribe(snapshots.append)

    ctx["set_context"]({"apiKey": "new"})

    assert len(snapshots) == 1
    assert snapshots[0]["get_weather"] == {"apiKey": "new"}


# -- resolveContext --------------------------------------------------------------


async def test_resolve_context_returns_empty_for_none() -> None:
    assert await resolve_context(None, {"number_of_turns": 0}) == {}


async def test_resolve_context_returns_static_value_as_is() -> None:
    value = {"get_weather": {"apiKey": "sk-123"}}

    assert await resolve_context(value, {"number_of_turns": 0}) == value


async def test_resolve_context_resolves_a_sync_function() -> None:
    def build(turn: Dict[str, Any]) -> Dict[str, Any]:
        return {"my_tool": {"turnNum": turn["number_of_turns"]}}

    assert await resolve_context(build, {"number_of_turns": 0}) == {"my_tool": {"turnNum": 0}}


async def test_resolve_context_resolves_an_async_function() -> None:
    async def build(turn: Dict[str, Any]) -> Dict[str, Any]:
        return {"my_tool": {"turnNum": turn["number_of_turns"] + 1}}

    assert await resolve_context(build, {"number_of_turns": 0}) == {"my_tool": {"turnNum": 1}}


# -- tool() rejects reserved name "shared" ---------------------------------------


def test_tool_rejects_the_reserved_name_shared() -> None:
    with pytest.raises(ValueError, match="reserved"):
        _tool("shared")


def test_tool_accepts_other_names() -> None:
    assert _tool("shared_thing")["function"]["name"] == "shared_thing"
