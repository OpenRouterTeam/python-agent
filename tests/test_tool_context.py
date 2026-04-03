"""Tests for ToolContextStore and context building."""

import pytest

from openrouter_agent import (
    SHARED_CONTEXT_KEY,
    ToolContextStore,
    TurnContext,
    build_tool_execute_context,
)


def test_context_store_get_set():
    store = ToolContextStore()
    store.set_tool_context("my_tool", {"count": 0})
    assert store.get_tool_context("my_tool") == {"count": 0}


def test_context_store_merge():
    store = ToolContextStore({"my_tool": {"count": 0, "name": "test"}})
    store.merge_tool_context("my_tool", {"count": 1})
    ctx = store.get_tool_context("my_tool")
    assert ctx["count"] == 1
    assert ctx["name"] == "test"


def test_context_store_subscribe():
    store = ToolContextStore()
    notifications = []
    unsub = store.subscribe(lambda: notifications.append(True))
    store.set_tool_context("x", {"a": 1})
    assert len(notifications) == 1
    unsub()
    store.set_tool_context("x", {"a": 2})
    assert len(notifications) == 1  # no more notifications after unsub


def test_context_store_snapshot():
    store = ToolContextStore({"a": {"x": 1}, "b": {"y": 2}})
    snap = store.get_snapshot()
    assert snap == {"a": {"x": 1}, "b": {"y": 2}}
    # Snapshot is a deep copy
    snap["a"]["x"] = 999
    assert store.get_tool_context("a") == {"x": 1}


def test_build_tool_execute_context():
    store = ToolContextStore({
        "my_tool": {"local_val": 42},
        SHARED_CONTEXT_KEY: {"shared_val": "hello"},
    })
    turn_ctx = TurnContext(number_of_turns=1)
    ctx = build_tool_execute_context(turn_ctx, store, "my_tool")
    assert ctx.local == {"local_val": 42}
    assert ctx.shared == {"shared_val": "hello"}
    assert ctx.number_of_turns == 1
    assert ctx.tool_name == "my_tool"


def test_build_tool_execute_context_set_context():
    store = ToolContextStore({"my_tool": {"val": 1}})
    turn_ctx = TurnContext()
    ctx = build_tool_execute_context(turn_ctx, store, "my_tool")
    ctx.set_context({"val": 2, "new": 3})
    assert store.get_tool_context("my_tool") == {"val": 2, "new": 3}


def test_local_is_readonly():
    """Directly mutating ctx.local should raise TypeError."""
    store = ToolContextStore({"my_tool": {"key": "original"}})
    turn_ctx = TurnContext()
    ctx = build_tool_execute_context(turn_ctx, store, "my_tool")

    with pytest.raises(TypeError):
        ctx.local["key"] = "val"  # type: ignore[index]


def test_shared_is_readonly():
    """Directly mutating ctx.shared should raise TypeError."""
    store = ToolContextStore({SHARED_CONTEXT_KEY: {"key": "original"}})
    turn_ctx = TurnContext()
    ctx = build_tool_execute_context(turn_ctx, store, "my_tool")

    with pytest.raises(TypeError):
        ctx.shared["key"] = "val"  # type: ignore[index]


def test_set_context_still_works():
    """set_context() should update the store even though local is readonly."""
    store = ToolContextStore({"my_tool": {"key": "original"}})
    turn_ctx = TurnContext()
    ctx = build_tool_execute_context(turn_ctx, store, "my_tool")
    ctx.set_context({"key": "val"})
    assert store.get_tool_context("my_tool") == {"key": "val"}


def test_set_shared_context_still_works():
    """set_shared_context() should update the store even though shared is readonly."""
    store = ToolContextStore({SHARED_CONTEXT_KEY: {"key": "original"}})
    turn_ctx = TurnContext()
    ctx = build_tool_execute_context(turn_ctx, store, "my_tool")
    ctx.set_shared_context({"key": "val"})
    assert store.get_tool_context(SHARED_CONTEXT_KEY) == {"key": "val"}


def test_reading_local_works():
    """Reading values from ctx.local should work normally."""
    store = ToolContextStore({"my_tool": {"key": "value"}})
    turn_ctx = TurnContext()
    ctx = build_tool_execute_context(turn_ctx, store, "my_tool")
    assert ctx.local["key"] == "value"
