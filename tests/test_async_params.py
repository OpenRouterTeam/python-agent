"""Tests for async parameter resolution."""

import pytest

from openrouter_agent import (
    TurnContext,
    has_async_functions,
    resolve_async_functions,
)


@pytest.mark.anyio
async def test_resolve_static_values():
    ctx = TurnContext(number_of_turns=0)
    result = await resolve_async_functions(
        {"model": "gpt-4", "temperature": 0.7}, ctx
    )
    assert result["model"] == "gpt-4"
    assert result["temperature"] == 0.7


@pytest.mark.anyio
async def test_resolve_sync_callable():
    ctx = TurnContext(number_of_turns=2)
    result = await resolve_async_functions(
        {"model": lambda ctx: f"model-{ctx.number_of_turns}"}, ctx
    )
    assert result["model"] == "model-2"


@pytest.mark.anyio
async def test_resolve_async_callable():
    ctx = TurnContext(number_of_turns=3)

    async def get_model(ctx: TurnContext) -> str:
        return f"async-model-{ctx.number_of_turns}"

    result = await resolve_async_functions({"model": get_model}, ctx)
    assert result["model"] == "async-model-3"


@pytest.mark.anyio
async def test_client_only_fields_not_resolved():
    ctx = TurnContext()
    def sentinel(ctx: object) -> str:
        return "should not be called"
    result = await resolve_async_functions(
        {"tools": sentinel, "stop_when": sentinel}, ctx
    )
    # Client-only fields should be passed through as-is
    assert result["tools"] is sentinel
    assert result["stop_when"] is sentinel


def test_has_async_functions():
    assert has_async_functions({"model": lambda ctx: "x"})
    assert not has_async_functions({"model": "gpt-4"})
    # Client-only fields don't count
    assert not has_async_functions({"tools": lambda ctx: []})
