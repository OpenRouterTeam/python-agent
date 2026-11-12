"""Port of upstream `tests/unit/next-turn-params.test.ts` AND
`src/lib/next-turn-params.test.ts` (both map here).

Python next-turn param keys are snake_case (`tool_choice`, `max_output_tokens`,
`top_p`, `top_k`, ...) and a tool declares them via
`tool(next_turn_params={"key": fn(args, ctx)})`.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel

from openrouter_agent import tool
from openrouter_agent.next_turn_params import (
    apply_next_turn_params_to_request,
    build_next_turn_params_context,
    execute_next_turn_params_functions,
)
from openrouter_agent.tool_types import ParsedToolCall


class EmptyInput(BaseModel):
    pass


class QInput(BaseModel):
    q: str


class DepthInput(BaseModel):
    depth: int


def create_request(**overrides: Any) -> Dict[str, Any]:
    request: Dict[str, Any] = {"model": "openai/gpt-4", "input": [{"role": "user", "content": "hello"}]}
    request.update(overrides)
    return request


def create_tool_call(name: str, args: Any) -> ParsedToolCall:
    return ParsedToolCall(id=f"call_{name}", name=name, arguments=args)


async def _done(params: Any, ctx: Any) -> str:
    return "done"


# ---------------------------------------------------------------------------
# build_next_turn_params_context (tests/unit/next-turn-params.test.ts)
# ---------------------------------------------------------------------------


def test_build_context_extracts_all_fields_from_a_fully_populated_request() -> None:
    request = create_request(
        models=["openai/gpt-4", "anthropic/claude"],
        temperature=0.7,
        max_output_tokens=500,
        top_p=0.9,
        top_k=40,
        instructions="Be terse",
    )

    context = build_next_turn_params_context(request)

    assert context == {
        "input": request["input"],
        "model": "openai/gpt-4",
        "models": ["openai/gpt-4", "anthropic/claude"],
        "temperature": 0.7,
        "max_output_tokens": 500,
        "top_p": 0.9,
        "top_k": 40,
        "instructions": "Be terse",
    }


def test_build_context_applies_defaults_for_missing_fields() -> None:
    context = build_next_turn_params_context({})

    # Upstream `topK: undefined` and `toolChoice: undefined` are absent keys here.
    assert context == {
        "input": [],
        "model": "",
        "models": [],
        "temperature": None,
        "max_output_tokens": None,
        "top_p": None,
        "instructions": None,
    }


# ---------------------------------------------------------------------------
# execute_next_turn_params_functions (tests/unit/next-turn-params.test.ts)
# ---------------------------------------------------------------------------


async def test_execute_returns_an_empty_dict_when_no_tools_have_next_turn_params() -> None:
    plain_tool = tool(name="plain", input_schema=QInput, execute=_done)

    result = await execute_next_turn_params_functions(
        [create_tool_call("plain", {"q": "x"})], [plain_tool], create_request()
    )

    assert result == {}


async def test_execute_computes_parameters_from_tool_call_arguments_and_context() -> None:
    search_tool = tool(
        name="search",
        input_schema=DepthInput,
        execute=_done,
        next_turn_params={
            "temperature": lambda params, ctx: 0.1 if params["depth"] > 3 else 0.9,
            "instructions": lambda params, ctx: f"{ctx.get('instructions') or ''} depth={params['depth']}",
        },
    )

    result = await execute_next_turn_params_functions(
        [create_tool_call("search", {"depth": 5})], [search_tool], create_request(instructions="Base.")
    )

    assert result == {"temperature": 0.1, "instructions": "Base. depth=5"}


async def test_execute_composes_later_tools_see_modifications_made_by_earlier_tools() -> None:
    first_tool = tool(
        name="first", input_schema=EmptyInput, execute=_done, next_turn_params={"temperature": lambda p, c: 0.5}
    )
    second_tool = tool(
        name="second",
        input_schema=EmptyInput,
        execute=_done,
        next_turn_params={"temperature": lambda p, c: (c.get("temperature") or 0) + 0.3},
    )

    result = await execute_next_turn_params_functions(
        [create_tool_call("first", {}), create_tool_call("second", {})],
        [first_tool, second_tool],
        create_request(temperature=0.7),
    )

    # first sets 0.5; second sees 0.5 (not the original 0.7) and adds 0.3
    assert result["temperature"] == pytest.approx(0.8)


async def test_execute_awaits_async_next_turn_params_functions() -> None:
    async def pick_model(params: Any, ctx: Any) -> str:
        return "openai/gpt-5"

    async_tool = tool(name="async_tool", input_schema=EmptyInput, execute=_done, next_turn_params={"model": pick_model})

    result = await execute_next_turn_params_functions(
        [create_tool_call("async_tool", {})], [async_tool], create_request()
    )

    assert result == {"model": "openai/gpt-5"}


async def test_execute_runs_a_tool_once_per_call_when_it_was_called_multiple_times() -> None:
    invocations: List[int] = []

    def count(params: Any, ctx: Any) -> int:
        invocations.append(1)
        return len(invocations)

    counting_tool = tool(
        name="counter", input_schema=EmptyInput, execute=_done, next_turn_params={"max_output_tokens": count}
    )

    result = await execute_next_turn_params_functions(
        [create_tool_call("counter", {}), create_tool_call("counter", {})], [counting_tool], create_request()
    )

    assert len(invocations) == 2
    assert result["max_output_tokens"] == 2


async def test_execute_ignores_tools_that_were_not_called() -> None:
    called_tool = tool(
        name="called", input_schema=EmptyInput, execute=_done, next_turn_params={"temperature": lambda p, c: 0.1}
    )
    uncalled_tool = tool(
        name="uncalled", input_schema=EmptyInput, execute=_done, next_turn_params={"temperature": lambda p, c: 0.9}
    )

    result = await execute_next_turn_params_functions(
        [create_tool_call("called", {})], [called_tool, uncalled_tool], create_request()
    )

    assert result == {"temperature": 0.1}


async def test_execute_skips_server_tools_entirely() -> None:
    invoked: List[int] = []

    def server_fn(params: Any, ctx: Any) -> float:
        invoked.append(1)
        return 0.0

    server_tool_like = {
        "_brand": "server-tool",
        "function": {"name": "server_search", "next_turn_params": {"temperature": server_fn}},
    }

    result = await execute_next_turn_params_functions(
        [create_tool_call("server_search", {})], [server_tool_like], create_request(temperature=0.7)
    )

    assert result == {}
    assert invoked == []


async def test_execute_throws_a_descriptive_error_when_tool_call_arguments_are_not_an_object() -> None:
    strict_tool = tool(
        name="strict", input_schema=EmptyInput, execute=_done, next_turn_params={"temperature": lambda p, c: 0.5}
    )

    with pytest.raises(Exception, match="Tool call arguments for strict must be an object, got array"):
        await execute_next_turn_params_functions(
            [create_tool_call("strict", ["not", "an", "object"])], [strict_tool], create_request()
        )
    with pytest.raises(Exception, match="Tool call arguments for strict must be an object, got string"):
        await execute_next_turn_params_functions(
            [create_tool_call("strict", "a-string")], [strict_tool], create_request()
        )


async def test_execute_warns_and_skips_invalid_next_turn_params_keys() -> None:
    odd_tool = tool(
        name="odd",
        input_schema=EmptyInput,
        execute=_done,
        next_turn_params={"bogus_key": lambda p, c: 42, "temperature": lambda p, c: 0.3},
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await execute_next_turn_params_functions([create_tool_call("odd", {})], [odd_tool], create_request())

    assert result == {"temperature": 0.3}
    messages = [str(w.message) for w in caught]
    assert sum('Invalid next_turn_params key "bogus_key" in tool "odd"' in m for m in messages) == 1


async def test_execute_ignores_non_function_values_in_next_turn_params() -> None:
    weird_tool = tool(name="weird", input_schema=EmptyInput, execute=_done, next_turn_params={"temperature": 0.5})

    result = await execute_next_turn_params_functions([create_tool_call("weird", {})], [weird_tool], create_request())

    assert result == {}


# ---------------------------------------------------------------------------
# apply_next_turn_params_to_request (src/lib/next-turn-params.test.ts)
# ---------------------------------------------------------------------------


def test_apply_passes_through_non_null_values_unchanged() -> None:
    result = apply_next_turn_params_to_request(
        create_request(), {"temperature": 0.5, "max_output_tokens": 1000, "instructions": "Be helpful"}
    )

    assert result["temperature"] == 0.5
    assert result["max_output_tokens"] == 1000
    assert result["instructions"] == "Be helpful"


def test_apply_converts_null_values_to_absent() -> None:
    result = apply_next_turn_params_to_request(
        create_request(temperature=0.7), {"temperature": None, "max_output_tokens": None, "instructions": None}
    )

    # Upstream maps null -> undefined (spread overrides with undefined). The
    # Python equivalent of an undefined property is an absent key.
    assert "temperature" not in result
    assert "max_output_tokens" not in result
    assert "instructions" not in result
    assert result["model"] == "openai/gpt-4"


def test_apply_preserves_original_request_fields_not_in_computed_params() -> None:
    result = apply_next_turn_params_to_request(
        create_request(model="anthropic/claude-3", temperature=0.9), {"max_output_tokens": 500}
    )

    assert result["model"] == "anthropic/claude-3"
    assert result["temperature"] == 0.9
    assert result["max_output_tokens"] == 500


def test_apply_handles_empty_computed_params_without_changing_the_request() -> None:
    request = create_request(temperature=0.5)

    result = apply_next_turn_params_to_request(request, {})

    assert result == request
    assert result["temperature"] == 0.5
    assert result["model"] == "openai/gpt-4"


def test_apply_handles_mixed_null_and_non_null_values() -> None:
    result = apply_next_turn_params_to_request(
        create_request(temperature=0.7, top_p=0.9),
        {"temperature": None, "top_p": 0.5, "max_output_tokens": None, "instructions": "Updated instructions"},
    )

    assert "temperature" not in result
    assert result["top_p"] == 0.5
    assert "max_output_tokens" not in result
    assert result["instructions"] == "Updated instructions"


def test_apply_preserves_zero_as_a_valid_non_null_value() -> None:
    result = apply_next_turn_params_to_request(create_request(), {"temperature": 0, "max_output_tokens": 0})

    assert result["temperature"] == 0
    assert result["max_output_tokens"] == 0


def test_apply_preserves_empty_string_as_a_valid_non_null_value() -> None:
    instructions: Optional[str] = ""
    result = apply_next_turn_params_to_request(create_request(), {"instructions": instructions})

    assert result["instructions"] == ""


# tool_choice is how a tool-search tool widens the model's reach mid-run: the
# `tools` array stays identical (preserving the provider's prompt-cache prefix)
# while `{"type": "allowed_tools", "tools": [...]}` grows.


def allowed(names: List[str]) -> Dict[str, Any]:
    return {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": n} for n in names]}


def test_apply_replaces_tool_choice_with_a_widened_allowed_tools_set() -> None:
    result = apply_next_turn_params_to_request(
        create_request(tool_choice=allowed(["tool_search"])),
        {"tool_choice": allowed(["tool_search", "get_weather"])},
    )

    assert result["tool_choice"] == allowed(["tool_search", "get_weather"])


def test_apply_leaves_the_tools_array_untouched_so_the_prompt_cache_prefix_survives() -> None:
    tools = [
        {"type": "function", "name": "tool_search", "parameters": {}},
        {"type": "function", "name": "get_weather", "parameters": {}},
    ]
    request = create_request(tools=tools, tool_choice=allowed(["tool_search"]))

    result = apply_next_turn_params_to_request(request, {"tool_choice": allowed(["tool_search", "get_weather"])})

    # Upstream `toBe(tools)`: same object identity.
    assert result["tools"] is tools


def test_apply_leaves_tool_choice_alone_when_no_tool_computed_one() -> None:
    result = apply_next_turn_params_to_request(
        create_request(tool_choice=allowed(["tool_search"])), {"temperature": 0.2}
    )

    assert result["tool_choice"] == allowed(["tool_search"])
    assert result["temperature"] == 0.2
