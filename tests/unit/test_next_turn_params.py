"""Ported from upstream tests/unit/next-turn-params.test.ts and
src/lib/next-turn-params.test.ts (the `toolChoice` key, #114)."""

from __future__ import annotations

from typing import Any, List

from openrouter_agent import tool
from openrouter_agent.next_turn_params import (
    apply_next_turn_params_to_request,
    build_next_turn_params_context,
    execute_next_turn_params_functions,
)
from openrouter_agent.tool_types import ParsedToolCall


def test_context_carries_the_current_tool_choice() -> None:
    choice = {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "a"}]}

    context = build_next_turn_params_context({"model": "m", "input": [], "tool_choice": choice})

    assert context["tool_choice"] == choice
    assert context["model"] == "m"


def test_context_always_has_a_tool_choice_key() -> None:
    context = build_next_turn_params_context({"model": "m"})

    assert "tool_choice" in context
    assert context["tool_choice"] is None


async def test_a_tool_can_compute_the_next_tool_choice_from_the_current_one() -> None:
    seen: List[Any] = []

    def widen(params: Any, context: Any) -> Any:
        seen.append(context["tool_choice"])
        names = [t["name"] for t in context["tool_choice"]["tools"]]
        return {
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [{"type": "function", "name": n} for n in [*names, params["add"]]],
        }

    search = tool(name="search", input_schema=dict, execute=lambda _p, _c: {}, next_turn_params={"tool_choice": widen})
    request = {
        "model": "m",
        "tool_choice": {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "search"}]},
    }

    computed = await execute_next_turn_params_functions(
        [ParsedToolCall(id="c1", name="search", arguments={"add": "weather"})], [search], request
    )
    updated = apply_next_turn_params_to_request(request, computed)

    assert seen == [request["tool_choice"]]
    assert [t["name"] for t in updated["tool_choice"]["tools"]] == ["search", "weather"]
    assert updated["model"] == "m"
