"""Port of upstream `tests/unit/next-turn-tool-choice-loop.test.ts`.

`next_turn_params["tool_choice"]` driven through the real `call_model` loop,
asserted against the DISPATCHED request rather than the helper in isolation:
the follow-up builder re-derives the wire tool choice from the caller-
configured value after next-turn params apply, so a hook that merges
correctly can still be discarded before dispatch. Only a loop-level assertion
catches that.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel

from openrouter_agent import call_model, tool
from tests._fixtures import QueuedClient, text_response, tool_call_response


class SearchInput(BaseModel):
    pattern: str


class EmptyInput(BaseModel):
    pass


def allowed(names: List[str]) -> Dict[str, Any]:
    return {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": n} for n in names]}


def search_call_response() -> Dict[str, Any]:
    return tool_call_response("resp_tool", "tool_search", call_id="call_1", arguments='{"pattern":"weather"}')


def make_search_tool(widened: List[str]) -> Dict[str, Any]:
    return tool(
        name="tool_search",
        input_schema=SearchInput,
        execute=lambda params, ctx: {"found": widened},
        next_turn_params={"tool_choice": lambda args, ctx: allowed(widened)},
    )


get_weather = tool(name="get_weather", input_schema=EmptyInput, execute=lambda params, ctx: {"ok": True})


def dispatched_tool_choices(client: QueuedClient) -> List[Any]:
    return [request.get("tool_choice") for request in client.requests]


async def test_reaches_the_dispatched_follow_up_request() -> None:
    client = QueuedClient([search_call_response(), text_response("resp_text", "done")])

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "hi",
            "tools": [make_search_tool(["tool_search", "get_weather"]), get_weather],
            "tool_choice": allowed(["tool_search"]),
        },
    ).get_text()

    choices = dispatched_tool_choices(client)
    assert len(choices) == 2
    # Turn 1 uses the caller's narrow set.
    assert choices[0] == allowed(["tool_search"])
    # Turn 2 carries the tool's widened set, not the caller's original.
    assert choices[1] == allowed(["tool_search", "get_weather"])


async def test_leaves_the_tools_array_untouched_while_the_choice_widens() -> None:
    client = QueuedClient([search_call_response(), text_response("resp_text", "done")])

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "hi",
            "tools": [make_search_tool(["tool_search", "get_weather"]), get_weather],
            "tool_choice": allowed(["tool_search"]),
        },
    ).get_text()

    assert len(client.requests) == 2
    tool_names = [[t.get("name") for t in (request.get("tools") or [])] for request in client.requests]
    # Identical across turns: widening must not change the request prefix, or
    # the provider's prompt cache is lost.
    assert tool_names[0] == ["tool_search", "get_weather"]
    assert tool_names[1] == tool_names[0]
    assert client.requests[1]["tools"] == client.requests[0]["tools"]


async def test_does_not_disturb_tool_choice_when_no_tool_computes_one() -> None:
    plain = tool(name="tool_search", input_schema=SearchInput, execute=lambda params, ctx: {"ok": True})
    client = QueuedClient([search_call_response(), text_response("resp_text", "done")])

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "hi",
            "tools": [plain, get_weather],
            "tool_choice": allowed(["tool_search"]),
        },
    ).get_text()

    assert dispatched_tool_choices(client) == [allowed(["tool_search"]), allowed(["tool_search"])]
