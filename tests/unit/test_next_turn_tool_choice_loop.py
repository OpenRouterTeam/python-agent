"""Ported from upstream tests/unit/next-turn-tool-choice-loop.test.ts (#114, #153).

`next_turn_params` may set `tool_choice` -- the hook a tool-search tool uses
to widen an `allowed_tools` choice without touching `tools` -- and it runs only
for calls whose tool body ran and returned without error, with the input the
tool actually ran on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel

from openrouter_agent import HookEntry, HookName, HooksManager, call_model, tool
from tests._fixtures import QueuedClient, text_response, tool_call_response, recording


def _allowed(names: List[str]) -> Dict[str, Any]:
    return {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": n} for n in names]}


class _Pattern(BaseModel):
    pattern: str


class _Empty(BaseModel):
    pass


_WIDENED = ["tool_search", "get_weather"]
_get_weather = tool(name="get_weather", input_schema=_Empty, execute=lambda _p, _c: {"ok": True})


def _search_tool(next_calls: List[Any], **overrides: Any) -> Dict[str, Any]:
    def next_choice(params: Any, context: Any) -> Any:
        next_calls.append((params, context.get("tool_choice")))
        return _allowed(_WIDENED)

    kwargs: Dict[str, Any] = {
        "name": "tool_search",
        "input_schema": _Pattern,
        "execute": lambda _p, _c: {"found": _WIDENED},
        "next_turn_params": {"tool_choice": next_choice},
    }
    kwargs.update(overrides)
    return tool(**kwargs)


async def _run(search: Dict[str, Any], **extra: Any) -> QueuedClient:
    client = QueuedClient(
        [
            tool_call_response("resp_tool", "tool_search", call_id="call_1", arguments='{"pattern":"weather"}'),
            text_response("resp_text", "done"),
        ]
    )
    await call_model(
        client,
        {
            "model": "test-model",
            "input": "hi",
            "tools": [search, _get_weather],
            "tool_choice": _allowed(["tool_search"]),
            **extra,
        },
    ).get_text()
    return client


async def test_reaches_the_dispatched_follow_up_request() -> None:
    next_calls: List[Any] = []
    client = await _run(_search_tool(next_calls))

    choices = [r.get("tool_choice") for r in client.requests]
    assert choices == [_allowed(["tool_search"]), _allowed(_WIDENED)]
    # The tool reads the current choice to widen it rather than rebuild it.
    assert len(next_calls) == 1
    assert next_calls[0][1] == _allowed(["tool_search"])


async def test_leaves_the_tools_array_untouched_while_the_choice_widens() -> None:
    client = await _run(_search_tool([]))

    names = [[t["name"] for t in r["tools"]] for r in client.requests]
    # Identical across turns: widening must not change the request prefix, or
    # the provider's prompt cache is lost.
    assert names[1] == names[0]


async def test_does_not_disturb_tool_choice_when_no_tool_computes_one() -> None:
    plain = tool(name="tool_search", input_schema=_Pattern, execute=lambda _p, _c: {"ok": True})
    client = await _run(plain)

    assert [r.get("tool_choice") for r in client.requests] == [_allowed(["tool_search"])] * 2


async def test_does_not_widen_tool_choice_when_pre_tool_use_blocks_the_call() -> None:
    executed: List[Any] = []
    next_calls: List[Any] = []
    hooks = HooksManager()
    hooks.on(
        HookName.PreToolUse.value,
        HookEntry(matcher="tool_search", handler=lambda _p, _c: {"block": "blocked by policy"}),
    )
    search = _search_tool(next_calls, execute=recording(executed, {"found": _WIDENED}))

    client = await _run(search, hooks=hooks)

    assert executed == []
    assert next_calls == []
    assert client.requests[1]["tool_choice"] == _allowed(["tool_search"])


async def test_does_not_widen_tool_choice_when_the_arguments_fail_input_schema() -> None:
    class _News(BaseModel):
        pattern: Literal["news"]

    executed: List[Any] = []
    next_calls: List[Any] = []
    search = _search_tool(next_calls, input_schema=_News, execute=recording(executed, {}))

    client = await _run(search)

    assert executed == []
    assert next_calls == []
    assert client.requests[1]["tool_choice"] == _allowed(["tool_search"])


async def test_does_not_widen_tool_choice_when_execute_throws() -> None:
    def boom(_p: Any, _c: Any) -> Any:
        raise RuntimeError("search backend unavailable")

    next_calls: List[Any] = []
    client = await _run(_search_tool(next_calls, execute=boom))

    assert next_calls == []
    assert client.requests[1]["tool_choice"] == _allowed(["tool_search"])


async def test_passes_the_pre_tool_use_mutated_input_that_the_tool_ran_on() -> None:
    ran_on: List[Any] = []
    next_calls: List[Any] = []
    hooks = HooksManager()
    hooks.on(
        HookName.PreToolUse.value,
        HookEntry(matcher="tool_search", handler=lambda _p, _c: {"mutated_input": {"pattern": "mutated"}}),
    )
    search = _search_tool(next_calls, execute=recording(ran_on, {"found": _WIDENED}, lambda p: p.pattern))

    client = await _run(search, hooks=hooks)

    assert ran_on == ["mutated"]
    assert [params for params, _choice in next_calls] == [{"pattern": "mutated"}]
    assert client.requests[1]["tool_choice"] == _allowed(_WIDENED)
