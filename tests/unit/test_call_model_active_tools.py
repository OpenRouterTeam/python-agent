"""Ported from upstream tests/unit/call-model-active-tools.test.ts (the
`activeTools` option from #31). The tool-set snapshot and async `task` helper
cases are not ported: this port has no tool-set module or async tools yet."""

from __future__ import annotations

from typing import Any, List

from openrouter_agent import call_model, server_tool, tool
from tests._fixtures import QueuedClient, text_response, tool_call_response, recording

_a = tool(name="a", input_schema=dict, execute=lambda _p, _c: {"ok": "a"})
_b = tool(name="b", input_schema=dict, execute=lambda _p, _c: {"ok": "b"})
_search = server_tool({"type": "web_search"})


def _sent_names(client: QueuedClient) -> List[Any]:
    return [t.get("name", t.get("type")) for t in client.requests[0].get("tools") or []]


async def test_sends_only_active_tools_when_active_tools_is_provided() -> None:
    client = QueuedClient([text_response("r1", "hi")])

    await call_model(
        client, {"model": "m", "input": "hi", "tools": [_a, _b, _search], "active_tools": ["b"]}
    ).get_text()

    # Server tools are always kept; the option never reaches the wire.
    assert _sent_names(client) == ["b", "web_search"]
    assert "active_tools" not in client.requests[0]


async def test_silently_ignores_unknown_active_tools_names() -> None:
    client = QueuedClient([text_response("r1", "hi")])

    await call_model(client, {"model": "m", "input": "hi", "tools": [_a, _b], "active_tools": ["a", "nope"]}).get_text()

    assert _sent_names(client) == ["a"]


async def test_sends_all_tools_when_active_tools_is_omitted() -> None:
    client = QueuedClient([text_response("r1", "hi")])

    await call_model(client, {"model": "m", "input": "hi", "tools": [_a, _b]}).get_text()

    assert _sent_names(client) == ["a", "b"]


async def test_omits_the_tools_key_when_tools_is_explicitly_empty() -> None:
    client = QueuedClient([text_response("r1", "hi")])

    await call_model(client, {"model": "m", "input": "hi", "tools": []}).get_text()

    assert "tools" not in client.requests[0]


async def test_omits_the_tools_key_when_active_tools_filters_out_every_tool() -> None:
    client = QueuedClient([text_response("r1", "hi")])

    await call_model(client, {"model": "m", "input": "hi", "tools": [_a, _b], "active_tools": []}).get_text()

    assert "tools" not in client.requests[0]


async def test_a_filtered_out_tool_is_not_executed_when_the_model_calls_it() -> None:
    executed: List[str] = []
    a = tool(name="a", input_schema=dict, execute=recording(executed, {}, lambda _p: "a"))
    b = tool(name="b", input_schema=dict, execute=recording(executed, {}, lambda _p: "b"))
    client = QueuedClient([tool_call_response("r1", "a"), text_response("r2", "done")])

    result = call_model(client, {"model": "m", "input": "hi", "tools": [a, b], "active_tools": ["b"]})
    await result.get_response()

    # The executor does not carry the filtered definition, so the call is
    # surfaced as unresolved instead of being run.
    assert executed == []
    assert len(client.requests) == 1
