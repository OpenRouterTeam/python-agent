from __future__ import annotations

from openrouter_agent import call_model, is_mcp_tool, mark_mcp, tool
from tests._fixtures import QueuedClient, function_call_item, make_response, text_response


def test_mark_mcp_is_non_mutating_and_is_mcp_tool_detects_the_brand() -> None:
    base = tool(name="remote_search", input_schema=dict, output_schema=dict, execute=lambda p, c: {"ok": True})

    branded = mark_mcp(base)

    assert base is not branded
    assert is_mcp_tool(base) is False
    assert is_mcp_tool(branded) is True
    # Runtime shape (type/function) unaffected -- only the additive brand.
    assert branded["type"] == base["type"]
    assert branded["function"] is base["function"]


def test_is_mcp_tool_false_for_plain_mappings_and_server_tools() -> None:
    from openrouter_agent import server_tool

    assert is_mcp_tool({"type": "function", "function": {"name": "x"}}) is False
    assert is_mcp_tool(server_tool({"type": "web_search"})) is False


async def test_mcp_branded_tool_result_carries_source_mcp_in_tool_result_event() -> None:
    remote_tool = mark_mcp(
        tool(name="remote_echo", input_schema=dict, output_schema=dict, execute=lambda p, c: {"ok": True})
    )
    client = QueuedClient(
        [make_response("r1", [function_call_item("call_1", "remote_echo", "{}")]), text_response("r2", "done")]
    )

    result = call_model(client, {"model": "test-model", "input": "hi", "tools": [remote_tool]})

    events = []
    async for event in result.get_tool_stream():
        if event.get("type") == "tool_result":
            events.append(event)
    await result.get_response()

    assert events[0]["source"] == "mcp"


async def test_regular_client_tool_result_carries_source_client() -> None:
    local_tool = tool(name="local_echo", input_schema=dict, output_schema=dict, execute=lambda p, c: {"ok": True})
    client = QueuedClient(
        [make_response("r1", [function_call_item("call_1", "local_echo", "{}")]), text_response("r2", "done")]
    )

    result = call_model(client, {"model": "test-model", "input": "hi", "tools": [local_tool]})

    events = []
    async for event in result.get_tool_stream():
        if event.get("type") == "tool_result":
            events.append(event)
    await result.get_response()

    assert events[0]["source"] == "client"
