from __future__ import annotations

from typing import Any, Dict, List

from openrouter_agent import call_model, is_mcp_tool, mark_mcp, tool


class QueuedResponses:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.requests: List[Dict[str, Any]] = []

    async def send_async(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return self._responses.pop(0)


class QueuedClient:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.beta = type("Beta", (), {"responses": QueuedResponses(responses)})()


def function_call_item(call_id: str, name: str, arguments: str) -> Dict[str, Any]:
    return {"type": "function_call", "id": f"fc_{call_id}", "callId": call_id, "name": name, "arguments": arguments}


def text_response(response_id: str, text: str) -> Dict[str, Any]:
    return {
        "id": response_id,
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
    }


def make_response(response_id: str, output: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"id": response_id, "output": output}


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
