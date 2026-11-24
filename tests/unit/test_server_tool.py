"""Ported from upstream tests/unit/server-tool.test.ts, the `strict`
serialization cases (#99). The run-lifecycle and `tool.agent()` variants are
not ported: this port has no unified `run` tools or subagent tools yet."""

from __future__ import annotations

from pydantic import BaseModel

from openrouter_agent import call_model, server_tool, tool
from openrouter_agent.tool_executor import convert_tools_to_api_format
from tests._fixtures import QueuedClient, text_response


class _Msg(BaseModel):
    msg: str


def test_serializes_strict_true_from_the_tool_definition() -> None:
    strict_tool = tool(name="echo", input_schema=_Msg, strict=True, execute=lambda p, _c: p.msg)

    api = convert_tools_to_api_format([strict_tool])

    assert api[0]["type"] == "function"
    assert api[0]["name"] == "echo"
    assert api[0]["strict"] is True


def test_serializes_strict_false_as_false_instead_of_null() -> None:
    api = convert_tools_to_api_format([tool(name="echo", input_schema=_Msg, strict=False, execute=lambda p, _c: p.msg)])

    assert api[0]["strict"] is False


def test_defaults_strict_to_null_when_not_declared() -> None:
    api = convert_tools_to_api_format([tool(name="echo", input_schema=_Msg, execute=lambda p, _c: p.msg)])

    assert "strict" in api[0]
    assert api[0]["strict"] is None


def test_server_tools_pass_their_config_through_unchanged() -> None:
    api = convert_tools_to_api_format(
        [
            tool(name="echo", input_schema=_Msg, strict=True, execute=lambda p, _c: p.msg),
            server_tool({"type": "web_search"}),
        ]
    )

    assert [entry.get("name", entry["type"]) for entry in api] == ["echo", "web_search"]
    assert api[1] == {"type": "web_search"}


async def test_strict_reaches_the_wire_request() -> None:
    client = QueuedClient([text_response("r1", "hi")])

    await call_model(
        client,
        {
            "model": "m",
            "input": "hi",
            "tools": [tool(name="echo", input_schema=_Msg, strict=True, execute=lambda p, _c: p.msg)],
        },
    ).get_text()

    assert client.requests[0]["tools"][0]["strict"] is True
