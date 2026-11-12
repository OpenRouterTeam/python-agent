"""Ports tests/unit/server-tool.test.ts.

Type-level cases (`expectTypeOf`, `@ts-expect-error` config narrowing, guard
narrowing, the `ToolResultItem` union) have no Python runtime counterpart
(contract, Idiomatic Divergences #4); their runtime halves are ported below.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from openrouter_agent import is_client_tool, is_server_tool, server_tool, tool
from openrouter_agent.tool_executor import convert_tools_to_api_format


class EchoInput(BaseModel):
    msg: str


def _echo(params: Any, context: Any) -> str:
    return params.msg


# -- server_tool() -------------------------------------------------------------


def test_creates_a_branded_server_tool_carrying_the_sdk_config_through() -> None:
    t = server_tool({"type": "web_search_2025_08_26", "engine": "exa", "maxResults": 10})

    assert t["_brand"] == "server-tool"
    assert t["config"] == {"type": "web_search_2025_08_26", "engine": "exa", "maxResults": 10}
    assert t["id"] == "server:web_search_2025_08_26"
    assert is_server_tool(t) is True
    assert is_client_tool(t) is False


def test_allows_overriding_the_stable_tool_set_id() -> None:
    t = server_tool({"type": "web_search_2025_08_26"}, {"id": "server:public_search"})

    assert t["id"] == "server:public_search"


def test_rejects_an_empty_stable_tool_set_id() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        server_tool({"type": "openrouter:datetime"}, {"id": ""})


def test_carries_type_specific_config_fields_through() -> None:
    # Runtime half of upstream "narrows config shape based on the chosen type
    # literal"; the narrowing itself is TS type-level only.
    dt = server_tool({"type": "openrouter:datetime", "parameters": {"timezone": "America/New_York"}})
    img = server_tool({"type": "image_generation", "size": "1024x1024", "quality": "high"})

    assert dt["config"]["parameters"]["timezone"] == "America/New_York"
    assert img["config"]["size"] == "1024x1024"
    assert img["config"]["quality"] == "high"


def test_server_tool_copies_the_caller_config() -> None:
    config = {"type": "openrouter:datetime"}
    t = server_tool(config)
    config["type"] = "mutated"

    assert t["config"] == {"type": "openrouter:datetime"}


# -- convert_tools_to_api_format ---------------------------------------------


def test_passes_server_tool_configs_through_verbatim() -> None:
    tools = [
        server_tool({"type": "openrouter:datetime"}),
        server_tool({"type": "web_search_2025_08_26", "engine": "native"}),
    ]

    api = convert_tools_to_api_format(tools)

    # The stable tool-set id and brand are client-only and never reach the wire.
    assert api == [
        {"type": "openrouter:datetime"},
        {"type": "web_search_2025_08_26", "engine": "native"},
    ]


def test_shapes_client_tools_into_the_function_call_api_format() -> None:
    client_tool = tool(name="echo", input_schema=EchoInput, execute=_echo)

    api = convert_tools_to_api_format([client_tool])

    assert len(api) == 1
    assert api[0]["type"] == "function"
    assert api[0]["name"] == "echo"
    assert api[0]["parameters"]["properties"]["msg"]["type"] == "string"
    # DIVERGENCE: upstream sends `description: null` for an undeclared
    # description (tool-executor.ts:130 `description || null`). This port omits
    # the key instead (tool_executor.py convert_tools_to_api_format). Both mean
    # "no description" to the Responses API.
    assert "description" not in api[0]


def test_serializes_strict_true_from_the_tool_definition() -> None:
    strict_tool = tool(name="echo", input_schema=EchoInput, strict=True, execute=_echo)

    api = convert_tools_to_api_format([strict_tool])

    assert api[0]["type"] == "function"
    assert api[0]["name"] == "echo"
    assert api[0]["strict"] is True


@pytest.mark.skip(reason="Out of scope: unified `run`/lifecycle tools are not ported (async/deferred tools).")
def test_serializes_strict_true_from_a_unified_run_tool() -> None:
    raise AssertionError("not ported")


@pytest.mark.skip(reason="Out of scope: `tool.agent` (agent tools) is not ported.")
def test_serializes_strict_true_from_an_agent_tool() -> None:
    raise AssertionError("not ported")


def test_serializes_strict_false_as_false_instead_of_null() -> None:
    non_strict_tool = tool(name="echo", input_schema=EchoInput, strict=False, execute=_echo)

    api = convert_tools_to_api_format([non_strict_tool])

    assert api[0]["type"] == "function"
    assert "strict" in api[0]
    assert api[0]["strict"] is False


def test_omits_strict_when_not_declared() -> None:
    plain_tool = tool(name="echo", input_schema=EchoInput, execute=_echo)

    api = convert_tools_to_api_format([plain_tool])

    assert api[0]["type"] == "function"
    # DIVERGENCE: upstream "defaults strict to null when not declared" and
    # sends `strict: null` (tool-executor.ts:131 `strict ?? null`). This port
    # omits the key when `strict` is unset (tool_executor.py
    # convert_tools_to_api_format: `if fn.get("strict") is not None`). An
    # absent key and an explicit null are equivalent on the wire; do not
    # "fix" this to assert `api[0]["strict"] is None`.
    assert "strict" not in api[0]


def test_mixes_client_and_server_tools_in_one_array() -> None:
    tools = [
        tool(name="echo", input_schema=EchoInput, execute=_echo),
        server_tool({"type": "image_generation", "size": "1024x1024"}),
    ]

    api = convert_tools_to_api_format(tools)

    assert len(api) == 2
    assert api[0]["type"] == "function"
    assert api[0]["name"] == "echo"
    assert api[1] == {"type": "image_generation", "size": "1024x1024"}


# -- guards (runtime half of the type-level narrowing test) ------------------


def test_guards_partition_a_mixed_tool_list() -> None:
    client = tool(name="x", input_schema=EchoInput, execute=lambda params, ctx: 1)
    server = server_tool({"type": "openrouter:datetime"})

    assert [is_server_tool(t) for t in (client, server)] == [False, True]
    assert [is_client_tool(t) for t in (client, server)] == [True, False]
