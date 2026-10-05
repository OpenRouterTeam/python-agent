"""Port of upstream tests/unit/server-tool.test.ts (runtime cases).

Not ported:
- "rejects unknown server tool types at the type level", "rejects wrong fields
  for a known type", "isServerTool narrows to ServerTool", "ToolResultItem
  union": type-only (`@ts-expect-error` / `expectTypeOf`).
- `expectTypeOf(t.id)` literal-type assertions: type-only; runtime halves kept.
- "serializes strict: ..." (5 cases), incl. unified `run` and `tool.agent`
  tools: this port's `tool()` has no `strict` option, `run` lifecycle, or
  agent builder yet — parity gap, not part of this module's port.
"""

from __future__ import annotations

import pytest

from openrouter_agent.tool import get_server_tool_id, server_tool, tool
from openrouter_agent.tool_executor import convert_tools_to_api_format
from openrouter_agent.tool_types import is_client_tool, is_server_tool


def test_creates_a_branded_server_tool_carrying_the_sdk_config_through() -> None:
    t = server_tool({"type": "web_search_2025_08_26", "engine": "exa", "max_results": 10})
    assert t["_brand"] == "server-tool"
    assert t["config"]["type"] == "web_search_2025_08_26"
    assert t["id"] == "server:web_search_2025_08_26"
    assert is_server_tool(t) is True
    assert is_client_tool(t) is False


def test_allows_overriding_the_stable_tool_set_id() -> None:
    t = server_tool({"type": "web_search_2025_08_26"}, {"id": "server:public_search"})
    assert t["id"] == "server:public_search"
    assert get_server_tool_id(t) == "server:public_search"


def test_rejects_an_empty_stable_tool_set_id() -> None:
    with pytest.raises(ValueError, match=r"must not be empty"):
        server_tool({"type": "openrouter:datetime"}, {"id": ""})


def test_explicit_none_id_falls_back_to_the_default() -> None:
    # Upstream: `options?.id ?? \`server:${config.type}\``.
    t = server_tool({"type": "openrouter:datetime"}, {})
    assert t["id"] == "server:openrouter:datetime"


def test_get_server_tool_id_synthesizes_the_default_for_hand_written_tools() -> None:
    # Upstream `defaultServerId`: legacy hand-built server tools have no `id`.
    hand_written = {"_brand": "server-tool", "config": {"type": "web_search_2025_08_26"}}
    assert get_server_tool_id(hand_written) == "server:web_search_2025_08_26"
    assert get_server_tool_id({**hand_written, "id": ""}) == "server:web_search_2025_08_26"


def test_carries_config_shape_for_the_chosen_type() -> None:
    dt = server_tool({"type": "openrouter:datetime", "parameters": {"timezone": "America/New_York"}})
    assert dt["config"]["parameters"]["timezone"] == "America/New_York"

    img = server_tool({"type": "image_generation", "size": "1024x1024", "quality": "high"})
    assert img["config"]["size"] == "1024x1024"


def test_passes_server_tool_configs_through_verbatim() -> None:
    tools = [
        server_tool({"type": "openrouter:datetime"}),
        server_tool({"type": "web_search_2025_08_26", "engine": "native"}, {"id": "server:custom"}),
    ]
    api = convert_tools_to_api_format(tools)
    # The tool-set `id` is client-side identity only; it must never hit the wire.
    assert api == [
        {"type": "openrouter:datetime"},
        {"type": "web_search_2025_08_26", "engine": "native"},
    ]


def test_mixes_client_and_server_tools_in_one_array() -> None:
    client_tool = tool(name="echo", input_schema={}, execute=lambda params, ctx=None: params)
    api = convert_tools_to_api_format([client_tool, server_tool({"type": "image_generation", "size": "1024x1024"})])
    assert len(api) == 2
    assert api[0]["type"] == "function"
    assert api[0]["name"] == "echo"
    assert api[1] == {"type": "image_generation", "size": "1024x1024"}
