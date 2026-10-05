"""Port of upstream tests/unit/call-model-active-tools.test.ts.

Upstream captures the outbound HTTP body; here `QueuedClient` records the
kwargs `call_model` passes to `client.beta.responses.send_async`, which is the
same boundary.

Not ported: "does not advertise the task helper when its background tool is
filtered out" — this port has no `lifecycle: 'background'` tools / `task`
helper yet.

Cases needing `active_tools` filtering / snapshot stripping inside
`call_model` are skipped until that wiring lands; remove the marker then.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import pytest

from openrouter_agent import call_model
from openrouter_agent.async_params import TOOL_SET_SNAPSHOT, strip_tool_set_snapshot_metadata
from openrouter_agent.tool import tool
from openrouter_agent.tool_set import create_tool_set
from tests._fixtures import QueuedClient, text_response

PENDING = pytest.mark.skip(reason="pending call_model active_tools wiring")


async def _ok(params: Any, ctx: Any = None) -> Dict[str, bool]:
    return {"ok": True}


tool_a = tool(name="a", input_schema={}, execute=_ok)
tool_b = tool(name="b", input_schema={}, execute=_ok)


async def capture_outbound_request(request: Mapping[str, Any]) -> Dict[str, Any]:
    client = QueuedClient([text_response("resp_1", "done")])
    result = call_model(client, request)
    assert await result.get_text() == "done"
    assert len(client.requests) == 1
    return client.requests[0]


def tool_names(sent: Mapping[str, Any]) -> List[str]:
    return [t["name"] for t in sent.get("tools") or [] if isinstance(t, Mapping) and "name" in t]


# ─── callModel activeTools filter ───────────────────────────────────────────


@PENDING
async def test_sends_only_active_tools_when_active_tools_is_provided() -> None:
    sent = await capture_outbound_request(
        {"model": "openai/gpt-4o-mini", "input": "hi", "tools": [tool_a, tool_b], "active_tools": ["a"]}
    )
    assert tool_names(sent) == ["a"]
    assert "active_tools" not in sent


@PENDING
async def test_silently_ignores_unknown_active_tools_names() -> None:
    sent = await capture_outbound_request(
        {"model": "openai/gpt-4o-mini", "input": "hi", "tools": [tool_a, tool_b], "active_tools": ["a", "missing"]}
    )
    assert tool_names(sent) == ["a"]


async def test_sends_all_tools_when_active_tools_is_omitted() -> None:
    sent = await capture_outbound_request({"model": "openai/gpt-4o-mini", "input": "hi", "tools": [tool_a, tool_b]})
    assert tool_names(sent) == ["a", "b"]


@PENDING
async def test_omits_the_tools_key_when_tools_is_explicitly_empty() -> None:
    sent = await capture_outbound_request({"model": "openai/gpt-4o-mini", "input": "hi", "tools": []})
    assert "tools" not in sent


@PENDING
async def test_omits_the_tools_key_entirely_when_active_tools_filters_out_every_tool() -> None:
    # Several providers reject an explicit `tools: []`; the key must be absent.
    sent = await capture_outbound_request(
        {"model": "openai/gpt-4o-mini", "input": "hi", "tools": [tool_a, tool_b], "active_tools": ["missing"]}
    )
    assert "tools" not in sent


@PENDING
async def test_filtered_tools_are_not_executable_by_the_model() -> None:
    # Upstream filters before registering tools for execution ("the model
    # cannot call filtered tools"). A call to a filtered tool must not run it.
    from tests._fixtures import tool_call_response

    calls: List[str] = []

    async def record_b(params: Any, ctx: Any = None) -> Dict[str, bool]:
        calls.append("b")
        return {"ok": True}

    b_recording = tool(name="b", input_schema={}, execute=record_b)
    client = QueuedClient([tool_call_response("resp_1", "b", call_id="call_1"), text_response("resp_2", "done")])
    result = call_model(client, {"model": "m", "input": "hi", "tools": [tool_a, b_recording], "active_tools": ["a"]})
    await result.get_text()
    assert calls == []
    assert tool_names(client.requests[0]) == ["a"]


# ─── callModel strips tool-set snapshot metadata ────────────────────────────


@PENDING
async def test_never_sends_snapshot_metadata_when_a_whole_snapshot_is_spread_in() -> None:
    snapshot_like_request = {
        "model": "openai/gpt-4o-mini",
        "input": "hi",
        "tools": [tool_a],
        "active_tools": ["a"],
        "enabled": ["a"],
        "disabled": [],
        "status_by_tool": {"a": "enabled"},
        "call_model": {"tools": [tool_a], "active_tools": ["a"]},
        TOOL_SET_SNAPSHOT: True,
    }
    sent = await capture_outbound_request(snapshot_like_request)
    for key in ("enabled", "disabled", "status_by_tool", "call_model", TOOL_SET_SNAPSHOT):
        assert key not in sent
    assert tool_names(sent) == ["a"]


@PENDING
async def test_never_sends_snapshot_metadata_for_a_real_tool_set_snapshot_spread() -> None:
    snapshot = create_tool_set(tools=[tool_a, tool_b]).deactivate("b").resolve()
    sent = await capture_outbound_request({**snapshot, "model": "openai/gpt-4o-mini", "input": "hi"})
    for key in ("enabled", "disabled", "status_by_tool", "call_model", "active_tools", TOOL_SET_SNAPSHOT):
        assert key not in sent
    assert tool_names(sent) == ["a"]


def test_preserves_identically_named_fields_when_the_request_is_not_a_snapshot() -> None:
    request: Dict[str, Any] = {
        "enabled": True,
        "disabled": False,
        "status_by_tool": {"a": "request-value"},
        "call_model": "request-value",
    }
    strip_tool_set_snapshot_metadata(request)
    assert request == {
        "enabled": True,
        "disabled": False,
        "status_by_tool": {"a": "request-value"},
        "call_model": "request-value",
    }


def test_strip_removes_metadata_and_marker_but_keeps_tools_and_active_tools() -> None:
    # Unit-level counterpart of the spread case, runnable before call_model wiring.
    snapshot = create_tool_set(tools=[tool_a, tool_b]).deactivate("b").resolve()
    request: Dict[str, Any] = {**snapshot, "model": "m"}
    strip_tool_set_snapshot_metadata(request)
    assert request == {"tools": [tool_a], "active_tools": ["a"], "model": "m"}


def test_strip_only_acts_on_a_literal_true_marker() -> None:
    request: Dict[str, Any] = {"enabled": [], TOOL_SET_SNAPSHOT: 1}
    strip_tool_set_snapshot_metadata(request)
    assert request == {"enabled": [], TOOL_SET_SNAPSHOT: 1}


@PENDING
async def test_still_sends_the_documented_spread_safe_pattern_unaffected() -> None:
    sent = await capture_outbound_request(
        {"model": "openai/gpt-4o-mini", "input": "hi", **create_tool_set(tools=[tool_a]).resolve()["call_model"]}
    )
    assert tool_names(sent) == ["a"]
