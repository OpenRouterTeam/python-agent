"""Ports `packages/agent/tests/unit/call-model-active-tools.test.ts`.

Upstream semantics (`src/inner-loop/call-model.ts:122-149`): `activeTools`
narrows `tools` *before* API conversion and before the tools are handed to
`ModelResult` for execution. Server tools always survive the filter (only
client `function.name`s are matched); unknown names are ignored; a list that
filters to empty -- or an explicitly empty `tools` -- omits the outbound
`tools` key entirely. `activeTools` itself is client-only and never sent.

Divergence from upstream's harness: upstream installs a capturing
`HTTPClient` and aborts the request; this port drives `call_model` through
`tests._fixtures.QueuedClient` and asserts on the recorded `send_async`
kwargs, which is the transport boundary this port binds to.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from openrouter_agent import call_model
from openrouter_agent.tool import server_tool, tool
from openrouter_agent.tool_set import TOOL_SET_SNAPSHOT, create_tool_set, strip_tool_set_snapshot_metadata
from tests._fixtures import MemoryStateAccessor, QueuedClient, text_response, tool_call_response


class ExecutionCounter:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, params: Any, ctx: Any) -> Dict[str, Any]:
        self.count += 1
        return {"ok": True}


def make_tool(name: str, execute: Any = None) -> Dict[str, Any]:
    return tool(name=name, input_schema=dict, execute=execute or ExecutionCounter())


tool_a = make_tool("a")
tool_b = make_tool("b")


def tool_names(request: Dict[str, Any]) -> List[str]:
    return [t["name"] for t in request.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("name"), str)]


async def capture_request(request: Dict[Any, Any]) -> Dict[str, Any]:
    """Run `call_model` to completion against one queued text turn and return
    the single request it sent."""
    client = QueuedClient([text_response("r1", "ok")])
    await call_model(client, {"model": "openai/gpt-4o-mini", "input": "hi", **request}).get_text()
    assert len(client.requests) == 1
    return client.requests[0]


class TestActiveToolsFilter:
    async def test_sends_only_active_tools_when_active_tools_is_provided(self) -> None:
        sent = await capture_request({"tools": [tool_a, tool_b], "active_tools": ["a"]})
        assert tool_names(sent) == ["a"]
        # Client-only field (async-params.ts clientOnlyFields): never on the wire.
        assert "active_tools" not in sent
        assert "activeTools" not in sent

    async def test_silently_ignores_unknown_active_tools_names(self) -> None:
        sent = await capture_request({"tools": [tool_a, tool_b], "active_tools": ["a", "missing"]})
        assert tool_names(sent) == ["a"]

    async def test_sends_all_tools_when_active_tools_is_omitted(self) -> None:
        sent = await capture_request({"tools": [tool_a, tool_b]})
        assert tool_names(sent) == ["a", "b"]

    @pytest.mark.skip(
        reason="this port has no long-running (lifecycle='background') tools or universal `task` tool to filter"
    )
    async def test_does_not_advertise_the_task_helper_when_its_background_tool_is_filtered_out(self) -> None:
        raise AssertionError("unreachable")

    async def test_omits_the_tools_key_when_tools_is_explicitly_empty(self) -> None:
        sent = await capture_request({"tools": []})
        assert "tools" not in sent

    async def test_omits_the_tools_key_when_active_tools_filters_out_every_tool(self) -> None:
        # Several providers reject an explicit empty tools array, so the key
        # must be absent rather than `[]`.
        sent = await capture_request({"tools": [tool_a, tool_b], "active_tools": ["missing"]})
        assert "tools" not in sent

    async def test_empty_active_tools_list_filters_every_client_tool(self) -> None:
        # Upstream: `activeTools ? new Set(activeTools) : undefined` -- an empty
        # array is truthy in JS, so `[]` means "no client tools", unlike None.
        sent = await capture_request({"tools": [tool_a, tool_b], "active_tools": []})
        assert "tools" not in sent

    async def test_server_tools_always_pass_the_active_tools_filter(self) -> None:
        # Upstream filter: `isServerTool(t) || activeSet.has(t.function.name)`.
        web_search = server_tool({"type": "web_search_2025_08_26"})
        sent = await capture_request({"tools": [tool_a, web_search, tool_b], "active_tools": ["b"]})
        assert [(t["type"], t.get("name")) for t in sent["tools"]] == [
            ("web_search_2025_08_26", None),
            ("function", "b"),
        ]

    async def test_filtered_tool_is_not_executable(self) -> None:
        # Upstream registers only the filtered tools on ModelResult
        # (call-model.ts `tools: filteredTools`), so a call to a filtered tool
        # is an unknown name: not executed, no error output, the run stops and
        # surfaces it as an unresolved client call (model-result.ts
        # executeSingleToolCall -> null; persistClientToolsPause).
        execute_b = ExecutionCounter()
        b_counted = make_tool("b", execute_b)
        state = MemoryStateAccessor()
        client = QueuedClient([tool_call_response("r1", "b", call_id="call_b"), text_response("r2", "unused")])

        await call_model(
            client,
            {"model": "m", "input": "hi", "tools": [tool_a, b_counted], "active_tools": ["a"], "state": state},
        ).get_response()

        assert execute_b.count == 0
        assert len(client.requests) == 1
        assert tool_names(client.requests[0]) == ["a"]
        assert state.stored is not None
        assert state.stored.status == "awaiting_client_tools"
        assert [c.name for c in state.stored.pending_tool_calls or []] == ["b"]


class TestToolSetSnapshotStripping:
    async def test_never_sends_snapshot_metadata_when_a_whole_snapshot_is_spread_in(self) -> None:
        # Mirrors the documented-but-dangerous pattern of spreading the full
        # `infer_tools()` result straight into call_model.
        snapshot_like_request: Dict[Any, Any] = {
            "tools": [tool_a],
            "active_tools": ["a"],
            "enabled": ["a"],
            "disabled": [],
            "status_by_tool": {"a": "enabled"},
            "call_model": {"tools": [tool_a], "active_tools": ["a"]},
            TOOL_SET_SNAPSHOT: True,
        }
        sent = await capture_request(snapshot_like_request)
        for key in ("enabled", "disabled", "status_by_tool", "call_model", TOOL_SET_SNAPSHOT):
            assert key not in sent
        assert tool_names(sent) == ["a"]

    async def test_whole_infer_tools_result_can_be_spread(self) -> None:
        snapshot = create_tool_set([tool_a, tool_b]).deactivate("b").infer_tools()
        sent = await capture_request(snapshot)
        assert tool_names(sent) == ["a"]
        for key in ("enabled", "disabled", "status_by_tool", TOOL_SET_SNAPSHOT):
            assert key not in sent

    def test_preserves_identically_named_fields_when_not_a_tool_set_snapshot(self) -> None:
        request: Dict[Any, Any] = {
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

    def test_strips_metadata_and_marker_from_a_marked_snapshot(self) -> None:
        request = create_tool_set([tool_a]).infer_tools()
        strip_tool_set_snapshot_metadata(request)
        assert request == {"tools": [tool_a], "active_tools": ["a"]}

    async def test_still_sends_the_documented_tools_active_tools_pattern(self) -> None:
        sent = await capture_request({"tools": [tool_a], "active_tools": ["a"]})
        assert tool_names(sent) == ["a"]
        assert "active_tools" not in sent

    async def test_resolved_call_model_mapping_spreads_into_call_model(self) -> None:
        resolved = create_tool_set([tool_a, tool_b]).deactivate("b").resolve()
        sent = await capture_request({**resolved.call_model})
        assert tool_names(sent) == ["a"]
        assert "active_tools" not in sent
