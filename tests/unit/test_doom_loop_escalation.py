"""Tests for the doom-loop `escalate` recovery rung.

Ports `packages/agent/tests/unit/doom-loop-escalation.test.ts`.

On detection, throw more intelligence at the NEXT turn -- a one-turn model swap
and/or a forced `openrouter:advisor` consult -- then return to the configured
model. All scenarios simulate the LLM with scripted `QueuedClient` responses.

Port notes:
- Request keys are the generated Python SDK's snake_case: `tool_choice`
  (upstream `toolChoice`) and the advisor's `forward_transcript` (upstream
  `forwardTranscript`; see `AdvisorServerToolConfig`). Deliberate.
- Config/state keys are snake_case (`max_escalations`, `escalations_used`).
- Upstream spies on `console.warn`; this port warns via `warnings.warn`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping

import pytest

from openrouter_agent import (
    DoomLoopLadder,
    DoomLoopMonitor,
    HookEntry,
    HooksManager,
    call_model,
    resolve_doom_loop_option,
    tool,
)
from tests._fixtures import MemoryStateAccessor, QueuedClient, function_call_item, make_response, message_item

pytestmark = pytest.mark.filterwarnings("ignore:\\[DoomLoop\\]:UserWarning")


class Script:
    def __init__(self) -> None:
        self.counter = 0

    def response(self, output: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.counter += 1
        return make_response(f"resp_{self.counter}", output)

    def call(self, name: str, args: Any) -> Dict[str, Any]:
        self.counter += 1
        return function_call_item(f"call_{self.counter}", name, json.dumps(args))

    def message(self, text: str) -> Dict[str, Any]:
        self.counter += 1
        return message_item(text, item_id=f"msg_{self.counter}")

    def tool_call_turn(self, name: str, args: Any) -> Dict[str, Any]:
        return self.response([self.call(name, args)])

    def text_turn(self, text: str) -> Dict[str, Any]:
        return self.response([self.message(text)])

    def text_and_call_turn(self, text: str, query: str) -> Dict[str, Any]:
        return self.response([self.message(text), self.call("web_search", {"query": query})])


def collect_detections(hooks: HooksManager) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []

    def handler(payload: Dict[str, Any], ctx: Any) -> None:
        detections.append(payload)

    hooks.on("DoomLoopDetected", HookEntry(handler=handler))
    return detections


search_tool = tool(
    name="web_search",
    input_schema=dict,
    output_schema=dict,
    execute=lambda p, c: {"results": f"nothing for {p['query']}"},
)

#: Escalate at streak 2, block/stop far away -- isolates the recovery rung.
ESCALATE_AT_2: DoomLoopLadder = {"observe": 1, "steer": False, "escalate": 2, "block": 5, "stop": 8}


def advisor_of(request: Mapping[str, Any]) -> Dict[str, Any]:
    tools = request.get("tools") or []
    matches = [t for t in tools if t.get("type") == "openrouter:advisor"]
    assert len(matches) == 1
    return matches[0]


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def test_resolves_a_mechanism_with_the_default_budget() -> None:
    resolved = resolve_doom_loop_option({"ladder": {"escalate": 3}, "escalation": {"model": "openai/gpt-5"}})
    assert resolved["escalation"] == {"model": "openai/gpt-5", "max_escalations": 2}
    assert resolved["ladder"]["escalate"] == 3


def test_warns_and_disables_the_rung_when_escalate_is_enabled_without_a_mechanism() -> None:
    with pytest.warns(UserWarning, match="no escalation config"):
        resolved = resolve_doom_loop_option({"ladder": {"escalate": 3}})
    assert resolved["escalation"] is None


def test_warns_when_a_mechanism_is_configured_but_the_rung_is_disabled() -> None:
    with pytest.warns(UserWarning, match="escalate ladder rung is disabled"):
        resolved = resolve_doom_loop_option({"escalation": {"model": "openai/gpt-5"}})
    # Mechanism kept (hook overrides may still use it); rung stays off.
    assert resolved["escalation"] is not None
    assert resolved["ladder"]["escalate"] is False


def test_clamps_an_invalid_max_escalations_to_the_default() -> None:
    resolved = resolve_doom_loop_option(
        {"ladder": {"escalate": 2}, "escalation": {"model": "openai/gpt-5", "max_escalations": 0}}
    )
    assert resolved["escalation"] is not None
    assert resolved["escalation"]["max_escalations"] == 2


# ---------------------------------------------------------------------------
# Monitor: ladder placement and budget
# ---------------------------------------------------------------------------


def make_monitor(max_escalations: int = 2) -> DoomLoopMonitor:
    config = resolve_doom_loop_option(
        {"ladder": ESCALATE_AT_2, "escalation": {"model": "openai/gpt-5", "max_escalations": max_escalations}}
    )
    return DoomLoopMonitor(config)


def _verdict(record: Mapping[str, Any]) -> Dict[str, Any]:
    verdict = record.get("verdict")
    assert verdict is not None
    return dict(verdict)


async def test_fires_escalate_between_steer_and_block_and_repeats_while_budget_remains() -> None:
    monitor = make_monitor()
    args = {"q": "same"}
    first = _verdict(await monitor.record_tool_call("search", args, 1))
    assert (first["action"], first["streak"]) == ("observe", 1)
    second = _verdict(await monitor.record_tool_call("search", args, 2))
    assert (second["action"], second["streak"]) == ("escalate", 2)
    # Budget not yet consumed (engine consumes at application time).
    third = _verdict(await monitor.record_tool_call("search", args, 3))
    assert (third["action"], third["streak"]) == ("escalate", 3)


async def test_falls_through_to_weaker_rungs_once_the_budget_is_exhausted() -> None:
    monitor = make_monitor(1)
    args = {"q": "same"}
    await monitor.record_tool_call("search", args, 1)
    assert _verdict(await monitor.record_tool_call("search", args, 2))["action"] == "escalate"
    monitor.consume_escalation()  # engine applied it
    assert monitor.can_escalate() is False
    # Streak 3: escalate unavailable -> observe (steer disabled, block at 5).
    third = _verdict(await monitor.record_tool_call("search", args, 3))
    assert (third["action"], third["streak"]) == ("observe", 3)
    await monitor.record_tool_call("search", args, 4)
    fifth = _verdict(await monitor.record_tool_call("search", args, 5))
    assert (fifth["action"], fifth["streak"]) == ("block", 5)


async def test_escalations_used_round_trips_through_serialized_state() -> None:
    monitor = make_monitor(2)
    monitor.consume_escalation()
    blob = json.loads(json.dumps(monitor.get_state()))
    assert blob["escalations_used"] == 1

    resumed = DoomLoopMonitor(
        resolve_doom_loop_option(
            {"ladder": ESCALATE_AT_2, "escalation": {"model": "openai/gpt-5", "max_escalations": 2}}
        ),
        blob,
    )
    assert resumed.can_escalate() is True
    resumed.consume_escalation()
    assert resumed.can_escalate() is False


# ---------------------------------------------------------------------------
# Engine: one-turn overrides
# ---------------------------------------------------------------------------


async def test_swaps_the_model_for_exactly_one_turn_then_reverts() -> None:
    script = Script()
    client = QueuedClient(
        [
            script.tool_call_turn("web_search", {"query": "same"}),
            script.tool_call_turn("web_search", {"query": "same"}),
            script.tool_call_turn("web_search", {"query": "different now"}),
            script.text_turn("recovered"),
        ]
    )

    text = await call_model(
        client,
        {
            "model": "cheap-model",
            "input": "go",
            "tools": [search_tool],
            "doom_loop": {"ladder": ESCALATE_AT_2, "escalation": {"model": "frontier-model"}},
        },
    ).get_text()

    assert text == "recovered"
    assert [request["model"] for request in client.requests] == [
        "cheap-model",
        "cheap-model",
        "frontier-model",  # follow-up after the escalate verdict
        "cheap-model",  # automatic revert
    ]


async def test_injects_the_advisor_tool_with_pinned_tool_choice_for_the_escalated_turn_only() -> None:
    script = Script()
    client = QueuedClient(
        [
            script.tool_call_turn("web_search", {"query": "same"}),
            script.tool_call_turn("web_search", {"query": "same"}),
            script.tool_call_turn("web_search", {"query": "unblocked"}),
            script.text_turn("done"),
        ]
    )

    await call_model(
        client,
        {
            "model": "cheap-model",
            "input": "go",
            "tools": [search_tool],
            "doom_loop": {"ladder": ESCALATE_AT_2, "escalation": {"advisor": True}},
        },
    ).get_text()

    escalated = client.requests[2]
    advisor = advisor_of(escalated)
    parameters = advisor["parameters"]
    assert parameters["forward_transcript"] is True
    assert "escalation advisor" in str(parameters["instructions"])
    # The configured function tool is still there, before the advisor.
    assert [t.get("name", t["type"]) for t in escalated["tools"]] == ["web_search", "openrouter:advisor"]
    assert escalated["tool_choice"] == {
        "type": "allowed_tools",
        "mode": "required",
        "tools": [{"type": "openrouter:advisor"}],
    }
    # Model unchanged (advisor-only escalation).
    assert escalated["model"] == "cheap-model"
    # Next dispatch reverts: no advisor, original tool choice.
    reverted = client.requests[3]
    assert all(t.get("type") != "openrouter:advisor" for t in reverted.get("tools") or [])
    assert reverted.get("tool_choice") is None
    # The dispatches before the escalation carried no advisor either.
    for request in client.requests[:2]:
        assert all(t.get("type") != "openrouter:advisor" for t in request.get("tools") or [])


async def test_does_not_consume_a_caller_choice_replaced_by_an_advisor_override() -> None:
    script = Script()
    client = QueuedClient(
        [
            script.tool_call_turn("web_search", {"query": "same"}),
            script.tool_call_turn("web_search", {"query": "same"}),
            script.tool_call_turn("web_search", {"query": "unblocked"}),
            script.text_turn("done"),
        ]
    )

    def tool_choice(ctx: Mapping[str, Any]) -> Any:
        return {"type": "function", "name": "web_search"} if ctx["number_of_turns"] >= 2 else "auto"

    await call_model(
        client,
        {
            "model": "cheap-model",
            "input": "go",
            "tools": [search_tool],
            "tool_choice": tool_choice,
            "doom_loop": {"ladder": ESCALATE_AT_2, "escalation": {"advisor": True}},
        },
    ).get_text()

    assert client.requests[2]["tool_choice"]["type"] == "allowed_tools"
    assert client.requests[2]["tool_choice"]["mode"] == "required"
    # The caller's web_search choice did not reach dispatch 2, so it remains
    # armed and is still forced after the one-turn advisor override.
    assert client.requests[3]["tool_choice"] == {"type": "function", "name": "web_search"}


async def test_advisor_object_config_merges_over_the_defaults() -> None:
    script = Script()
    client = QueuedClient(
        [
            script.tool_call_turn("web_search", {"query": "same"}),
            script.tool_call_turn("web_search", {"query": "same"}),
            script.text_turn("done"),
        ]
    )

    await call_model(
        client,
        {
            "model": "cheap-model",
            "input": "go",
            "tools": [search_tool],
            "doom_loop": {
                "ladder": ESCALATE_AT_2,
                "escalation": {"advisor": {"model": "anthropic/claude-opus-4.6", "forward_transcript": False}},
            },
        },
    ).get_text()

    parameters = advisor_of(client.requests[2])["parameters"]
    assert parameters["model"] == "anthropic/claude-opus-4.6"
    # Explicit config overrides the forward_transcript default.
    assert parameters["forward_transcript"] is False
    # Default instructions kept.
    assert "escalation advisor" in str(parameters["instructions"])


async def test_a_steer_notice_accompanies_the_escalated_turn_so_the_model_knows_why() -> None:
    script = Script()
    client = QueuedClient(
        [
            script.tool_call_turn("web_search", {"query": "same"}),
            script.tool_call_turn("web_search", {"query": "same"}),
            script.text_turn("done"),
        ]
    )

    await call_model(
        client,
        {
            "model": "cheap-model",
            "input": "go",
            "tools": [search_tool],
            "doom_loop": {"ladder": ESCALATE_AT_2, "escalation": {"model": "frontier-model"}},
        },
    ).get_text()

    escalated_input = client.requests[2]["input"]
    notices = [
        item
        for item in escalated_input
        if isinstance(item, Mapping)
        and item.get("role") == "user"
        and isinstance(item.get("content"), str)
        and "escalated turn follows" in item["content"]
    ]
    assert len(notices) == 1


async def test_consumes_budget_per_applied_escalation_and_falls_through_when_exhausted() -> None:
    script = Script()
    hooks = HooksManager()
    detections = collect_detections(hooks)
    client = QueuedClient(
        [*[script.tool_call_turn("web_search", {"query": "same"}) for _ in range(6)], script.text_turn("gave up")]
    )
    accessor = MemoryStateAccessor()

    await call_model(
        client,
        {
            "model": "cheap-model",
            "input": "go",
            "tools": [search_tool],
            "doom_loop": {
                "ladder": ESCALATE_AT_2,
                "escalation": {"model": "frontier-model", "max_escalations": 1},
            },
            "state": accessor,
            "hooks": hooks,
        },
    ).get_text()

    # The final "gave up" text also crosses observe@1 via the text-streak
    # detector; this asserts the TOOL streak ladder.
    assert [[d["action"], d["streak"]] for d in detections if d["detector"] == "tool-fingerprint"] == [
        ["observe", 1],
        ["escalate", 2],
        ["observe", 3],
        ["observe", 4],
        ["block", 5],
        ["block", 6],
    ]
    # Exactly ONE dispatch used the frontier model.
    assert [request["model"] for request in client.requests].count("frontier-model") == 1
    assert len(client.requests) == 7
    # Consumed budget persisted for resumes.
    assert accessor.stored is not None
    assert accessor.stored.doom_loop is not None
    assert accessor.stored.doom_loop["escalations_used"] == 1


async def test_hook_override_action_escalate_is_honored_with_budget_and_downgraded_without_config() -> None:
    ladder = {"observe": 2, "block": 5, "stop": 8}

    script = Script()
    hooks = HooksManager()
    hooks.on("DoomLoopDetected", HookEntry(handler=lambda p, c: {"override_action": "escalate"}))
    client = QueuedClient(
        [
            script.tool_call_turn("web_search", {"query": "same"}),
            script.tool_call_turn("web_search", {"query": "same"}),
            script.text_turn("done"),
        ]
    )
    await call_model(
        client,
        {
            "model": "cheap-model",
            "input": "go",
            "tools": [search_tool],
            "doom_loop": {"ladder": ladder, "escalation": {"model": "frontier-model"}},
            "hooks": hooks,
        },
    ).get_text()
    assert client.requests[2]["model"] == "frontier-model"

    # Without a config: the override downgrades to observe -- no crash, no
    # model change.
    script2 = Script()
    hooks2 = HooksManager()
    hooks2.on("DoomLoopDetected", HookEntry(handler=lambda p, c: {"override_action": "escalate"}))
    client2 = QueuedClient(
        [
            script2.tool_call_turn("web_search", {"query": "same"}),
            script2.tool_call_turn("web_search", {"query": "same"}),
            script2.text_turn("done"),
        ]
    )
    text = await call_model(
        client2,
        {
            "model": "cheap-model",
            "input": "go",
            "tools": [search_tool],
            "doom_loop": {"ladder": ladder},
            "hooks": hooks2,
        },
    ).get_text()
    assert text == "done"
    assert [request["model"] for request in client2.requests] == ["cheap-model", "cheap-model", "cheap-model"]


async def test_text_detector_verdicts_can_escalate_too() -> None:
    script = Script()
    filler = "Let me look somewhere else."
    client = QueuedClient(
        [
            script.text_and_call_turn(filler, "q1"),
            script.text_and_call_turn(filler, "q2"),
            script.tool_call_turn("web_search", {"query": "q3"}),
            script.text_turn("unblocked"),
        ]
    )

    await call_model(
        client,
        {
            "model": "cheap-model",
            "input": "go",
            "tools": [search_tool],
            "doom_loop": {"ladder": ESCALATE_AT_2, "escalation": {"model": "frontier-model"}, "text": True},
        },
    ).get_text()

    # The text streak hits 2 on the second response; the NEXT follow-up is
    # the (only) escalated dispatch.
    assert [request["model"] for request in client.requests] == [
        "cheap-model",
        "cheap-model",
        "frontier-model",
        "cheap-model",
    ]


async def test_two_detectors_firing_in_one_window_escalate_once() -> None:
    script = Script()

    def same_everything() -> Dict[str, Any]:
        return script.text_and_call_turn("Trying the same thing.", "same")

    accessor = MemoryStateAccessor()
    client = QueuedClient([same_everything(), same_everything(), script.text_turn("done")])

    await call_model(
        client,
        {
            "model": "cheap-model",
            "input": "go",
            "tools": [search_tool],
            "doom_loop": {
                "ladder": ESCALATE_AT_2,
                "escalation": {"model": "frontier-model", "max_escalations": 5},
            },
            "state": accessor,
        },
    ).get_text()

    assert accessor.stored is not None
    assert accessor.stored.doom_loop is not None
    assert accessor.stored.doom_loop["escalations_used"] == 1
    assert [request["model"] for request in client.requests].count("frontier-model") == 1
