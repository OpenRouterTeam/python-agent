"""Consumer-facing contract for driving ``DoomLoopMonitor`` directly (port of
upstream ``tests/unit/doom-loop-public-api.test.ts``).

Upstream imports everything from the package entrypoint ONLY, exactly as a
consumer or SDK port would. This port's ``openrouter_agent/__init__.py`` does
not re-export the doom-loop surface yet (being wired up separately), so these
cases import from ``openrouter_agent.doom_loop``; the entrypoint presence check
is the skipped test at the bottom.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from openrouter_agent.doom_loop import DEFAULT_DOOM_LOOP_LADDER, DoomLoopMonitor, resolve_doom_loop_option


def _calls(paths: List[str]) -> List[Dict[str, Any]]:
    return [{"tool_name": "read", "key_material": {"path": path}} for path in paths]


async def test_constructs_with_defaults_detects_a_repeating_fanout_and_honors_the_exported_ladder() -> None:
    monitor = DoomLoopMonitor(resolve_doom_loop_option(True))
    paths = ["a", "b", "c"]

    verdicts: List[str] = []
    verdict_streak: Optional[int] = None
    for round_ in (0, 1, 2):
        await monitor.declare_round(round_, _calls(paths))
        for path in paths:
            record = await monitor.record_tool_call("read", {"path": path}, round_)
            verdict = record.get("verdict")
            if verdict is not None:
                verdicts.append(verdict["action"])
                verdict_streak = verdict["streak"]

    # Third identical round crosses the exported default block threshold.
    assert verdict_streak == DEFAULT_DOOM_LOOP_LADDER["block"]
    assert verdicts == ["observe"] * 3 + ["block"] * 3


async def test_accepts_a_config_object_and_honors_a_custom_ladder() -> None:
    monitor = DoomLoopMonitor(resolve_doom_loop_option({"ladder": {"observe": 2, "block": False, "stop": 4}}))
    actions: List[Optional[str]] = []
    for round_ in (0, 1, 2, 3):
        record = await monitor.record_tool_call("search", {"q": "same"}, round_)
        verdict = record.get("verdict")
        actions.append(verdict["action"] if verdict is not None else None)
    # block disabled; the streak of 4 reaches the custom stop rung.
    assert actions == [None, "observe", "observe", "stop"]


async def test_accumulates_a_fanout_streak_across_per_turn_process_boundaries() -> None:
    # The serverless pattern: one round per call_model run, state persisted
    # between turns. The round's fingerprint SET is persisted for this.
    paths = ["a", "b", "c"]
    wire: Any = None
    per_turn: List[str] = []
    for _turn in range(4):
        monitor = DoomLoopMonitor(resolve_doom_loop_option(True), wire)
        await monitor.declare_round(0, _calls(paths))
        last = ""
        for path in paths:
            record = await monitor.record_tool_call("read", {"path": path}, 0)
            verdict = record.get("verdict")
            last = f"{record['streak']}:{verdict['action'] if verdict is not None else 'none'}"
        per_turn.append(last)
        wire = json.loads(json.dumps(monitor.get_state()))

    assert per_turn == ["1:none", "2:observe", "3:block", "4:block"]


async def test_restores_a_legacy_blob_without_round_fingerprints_with_single_call_semantics() -> None:
    legacy = {"tools": {"read": {"fingerprint": "not-a-real-fingerprint", "streak": 2}}}
    monitor = DoomLoopMonitor(resolve_doom_loop_option(True), legacy)
    different = await monitor.record_tool_call("read", {"path": "x"}, 0)
    assert different["streak"] == 1

    hostile = {"tools": {"read": {"fingerprint": "ab", "streak": 2, "round_fingerprints": [1, {}, None]}}}
    survives = DoomLoopMonitor(resolve_doom_loop_option(True), hostile)
    record = await survives.record_tool_call("read", {"path": "x"}, 0)
    assert record["streak"] == 1


async def test_isolates_the_live_detector_from_mutations_of_a_saved_snapshot() -> None:
    paths = ["a", "b"]
    monitor = DoomLoopMonitor(resolve_doom_loop_option(True))
    for round_ in (0, 1):
        await monitor.declare_round(round_, _calls(paths))
        for path in paths:
            await monitor.record_tool_call("read", {"path": path}, round_)

    # A careless (or hostile) caller mutates the saved snapshot in place.
    saved = monitor.get_state()
    saved["tools"]["read"]["round_fingerprints"].append("INJECTED")

    await monitor.declare_round(2, _calls(paths))
    last: Dict[str, Any] = {"streak": 0}
    for path in paths:
        record = await monitor.record_tool_call("read", {"path": path}, 2)
        verdict = record.get("verdict")
        last = {"streak": record["streak"], "action": verdict["action"] if verdict is not None else None}
    assert last == {"streak": 3, "action": "block"}


async def test_round_trips_state_across_a_process_boundary_via_plain_json() -> None:
    first = DoomLoopMonitor(resolve_doom_loop_option(True))
    await first.record_tool_call("search", {"q": "same"}, 0)
    await first.record_tool_call("search", {"q": "same"}, 1)

    wire = json.dumps(first.get_state())
    second = DoomLoopMonitor(resolve_doom_loop_option(True), json.loads(wire))
    resumed = await second.record_tool_call("search", {"q": "same"}, 0)

    # Single-call streak continues across the boundary: 2 -> 3.
    assert resumed["streak"] == 3


@pytest.mark.skip(reason="pending openrouter_agent/__init__.py doom-loop re-exports")
def test_doom_loop_surface_is_importable_from_the_package_entrypoint() -> None:
    import openrouter_agent

    from openrouter_agent import doom_loop

    for name in ("DEFAULT_DOOM_LOOP_LADDER", "DoomLoopMonitor", "resolve_doom_loop_option"):
        assert getattr(openrouter_agent, name) is getattr(doom_loop, name)
