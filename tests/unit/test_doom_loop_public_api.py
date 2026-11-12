"""Consumer-facing contract for driving `DoomLoopMonitor` directly.

Port of tests/unit/doom-loop-public-api.test.ts. Upstream imports from the
package entrypoint; until `openrouter_agent/__init__.py` re-exports the
doom-loop symbols this imports from `openrouter_agent.doom_loop`. Once the
re-export lands, switch the import to `openrouter_agent` so the test pins the
public surface again.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

from openrouter_agent.doom_loop import (
    DEFAULT_DOOM_LOOP_LADDER,
    DoomLoopMonitor,
    DoomLoopRoundCall,
    resolve_doom_loop_option,
)


def _read_calls(paths: List[str]) -> List[DoomLoopRoundCall]:
    return [{"tool_name": "read", "key_material": {"path": path}} for path in paths]


async def test_constructs_with_defaults_detects_a_repeating_fan_out_and_honors_the_exported_ladder() -> None:
    monitor = DoomLoopMonitor(resolve_doom_loop_option(True))
    paths = ["a", "b", "c"]

    verdict_action: Optional[str] = None
    verdict_streak: Optional[int] = None
    for round_number in (0, 1, 2):
        await monitor.declare_round(round_number, _read_calls(paths))
        for path in paths:
            record = await monitor.record_tool_call("read", {"path": path}, round_number)
            verdict = record.get("verdict")
            if verdict is not None:
                verdict_action = verdict["action"]
                verdict_streak = verdict["streak"]

    # Third identical round crosses the exported default block threshold.
    assert verdict_streak == DEFAULT_DOOM_LOOP_LADDER["block"]
    assert verdict_action == "block"


async def test_accepts_a_config_object_and_honors_a_custom_ladder() -> None:
    monitor = DoomLoopMonitor(resolve_doom_loop_option({"ladder": {"observe": 2, "block": False, "stop": 4}}))

    actions: List[Optional[str]] = []
    for round_number in (0, 1, 2, 3):
        record = await monitor.record_tool_call("search", {"q": "same"}, round_number)
        verdict = record.get("verdict")
        actions.append(verdict["action"] if verdict else None)
    # block disabled; the streak of 4 reaches the custom stop rung.
    assert actions[-1] == "stop"
    assert actions == [None, "observe", "observe", "stop"]


async def test_accumulates_a_fan_out_streak_across_per_turn_process_boundaries() -> None:
    # The serverless pattern: one round per call_model run, state persisted
    # between turns. A repeating fan-out must accumulate across boundaries —
    # this is why the round's fingerprint SET is persisted.
    paths = ["a", "b", "c"]
    wire: Any = None
    per_turn: List[str] = []
    for _turn in range(4):
        monitor = DoomLoopMonitor(resolve_doom_loop_option(True), wire)
        await monitor.declare_round(0, _read_calls(paths))
        last = ""
        for path in paths:
            record = await monitor.record_tool_call("read", {"path": path}, 0)
            verdict = record.get("verdict")
            last = f"{record['streak']}:{verdict['action'] if verdict else 'none'}"
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
    # The malformed set degraded to the lone fingerprint rather than dropping the entry.
    assert survives.get_state()["tools"]["read"]["streak"] == 1


async def test_isolates_the_live_detector_from_mutations_of_a_saved_snapshot() -> None:
    paths = ["a", "b"]
    monitor = DoomLoopMonitor(resolve_doom_loop_option(True))
    for round_number in (0, 1):
        await monitor.declare_round(round_number, _read_calls(paths))
        for path in paths:
            await monitor.record_tool_call("read", {"path": path}, round_number)

    # A careless (or hostile) caller mutates the saved snapshot in place.
    saved = monitor.get_state()
    round_fingerprints = saved["tools"]["read"].get("round_fingerprints")
    assert round_fingerprints is not None
    assert len(round_fingerprints) == 2
    round_fingerprints.append("INJECTED")

    # The live detector must be unaffected: round 3 still blocks.
    await monitor.declare_round(2, _read_calls(paths))
    last_streak = 0
    last_action: Optional[str] = None
    for path in paths:
        record = await monitor.record_tool_call("read", {"path": path}, 2)
        verdict = record.get("verdict")
        last_streak = record["streak"]
        last_action = verdict["action"] if verdict else None
    assert last_streak == 3
    assert last_action == "block"


async def test_round_trips_state_across_a_process_boundary_via_plain_json() -> None:
    first = DoomLoopMonitor(resolve_doom_loop_option(True))
    await first.record_tool_call("search", {"q": "same"}, 0)
    await first.record_tool_call("search", {"q": "same"}, 1)

    # Simulate the serverless pattern: serialize, "new process", restore.
    wire = json.dumps(first.get_state())
    second = DoomLoopMonitor(resolve_doom_loop_option(True), json.loads(wire))
    resumed = await second.record_tool_call("search", {"q": "same"}, 0)

    # Single-call streak continues across the boundary: 2 -> 3.
    assert resumed["streak"] == 3
