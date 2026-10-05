"""Regression suite for the same-tool fan-out gap (port of upstream
``tests/unit/doom-loop-fanout.test.ts``).

A streak keyed on a tool's *last* fingerprint cannot see a repeating fan-out:
``read(a), read(b), read(c)`` reissued verbatim has a different last call every
round. A round's identity for one tool is therefore the *set* of fingerprints
it was called with, declared before any of the round's calls is scored. The
superset, order-permutation, and expanding-fan-out cases guard the
accumulate-as-you-go regression (a superset round transiently matching the
previous one, order-dependently).

Upstream's unhashable stand-in is a bigint (``{ size: 1n }``); the Python
analogue is a set (``{"size": {1}}``), which ``canonicalize_key_material``
rejects the same way.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

import pytest

from openrouter_agent.doom_loop import DoomLoopCallRecord, DoomLoopMonitor, resolve_doom_loop_option

UNHASHABLE: Dict[str, Any] = {"size": {1}}


def monitor() -> DoomLoopMonitor:
    return DoomLoopMonitor(resolve_doom_loop_option(True))


def steer_monitor() -> DoomLoopMonitor:
    """``{"ladder": {"steer": 2}}`` — as upstream. That ladder makes the default
    observe rung (2) dead, which resolve_doom_loop_option warns about exactly
    once; asserted here so the expected warning is pinned, not leaked."""
    with pytest.warns(UserWarning) as caught:
        config = resolve_doom_loop_option({"ladder": {"steer": 2}})
    assert [str(w.message) for w in caught] == [
        '[DoomLoop] ladder rung "observe" (2) can never fire: stronger rung "steer" (2) already wins at that '
        "streak (strongest crossed rung is applied)."
    ]
    return DoomLoopMonitor(config)


def _calls(paths: Sequence[str], tool_name: str = "read") -> List[Dict[str, Any]]:
    return [{"tool_name": tool_name, "key_material": {"path": path}} for path in paths]


def _action(record: DoomLoopCallRecord) -> str:
    verdict = record.get("verdict")
    return verdict["action"] if verdict is not None else "none"


async def play_rounds(fanouts: Sequence[Sequence[str]]) -> List[List[str]]:
    """Plays each fan-out as one round, declaring the round's calls first — the
    engine does the same at every execution-batch boundary."""
    detector = monitor()
    actions: List[List[str]] = []
    for round_, paths in enumerate(fanouts):
        await detector.declare_round(round_, _calls(paths))
        round_actions: List[str] = []
        for path in paths:
            record = await detector.record_tool_call("read", {"path": path}, round_)
            round_actions.append(_action(record))
        actions.append(round_actions)
    return actions


async def test_accumulates_across_rounds_that_repeat_a_distinct_argument_fanout() -> None:
    actions = await play_rounds([["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]])
    # Every call in a repeating round reports that round's streak.
    assert actions == [
        ["none", "none", "none"],
        ["observe", "observe", "observe"],
        ["block", "block", "block"],
    ]


async def test_is_order_insensitive_within_the_round() -> None:
    actions = await play_rounds([["a", "b", "c"], ["c", "a", "b"], ["b", "c", "a"]])
    assert actions[1] == ["observe", "observe", "observe"]
    assert actions[2] == ["block", "block", "block"]


async def test_scores_each_call_individually_when_fanout_membership_changes() -> None:
    actions = await play_rounds([["a", "b"], ["a", "b"], ["a", "z"], ["a", "z"]])
    assert actions[1] == ["observe", "observe"]
    # The ROUND identity resets, but `a` is on its third consecutive round.
    assert actions[2] == ["block", "none"]
    assert actions[3] == ["block", "observe"]


async def test_flags_the_calls_of_a_partial_repeat_that_actually_repeated() -> None:
    actions = await play_rounds([["a", "b", "c"], ["a", "b"]])
    assert actions[1] == ["observe", "observe"]


async def test_flags_repeated_members_of_a_superset_round_never_the_new_one() -> None:
    actions = await play_rounds([["a", "b"], ["a", "b"], ["a", "b", "c"]])
    assert actions[1] == ["observe", "observe"]
    assert actions[2] == ["block", "block", "none"]


async def test_scores_a_superset_round_the_same_whatever_order_it_is_emitted_in() -> None:
    in_order = await play_rounds([["a", "b"], ["a", "b"], ["a", "b", "c"]])
    permuted = await play_rounds([["a", "b"], ["a", "b"], ["c", "a", "b"]])
    assert in_order[2] == ["block", "block", "none"]
    assert permuted[2] == ["none", "block", "block"]


async def test_scores_an_expanding_fanout_per_call() -> None:
    actions = await play_rounds([["a"], ["a", "b"], ["a", "b", "c"], ["a", "b", "c", "d"]])
    assert actions == [
        ["none"],
        ["observe", "none"],
        ["block", "observe", "none"],
        ["block", "block", "observe", "none"],
    ]


async def test_leaves_single_call_rounds_behaving_exactly_as_before() -> None:
    actions = await play_rounds([["a"], ["a"], ["a"], ["a"]])
    assert [action for round_ in actions for action in round_] == ["none", "observe", "block", "block"]


async def test_still_counts_identical_duplicates_within_one_round_only_once() -> None:
    detector = monitor()
    first = await detector.record_tool_call("read", {"path": "a"}, 0)
    second = await detector.record_tool_call("read", {"path": "a"}, 0)
    assert first["duplicate_in_round"] is False
    assert second["duplicate_in_round"] is True
    assert second["streak"] == first["streak"] == 1


async def test_keeps_a_resumed_single_call_streak_incrementing_after_restore() -> None:
    detector = monitor()
    await detector.record_tool_call("read", {"path": "a"}, 0)
    await detector.record_tool_call("read", {"path": "a"}, 1)

    resumed = DoomLoopMonitor(resolve_doom_loop_option(True), detector.get_state())
    record = await resumed.record_tool_call("read", {"path": "a"}, 0)
    assert record["streak"] == 3


async def test_falls_back_to_per_call_scoring_when_a_round_is_never_declared() -> None:
    detector = monitor()
    await detector.record_tool_call("server:web_search", {"q": "x"}, 0)
    repeated = await detector.record_tool_call("server:web_search", {"q": "x"}, 1)
    fresh = await detector.record_tool_call("server:web_search", {"q": "y"}, 1)

    assert repeated["streak"] == 2
    assert fresh["streak"] == 1
    assert "verdict" not in fresh


async def test_scores_an_undeclared_multi_call_round_per_call_order_independently() -> None:
    async def undeclared(rounds: Sequence[Sequence[str]]) -> List[List[str]]:
        detector = monitor()
        out: List[List[str]] = []
        for round_, paths in enumerate(rounds):
            scored: List[str] = []
            for path in paths:
                record = await detector.record_tool_call("read", {"path": path}, round_)
                scored.append(f"{record['streak']}:{_action(record)}")
            out.append(scored)
        return out

    assert await undeclared([["a", "b"], ["a", "b"]]) == [["1:none", "1:none"], ["2:observe", "2:observe"]]
    assert await undeclared([["a", "b"], ["b", "a"]]) == [["1:none", "1:none"], ["2:observe", "2:observe"]]
    assert await undeclared([["b", "a"], ["b", "c"], ["b", "d"]]) == [
        ["1:none", "1:none"],
        ["2:observe", "1:none"],
        ["3:block", "1:none"],
    ]


@pytest.mark.filterwarnings("ignore:.*could not fingerprint")
async def test_does_not_let_a_call_outside_the_declared_set_inherit_the_round_streak() -> None:
    detector = monitor()
    hashable = {"path": "a"}
    for round_ in (0, 1):
        await detector.declare_round(
            round_,
            [
                {"tool_name": "read", "key_material": hashable},
                # Unhashable: dropped from the declared set.
                {"tool_name": "read", "key_material": UNHASHABLE},
            ],
        )
        await detector.record_tool_call("read", hashable, round_)

    await detector.declare_round(
        2,
        [
            {"tool_name": "read", "key_material": hashable},
            {"tool_name": "read", "key_material": UNHASHABLE},
        ],
    )
    member = await detector.record_tool_call("read", hashable, 2)
    # Recorded with a fallback identity, for the FIRST time.
    dropped = await detector.record_tool_call("read", {"size": "fallback-identity"}, 2)

    assert member["streak"] == 3
    assert dropped["streak"] == 1
    assert "verdict" not in dropped


async def test_declare_round_warns_once_per_unhashable_call() -> None:
    # Port-side: upstream logs via console.warn (not asserted upstream); pin
    # that the warning names the tool and the round.
    detector = monitor()
    with pytest.warns(UserWarning) as caught:
        await detector.declare_round(
            7,
            [
                {"tool_name": "read", "key_material": {"path": "a"}},
                {"tool_name": "read", "key_material": UNHASHABLE},
            ],
        )
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1
    assert messages[0].startswith('[DoomLoop] could not fingerprint a "read" call while declaring round 7;')


@pytest.mark.filterwarnings("ignore:.*could not fingerprint")
async def test_accumulates_per_call_evidence_for_an_unhashable_call_reissued_verbatim() -> None:
    detector = monitor()
    dropped_streaks: List[int] = []
    member_streaks: List[int] = []
    for round_ in (0, 1, 2):
        await detector.declare_round(
            round_,
            [
                {"tool_name": "read", "key_material": {"path": "a"}},
                {"tool_name": "read", "key_material": UNHASHABLE},
            ],
        )
        member_streaks.append((await detector.record_tool_call("read", {"path": "a"}, round_))["streak"])
        dropped_streaks.append(
            (await detector.record_tool_call("read", {"size": "fallback-identity"}, round_))["streak"]
        )

    assert member_streaks == [1, 2, 3]
    assert dropped_streaks == [1, 2, 3]


@pytest.mark.filterwarnings("ignore:.*could not fingerprint")
async def test_keeps_accumulating_when_an_unhashable_call_rides_along_every_round() -> None:
    async def streaks_for(dropped_first: bool) -> List[int]:
        detector = monitor()
        streaks: List[int] = []
        for round_ in (0, 1, 2, 3):
            await detector.declare_round(
                round_,
                [
                    {"tool_name": "read", "key_material": {"path": "a"}},
                    {"tool_name": "read", "key_material": UNHASHABLE},
                ],
            )
            if dropped_first:
                await detector.record_tool_call("read", {"size": "fallback-identity"}, round_)
            member = await detector.record_tool_call("read", {"path": "a"}, round_)
            if not dropped_first:
                await detector.record_tool_call("read", {"size": "fallback-identity"}, round_)
            streaks.append(member["streak"])
        return streaks

    assert await streaks_for(False) == [1, 2, 3, 4]
    assert await streaks_for(True) == [1, 2, 3, 4]


@pytest.mark.filterwarnings("ignore:.*could not fingerprint")
async def test_persists_the_streak_against_the_member_that_earned_it() -> None:
    detector = monitor()
    for round_ in (0, 1):
        await detector.declare_round(
            round_,
            [
                {"tool_name": "read", "key_material": {"path": "a"}},
                {"tool_name": "read", "key_material": UNHASHABLE},
            ],
        )
        await detector.record_tool_call("read", {"path": "a"}, round_)
        # Recorded LAST, so it used to become the persisted identity.
        await detector.record_tool_call("read", {"size": "fallback-identity"}, round_)

    resumed_non_member = DoomLoopMonitor(resolve_doom_loop_option(True), detector.get_state())
    own_repeat = await resumed_non_member.record_tool_call("read", {"size": "fallback-identity"}, 0)
    assert own_repeat["streak"] == 3

    resumed_fresh = DoomLoopMonitor(resolve_doom_loop_option(True), detector.get_state())
    fresh = await resumed_fresh.record_tool_call("read", {"size": "never-seen-before"}, 0)
    assert fresh["streak"] == 1
    assert "verdict" not in fresh

    resumed_member = DoomLoopMonitor(resolve_doom_loop_option(True), detector.get_state())
    await resumed_member.declare_round(0, _calls(["a"]))
    real_repeat = await resumed_member.record_tool_call("read", {"path": "a"}, 0)
    assert real_repeat["streak"] == 3


async def test_keeps_counting_a_recorded_call_when_a_phantom_member_disappears() -> None:
    detector = monitor()
    streaks: List[int] = []
    for round_ in (0, 1, 2, 3):
        # Rounds 0-1 declare a phantom alongside the real call; 2-3 do not.
        await detector.declare_round(round_, _calls(["a", "phantom"] if round_ < 2 else ["a"]))
        streaks.append((await detector.record_tool_call("read", {"path": "a"}, round_))["streak"])
    assert streaks == [1, 2, 3, 4]


async def test_gives_every_call_of_a_repeating_round_the_same_message_so_steer_dedupes() -> None:
    detector = steer_monitor()
    paths = ["a", "b", "c"]
    messages: List[str] = []
    for round_ in (0, 1):
        await detector.declare_round(round_, _calls(paths))
        for path in paths:
            record = await detector.record_tool_call("read", {"path": path}, round_)
            verdict = record.get("verdict")
            if round_ == 1 and verdict is not None:
                messages.append(verdict["message"])

    assert len(messages) == 3
    assert len(set(messages)) == 1
    assert "3 parallel calls" in messages[0]


async def test_continues_a_resumed_fanout_streak_exactly_like_a_single_call_streak() -> None:
    detector = monitor()
    for round_ in (0, 1):
        await detector.declare_round(round_, _calls(["a", "b"]))
        for path in ("a", "b"):
            await detector.record_tool_call("read", {"path": path}, round_)

    wire = json.loads(json.dumps(detector.get_state()))
    resumed = DoomLoopMonitor(resolve_doom_loop_option(True), wire)
    await resumed.declare_round(0, _calls(["a", "b"]))
    record = await resumed.record_tool_call("read", {"path": "a"}, 0)

    assert record["streak"] == 3
    assert _action(record) == "block"


async def test_keeps_counting_a_repeat_when_a_paused_hitl_member_drops_from_the_resumed_round() -> None:
    detector = monitor()
    work_streaks: List[str] = []
    for round_ in (0, 1, 2):
        members = ["work", "gated"] if round_ == 0 else ["work"]
        await detector.declare_round(round_, _calls(members, "deploy"))
        for path in members:
            record = await detector.record_tool_call("deploy", {"path": path}, round_)
            if path == "work":
                work_streaks.append(f"{record['streak']}:{_action(record)}")
    assert work_streaks == ["1:none", "2:observe", "3:block"]


async def test_collapses_staggered_per_call_counts_in_one_round_to_a_single_message() -> None:
    detector = steer_monitor()
    rounds = [["a"], ["a", "b"], ["a", "b", "c"], ["a", "b", "c", "d"]]
    messages: List[str] = []
    streaks: List[int] = []
    for round_, paths in enumerate(rounds):
        await detector.declare_round(round_, _calls(paths))
        for path in paths:
            record = await detector.record_tool_call("read", {"path": path}, round_)
            verdict = record.get("verdict")
            if round_ == 3 and verdict is not None:
                messages.append(verdict["message"])
                streaks.append(verdict["streak"])

    assert streaks == [4, 3, 2]
    assert len(messages) == 3
    assert len(set(messages)) == 1


async def test_bounds_a_mixed_evidence_round_to_one_message_per_distinct_fact() -> None:
    detector = steer_monitor()
    rounds = [["a"], ["a", "b"], ["a", "b"]]
    last_round = len(rounds) - 1
    messages: List[str] = []
    for round_, paths in enumerate(rounds):
        await detector.declare_round(round_, _calls(paths))
        for path in paths:
            record = await detector.record_tool_call("read", {"path": path}, round_)
            verdict = record.get("verdict")
            if round_ == last_round and verdict is not None:
                messages.append(verdict["message"])

    assert len(messages) == 2
    assert len(set(messages)) == 2
    assert 'this exact "read" call has been repeated' in messages[0]
    assert "same set of 2 parallel calls" in messages[1]


async def test_collapses_per_call_steer_messages_across_a_wide_repeating_round() -> None:
    detector = steer_monitor()
    wide = [f"f{index}" for index in range(20)]
    messages: set[str] = set()
    verdict_count = 0
    for round_, paths in enumerate([wide, wide, [*wide, "new"]]):
        await detector.declare_round(round_, _calls(paths))
        for path in paths:
            record = await detector.record_tool_call("read", {"path": path}, round_)
            verdict = record.get("verdict")
            if round_ == 2 and verdict is not None:
                verdict_count += 1
                messages.add(verdict["message"])

    assert verdict_count == 20
    assert len(messages) == 1


async def test_persists_per_call_evidence_when_the_round_has_shrunk_to_a_single_call() -> None:
    detector = monitor()
    await detector.declare_round(0, _calls(["work", "gated"], "deploy"))
    await detector.record_tool_call("deploy", {"path": "work"}, 0)
    await detector.record_tool_call("deploy", {"path": "gated"}, 0)
    # The gated call paused; the resumed round is just the working call.
    await detector.declare_round(1, _calls(["work"], "deploy"))
    before_save = await detector.record_tool_call("deploy", {"path": "work"}, 1)
    assert before_save["streak"] == 2

    wire = json.loads(json.dumps(detector.get_state()))
    resumed = DoomLoopMonitor(resolve_doom_loop_option(True), wire)
    await resumed.declare_round(0, _calls(["work"], "deploy"))
    after_resume = await resumed.record_tool_call("deploy", {"path": "work"}, 0)
    assert after_resume["streak"] == 3
    assert _action(after_resume) == "block"


async def test_scores_a_resumed_single_call_on_its_own_earned_evidence_never_inherited() -> None:
    detector = monitor()
    paths = ["a", "b", "c"]
    for round_ in (0, 1):
        await detector.declare_round(round_, _calls(paths))
        for path in paths:
            await detector.record_tool_call("read", {"path": path}, round_)

    saved = detector.get_state()
    assert saved["tools"]["read"]["streak"] == 2
    assert len(saved["tools"]["read"].get("round_fingerprints", [])) == 3
    # Steady state: per-call counts equal the round streak, so they are omitted.
    assert "call_streaks" not in saved["tools"]["read"]

    for path in paths:
        resumed = DoomLoopMonitor(resolve_doom_loop_option(True), detector.get_state())
        await resumed.declare_round(0, _calls([path]))
        solo = await resumed.record_tool_call("read", {"path": path}, 0)
        assert solo["streak"] == 3
        assert _action(solo) == "block"

    resumed_fresh = DoomLoopMonitor(resolve_doom_loop_option(True), detector.get_state())
    await resumed_fresh.declare_round(0, _calls(["never-before"]))
    fresh = await resumed_fresh.record_tool_call("read", {"path": "never-before"}, 0)
    assert fresh["streak"] == 1
    assert "verdict" not in fresh


async def test_renders_one_verdict_text_per_undeclared_multi_call_round() -> None:
    detector = DoomLoopMonitor(resolve_doom_loop_option(True))
    distinct_per_round: List[int] = []
    for round_ in range(1, 5):
        texts: List[str] = []
        for cmd in ("a", "b"):
            result = await detector.record_tool_call("sh", {"cmd": cmd}, round_, allow_block=False)
            verdict = result.get("verdict")
            if verdict is not None:
                texts.append(verdict["message"])
        distinct_per_round.append(len(set(texts)))
    # Round 1: no verdict yet; every later round: both members, ONE text.
    assert distinct_per_round == [0, 1, 1, 1]


async def test_keeps_the_fingerprint_bearing_text_for_a_genuine_single_call_round() -> None:
    detector = DoomLoopMonitor(resolve_doom_loop_option(True))
    last = None
    for round_ in range(1, 4):
        await detector.declare_round(round_, _calls(["same"]))
        result = await detector.record_tool_call("read", {"path": "same"}, round_)
        verdict = result.get("verdict")
        last = verdict["message"] if verdict is not None else last
    assert last is not None
    assert "identical arguments (fingerprint" in last
