"""Unit tests for the doom-loop detection primitives (port of tests/unit/doom-loop.test.ts).

Everything here is pure and deterministic: JCS canonicalization, SHA-256
fingerprints (asserted against the cross-port vector file), text-repetition
detection, ladder resolution, loop_key resolution, and the DoomLoopMonitor
state machine (round-scoped streaks). Loop integration (blocking, steering,
stopping a real run) is upstream's doom-loop-integration.test.ts, which needs
the engine wiring and is not part of this file.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from openrouter_agent.doom_loop import (
    DEFAULT_DOOM_LOOP_LADDER,
    MAX_CANONICALIZE_DEPTH,
    DoomLoopCallRecord,
    DoomLoopLadder,
    DoomLoopMonitor,
    DoomLoopOption,
    DoomLoopVerdict,
    canonicalize_key_material,
    detect_text_repetition,
    fingerprint_key_material,
    fingerprint_tool_call,
    resolve_doom_loop_option,
    resolve_ladder_action,
    resolve_loop_key_material,
)

VECTORS_PATH = Path(__file__).resolve().parent.parent / "vectors" / "doom-loop-fingerprints.json"
VECTORS: Dict[str, Any] = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))


def _subset(actual: Optional[Any], expected: Dict[str, Any]) -> None:
    """`toMatchObject`: every expected key present with an equal value."""
    assert actual is not None
    for key, value in expected.items():
        assert key in actual, key
        assert actual[key] == value, (key, actual[key], value)


def _make_monitor(overrides: Optional[DoomLoopOption] = None) -> DoomLoopMonitor:
    return DoomLoopMonitor(resolve_doom_loop_option(True if overrides is None else overrides))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Canonicalization (RFC 8785 / JCS semantics)
# ---------------------------------------------------------------------------


class TestCanonicalizeKeyMaterial:
    def test_is_insensitive_to_object_key_order_recursively(self) -> None:
        assert canonicalize_key_material({"b": 2, "a": {"d": 4, "c": 3}}) == canonicalize_key_material(
            {"a": {"c": 3, "d": 4}, "b": 2}
        )
        assert canonicalize_key_material({"b": 2, "a": {"d": 4, "c": 3}}) == '{"a":{"c":3,"d":4},"b":2}'

    def test_distinguishes_array_order(self) -> None:
        assert canonicalize_key_material([1, 2]) != canonicalize_key_material([2, 1])

    def test_drops_function_valued_object_entries_matching_json_semantics(self) -> None:
        # Upstream drops `undefined` (and function/symbol) entries. Python has no
        # `undefined`; callables are the remaining JS-only analog and are dropped
        # from objects / serialized as null in arrays, as JSON.stringify does.
        assert canonicalize_key_material({"a": 1, "b": len}) == canonicalize_key_material({"a": 1})
        assert canonicalize_key_material([1, len]) == "[1,null]"

    def test_canonicalizes_primitives_per_jcs(self) -> None:
        assert canonicalize_key_material("query") == '"query"'
        assert canonicalize_key_material(42) == "42"
        assert canonicalize_key_material(None) == "null"
        # JCS number serialization is ECMAScript JSON.stringify:
        assert canonicalize_key_material(-0.0) == "0"
        assert canonicalize_key_material(1e21) == "1e+21"

    def test_number_formatting_follows_ecmascript(self) -> None:
        # Python-specific: ints go through IEEE-754 doubles, exactly as
        # JSON.parse would have rounded them on the TypeScript side.
        assert canonicalize_key_material(2**53 + 1) == "9007199254740992"
        assert canonicalize_key_material(1.0) == "1"
        assert canonicalize_key_material(1e-7) == "1e-7"
        assert canonicalize_key_material(0.000001) == "0.000001"
        assert canonicalize_key_material(1.5e20) == "150000000000000000000"
        assert canonicalize_key_material(True) == "true"
        assert canonicalize_key_material(False) == "false"

    def test_sorts_keys_by_utf16_code_units(self) -> None:
        # U+FF61 sorts AFTER U+1F600 by code point but BEFORE it by UTF-16 code
        # unit (0xFF61 > 0xD83D). JCS mandates UTF-16 order.
        assert canonicalize_key_material({"\U0001f600": 1, "\uff61": 2}) == '{"\U0001f600":1,"\uff61":2}'

    def test_rejects_values_rfc8785_cannot_represent(self) -> None:
        with pytest.raises(ValueError, match=re.compile("non-finite", re.I)):
            canonicalize_key_material(float("nan"))
        with pytest.raises(ValueError, match=re.compile("non-finite", re.I)):
            canonicalize_key_material(float("inf"))
        # Upstream's third case is a `bigint`. Python ints are not bigints, but
        # an int outside the double range is equally unrepresentable.
        with pytest.raises(ValueError, match=re.compile("non-finite", re.I)):
            canonicalize_key_material({"id": 10**400})

    def test_rejects_non_json_types(self) -> None:
        with pytest.raises(TypeError):
            canonicalize_key_material({"blob": b"bytes"})
        with pytest.raises(TypeError):
            canonicalize_key_material({1: "non-string key"})

    def test_throws_on_circular_key_material_instead_of_hanging(self) -> None:
        circular: Dict[str, Any] = {}
        circular["self"] = circular
        with pytest.raises(ValueError, match=re.compile("circular", re.I)):
            canonicalize_key_material(circular)

    def test_shared_non_circular_references_are_fine(self) -> None:
        shared = {"x": 1}
        assert canonicalize_key_material({"a": shared, "b": shared}) == '{"a":{"x":1},"b":{"x":1}}'

    def test_throws_on_nesting_past_max_depth_instead_of_overflowing(self) -> None:
        deep: Any = "leaf"
        for _ in range(MAX_CANONICALIZE_DEPTH + 10):
            deep = [deep]
        with pytest.raises(ValueError, match=re.compile("deeper than", re.I)):
            canonicalize_key_material(deep)


# ---------------------------------------------------------------------------
# Fingerprints — the cross-port contract
# ---------------------------------------------------------------------------


class TestFingerprints:
    async def test_reproduces_every_tool_call_vector(self) -> None:
        assert len(VECTORS["toolCallVectors"]) == 10
        for vector in VECTORS["toolCallVectors"]:
            assert canonicalize_key_material(vector["keyMaterial"]) == vector["jcs"], vector["name"]
            assert await fingerprint_tool_call(vector["toolName"], vector["keyMaterial"]) == vector["fingerprint"], (
                vector["name"]
            )

    async def test_reproduces_every_bare_key_material_vector(self) -> None:
        assert len(VECTORS["keyMaterialVectors"]) == 2
        for vector in VECTORS["keyMaterialVectors"]:
            assert canonicalize_key_material(vector["keyMaterial"]) == vector["jcs"], vector["name"]
            assert await fingerprint_key_material(vector["keyMaterial"]) == vector["fingerprint"], vector["name"]

    async def test_rejected_vector_classes_raise(self) -> None:
        names = [entry["name"] for entry in VECTORS["rejected"]]
        assert names == ["bigint", "NaN / Infinity", "circular reference", "nesting > 64 levels"]
        circular: List[Any] = []
        circular.append(circular)
        deep: Any = 0
        for _ in range(MAX_CANONICALIZE_DEPTH + 1):
            deep = [deep]
        # "bigint" has no Python type; an int outside the double range stands in.
        for rejected in (10**400, float("nan"), circular, deep):
            with pytest.raises(ValueError):
                await fingerprint_tool_call("t", rejected)

    async def test_produces_64_char_lowercase_hex(self) -> None:
        fp = await fingerprint_tool_call("t", {})
        assert re.fullmatch(r"[0-9a-f]{64}", fp)

    async def test_includes_the_tool_name(self) -> None:
        assert await fingerprint_tool_call("web_search", {"q": "x"}) != await fingerprint_tool_call(
            "fetch_page", {"q": "x"}
        )

    async def test_non_ascii_key_material_hashes_over_utf8_bytes(self) -> None:
        matching = [v for v in VECTORS["toolCallVectors"] if "non-ascii" in v["name"]]
        assert len(matching) == 1
        vector = matching[0]
        assert await fingerprint_tool_call(vector["toolName"], vector["keyMaterial"]) == vector["fingerprint"]

    async def test_lone_surrogate_vector_is_escaped(self) -> None:
        matching = [v for v in VECTORS["toolCallVectors"] if "lone surrogate" in v["name"]]
        assert len(matching) == 1
        assert matching[0]["keyMaterial"] == {"s": "\ud800"}
        assert canonicalize_key_material(matching[0]["keyMaterial"]) == '{"s":"\\ud800"}'

    async def test_surrogate_pair_code_points_combine_like_utf16(self) -> None:
        # A pair stored as two Python code points is ONE valid character in JS.
        assert canonicalize_key_material("\ud83c\udf89") == canonicalize_key_material("\U0001f389")


# ---------------------------------------------------------------------------
# loop_key resolution (function | False | absent | field list)
# ---------------------------------------------------------------------------

ARGS: Dict[str, Any] = {"command": "ls", "cwd": "/tmp", "verbose": True}


class TestResolveLoopKeyMaterial:
    def test_absent_declaration_is_full_arguments(self) -> None:
        assert resolve_loop_key_material(None, ARGS) == {"kind": "key", "key_material": ARGS}

    def test_false_is_statically_exempt(self) -> None:
        assert resolve_loop_key_material(False, ARGS) == {"kind": "exempt"}

    def test_field_array_is_declarative_subset(self) -> None:
        assert resolve_loop_key_material(["command", "cwd", "not_a_field"], ARGS) == {
            "kind": "key",
            "key_material": {"command": "ls", "cwd": "/tmp"},
        }

    def test_empty_field_array_warns_and_falls_back(self) -> None:
        resolution = resolve_loop_key_material([], ARGS)
        assert resolution["kind"] == "fallback"
        assert resolution["key_material"] is ARGS
        assert "empty field list" in resolution["warning"]

    def test_empty_field_array_does_not_collapse_unrelated_calls(self) -> None:
        ls = resolve_loop_key_material([], {"command": "ls"})
        rm = resolve_loop_key_material([], {"command": "rm -rf /"})
        assert ls["kind"] == "fallback" and rm["kind"] == "fallback"
        assert ls["key_material"] != rm["key_material"]

    def test_field_array_whose_every_field_is_absent_warns_and_falls_back(self) -> None:
        resolution = resolve_loop_key_material(["nope", "also_nope"], ARGS)
        assert resolution["kind"] == "fallback"
        assert resolution["key_material"] is ARGS
        assert "absent from the arguments" in resolution["warning"]
        assert "[nope, also_nope]" in resolution["warning"]

    def test_preserves_proto_as_a_declared_field(self) -> None:
        # Upstream guards against JS prototype pollution; for Python it is just
        # an ordinary key and must survive the subset.
        args = json.loads('{"__proto__":"declared"}')
        assert resolve_loop_key_material(["__proto__"], args) == {
            "kind": "key",
            "key_material": {"__proto__": "declared"},
        }

    def test_function_returning_a_value_is_that_value(self) -> None:
        resolution = resolve_loop_key_material(lambda a: str(a["command"]).strip(), ARGS)
        assert resolution == {"kind": "key", "key_material": "ls"}

    def test_function_returning_none_is_per_call_exemption(self) -> None:
        # Upstream: `null` exempts, `undefined` falls back with a warning.
        # Python has only `None`, which maps to upstream `null` (exempt) —
        # deliberate divergence documented in doom_loop.py's module docstring.
        # Upstream's "function returning undefined → FALLBACK" case has no
        # Python counterpart.
        assert resolve_loop_key_material(lambda a: None, ARGS) == {"kind": "exempt"}

    def test_throwing_function_falls_back_with_warning(self) -> None:
        def broken(_: Dict[str, Any]) -> Any:
            raise RuntimeError("loopKey bug")

        resolution = resolve_loop_key_material(broken, ARGS)
        assert resolution["kind"] == "fallback"
        assert resolution["key_material"] is ARGS
        assert re.search("threw", resolution["warning"])
        assert "loopKey bug" in resolution["warning"]

    def test_unsupported_shape_falls_back(self) -> None:
        # A bare string is not a field list (upstream: typeof 'string' → unsupported).
        for shape in ("command", True, 3):
            resolution = resolve_loop_key_material(shape, ARGS)
            assert resolution["kind"] == "fallback"
            assert "unsupported shape" in resolution["warning"]


# ---------------------------------------------------------------------------
# Text repetition
# ---------------------------------------------------------------------------


class TestDetectTextRepetition:
    def test_detects_a_single_repeated_token(self) -> None:
        result = detect_text_repetition("no no no no no no no no no no no no")
        assert result == {"repeats": 12, "period_tokens": 1, "covered_tokens": 12, "sample": "no"}

    def test_detects_a_repeating_phrase_block(self) -> None:
        phrase = "I am stuck in a loop."
        result = detect_text_repetition(" ".join([phrase] * 6))
        assert result is not None
        assert result["period_tokens"] == 6
        assert result["repeats"] == 6
        assert result["sample"] == phrase

    def test_only_counts_repetition_at_the_tail(self) -> None:
        looping = " ".join(["retry now"] * 6)
        recovered = f"{looping} but then I found the actual answer to your question about databases"
        assert detect_text_repetition(recovered) is None

    def test_stays_quiet_on_ordinary_prose(self) -> None:
        assert (
            detect_text_repetition("The quick brown fox jumps over the lazy dog and then runs far away into the woods.")
            is None
        )

    def test_stays_quiet_below_the_repeat_threshold(self) -> None:
        assert detect_text_repetition("this is very very very good and quite long enough overall") is None

    def test_documented_miss_paraphrased_repetition(self) -> None:
        paraphrased = " ".join(
            [
                "I am unable to find the file.",
                "I cannot locate the file.",
                "I am not able to find that file.",
                "The file cannot be found by me.",
                "Finding the file is not possible.",
                "I could not locate that file.",
            ]
        )
        assert detect_text_repetition(paraphrased) is None

    def test_is_deterministic(self) -> None:
        text = " ".join(["loop detected"] * 8)
        first = detect_text_repetition(text)
        assert first is not None
        assert first == detect_text_repetition(text)

    def test_handles_multi_megabyte_input_via_tail_slice(self) -> None:
        filler = " ".join(f"w{i}" for i in range(400_000))
        doom_tail = " ".join(["stuck"] * 20)
        result = detect_text_repetition(f"{filler} {doom_tail}")
        _subset(result, {"period_tokens": 1, "repeats": 20})

    def test_honors_custom_thresholds(self) -> None:
        text = "ha ha ha"
        assert detect_text_repetition(text) is None
        _subset(
            detect_text_repetition(text, {"min_repeats": 3, "min_covered_tokens": 3}),
            {"repeats": 3, "period_tokens": 1},
        )

    def test_prefers_smallest_period_on_ties(self) -> None:
        result = detect_text_repetition("no no no no no no", {"min_repeats": 3, "min_covered_tokens": 6})
        _subset(result, {"period_tokens": 1, "repeats": 6})


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------

LADDER: DoomLoopLadder = {"observe": 2, "steer": False, "block": 3, "stop": 6}


class TestResolveLadderAction:
    def test_maps_streaks_onto_rungs_strongest_wins(self) -> None:
        assert [resolve_ladder_action(LADDER, streak, allow_block=True) for streak in (1, 2, 3, 5, 6)] == [
            None,
            "observe",
            "block",
            "block",
            "stop",
        ]

    def test_falls_through_block_for_non_blockable_verdicts(self) -> None:
        assert resolve_ladder_action(LADDER, 3, allow_block=False) == "observe"
        assert resolve_ladder_action(LADDER, 6, allow_block=False) == "stop"

    def test_respects_disabled_rungs(self) -> None:
        assert (
            resolve_ladder_action({"observe": False, "steer": False, "block": False, "stop": 4}, 3, allow_block=True)
            is None
        )

    def test_escalate_rung_and_its_gate(self) -> None:
        ladder: DoomLoopLadder = {"observe": 1, "steer": 2, "escalate": 3, "block": 5, "stop": 8}
        assert resolve_ladder_action(ladder, 3, allow_block=True) == "escalate"
        assert resolve_ladder_action(ladder, 3, allow_block=True, allow_escalate=False) == "steer"
        assert resolve_ladder_action(ladder, 5, allow_block=False) == "escalate"


class TestResolveDoomLoopOption:
    def test_returns_none_for_none_or_false(self) -> None:
        assert resolve_doom_loop_option(None) is None
        assert resolve_doom_loop_option(False) is None

    def test_true_resolves_to_the_documented_defaults(self) -> None:
        resolved = resolve_doom_loop_option(True)
        assert resolved["ladder"] == DEFAULT_DOOM_LOOP_LADDER
        assert resolved["ladder"] == {"observe": 2, "steer": False, "escalate": False, "block": 3, "stop": 6}
        assert resolved["text"] == {
            "enabled": True,
            "max_period_tokens": 16,
            "min_repeats": 4,
            "min_covered_tokens": 12,
            "max_window_tokens": 400,
        }
        assert resolved["escalation"] is None

    def test_default_ladder_is_read_only(self) -> None:
        with pytest.raises(TypeError):
            DEFAULT_DOOM_LOOP_LADDER["block"] = 1  # type: ignore[index]

    def test_merges_partial_ladders_and_clamps_invalid_thresholds(self) -> None:
        resolved = resolve_doom_loop_option({"ladder": {"block": 5, "stop": 0}})
        assert resolved["ladder"]["block"] == 5
        assert resolved["ladder"]["stop"] == DEFAULT_DOOM_LOOP_LADDER["stop"]
        assert resolved["ladder"]["observe"] == DEFAULT_DOOM_LOOP_LADDER["observe"]

    def test_floors_fractional_thresholds_and_rejects_bools(self) -> None:
        # `True` is an int in Python but `typeof true !== 'number'` upstream.
        # Deliberately off-type input (callers may pass untyped config).
        option: Any = {"ladder": {"block": 4.7, "observe": True}}
        resolved = resolve_doom_loop_option(option)
        assert resolved["ladder"]["block"] == 4
        assert resolved["ladder"]["observe"] == 2

    def test_advisor_false_is_not_a_recovery_mechanism(self) -> None:
        assert resolve_doom_loop_option({"escalation": {"advisor": False}})["escalation"] is None

    def test_advisor_false_does_not_resolve_alongside_a_real_mechanism(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resolved = resolve_doom_loop_option({"escalation": {"model": "stronger-model", "advisor": False}})
        escalation = resolved["escalation"]
        assert escalation is not None
        assert escalation["model"] == "stronger-model"
        assert "advisor" not in escalation

    def test_advisor_true_is_a_recovery_mechanism(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            escalation = resolve_doom_loop_option({"escalation": {"advisor": True}})["escalation"]
        assert escalation == {"advisor": True, "max_escalations": 2}

    def test_escalation_budget_resolution(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fractional: Any = {"escalation": {"model": "m", "max_escalations": 3.9}}
            assert resolve_doom_loop_option(fractional)["escalation"] == {
                "model": "m",
                "max_escalations": 3,
            }
            assert resolve_doom_loop_option({"escalation": {"model": "m", "max_escalations": 0}})["escalation"] == {
                "model": "m",
                "max_escalations": 2,
            }

    def test_warns_when_escalation_config_has_no_rung(self) -> None:
        with pytest.warns(UserWarning, match="escalate ladder rung is disabled"):
            resolve_doom_loop_option({"escalation": {"model": "m"}})

    def test_warns_when_escalate_rung_has_no_config(self) -> None:
        with pytest.warns(UserWarning, match="no escalation config"):
            resolved = resolve_doom_loop_option({"ladder": {"observe": 1, "escalate": 2, "block": 3}})
        assert resolved["escalation"] is None

    def test_text_false_disables_text_detection(self) -> None:
        assert resolve_doom_loop_option({"text": False})["text"]["enabled"] is False

    def test_text_options_merge_with_defaults(self) -> None:
        text = resolve_doom_loop_option({"text": {"min_repeats": 3}})["text"]
        assert text["min_repeats"] == 3
        assert text["max_period_tokens"] == 16

    def test_warns_when_block_enabled_with_stop_disabled(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_doom_loop_option({"ladder": {"block": 3, "stop": False}})
        messages = [str(w.message) for w in caught]
        assert len(messages) == 1
        assert "stop disabled" in messages[0]

    def test_warns_on_dead_rungs(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_doom_loop_option({"ladder": {"observe": 5, "block": 2}})
        messages = [str(w.message) for w in caught]
        # One warning per dead rung: observe (5) is dominated by block (2).
        assert len(messages) == 1
        assert '"observe" (5) can never fire' in messages[0]

    def test_does_not_warn_on_the_default_ladder(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_doom_loop_option(True)
        assert caught == []


# ---------------------------------------------------------------------------
# Monitor state machine (round-scoped streaks)
# ---------------------------------------------------------------------------


class TestMonitorToolCallStreaks:
    async def test_fires_observe_at_2_and_block_at_3(self) -> None:
        monitor = _make_monitor()
        args = {"query": "same thing"}
        assert "verdict" not in await monitor.record_tool_call("search", args, 1)
        second = await monitor.record_tool_call("search", args, 2)
        _subset(
            second.get("verdict"),
            {"action": "observe", "streak": 2, "detector": "tool-fingerprint", "tool_name": "search"},
        )
        third = await monitor.record_tool_call("search", args, 3)
        _subset(third.get("verdict"), {"action": "block", "streak": 3})
        verdict = third.get("verdict")
        assert verdict is not None
        assert verdict["fingerprint"] == await fingerprint_tool_call("search", args)
        # Undeclared round: the true round membership is unknown, so the text
        # is the fingerprint-free per-call form (upstream buildToolVerdictMessage).
        assert verdict["message"] == (
            'Doom loop suspected: this exact "search" call has been repeated across consecutive rounds. '
            "Repeating it will not change the result. Take a different approach, or explain why "
            "repetition is required."
        )

    async def test_declared_single_call_round_quotes_the_fingerprint(self) -> None:
        monitor = _make_monitor()
        args = {"query": "same thing"}
        record: Optional[DoomLoopCallRecord] = None
        for round_number in (1, 2, 3):
            await monitor.declare_round(round_number, [{"tool_name": "search", "key_material": args}])
            record = await monitor.record_tool_call("search", args, round_number)
        assert record is not None
        verdict = record.get("verdict")
        assert verdict is not None
        assert verdict["action"] == "block"
        assert verdict["message"] == (
            'Doom loop suspected: tool "search" was invoked in 3 consecutive rounds with identical '
            "arguments (fingerprint " + verdict["fingerprint"][:16] + "…). Repeating the call will "
            "not change the result. Take a different approach, or explain why repetition is required."
        )

    async def test_escalates_to_stop_at_the_stop_threshold(self) -> None:
        monitor = _make_monitor()
        actions: List[Optional[str]] = []
        last: Optional[DoomLoopCallRecord] = None
        for round_number in range(1, 7):
            last = await monitor.record_tool_call("search", {"q": "x"}, round_number)
            verdict = last.get("verdict")
            actions.append(verdict["action"] if verdict else None)
        assert last is not None
        _subset(last.get("verdict"), {"action": "stop", "streak": 6})
        assert actions == [None, "observe", "block", "block", "block", "stop"]

    async def test_identical_calls_in_the_same_round_are_duplicates(self) -> None:
        monitor = _make_monitor()
        args = {"q": "parallel"}
        first = await monitor.record_tool_call("search", args, 1)
        _subset(first, {"streak": 1, "duplicate_in_round": False})
        for _ in range(5):
            dup = await monitor.record_tool_call("search", args, 1)
            _subset(dup, {"streak": 1, "duplicate_in_round": True})
            assert "verdict" not in dup
        _subset(await monitor.record_tool_call("search", args, 2), {"streak": 2, "duplicate_in_round": False})

    async def test_different_arguments_reset_the_streak(self) -> None:
        monitor = _make_monitor()
        await monitor.record_tool_call("search", {"q": "a"}, 1)
        await monitor.record_tool_call("search", {"q": "a"}, 2)
        reset = await monitor.record_tool_call("search", {"q": "b"}, 3)
        assert "verdict" not in reset
        assert reset["streak"] == 1
        _subset((await monitor.record_tool_call("search", {"q": "b"}, 4)).get("verdict"), {"streak": 2})

    async def test_interleaved_calls_to_other_tools_do_not_reset(self) -> None:
        monitor = _make_monitor()
        search = {"q": "x"}
        await monitor.record_tool_call("search", search, 1)
        await monitor.record_tool_call("read_file", {"path": "/a"}, 2)
        _subset(
            (await monitor.record_tool_call("search", search, 3)).get("verdict"),
            {"action": "observe", "streak": 2},
        )
        await monitor.record_tool_call("read_file", {"path": "/b"}, 4)
        _subset(
            (await monitor.record_tool_call("search", search, 5)).get("verdict"),
            {"action": "block", "streak": 3},
        )

    async def test_argument_key_order_does_not_defeat_detection(self) -> None:
        monitor = _make_monitor()
        await monitor.record_tool_call("run", {"cmd": "ls", "cwd": "/tmp"}, 1)
        _subset((await monitor.record_tool_call("run", {"cwd": "/tmp", "cmd": "ls"}, 2)).get("verdict"), {"streak": 2})

    async def test_repeated_empty_calls_trip_the_detector(self) -> None:
        monitor = _make_monitor()
        assert "verdict" not in await monitor.record_tool_call("list_tasks", {}, 1)
        _subset(
            (await monitor.record_tool_call("list_tasks", {}, 2)).get("verdict"),
            {"action": "observe", "streak": 2},
        )
        _subset(
            (await monitor.record_tool_call("list_tasks", {}, 3)).get("verdict"),
            {"action": "block", "streak": 3},
        )

    async def test_propagates_canonicalization_errors_for_engine_fallback(self) -> None:
        # Upstream uses a bigint; an out-of-double-range int is the Python analog.
        monitor = _make_monitor()
        with pytest.raises(ValueError, match=re.compile("non-finite", re.I)):
            await monitor.record_tool_call("search", {"id": 10**400}, 1)
        # A failed record leaves no trace: the next valid call starts at 1.
        assert (await monitor.record_tool_call("search", {"id": 1}, 1))["streak"] == 1

    async def test_server_tool_records_use_non_blockable_path_and_label(self) -> None:
        monitor = _make_monitor()
        last: Optional[DoomLoopCallRecord] = None
        for round_number in range(1, 4):
            last = await monitor.record_tool_call(
                "server:web_search_call",
                {"query": "x"},
                round_number,
                allow_block=False,
                detector="server-tool-fingerprint",
            )
        assert last is not None
        _subset(last.get("verdict"), {"detector": "server-tool-fingerprint", "action": "observe", "streak": 3})

    async def test_is_deterministic_across_monitors(self) -> None:
        script: List[Tuple[str, Any, int]] = [
            ("a", {"x": 1}, 1),
            ("b", {}, 1),
            ("a", {"x": 1}, 2),
            ("a", {"x": 2}, 3),
            ("a", {"x": 2}, 4),
        ]

        async def run() -> List[Optional[DoomLoopVerdict]]:
            monitor = _make_monitor()
            return [(await monitor.record_tool_call(name, args, r)).get("verdict") for name, args, r in script]

        first = await run()
        assert first == await run()
        assert [v["action"] if v else None for v in first] == [None, None, "observe", None, "observe"]


class TestMonitorRoundSets:
    """Round-set / per-call semantics (exercised upstream mainly via the fan-out suite)."""

    async def _round(
        self, monitor: DoomLoopMonitor, round_number: int, paths: List[str], declare: bool = True
    ) -> List[DoomLoopCallRecord]:
        if declare:
            await monitor.declare_round(
                round_number, [{"tool_name": "read", "key_material": {"path": p}} for p in paths]
            )
        return [await monitor.record_tool_call("read", {"path": p}, round_number) for p in paths]

    async def test_declared_fan_out_shares_the_round_streak_and_message(self) -> None:
        monitor = _make_monitor()
        await self._round(monitor, 1, ["a", "b", "c"])
        await self._round(monitor, 2, ["a", "b", "c"])
        third = await self._round(monitor, 3, ["c", "a", "b"])  # emission order does not matter
        assert [r["streak"] for r in third] == [3, 3, 3]
        verdicts = [r.get("verdict") for r in third]
        assert [v["action"] if v else None for v in verdicts] == ["block", "block", "block"]
        messages = {v["message"] for v in verdicts if v}
        assert len(messages) == 1
        assert "same set of 3 parallel calls" in messages.pop()

    async def test_superset_round_resets_and_does_not_block_progress(self) -> None:
        monitor = _make_monitor()
        await self._round(monitor, 1, ["a", "b"])
        second = await self._round(monitor, 2, ["a", "b"])
        assert [r["streak"] for r in second] == [2, 2]
        third = await self._round(monitor, 3, ["a", "b", "c"])
        # Round set differs (reset to 1) but a and b are 3-peats per call.
        assert [r["streak"] for r in third] == [3, 3, 1]
        actions = [r["verdict"]["action"] if "verdict" in r else None for r in third]
        assert actions == ["block", "block", None]

    async def test_per_call_streak_catches_repeat_in_varying_company(self) -> None:
        monitor = _make_monitor()
        results = []
        for round_number, other in enumerate(["b", "c", "d"], start=1):
            results.append(await self._round(monitor, round_number, ["a", other]))
        last_a, last_other = results[-1]
        assert last_a["streak"] == 3
        assert last_other["streak"] == 1
        verdict = last_a.get("verdict")
        assert verdict is not None
        assert verdict["action"] == "block"
        # Per-call verdict text is fingerprint- and count-free.
        assert verdict["message"] == (
            'Doom loop suspected: this exact "read" call has been repeated across consecutive rounds. '
            "Repeating it will not change the result. Take a different approach, or explain why "
            "repetition is required."
        )

    async def test_undeclared_round_uses_per_call_text(self) -> None:
        monitor = _make_monitor()
        for round_number in (1, 2):
            records = await self._round(monitor, round_number, ["a", "b"], declare=False)
        messages = [r["verdict"]["message"] for r in records if "verdict" in r]
        assert len(messages) == 2
        assert len(set(messages)) == 1
        assert 'this exact "read" call' in messages[0]

    async def test_unhashable_declared_call_is_excluded_with_warning(self) -> None:
        monitor = _make_monitor()
        with pytest.warns(UserWarning, match='could not fingerprint a "read" call while declaring round 1'):
            await monitor.declare_round(
                1,
                [
                    {"tool_name": "read", "key_material": {"path": "a"}},
                    {"tool_name": "read", "key_material": {"bad": float("nan")}},
                ],
            )
        record = await monitor.record_tool_call("read", {"path": "a"}, 1)
        assert record["streak"] == 1

    async def test_non_member_does_not_overwrite_round_identity(self) -> None:
        monitor = _make_monitor()
        for round_number in (1, 2, 3):
            await monitor.declare_round(round_number, [{"tool_name": "read", "key_material": {"path": "a"}}])
            member = await monitor.record_tool_call("read", {"path": "a"}, round_number)
            await monitor.record_tool_call("read", {"path": "fallback"}, round_number)
        assert member["streak"] == 3
        assert member.get("verdict", {}).get("action") == "block"
        state = monitor.get_state()
        # The persisted identity pairs with the round streak: the member, not the non-member.
        assert state["tools"]["read"]["fingerprint"] == await fingerprint_tool_call("read", {"path": "a"})


class TestMonitorEscalation:
    def _config(self) -> Any:
        return {
            "ladder": {"observe": False, "escalate": 2, "block": False, "stop": 10},
            "escalation": {"model": "frontier", "max_escalations": 1},
        }

    def _monitor(self) -> DoomLoopMonitor:
        return _make_monitor(self._config())

    async def test_escalate_until_budget_is_consumed_then_falls_through(self) -> None:
        monitor = self._monitor()
        assert monitor.can_escalate() is True
        await monitor.record_tool_call("search", {"q": "x"}, 1)
        second = await monitor.record_tool_call("search", {"q": "x"}, 2)
        assert second.get("verdict", {}).get("action") == "escalate"
        monitor.consume_escalation()
        assert monitor.can_escalate() is False
        third = await monitor.record_tool_call("search", {"q": "x"}, 3)
        # Budget exhausted; steer/observe disabled → no verdict.
        assert "verdict" not in third

    async def test_escalation_budget_survives_state_round_trip(self) -> None:
        monitor = self._monitor()
        monitor.consume_escalation()
        state = json.loads(json.dumps(monitor.get_state()))
        assert state["escalations_used"] == 1
        resumed = DoomLoopMonitor(resolve_doom_loop_option(self._config()), state)
        assert resumed.can_escalate() is False

    async def test_no_mechanism_means_never_escalate(self) -> None:
        assert _make_monitor().can_escalate() is False


class TestMonitorText:
    async def test_within_response_repetition_can_stop_immediately(self) -> None:
        monitor = _make_monitor()
        verdict = await monitor.record_assistant_text(" ".join(["I am stuck."] * 20))
        _subset(verdict, {"detector": "text-repetition", "action": "stop", "streak": 20})
        assert verdict is not None
        assert verdict["fingerprint"] == await fingerprint_key_material("I am stuck.")
        assert "tool_name" not in verdict

    async def test_cross_step_identical_text_builds_a_streak(self) -> None:
        monitor = _make_monitor()
        assert await monitor.record_assistant_text("Let me try that again.") is None
        _subset(
            await monitor.record_assistant_text("Let me try that again."),
            {"detector": "text-streak", "action": "observe", "streak": 2},
        )
        # Streak 3 crosses the block rung, but text cannot be blocked → observe.
        _subset(
            await monitor.record_assistant_text("Let me try that again."),
            {"detector": "text-streak", "action": "observe", "streak": 3},
        )
        for _ in range(2):
            await monitor.record_assistant_text("Let me try that again.")
        _subset(await monitor.record_assistant_text("Let me try that again."), {"action": "stop", "streak": 6})

    async def test_whitespace_only_differences_do_not_defeat_the_streak(self) -> None:
        monitor = _make_monitor()
        await monitor.record_assistant_text("Retrying   the\nsame plan.")
        verdict = await monitor.record_assistant_text("Retrying the same plan.")
        _subset(verdict, {"streak": 2})
        assert verdict is not None
        # Cross-port: the whitespace-normalized text vector.
        assert verdict["fingerprint"] == VECTORS["keyMaterialVectors"][1]["fingerprint"]

    async def test_documented_miss_paraphrased_cross_step_text(self) -> None:
        monitor = _make_monitor()
        paraphrases = [
            "I am unable to find the file.",
            "I cannot locate the file.",
            "I am not able to find that file.",
            "The file cannot be found.",
            "Locating the file is not possible.",
            "I could not find that file.",
        ]
        assert [await monitor.record_assistant_text(text) for text in paraphrases] == [None] * 6

    async def test_empty_text_neither_counts_nor_resets(self) -> None:
        monitor = _make_monitor()
        await monitor.record_assistant_text("same")
        assert await monitor.record_assistant_text("") is None
        assert await monitor.record_assistant_text("   ") is None
        _subset(await monitor.record_assistant_text("same"), {"streak": 2})

    async def test_reset_text_streak(self) -> None:
        monitor = _make_monitor()
        await monitor.record_assistant_text("same")
        monitor.reset_text_streak()
        assert await monitor.record_assistant_text("same") is None
        assert "text" in monitor.get_state()

    async def test_text_false_disables_both_text_detectors(self) -> None:
        monitor = _make_monitor({"text": False})
        assert await monitor.record_assistant_text(" ".join(["loop"] * 30)) is None


class TestMonitorStateRoundTrip:
    async def test_serializes_to_plain_json_and_resumes_counting(self) -> None:
        monitor = _make_monitor()
        await monitor.record_tool_call("search", {"q": "x"}, 1)
        await monitor.record_tool_call("search", {"q": "x"}, 2)
        blob = json.loads(json.dumps(monitor.get_state()))
        resumed = DoomLoopMonitor(resolve_doom_loop_option(True), blob)
        # Round markers are NOT serialized, so the first resumed record increments.
        _subset(
            (await resumed.record_tool_call("search", {"q": "x"}, 1)).get("verdict"),
            {"action": "block", "streak": 3},
        )

    async def test_ignores_corrupt_persisted_state(self) -> None:
        config = resolve_doom_loop_option(True)
        with pytest.warns(UserWarning, match="Ignoring invalid persisted doom-loop state"):
            monitor = DoomLoopMonitor(config, "not an object")
        record = await monitor.record_tool_call("search", {"q": "x"}, 1)
        assert record["streak"] == 1
        assert "verdict" not in record

    async def test_persisted_proto_tool_name_cannot_pollute_bookkeeping(self) -> None:
        hostile = json.loads('{"tools":{"__proto__":{"fingerprint":"x","streak":999}}}')
        monitor = DoomLoopMonitor(resolve_doom_loop_option(True), hostile)
        record = await monitor.record_tool_call("unrelated_tool", {"q": "first call"}, 1)
        assert record["streak"] == 1
        assert "verdict" not in record

    async def test_a_tool_named_proto_is_tracked_as_ordinary_data(self) -> None:
        monitor = _make_monitor()
        await monitor.record_tool_call("__proto__", {}, 1)
        record = await monitor.record_tool_call("__proto__", {}, 2)
        assert record["streak"] == 2
        snapshot = monitor.get_state()
        assert list(snapshot["tools"]) == ["__proto__"]
        revived = json.loads(json.dumps(snapshot))
        assert list(revived["tools"]) == ["__proto__"]

    async def test_get_state_returns_a_snapshot_not_a_live_reference(self) -> None:
        monitor = _make_monitor()
        await monitor.record_tool_call("a", {}, 1)
        snapshot = monitor.get_state()
        await monitor.record_tool_call("a", {}, 2)
        assert snapshot["tools"]["a"]["streak"] == 1
        assert monitor.get_state()["tools"]["a"]["streak"] == 2

    async def test_get_state_shape(self) -> None:
        monitor = _make_monitor()
        await monitor.declare_round(1, [{"tool_name": "read", "key_material": {"path": p}} for p in ("a", "b")])
        for p in ("a", "b"):
            await monitor.record_tool_call("read", {"path": p}, 1)
        await monitor.record_tool_call("search", {"q": "x"}, 1)
        await monitor.record_assistant_text("hello")
        state = monitor.get_state()
        fa = await fingerprint_tool_call("read", {"path": "a"})
        fb = await fingerprint_tool_call("read", {"path": "b"})
        assert state == {
            "tools": {
                "read": {"fingerprint": fb, "streak": 1, "round_fingerprints": sorted([fa, fb])},
                "search": {"fingerprint": await fingerprint_tool_call("search", {"q": "x"}), "streak": 1},
            },
            "text": {"fingerprint": await fingerprint_key_material("hello"), "streak": 1},
        }

    async def test_shrunk_round_persists_call_streaks_across_resume(self) -> None:
        monitor = _make_monitor()
        for round_number in (1, 2):
            await monitor.declare_round(
                round_number, [{"tool_name": "read", "key_material": {"path": p}} for p in ("a", "b")]
            )
            for p in ("a", "b"):
                await monitor.record_tool_call("read", {"path": p}, round_number)
        # Round shrinks to one call: round streak resets, per-call count keeps climbing.
        await monitor.declare_round(3, [{"tool_name": "read", "key_material": {"path": "a"}}])
        await monitor.record_tool_call("read", {"path": "a"}, 3)
        state = json.loads(json.dumps(monitor.get_state()))
        fa = await fingerprint_tool_call("read", {"path": "a"})
        assert state["tools"]["read"] == {"fingerprint": fa, "streak": 1, "call_streaks": {fa: 3}}
        resumed = DoomLoopMonitor(resolve_doom_loop_option(True), state)
        record = await resumed.record_tool_call("read", {"path": "a"}, 1)
        assert record["streak"] == 4
        assert record.get("verdict", {}).get("action") == "block"

    async def test_restore_accepts_upstream_camel_case_blob(self) -> None:
        fa = await fingerprint_tool_call("read", {"path": "a"})
        fb = await fingerprint_tool_call("read", {"path": "b"})
        blob = {
            "tools": {"read": {"fingerprint": fb, "streak": 2, "roundFingerprints": [fb, fa]}},
            "escalationsUsed": 1,
        }
        monitor = DoomLoopMonitor(resolve_doom_loop_option(True), blob)
        await monitor.declare_round(1, [{"tool_name": "read", "key_material": {"path": p}} for p in ("a", "b")])
        streaks = [(await monitor.record_tool_call("read", {"path": p}, 1))["streak"] for p in ("a", "b")]
        assert streaks == [3, 3]
        assert monitor.get_state().get("escalations_used") == 1

    async def test_restore_clamps_corrupt_streaks_and_drops_invalid_entries(self) -> None:
        blob = {
            "tools": {
                "neg": {"fingerprint": "x", "streak": -50},
                "bad": {"fingerprint": 1, "streak": 2},
                "boolean": {"fingerprint": "x", "streak": True},
            },
            "text": {"fingerprint": "t", "streak": "nope"},
            "escalations_used": -1,
        }
        monitor = DoomLoopMonitor(resolve_doom_loop_option(True), blob)
        assert monitor.get_state() == {"tools": {"neg": {"fingerprint": "x", "streak": 1}}}
