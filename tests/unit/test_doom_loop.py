"""Unit tests for the doom-loop detection primitives (port of upstream
``tests/unit/doom-loop.test.ts``, module ``src/lib/doom-loop.ts``).

Everything here is pure and deterministic: JCS canonicalization, SHA-256
fingerprints (asserted against the cross-port vector file), text-repetition
detection, ladder resolution, loop_key resolution, and the DoomLoopMonitor
state machine (round-scoped streaks).
"""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from openrouter_agent.doom_loop import (
    DEFAULT_DOOM_LOOP_LADDER,
    MAX_CANONICALIZE_DEPTH,
    DoomLoopCallRecord,
    DoomLoopCanonicalizationError,
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

VECTORS_PATH = Path(__file__).resolve().parent.parent / "vectors" / "doom_loop_fingerprints.json"


def _subset(actual: Any, expected: Dict[str, Any]) -> bool:
    """vitest ``toMatchObject``: every expected key present with an equal value."""
    return actual is not None and all(key in actual and actual[key] == value for key, value in expected.items())


# ---------------------------------------------------------------------------
# Canonicalization (RFC 8785 / JCS semantics)
# ---------------------------------------------------------------------------


class TestCanonicalizeKeyMaterial:
    def test_is_insensitive_to_object_key_order_recursively(self) -> None:
        assert canonicalize_key_material({"b": 2, "a": {"d": 4, "c": 3}}) == canonicalize_key_material(
            {"a": {"c": 3, "d": 4}, "b": 2}
        )

    def test_distinguishes_array_order(self) -> None:
        assert canonicalize_key_material([1, 2]) != canonicalize_key_material([2, 1])

    def test_drops_function_object_entries_matching_json_semantics(self) -> None:
        # Upstream drops `undefined` entries; Python has no `undefined` (None is
        # JSON null), so the equivalent JSON-semantics drop is a callable value
        # — upstream drops function entries identically.
        assert canonicalize_key_material({"a": 1, "b": lambda: None}) == canonicalize_key_material({"a": 1})
        assert canonicalize_key_material([lambda: None]) == "[null]"

    def test_canonicalizes_primitives_per_jcs(self) -> None:
        assert canonicalize_key_material("query") == '"query"'
        assert canonicalize_key_material(42) == "42"
        assert canonicalize_key_material(None) == "null"
        # JCS number serialization is ECMAScript JSON.stringify:
        assert canonicalize_key_material(-0.0) == "0"
        assert canonicalize_key_material(1e21) == "1e+21"

    def test_rejects_values_rfc8785_cannot_represent(self) -> None:
        with pytest.raises(DoomLoopCanonicalizationError, match=re.compile("non-finite", re.I)):
            canonicalize_key_material(math.nan)
        with pytest.raises(DoomLoopCanonicalizationError, match=re.compile("non-finite", re.I)):
            canonicalize_key_material(math.inf)
        # Upstream's `{ id: 10n }` (bigint) case. Python ints are JSON numbers
        # (serialized as the IEEE double JS would parse), so the non-JSON
        # analogues are types with no JSON form, and ints beyond double range
        # (JS would parse those to Infinity).
        with pytest.raises(DoomLoopCanonicalizationError, match="set"):
            canonicalize_key_material({"id": {10}})
        with pytest.raises(DoomLoopCanonicalizationError, match="bytes"):
            canonicalize_key_material({"id": b"10"})
        with pytest.raises(DoomLoopCanonicalizationError, match=re.compile("non-finite", re.I)):
            canonicalize_key_material({"id": 10**400})

    def test_throws_on_circular_key_material_instead_of_hanging(self) -> None:
        circular: Dict[str, Any] = {}
        circular["self"] = circular
        with pytest.raises(DoomLoopCanonicalizationError, match=re.compile("circular", re.I)):
            canonicalize_key_material(circular)

    def test_throws_on_nesting_past_max_depth_instead_of_overflowing(self) -> None:
        deep: Any = "leaf"
        for _ in range(MAX_CANONICALIZE_DEPTH + 10):
            deep = [deep]
        with pytest.raises(DoomLoopCanonicalizationError, match=re.compile("deeper than", re.I)):
            canonicalize_key_material(deep)

    # -- Python-port-specific JCS edges (no upstream counterpart: they pin that
    # the port reproduces ECMAScript serialization rather than json.dumps).

    def test_number_layout_matches_ecmascript(self) -> None:
        cases: List[Tuple[Any, str]] = [
            (1e-7, "1e-7"),
            (1.5e-7, "1.5e-7"),
            (0.000001, "0.000001"),
            (0.1, "0.1"),
            (123.0, "123"),
            (1e20, "100000000000000000000"),
            (5e-324, "5e-324"),
            (-2.5, "-2.5"),
            # JSON integers parse to doubles in JS: beyond 2**53 they round.
            (2**53 + 1, "9007199254740992"),
            (12345678901234567890, "12345678901234567000"),
        ]
        assert [canonicalize_key_material(value) for value, _ in cases] == [text for _, text in cases]

    def test_keys_sort_by_utf16_code_units_and_strings_escape_like_json_stringify(self) -> None:
        # U+1F600 is the surrogate pair D83D DE00, which sorts BEFORE U+FFFF in
        # UTF-16 even though its code point is larger.
        assert (
            canonicalize_key_material({"\uffff": 2, "\U0001f600": 1, "b\x01": '\x1f"\\\n'})
            == '{"b\\u0001":"\\u001f\\"\\\\\\n","\U0001f600":1,"\uffff":2}'
        )
        # A surrogate PAIR spelled as two code points is one character in JS.
        assert canonicalize_key_material("\ud83d\ude00") == '"\U0001f600"'

    def test_rejects_non_string_mapping_keys(self) -> None:
        with pytest.raises(DoomLoopCanonicalizationError, match="non-string mapping key"):
            canonicalize_key_material({1: "a"})


# ---------------------------------------------------------------------------
# Fingerprints — the cross-port contract
# ---------------------------------------------------------------------------


def _load_vectors() -> Dict[str, Any]:
    with VECTORS_PATH.open(encoding="utf-8") as handle:
        data: Dict[str, Any] = json.load(handle)
    return data


class TestFingerprintVectors:
    async def test_reproduces_every_tool_call_vector(self) -> None:
        vectors = _load_vectors()["toolCallVectors"]
        assert len(vectors) == 10
        for vector in vectors:
            assert canonicalize_key_material(vector["keyMaterial"]) == vector["jcs"], vector["name"]
            assert await fingerprint_tool_call(vector["toolName"], vector["keyMaterial"]) == vector["fingerprint"], (
                vector["name"]
            )

    async def test_reproduces_every_bare_key_material_vector(self) -> None:
        vectors = _load_vectors()["keyMaterialVectors"]
        assert len(vectors) == 2
        for vector in vectors:
            assert canonicalize_key_material(vector["keyMaterial"]) == vector["jcs"], vector["name"]
            assert await fingerprint_key_material(vector["keyMaterial"]) == vector["fingerprint"], vector["name"]

    def test_documents_the_rejected_classes(self) -> None:
        rejected = [entry["name"] for entry in _load_vectors()["rejected"]]
        assert rejected == ["bigint", "NaN / Infinity", "circular reference", "nesting > 64 levels"]

    async def test_produces_64_char_lowercase_hex(self) -> None:
        fp = await fingerprint_tool_call("t", {})
        assert re.fullmatch(r"[0-9a-f]{64}", fp) is not None

    async def test_includes_the_tool_name(self) -> None:
        assert await fingerprint_tool_call("web_search", {"q": "x"}) != await fingerprint_tool_call(
            "fetch_page", {"q": "x"}
        )

    async def test_non_ascii_key_material_hashes_over_utf8(self) -> None:
        matches = [v for v in _load_vectors()["toolCallVectors"] if "non-ascii" in v["name"]]
        assert len(matches) == 1
        vector = matches[0]
        assert await fingerprint_tool_call(vector["toolName"], vector["keyMaterial"]) == vector["fingerprint"]


# ---------------------------------------------------------------------------
# loop_key resolution (function | False | absent)
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
        assert sorted(resolution.keys()) == ["key_material", "kind", "warning"]

    def test_empty_field_array_does_not_collapse_unrelated_calls(self) -> None:
        ls = resolve_loop_key_material([], {"command": "ls"})
        rm = resolve_loop_key_material([], {"command": "rm -rf /"})
        assert ls["key_material"] != rm["key_material"]

    def test_field_array_whose_every_field_is_absent_warns_and_falls_back(self) -> None:
        resolution = resolve_loop_key_material(["nope", "also_nope"], ARGS)
        assert resolution["kind"] == "fallback"
        assert resolution["key_material"] is ARGS
        assert "absent from the arguments" in resolution["warning"]

    def test_preserves_dunder_proto_as_a_declared_field(self) -> None:
        args = json.loads('{"__proto__":"declared"}')
        assert resolve_loop_key_material(["__proto__"], args) == {
            "kind": "key",
            "key_material": {"__proto__": "declared"},
        }

    def test_function_returning_a_value(self) -> None:
        resolution = resolve_loop_key_material(lambda a: str(a["command"]).strip(), ARGS)
        assert resolution == {"kind": "key", "key_material": "ls"}

    def test_function_returning_none_is_per_call_exemption(self) -> None:
        # Upstream `() => null`. Python's None is JSON null.
        assert resolve_loop_key_material(lambda _a: None, ARGS) == {"kind": "exempt"}

    def test_throwing_function_falls_back_with_warning(self) -> None:
        def broken(_args: Dict[str, Any]) -> Any:
            raise RuntimeError("loopKey bug")

        resolution = resolve_loop_key_material(broken, ARGS)
        assert resolution["kind"] == "fallback"
        assert resolution["key_material"] is ARGS
        assert re.search("threw", resolution["warning"]) is not None
        assert "loopKey bug" in resolution["warning"]

    def test_unsupported_shape_falls_back_with_warning(self) -> None:
        resolution = resolve_loop_key_material(True, ARGS)
        assert resolution["kind"] == "fallback"
        assert resolution["key_material"] is ARGS
        assert "unsupported shape (bool)" in resolution["warning"]

    def test_async_function_falls_back_instead_of_hashing_a_coroutine(self) -> None:
        # Python-only: an `async def` loop_key returns a coroutine; upstream's
        # loopKey is synchronous, so the port falls back (and closes it).
        async def async_key(_args: Dict[str, Any]) -> Any:
            return "never"

        resolution = resolve_loop_key_material(async_key, ARGS)
        assert resolution["kind"] == "fallback"
        assert resolution["key_material"] is ARGS
        assert "awaitable" in resolution["warning"]


# ---------------------------------------------------------------------------
# Text repetition
# ---------------------------------------------------------------------------


class TestDetectTextRepetition:
    def test_detects_a_single_repeated_token(self) -> None:
        result = detect_text_repetition("no no no no no no no no no no no no")
        assert result is not None
        assert result["period_tokens"] == 1
        assert result["repeats"] == 12
        assert result["sample"] == "no"

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

    def test_handles_multi_megabyte_input_via_char_budget_tail_slice(self) -> None:
        filler = " ".join(f"w{i}" for i in range(400_000))
        doom_tail = " ".join(["stuck"] * 20)
        result = detect_text_repetition(f"{filler} {doom_tail}")
        assert result is not None
        assert result["period_tokens"] == 1
        assert result["repeats"] == 20

    def test_honors_custom_thresholds(self) -> None:
        text = "ha ha ha"
        assert detect_text_repetition(text) is None  # default min_repeats=4, min_covered_tokens=12
        result = detect_text_repetition(text, {"min_repeats": 3, "min_covered_tokens": 3})
        assert result is not None
        assert result["repeats"] == 3
        assert result["period_tokens"] == 1

    def test_tokenizes_on_ecmascript_whitespace(self) -> None:
        # Python-port-specific: U+FEFF is JS whitespace (Python's str.split
        # disagrees); \x1c is Python whitespace but NOT JS whitespace.
        result = detect_text_repetition("\ufeff".join(["go"] * 12))
        assert result is not None
        assert result["repeats"] == 12
        assert detect_text_repetition("\x1c".join(["go"] * 12)) is None


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------

LADDER: Dict[str, Any] = {"observe": 2, "steer": False, "block": 3, "stop": 6}


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

    def test_escalate_rung_and_allow_escalate(self) -> None:
        ladder = {"observe": 2, "steer": False, "escalate": 3, "block": 4, "stop": 6}
        assert resolve_ladder_action(ladder, 3, allow_block=True) == "escalate"
        assert resolve_ladder_action(ladder, 3, allow_block=True, allow_escalate=False) == "observe"
        assert resolve_ladder_action(ladder, 4, allow_block=False) == "escalate"


class TestResolveDoomLoopOption:
    def test_returns_none_for_none_or_false(self) -> None:
        assert resolve_doom_loop_option(None) is None
        assert resolve_doom_loop_option(False) is None

    def test_true_resolves_to_documented_defaults(self) -> None:
        resolved = resolve_doom_loop_option(True)
        assert resolved["ladder"] == dict(DEFAULT_DOOM_LOOP_LADDER)
        assert resolved["text"]["enabled"] is True
        assert resolved["escalation"] is None

    def test_merges_partial_ladders_and_clamps_invalid_thresholds(self) -> None:
        resolved = resolve_doom_loop_option({"ladder": {"block": 5, "stop": 0}})
        assert resolved["ladder"]["block"] == 5
        assert resolved["ladder"]["stop"] == DEFAULT_DOOM_LOOP_LADDER["stop"]
        assert resolved["ladder"]["observe"] == DEFAULT_DOOM_LOOP_LADDER["observe"]

    def test_advisor_false_is_not_a_recovery_mechanism(self) -> None:
        assert resolve_doom_loop_option({"escalation": {"advisor": False}})["escalation"] is None

    @pytest.mark.filterwarnings("ignore:.*escalate ladder rung is disabled")
    def test_advisor_false_does_not_resolve_alongside_a_real_mechanism(self) -> None:
        resolved = resolve_doom_loop_option({"escalation": {"model": "stronger-model", "advisor": False}})
        escalation = resolved["escalation"]
        assert escalation is not None
        assert escalation.get("model") == "stronger-model"
        assert "advisor" not in escalation

    @pytest.mark.filterwarnings("ignore:.*escalate ladder rung is disabled")
    def test_advisor_true_is_a_recovery_mechanism(self) -> None:
        escalation = resolve_doom_loop_option({"escalation": {"advisor": True}})["escalation"]
        assert escalation is not None
        assert escalation.get("advisor") is True

    def test_text_false_disables_text_detection(self) -> None:
        assert resolve_doom_loop_option({"text": False})["text"]["enabled"] is False

    def test_warns_when_block_enabled_with_stop_disabled(self) -> None:
        with pytest.warns(UserWarning, match="stop disabled"):
            resolve_doom_loop_option({"ladder": {"block": 3, "stop": False}})

    def test_warns_on_dead_rungs(self) -> None:
        with pytest.warns(UserWarning, match=re.escape('"observe" (5) can never fire')):
            resolve_doom_loop_option({"ladder": {"observe": 5, "block": 2}})

    def test_does_not_warn_on_the_default_ladder(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_doom_loop_option(True)
        assert [str(w.message) for w in caught] == []

    # -- Escalation resolution (exercised upstream only via the escalation
    # suite; pinned here because the monitor's budget depends on it).

    def test_escalation_budget_defaults_and_clamps(self) -> None:
        ladder = {"observe": 1, "escalate": 2, "block": 3}
        resolved = resolve_doom_loop_option({"ladder": ladder, "escalation": {"model": "m"}})
        assert resolved["escalation"] == {"model": "m", "max_escalations": 2}
        resolved = resolve_doom_loop_option(
            {"ladder": ladder, "escalation": {"advisor": {"x": 1}, "max_escalations": 3.9}}
        )
        assert resolved["escalation"] == {"advisor": {"x": 1}, "max_escalations": 3}

    def test_warns_on_escalation_config_without_rung_and_vice_versa(self) -> None:
        with pytest.warns(UserWarning, match="escalate ladder rung is disabled"):
            resolve_doom_loop_option({"escalation": {"model": "m"}})
        with pytest.warns(UserWarning, match="no escalation config"):
            resolved = resolve_doom_loop_option({"ladder": {"observe": 1, "escalate": 2, "block": 3}})
        assert resolved["escalation"] is None


# ---------------------------------------------------------------------------
# Monitor state machine (round-scoped streaks)
# ---------------------------------------------------------------------------


def make_monitor(overrides: Optional[DoomLoopOption] = None) -> DoomLoopMonitor:
    config = resolve_doom_loop_option(overrides if overrides is not None else True)
    if config is None:
        raise AssertionError("test setup: config must resolve")
    return DoomLoopMonitor(config)


class TestDoomLoopMonitorToolCallStreaks:
    async def test_fires_observe_at_2_and_block_at_3(self) -> None:
        monitor = make_monitor()
        args = {"query": "same thing"}
        assert "verdict" not in await monitor.record_tool_call("search", args, 1)
        assert _subset(
            (await monitor.record_tool_call("search", args, 2)).get("verdict"),
            {"action": "observe", "streak": 2, "detector": "tool-fingerprint", "tool_name": "search"},
        )
        assert _subset(
            (await monitor.record_tool_call("search", args, 3)).get("verdict"),
            {"action": "block", "streak": 3},
        )

    async def test_escalates_to_stop_at_the_stop_threshold(self) -> None:
        monitor = make_monitor()
        last: Optional[DoomLoopCallRecord] = None
        for round_ in range(1, 7):
            last = await monitor.record_tool_call("search", {"q": "x"}, round_)
        assert last is not None
        assert _subset(last.get("verdict"), {"action": "stop", "streak": 6})

    async def test_identical_calls_in_same_round_are_duplicates(self) -> None:
        monitor = make_monitor()
        args = {"q": "parallel"}
        first = await monitor.record_tool_call("search", args, 1)
        assert first["streak"] == 1
        assert first["duplicate_in_round"] is False
        for _ in range(5):
            dup = await monitor.record_tool_call("search", args, 1)
            assert dup["streak"] == 1
            assert dup["duplicate_in_round"] is True
            assert "verdict" not in dup
        nxt = await monitor.record_tool_call("search", args, 2)
        assert nxt["streak"] == 2
        assert nxt["duplicate_in_round"] is False

    async def test_different_arguments_reset_the_streak(self) -> None:
        monitor = make_monitor()
        await monitor.record_tool_call("search", {"q": "a"}, 1)
        await monitor.record_tool_call("search", {"q": "a"}, 2)
        assert "verdict" not in await monitor.record_tool_call("search", {"q": "b"}, 3)
        record = await monitor.record_tool_call("search", {"q": "b"}, 4)
        assert _subset(record.get("verdict"), {"streak": 2})

    async def test_interleaved_calls_to_other_tools_do_not_reset(self) -> None:
        monitor = make_monitor()
        search = {"q": "x"}
        await monitor.record_tool_call("search", search, 1)
        await monitor.record_tool_call("read_file", {"path": "/a"}, 2)
        record = await monitor.record_tool_call("search", search, 3)
        assert _subset(record.get("verdict"), {"action": "observe", "streak": 2})
        await monitor.record_tool_call("read_file", {"path": "/b"}, 4)
        record = await monitor.record_tool_call("search", search, 5)
        assert _subset(record.get("verdict"), {"action": "block", "streak": 3})

    async def test_argument_key_order_does_not_defeat_detection(self) -> None:
        monitor = make_monitor()
        await monitor.record_tool_call("run", {"cmd": "ls", "cwd": "/tmp"}, 1)
        record = await monitor.record_tool_call("run", {"cwd": "/tmp", "cmd": "ls"}, 2)
        assert _subset(record.get("verdict"), {"streak": 2})

    async def test_repeated_empty_calls_trip_the_detector(self) -> None:
        monitor = make_monitor()
        assert "verdict" not in await monitor.record_tool_call("list_tasks", {}, 1)
        record = await monitor.record_tool_call("list_tasks", {}, 2)
        assert _subset(record.get("verdict"), {"action": "observe", "streak": 2})
        record = await monitor.record_tool_call("list_tasks", {}, 3)
        assert _subset(record.get("verdict"), {"action": "block", "streak": 3})

    async def test_propagates_canonicalization_errors_for_the_engine_fallback(self) -> None:
        # Upstream: `{ id: 10n }` rejects with /bigint/. Python analogue: a set.
        monitor = make_monitor()
        with pytest.raises(DoomLoopCanonicalizationError, match="set"):
            await monitor.record_tool_call("search", {"id": {10}}, 1)

    async def test_server_tool_records_use_non_blockable_path_and_server_label(self) -> None:
        monitor = make_monitor()
        last: Optional[DoomLoopCallRecord] = None
        for round_ in range(1, 4):
            last = await monitor.record_tool_call(
                "server:web_search_call",
                {"query": "x"},
                round_,
                allow_block=False,
                detector="server-tool-fingerprint",
            )
        assert last is not None
        assert _subset(
            last.get("verdict"),
            {"detector": "server-tool-fingerprint", "action": "observe", "streak": 3},
        )

    async def test_is_deterministic_two_monitors_agree(self) -> None:
        script: List[Tuple[str, Any, int]] = [
            ("a", {"x": 1}, 1),
            ("b", {}, 1),
            ("a", {"x": 1}, 2),
            ("a", {"x": 2}, 3),
            ("a", {"x": 2}, 4),
        ]

        async def run() -> List[Optional[DoomLoopVerdict]]:
            monitor = make_monitor()
            results: List[Optional[DoomLoopVerdict]] = []
            for name, args, round_ in script:
                results.append((await monitor.record_tool_call(name, args, round_)).get("verdict"))
            return results

        first = await run()
        assert first == await run()
        assert [v["action"] if v else None for v in first] == [None, None, "observe", None, "observe"]


class TestDoomLoopMonitorText:
    async def test_within_response_repetition_can_stop_immediately(self) -> None:
        monitor = make_monitor()
        verdict = await monitor.record_assistant_text(" ".join(["I am stuck."] * 20))
        assert _subset(verdict, {"detector": "text-repetition", "action": "stop", "streak": 20})

    async def test_cross_step_identical_text_builds_a_streak(self) -> None:
        monitor = make_monitor()
        text = "Let me try that again."
        assert await monitor.record_assistant_text(text) is None
        assert _subset(
            await monitor.record_assistant_text(text),
            {"detector": "text-streak", "action": "observe", "streak": 2},
        )
        # Streak 3 crosses the block rung, but text cannot be blocked -> observe.
        assert _subset(
            await monitor.record_assistant_text(text),
            {"detector": "text-streak", "action": "observe", "streak": 3},
        )
        for _ in range(2):
            await monitor.record_assistant_text(text)
        assert _subset(await monitor.record_assistant_text(text), {"action": "stop", "streak": 6})

    async def test_whitespace_only_differences_do_not_defeat_the_streak(self) -> None:
        monitor = make_monitor()
        await monitor.record_assistant_text("Retrying   the\nsame plan.")
        assert _subset(await monitor.record_assistant_text("Retrying the same plan."), {"streak": 2})

    async def test_documented_miss_paraphrased_cross_step_text(self) -> None:
        monitor = make_monitor()
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
        monitor = make_monitor()
        await monitor.record_assistant_text("same")
        assert await monitor.record_assistant_text("") is None
        assert await monitor.record_assistant_text("   ") is None
        assert _subset(await monitor.record_assistant_text("same"), {"streak": 2})

    async def test_text_false_disables_both_text_detectors(self) -> None:
        monitor = make_monitor({"text": False})
        assert await monitor.record_assistant_text(" ".join(["loop"] * 30)) is None

    async def test_reset_text_streak_clears_only_the_text_streak(self) -> None:
        # Port-side coverage for `resetTextStreak` (upstream exercises it via
        # the async-tool integration suite).
        monitor = make_monitor()
        await monitor.record_tool_call("a", {}, 1)
        await monitor.record_assistant_text("waiting")
        monitor.reset_text_streak()
        assert "text" not in monitor.get_state()
        assert await monitor.record_assistant_text("waiting") is None
        assert (await monitor.record_tool_call("a", {}, 2))["streak"] == 2


class TestDoomLoopMonitorStateRoundTrip:
    async def test_serializes_to_plain_json_and_resumes_counting(self) -> None:
        monitor = make_monitor()
        await monitor.record_tool_call("search", {"q": "x"}, 1)
        await monitor.record_tool_call("search", {"q": "x"}, 2)

        blob = json.loads(json.dumps(monitor.get_state()))
        resumed = DoomLoopMonitor(resolve_doom_loop_option(True), blob)

        # Round markers are NOT serialized: the first resumed record increments.
        record = await resumed.record_tool_call("search", {"q": "x"}, 1)
        assert _subset(record.get("verdict"), {"action": "block", "streak": 3})

    async def test_ignores_corrupt_persisted_state_instead_of_crashing(self) -> None:
        config = resolve_doom_loop_option(True)
        with pytest.warns(UserWarning, match="Ignoring invalid persisted doom-loop state"):
            monitor = DoomLoopMonitor(config, "not an object")
        record = await monitor.record_tool_call("search", {"q": "x"}, 1)
        assert record["streak"] == 1
        assert "verdict" not in record

    async def test_a_persisted_dunder_proto_tool_name_cannot_pollute_bookkeeping(self) -> None:
        hostile = json.loads('{"tools":{"__proto__":{"fingerprint":"x","streak":999}}}')
        monitor = DoomLoopMonitor(resolve_doom_loop_option(True), hostile)
        record = await monitor.record_tool_call("unrelated_tool", {"q": "first call"}, 1)
        assert record["streak"] == 1
        assert "verdict" not in record

    async def test_a_tool_legitimately_named_dunder_proto_is_ordinary_data(self) -> None:
        monitor = make_monitor()
        await monitor.record_tool_call("__proto__", {}, 1)
        record = await monitor.record_tool_call("__proto__", {}, 2)
        assert record["streak"] == 2
        snapshot = monitor.get_state()
        assert list(snapshot["tools"].keys()) == ["__proto__"]
        revived = json.loads(json.dumps(snapshot))
        assert list(revived["tools"].keys()) == ["__proto__"]

    async def test_get_state_returns_a_snapshot_not_a_live_reference(self) -> None:
        monitor = make_monitor()
        await monitor.record_tool_call("a", {}, 1)
        snapshot = monitor.get_state()
        await monitor.record_tool_call("a", {}, 2)
        assert snapshot["tools"]["a"]["streak"] == 1
        assert monitor.get_state()["tools"]["a"]["streak"] == 2

    # -- Port-side coverage of the escalation budget and restore validation
    # (upstream covers these through doom-loop-escalation.test.ts, which drives
    # the full loop).

    async def test_escalation_budget_persists_and_gates_the_escalate_rung(self) -> None:
        option = {
            "ladder": {"observe": 1, "escalate": 2, "block": 3},
            "escalation": {"model": "big", "max_escalations": 1},
        }
        monitor = make_monitor(option)
        assert monitor.can_escalate() is True
        await monitor.record_tool_call("s", {}, 1)
        first = await monitor.record_tool_call("s", {}, 2)
        assert first.get("verdict", {}).get("action") == "escalate"
        monitor.consume_escalation()
        assert monitor.can_escalate() is False
        state = monitor.get_state()
        assert state.get("escalations_used") == 1

        resumed = DoomLoopMonitor(resolve_doom_loop_option(option), json.loads(json.dumps(state)))
        assert resumed.can_escalate() is False
        # Budget exhausted: a streak of 2 falls through to observe.
        await resumed.record_tool_call("t", {}, 1)
        assert (await resumed.record_tool_call("t", {}, 2)).get("verdict", {}).get("action") == "observe"

    async def test_restore_clamps_and_validates_fields(self) -> None:
        blob = {
            "tools": {
                "neg": {"fingerprint": "f", "streak": -5},
                "bool": {"fingerprint": "f", "streak": True},
                "huge": {"fingerprint": "f", "streak": 1e300},
            },
            "text": {"fingerprint": 3, "streak": 1},
            "escalations_used": -1,
            "stop_verdict": {"ignored": True},
        }
        monitor = DoomLoopMonitor(resolve_doom_loop_option(True), blob)
        state = monitor.get_state()
        assert state == {
            "tools": {
                "neg": {"fingerprint": "f", "streak": 1},
                "huge": {"fingerprint": "f", "streak": 1_000_000},
            }
        }
