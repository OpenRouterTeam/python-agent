"""Port of upstream tests/unit/tool-set.test.ts.

Dropped (type-only, `expectTypeOf` with no runtime assertion — contract
divergence #4, no compile-time partition tracking in Python):
- clone: "widens the partition/situation types when cloning to mutable"
- clone: "preserves the exact source partition type on clone() and clone({mutable: false})"
- mutable aliasing soundness: "gives createToolSet({mutable: true}) the widened partition/situation types"
- compile-time partition inference: "tracks static activate/deactivate transitions",
  "moves IDs into conditional via activateWhen/deactivateWhen"
- defineSituations: "keeps situation names literal at compile time"
- InferToolSet / event narrowing: "aliases CorrelatedToolEventUnion from @openrouter/agent"
- server tools / erased ServerToolBase: "accepts the real runtime ID ... (type-level)"
Cases mixing type and runtime assertions keep their runtime half.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from openrouter_agent.async_params import TOOL_SET_SNAPSHOT
from openrouter_agent.tool import server_tool, tool
from openrouter_agent.tool_set import ActivationInput, create_tool_set, get_tool_id
from openrouter_agent.tool_types import ConversationState


def make_tool(name: str) -> Dict[str, Any]:
    async def execute(params: Any, ctx: Any = None) -> Dict[str, str]:
        return {"name": name}

    return tool(name=name, description=f"{name} tool", input_schema={}, execute=execute)


a = make_tool("a")
b = make_tool("b")
c = make_tool("c")


def minimal_state(**partial: Any) -> ConversationState:
    fields: Dict[str, Any] = {"id": "conv_test", "messages": [], "status": "complete", "created_at": 0, "updated_at": 0}
    fields.update(partial)
    return ConversationState(**fields)


# ─── createToolSet ──────────────────────────────────────────────────────────


def test_preserves_tool_order_via_tools_property() -> None:
    ts = create_tool_set(tools=[a, b, c])
    assert [get_tool_id(t) for t in ts.tools] == ["a", "b", "c"]
    assert ts.tools == [a, b, c]


def test_throws_on_duplicate_tool_names_at_construction() -> None:
    dup = make_tool("a")
    with pytest.raises(ValueError, match=r'Duplicate tool ID: "a"'):
        create_tool_set(tools=[a, dup])


def test_constructs_an_empty_set_without_tools() -> None:
    ts = create_tool_set(tools=[])
    assert ts.tools == []
    snapshot = ts.resolve()
    assert snapshot["tools"] == []
    assert snapshot["active_tools"] == []
    assert snapshot["enabled"] == []
    assert snapshot["disabled"] == []
    assert snapshot["status_by_tool"] == {}


def test_defaults_all_tools_to_active_when_no_directives_are_set() -> None:
    ts = create_tool_set(tools=[a, b])
    snapshot = ts.resolve()
    assert snapshot["tools"] == [a, b]
    assert snapshot["active_tools"] == ["a", "b"]
    assert snapshot["enabled"] == ["a", "b"]
    assert snapshot["disabled"] == []
    assert snapshot["status_by_tool"] == {
        "a": {"enabled": True, "reason": "default"},
        "b": {"enabled": True, "reason": "default"},
    }


# ─── activate / deactivate ──────────────────────────────────────────────────


def test_deactivates_a_single_tool_by_name() -> None:
    ts = create_tool_set(tools=[a, b, c]).deactivate("b")
    assert ts.resolve()["active_tools"] == ["a", "c"]
    assert ts.resolve()["disabled"] == ["b"]


def test_activates_and_deactivates_arrays_of_names() -> None:
    ts = create_tool_set(tools=[a, b, c]).deactivate(["a", "b"]).activate(["b"])
    assert ts.resolve()["active_tools"] == ["b", "c"]


def test_throws_on_unknown_names() -> None:
    ts = create_tool_set(tools=[a])
    with pytest.raises(ValueError, match=r'Unknown tool: "missing"'):
        ts.activate("missing")
    with pytest.raises(ValueError, match=r'Unknown tool: "missing"'):
        ts.deactivate(["a", "missing"])


# ─── activateWhen ───────────────────────────────────────────────────────────


def test_activate_when_defaults_to_inactive_and_flips_based_on_predicate() -> None:
    ts = create_tool_set(tools=[a, b]).activate_when("a", lambda inp: (inp.get("context") or {}).get("enabled") is True)
    assert ts.resolve()["active_tools"] == ["b"]
    assert ts.resolve({"context": {"enabled": True}})["active_tools"] == ["a", "b"]


def test_activate_when_accepts_a_predicate_map() -> None:
    ts = create_tool_set(tools=[a, b]).activate_when({"a": lambda _: True, "b": lambda _: False})
    assert ts.resolve()["active_tools"] == ["a"]


def test_activate_when_validates_every_name_in_the_map_before_applying() -> None:
    ts = create_tool_set(tools=[a, b])
    with pytest.raises(ValueError, match=r'Unknown tool: "nope"'):
        ts.activate_when({"a": lambda _: True, "nope": lambda _: True})
    # original untouched
    assert ts.resolve()["active_tools"] == ["a", "b"]


# ─── deactivateWhen ─────────────────────────────────────────────────────────


def test_deactivate_when_defaults_to_active_and_flips_inactive_when_predicate_is_true() -> None:
    ts = create_tool_set(tools=[a, b]).deactivate_when("a", lambda _: True)
    assert ts.resolve()["active_tools"] == ["b"]


def test_deactivate_when_accepts_a_predicate_map() -> None:
    ts = create_tool_set(tools=[a, b]).deactivate_when({"a": lambda _: True, "b": lambda _: False})
    assert ts.resolve()["active_tools"] == ["b"]


# ─── last-call-wins semantics ───────────────────────────────────────────────


def test_resolves_to_the_most_recent_directive_per_tool() -> None:
    ts = create_tool_set(tools=[a, b]).activate("a").deactivate_when("a", lambda _: True)
    assert ts.resolve()["active_tools"] == ["b"]

    ts2 = create_tool_set(tools=[a, b]).deactivate_when("a", lambda _: True).activate("a")
    assert ts2.resolve()["active_tools"] == ["a", "b"]


# ─── immutability vs mutability ─────────────────────────────────────────────


def test_is_immutable_by_default_mutators_return_a_new_instance() -> None:
    base = create_tool_set(tools=[a, b])
    nxt = base.deactivate("a")
    assert nxt is not base
    assert base.resolve()["active_tools"] == ["a", "b"]
    assert nxt.resolve()["active_tools"] == ["b"]


def test_mutates_in_place_when_mutable_true() -> None:
    base = create_tool_set(tools=[a, b], mutable=True)
    nxt = base.deactivate("a")
    assert nxt is base
    assert base.resolve()["active_tools"] == ["b"]


# ─── clone ──────────────────────────────────────────────────────────────────


def test_clone_copies_state_and_can_flip_mode() -> None:
    immutable = create_tool_set(tools=[a, b]).deactivate("a")
    mutable_copy = immutable.clone(mutable=True)
    mutable_copy.activate("a")
    assert mutable_copy.resolve()["active_tools"] == ["a", "b"]
    # original untouched
    assert immutable.resolve()["active_tools"] == ["b"]


def test_clone_inherits_mode_when_not_overridden() -> None:
    mutable = create_tool_set(tools=[a], mutable=True)
    clone = mutable.clone()
    after = clone.deactivate("a")
    assert after is clone
    # Independent of the source (Python-side addition: guards the copy).
    assert mutable.resolve()["active_tools"] == ["a"]


def test_clone_to_immutable_returns_new_instances_on_mutation() -> None:
    # Runtime half of upstream's type-only "clone({mutable: false})" case.
    mutable = create_tool_set(tools=[a, b], mutable=True).deactivate("a")
    immutable = mutable.clone(mutable=False)
    after = immutable.activate("a")
    assert after is not immutable
    assert immutable.resolve()["active_tools"] == ["b"]
    assert after.resolve()["active_tools"] == ["a", "b"]


# ─── mutable aliasing soundness (runtime halves) ────────────────────────────


def test_every_alias_of_a_mutable_instance_is_the_same_object_after_divergent_mutations() -> None:
    base = create_tool_set(tools=[a, b, c], mutable=True)
    alias_one = base
    alias_two = base

    after_activate = alias_one.activate("a")
    after_deactivate = alias_two.deactivate("b")
    after_activate_when = after_activate.activate_when("c", lambda _: True)
    after_deactivate_when = after_deactivate.deactivate_when("c", lambda _: False)

    assert after_activate is base
    assert after_deactivate is base
    assert after_activate_when is base
    assert after_deactivate_when is base


def test_mutable_alias_observes_mutations_made_through_another_alias() -> None:
    mutable = create_tool_set(tools=[a, b], mutable=True)
    alias = mutable
    mutable.deactivate("a")
    assert alias.resolve()["active_tools"] == ["b"]


def test_immutable_chain_steps_are_distinct_instances() -> None:
    base = create_tool_set(tools=[a, b, c])
    after_deactivate = base.deactivate("b")
    after_activate_when = after_deactivate.activate_when("a", lambda _: True)
    assert after_deactivate is not base
    assert after_activate_when is not after_deactivate


# ─── resolve / inferTools input shapes ──────────────────────────────────────


def test_handles_none_and_empty_input() -> None:
    ts = create_tool_set(tools=[a]).activate_when(
        "a", lambda inp: inp.get("state") is None and inp.get("context") is None
    )
    assert ts.resolve()["active_tools"] == ["a"]
    assert ts.resolve({})["active_tools"] == ["a"]


def test_passes_state_and_context_to_the_predicate() -> None:
    calls: List[ActivationInput] = []

    def spy(inp: ActivationInput) -> bool:
        calls.append(inp)
        return True

    ts = create_tool_set(tools=[a]).activate_when("a", spy)
    state = minimal_state(messages=[{"role": "user", "content": "hi"}])
    ts.resolve({"state": state, "context": {"foo": "bar"}})
    assert calls == [{"state": state, "context": {"foo": "bar"}}]


def test_keeps_infer_tools_as_a_back_compat_alias_of_resolve() -> None:
    ts = create_tool_set(tools=[a, b]).deactivate("a")
    via_resolve = ts.resolve()
    via_infer = ts.infer_tools()
    assert via_infer["tools"] == via_resolve["tools"]
    assert via_infer["active_tools"] == via_resolve["active_tools"]
    assert via_infer["enabled"] == via_resolve["enabled"]
    assert via_infer["disabled"] == via_resolve["disabled"]
    assert via_infer["status_by_tool"] == via_resolve["status_by_tool"]
    # Both carry the snapshot marker so call_model can strip their metadata.
    assert via_infer[TOOL_SET_SNAPSHOT] is True
    assert via_resolve[TOOL_SET_SNAPSHOT] is True
    assert "call_model" not in via_infer


# ─── exhaustive statusByTool snapshot ───────────────────────────────────────


def test_status_by_tool_includes_every_id_with_reason_directive_predicate_metadata() -> None:
    ts = create_tool_set(tools=[a, b, c]).deactivate("b").activate_when("c", lambda _: True)
    snapshot = ts.resolve()
    status = snapshot["status_by_tool"]
    assert sorted(status) == ["a", "b", "c"]
    assert status["a"] == {"enabled": True, "reason": "default"}
    assert status["b"] == {"enabled": False, "reason": "deactivate", "directive": "deactivate"}
    assert status["c"] == {"enabled": True, "reason": "activateWhen", "directive": "activateWhen", "predicate": True}
    assert snapshot["enabled"] == ["a", "c"]
    assert snapshot["disabled"] == ["b"]


def test_keeps_prototype_sensitive_ids_as_real_keys_of_status_by_tool() -> None:
    # Upstream guards JS `__proto__` setter semantics; Python dicts have none,
    # but the ID-handling invariants (exhaustive keys, order) still hold. The
    # `Object.getPrototypeOf(statusByTool) === null` assertion is JS-only.
    dunder_proto = make_tool("__proto__")
    ctor = make_tool("constructor")
    proto = make_tool("prototype")

    ts = create_tool_set(tools=[a, dunder_proto, ctor, proto, b]).deactivate("constructor")
    snapshot = ts.resolve()
    status = snapshot["status_by_tool"]

    assert sorted(status) == ["__proto__", "a", "b", "constructor", "prototype"]
    assert type(status) is dict
    assert status["constructor"] == {"enabled": False, "reason": "deactivate", "directive": "deactivate"}
    assert status["prototype"] == {"enabled": True, "reason": "default"}
    assert status["__proto__"] == {"enabled": True, "reason": "default"}
    assert status["a"] == {"enabled": True, "reason": "default"}
    assert status["b"] == {"enabled": True, "reason": "default"}

    assert snapshot["enabled"] == ["a", "__proto__", "prototype", "b"]
    assert snapshot["disabled"] == ["constructor"]
    assert snapshot["tools"] == [a, dunder_proto, proto, b]
    assert snapshot["active_tools"] == ["a", "__proto__", "prototype", "b"]


def test_resolves_exotic_ids_individually_via_activate_deactivate_activate_when() -> None:
    dunder_proto = make_tool("__proto__")
    ctor = make_tool("constructor")
    proto = make_tool("prototype")

    ts = (
        create_tool_set(tools=[dunder_proto, ctor, proto])
        .activate("__proto__")
        .deactivate("prototype")
        .activate_when("constructor", lambda _: False)
    )
    snapshot = ts.resolve()
    status = snapshot["status_by_tool"]

    assert status["__proto__"] == {"enabled": True, "reason": "activate", "directive": "activate"}
    assert status["constructor"] == {
        "enabled": False,
        "reason": "activateWhen",
        "directive": "activateWhen",
        "predicate": True,
    }
    assert status["prototype"] == {"enabled": False, "reason": "deactivate", "directive": "deactivate"}
    assert sorted(snapshot["enabled"]) == ["__proto__"]
    assert sorted(snapshot["disabled"]) == ["constructor", "prototype"]


# ─── compile-time partition inference (runtime halves) ──────────────────────


def test_includes_a_conditional_id_in_the_runtime_disabled_list_when_predicate_resolves_false() -> None:
    ts = create_tool_set(tools=[a, b]).deactivate("b").activate_when("a", lambda _: False)
    assert ts.resolve()["disabled"] == ["a", "b"]


def test_returns_the_active_tools_for_static_partitions() -> None:
    ts = create_tool_set(tools=[a, b, c]).deactivate("b")
    snapshot = ts.resolve()
    assert snapshot["tools"] == [a, c]
    assert snapshot["active_tools"] == ["a", "c"]


# ─── server tools ───────────────────────────────────────────────────────────

web_search = server_tool({"type": "web_search_2025_08_26"})
datetime_tool = server_tool({"type": "openrouter:datetime"})
public_search = server_tool({"type": "web_search_2025_08_26"}, {"id": "server:public_search"})


def test_assigns_default_server_ids_from_config_type() -> None:
    assert web_search["id"] == "server:web_search_2025_08_26"
    assert datetime_tool["id"] == "server:openrouter:datetime"
    assert public_search["id"] == "server:public_search"


def test_preserves_server_tools_in_tools_in_construction_order() -> None:
    ts = create_tool_set(tools=[a, web_search, b, datetime_tool])
    assert ts.tools == [a, web_search, b, datetime_tool]
    assert [get_tool_id(t) for t in ts.tools] == [
        "a",
        "server:web_search_2025_08_26",
        "b",
        "server:openrouter:datetime",
    ]


def test_includes_active_server_tools_in_tools_enabled_status_but_not_active_tools() -> None:
    ts = create_tool_set(tools=[a, web_search, b]).deactivate("a")
    snapshot = ts.resolve()
    assert snapshot["tools"] == [web_search, b]
    assert snapshot["active_tools"] == ["b"]
    assert snapshot["enabled"] == ["server:web_search_2025_08_26", "b"]
    assert snapshot["status_by_tool"]["server:web_search_2025_08_26"] == {"enabled": True, "reason": "default"}


def test_can_deactivate_server_tools_by_stable_id() -> None:
    ts = create_tool_set(tools=[a, web_search, b]).deactivate("server:web_search_2025_08_26")
    snapshot = ts.resolve()
    assert snapshot["tools"] == [a, b]
    assert snapshot["enabled"] == ["a", "b"]
    assert snapshot["disabled"] == ["server:web_search_2025_08_26"]
    assert snapshot["status_by_tool"]["server:web_search_2025_08_26"] == {
        "enabled": False,
        "reason": "deactivate",
        "directive": "deactivate",
    }


def test_supports_override_ids_and_rejects_duplicates() -> None:
    ts = create_tool_set(tools=[a, public_search])
    assert ts.resolve()["enabled"] == ["a", "server:public_search"]

    with pytest.raises(ValueError, match=r'Duplicate tool ID: "server:web_search_2025_08_26"'):
        create_tool_set(tools=[web_search, server_tool({"type": "web_search_2025_08_26"})])


def test_rejects_activate_on_raw_server_type_strings() -> None:
    ts = create_tool_set(tools=[a, web_search])
    with pytest.raises(ValueError, match=r"Unknown tool"):
        ts.activate("web_search_2025_08_26")


HAND_WRITTEN: Dict[str, Any] = {"_brand": "server-tool", "config": {"type": "web_search_2025_08_26"}}
HAND_WRITTEN_ID = "server:web_search_2025_08_26"


def test_hand_written_server_tool_uses_the_synthesized_id_for_activation_status_and_filtering() -> None:
    ts = create_tool_set(tools=[a, HAND_WRITTEN]).deactivate(HAND_WRITTEN_ID)
    resolved = ts.resolve()
    assert resolved["enabled"] == ["a"]
    assert resolved["disabled"] == [HAND_WRITTEN_ID]
    assert resolved["status_by_tool"][HAND_WRITTEN_ID] == {
        "enabled": False,
        "reason": "deactivate",
        "directive": "deactivate",
    }
    assert resolved["tools"] == [a]


def test_hand_written_server_tool_can_be_activated_by_synthesized_id() -> None:
    # Runtime half of upstream server-tool-id.test-d.ts (`activate(handWrittenId)`).
    ts = create_tool_set(tools=[HAND_WRITTEN]).deactivate(HAND_WRITTEN_ID).activate(HAND_WRITTEN_ID)
    assert ts.resolve()["status_by_tool"] == {
        HAND_WRITTEN_ID: {"enabled": True, "reason": "activate", "directive": "activate"}
    }


def test_custom_id_server_tool_accepts_the_real_runtime_id() -> None:
    erased = server_tool({"type": "web_search_2025_08_26"}, {"id": "server:public_search"})
    ts = create_tool_set(tools=[a, erased]).deactivate("server:public_search")
    snapshot = ts.resolve()
    assert snapshot["enabled"] == ["a"]
    assert snapshot["disabled"] == ["server:public_search"]


def test_custom_id_server_tool_rejects_the_synthesized_default_id() -> None:
    erased = server_tool({"type": "web_search_2025_08_26"}, {"id": "server:public_search"})
    ts = create_tool_set(tools=[a, erased])
    with pytest.raises(ValueError, match=r"Unknown tool"):
        ts.deactivate("server:web_search_2025_08_26")


# ─── defineSituations / resolveSituation ────────────────────────────────────


def test_situations_overlay_static_enabled_disabled() -> None:
    ts = (
        create_tool_set(tools=[a, b, c])
        .deactivate("c")
        .define_situations(
            {
                "guest": {"enabled": ["a"], "disabled": ["b", "c"]},
                "full": {"enabled": ["a", "b", "c"]},
            }
        )
    )

    guest = ts.resolve_situation("guest")
    assert guest["tools"] == [a]
    assert guest["active_tools"] == ["a"]
    assert guest["enabled"] == ["a"]
    assert guest["disabled"] == ["b", "c"]
    assert guest["status_by_tool"] == {
        "a": {"enabled": True, "reason": "situation", "directive": "activate"},
        "b": {"enabled": False, "reason": "situation", "directive": "deactivate"},
        "c": {"enabled": False, "reason": "situation", "directive": "deactivate"},
    }

    full = ts.resolve_situation("full")
    assert full["tools"] == [a, b, c]


def test_supports_conditional_situation_rules_with_runtime_exact_status() -> None:
    ts = create_tool_set(tools=[a, b, c]).define_situations(
        {
            "authed": {
                "enabled": ["a"],
                "disabled": ["b"],
                "conditional": {"c": lambda inp: (inp.get("context") or {}).get("admin") is True},
            }
        }
    )

    denied = ts.resolve_situation("authed", {"context": {"admin": False}})
    assert denied["tools"] == [a]
    assert denied["enabled"] == ["a"]
    assert denied["disabled"] == ["b", "c"]
    assert denied["status_by_tool"]["c"] == {
        "enabled": False,
        "reason": "situation",
        "directive": "activateWhen",
        "predicate": True,
    }

    allowed = ts.resolve_situation("authed", {"context": {"admin": True}})
    assert allowed["tools"] == [a, c]
    assert allowed["enabled"] == ["a", "c"]


def test_supports_deactivate_when_situation_rules() -> None:
    ts = create_tool_set(tools=[a]).define_situations(
        {
            "guarded": {
                "conditional": {
                    "a": {
                        "mode": "deactivateWhen",
                        "predicate": lambda inp: (inp.get("context") or {}).get("blocked") is True,
                    }
                }
            }
        }
    )

    assert ts.resolve_situation("guarded")["enabled"] == ["a"]
    blocked = ts.resolve_situation("guarded", {"context": {"blocked": True}})
    assert blocked["disabled"] == ["a"]
    assert blocked["status_by_tool"]["a"] == {
        "enabled": False,
        "reason": "situation",
        "directive": "deactivateWhen",
        "predicate": True,
    }


def test_validates_unknown_duplicate_conflicting_ids_in_a_situation() -> None:
    base = create_tool_set(tools=[a, b])

    with pytest.raises(ValueError, match=r'Unknown tool: "nope"'):
        base.define_situations({"bad": {"enabled": ["nope"]}})
    with pytest.raises(ValueError, match=r'lists tool "a" more than once'):
        base.define_situations({"bad": {"enabled": ["a"], "disabled": ["a"]}})
    with pytest.raises(ValueError, match=r'lists tool "a" more than once'):
        base.define_situations({"bad": {"enabled": ["a"], "conditional": {"a": lambda _: True}}})


def test_ignores_none_conditional_entries() -> None:
    ts = create_tool_set(tools=[a, b]).define_situations(
        {"optional": {"conditional": {"a": None, "b": lambda _: False}}}
    )
    assert ts.resolve_situation("optional")["active_tools"] == ["a"]


_MALFORMED = 'Situation "checkout": conditional rule for tool "a" must be a function or { mode, predicate } object'


def test_rejects_malformed_conditional_rules_at_definition_time() -> None:
    base = create_tool_set(tools=[a])
    with pytest.raises(ValueError) as exc:
        base.define_situations({"checkout": {"conditional": {"a": "invalid"}}})  # type: ignore[dict-item]
    assert str(exc.value) == _MALFORMED


def test_rejects_a_missing_conditional_predicate_at_definition_time() -> None:
    base = create_tool_set(tools=[a])
    with pytest.raises(ValueError) as exc:
        base.define_situations({"checkout": {"conditional": {"a": {"mode": "activateWhen"}}}})
    assert str(exc.value) == _MALFORMED


def test_throws_on_unknown_situation_names_at_resolve_time() -> None:
    ts = create_tool_set(tools=[a]).define_situations({"guest": {"enabled": ["a"]}})
    with pytest.raises(ValueError, match=r'Unknown situation: "missing"'):
        ts.resolve_situation("missing")


def test_leaves_unmentioned_ids_on_the_base_partition() -> None:
    ts = create_tool_set(tools=[a, b, c]).deactivate("c").define_situations({"onlyB": {"disabled": ["b"]}})
    snapshot = ts.resolve_situation("onlyB")
    # a stays default-enabled, b disabled by situation, c disabled by base
    assert snapshot["enabled"] == ["a"]
    assert snapshot["disabled"] == ["b", "c"]


# ─── TShared generic (runtime halves) ───────────────────────────────────────


def test_predicate_reads_typed_shared_context() -> None:
    def is_authenticated(inp: ActivationInput) -> bool:
        context = inp.get("context")
        if not context:
            return False
        return bool(context["isAuthenticated"])

    ts = create_tool_set(tools=[a]).activate_when("a", is_authenticated)
    assert ts.resolve({"context": {"isAuthenticated": True, "userId": "u1"}})["active_tools"] == ["a"]
    assert ts.resolve({"context": {"isAuthenticated": False, "userId": "u1"}})["active_tools"] == []


def test_predicate_reads_untyped_context() -> None:
    def enabled(inp: ActivationInput) -> bool:
        context = inp.get("context")
        if not context:
            return False
        return context.get("enabled") is True

    ts = create_tool_set(tools=[a]).activate_when("a", enabled)
    assert ts.resolve({"context": {"enabled": True}})["active_tools"] == ["a"]


# ─── callModel-oriented spread shape ────────────────────────────────────────


def test_produces_tools_and_active_tools_suitable_for_call_model_spread() -> None:
    ts = create_tool_set(tools=[a, b, c]).deactivate("b")
    snapshot = ts.resolve()
    for_call_model = snapshot["call_model"]
    assert set(for_call_model) == {"tools", "active_tools"}
    assert for_call_model["tools"] == [a, c]
    assert for_call_model["active_tools"] == ["a", "c"]


# ─── Python-port additions (argument validation paths) ──────────────────────


def test_activate_when_with_a_name_requires_a_predicate() -> None:
    ts = create_tool_set(tools=[a])
    with pytest.raises(ValueError, match=r"requires a predicate when called with a name"):
        ts.activate_when("a")


def test_activate_when_rejects_a_non_mapping_non_string_argument() -> None:
    ts = create_tool_set(tools=[a])
    with pytest.raises(ValueError, match=r"requires a name\+predicate or predicate map"):
        ts.deactivate_when(["a"])  # type: ignore[arg-type]


def test_predicate_map_skips_non_callable_values_without_validating_them() -> None:
    # Upstream filters entries to functions before `#assertKnown`.
    ts = create_tool_set(tools=[a, b]).activate_when({"a": lambda _: False, "nope": None})
    assert ts.resolve()["active_tools"] == ["b"]


def test_only_a_literal_true_predicate_result_counts_as_true() -> None:
    # Upstream compares `predicate(input) === true`; truthy non-True is false.
    ts = create_tool_set(tools=[a, b]).activate_when("a", lambda _: 1).deactivate_when("b", lambda _: "yes")
    assert ts.resolve()["active_tools"] == ["b"]


def test_mutable_define_situations_replaces_registry_in_place() -> None:
    ts = create_tool_set(tools=[a, b], mutable=True)
    assert ts.define_situations({"one": {"disabled": ["a"]}}) is ts
    ts.define_situations({"two": {"disabled": ["b"]}})
    assert ts.resolve_situation("two")["active_tools"] == ["a"]
    with pytest.raises(ValueError, match=r'Unknown situation: "one"'):
        ts.resolve_situation("one")


def test_immutable_define_situations_does_not_leak_into_the_source() -> None:
    base = create_tool_set(tools=[a])
    with_situations = base.define_situations({"off": {"disabled": ["a"]}})
    assert with_situations is not base
    assert with_situations.resolve_situation("off")["active_tools"] == []
    with pytest.raises(ValueError, match=r'Unknown situation: "off"'):
        base.resolve_situation("off")


def test_mutable_clone_situations_are_independent_of_source() -> None:
    base = create_tool_set(tools=[a]).define_situations({"off": {"disabled": ["a"]}})
    mutable_copy = base.clone(mutable=True)
    mutable_copy.define_situations({"on": {"enabled": ["a"]}})
    assert base.resolve_situation("off")["active_tools"] == []
    with pytest.raises(ValueError, match=r'Unknown situation: "off"'):
        mutable_copy.resolve_situation("off")
