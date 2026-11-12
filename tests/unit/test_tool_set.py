"""Ports `packages/agent/tests/unit/tool-set.test.ts`.

Runtime cases are ported 1:1 (same `describe` grouping, as classes). Upstream
cases that are *purely* `expectTypeOf` assertions are not reproducible in
Python (contract divergence 4 -- typing is looser) and are omitted:

- clone: "widens the partition/situation types when cloning to mutable",
  "preserves the exact source partition type on clone() and clone({mutable: false})"
- mutable aliasing soundness: "gives createToolSet({mutable: true}) the widened
  partition/situation types"
- compile-time partition inference: "tracks static activate/deactivate
  transitions", "moves IDs into conditional via activateWhen/deactivateWhen"
- defineSituations: "keeps situation names literal at compile time"
- InferToolSet / event narrowing: "aliases CorrelatedToolEventUnion"

Mixed type/runtime cases keep only their runtime assertions.

Server-tool ids: upstream `serverTool(config, { id })` stamps `id` on the tool
(`src/lib/tool.ts` serverTool). This port's `server_tool` has no `id` option,
so custom-id server tools are built as `{**server_tool(cfg), "id": ...}` and the
default id is read via `get_tool_id` (same `server:${type}` rule).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from openrouter_agent.tool import server_tool, tool
from openrouter_agent.tool_set import (
    TOOL_SET_SNAPSHOT,
    ActivationInput,
    ToolSet,
    create_tool_set,
    get_tool_id,
)
from openrouter_agent.tool_types import ConversationState


def make_tool(name: str) -> Dict[str, Any]:
    async def execute(params: Any, ctx: Any) -> Dict[str, Any]:
        return {"name": name}

    return tool(name=name, description=f"{name} tool", input_schema=dict, execute=execute)


a = make_tool("a")
b = make_tool("b")
c = make_tool("c")


def minimal_state(**partial: Any) -> ConversationState:
    base: Dict[str, Any] = {"id": "conv_test", "messages": [], "status": "complete", "created_at": 0, "updated_at": 0}
    base.update(partial)
    return ConversationState(**base)


def ctx(inp: ActivationInput) -> Dict[str, Any]:
    return dict(inp.get("context") or {})


class PredicateSpy:
    def __init__(self, result: bool = True) -> None:
        self.calls: List[ActivationInput] = []
        self.result = result

    def __call__(self, inp: ActivationInput) -> bool:
        self.calls.append(inp)
        return self.result


class TestCreateToolSet:
    def test_preserves_tool_order_via_tools(self) -> None:
        ts = create_tool_set([a, b, c])
        assert [get_tool_id(t) for t in ts.tools] == ["a", "b", "c"]
        assert ts.tools == (a, b, c)

    def test_throws_on_duplicate_tool_names_at_construction(self) -> None:
        dup = make_tool("a")
        with pytest.raises(ValueError, match='Duplicate tool ID: "a"'):
            create_tool_set([a, dup])

    def test_constructs_an_empty_set_without_tools(self) -> None:
        ts = create_tool_set([])
        assert ts.tools == ()
        resolved = ts.resolve()
        assert resolved.tools == []
        assert resolved.active_tools == []
        assert resolved.enabled == []
        assert resolved.disabled == []
        assert resolved.status_by_tool == {}

    def test_defaults_all_tools_to_active_when_no_directives_are_set(self) -> None:
        resolved = create_tool_set([a, b]).resolve()
        assert resolved.tools == [a, b]
        assert resolved.active_tools == ["a", "b"]
        assert resolved.enabled == ["a", "b"]
        assert resolved.disabled == []
        assert resolved.status_by_tool == {
            "a": {"enabled": True, "reason": "default"},
            "b": {"enabled": True, "reason": "default"},
        }


class TestActivateDeactivate:
    def test_deactivates_a_single_tool_by_name(self) -> None:
        ts = create_tool_set([a, b, c]).deactivate("b")
        assert ts.resolve().active_tools == ["a", "c"]
        assert ts.resolve().disabled == ["b"]

    def test_activates_deactivates_arrays_of_names(self) -> None:
        ts = create_tool_set([a, b, c]).deactivate(["a", "b"]).activate(["b"])
        assert ts.resolve().active_tools == ["b", "c"]

    def test_throws_on_unknown_names(self) -> None:
        ts = create_tool_set([a])
        with pytest.raises(ValueError, match='Unknown tool: "missing"'):
            ts.activate("missing")
        with pytest.raises(ValueError, match='Unknown tool: "missing"'):
            ts.deactivate(["a", "missing"])


class TestActivateWhen:
    def test_defaults_to_inactive_and_flips_based_on_predicate(self) -> None:
        ts = create_tool_set([a, b]).activate_when("a", lambda i: ctx(i).get("enabled") is True)
        assert ts.resolve().active_tools == ["b"]
        assert ts.resolve({"context": {"enabled": True}}).active_tools == ["a", "b"]

    def test_accepts_a_predicate_map(self) -> None:
        ts = create_tool_set([a, b]).activate_when({"a": lambda _i: True, "b": lambda _i: False})
        assert ts.resolve().active_tools == ["a"]

    def test_validates_every_name_in_the_map_before_applying(self) -> None:
        ts = create_tool_set([a, b])
        with pytest.raises(ValueError, match='Unknown tool: "nope"'):
            ts.activate_when({"a": lambda _i: True, "nope": lambda _i: True})
        # original untouched
        assert ts.resolve().active_tools == ["a", "b"]


class TestDeactivateWhen:
    def test_defaults_to_active_and_flips_inactive_when_predicate_is_true(self) -> None:
        ts = create_tool_set([a, b]).deactivate_when("a", lambda _i: True)
        assert ts.resolve().active_tools == ["b"]

    def test_accepts_a_predicate_map(self) -> None:
        ts = create_tool_set([a, b]).deactivate_when({"a": lambda _i: True, "b": lambda _i: False})
        assert ts.resolve().active_tools == ["b"]


class TestLastCallWins:
    def test_resolves_to_the_most_recent_directive_per_tool(self) -> None:
        ts = create_tool_set([a, b]).activate("a").deactivate_when("a", lambda _i: True)
        assert ts.resolve().active_tools == ["b"]

        ts2 = create_tool_set([a, b]).deactivate_when("a", lambda _i: True).activate("a")
        assert ts2.resolve().active_tools == ["a", "b"]


class TestImmutabilityVsMutability:
    def test_is_immutable_by_default_mutators_return_a_new_instance(self) -> None:
        base = create_tool_set([a, b])
        nxt = base.deactivate("a")
        assert nxt is not base
        assert base.resolve().active_tools == ["a", "b"]
        assert nxt.resolve().active_tools == ["b"]

    def test_mutates_in_place_when_mutable_true(self) -> None:
        base = create_tool_set([a, b], mutable=True)
        nxt = base.deactivate("a")
        assert nxt is base
        assert base.resolve().active_tools == ["b"]


class TestClone:
    def test_copies_state_and_can_flip_mode(self) -> None:
        immutable = create_tool_set([a, b]).deactivate("a")
        mutable_copy = immutable.clone(mutable=True)
        mutable_copy.activate("a")
        assert mutable_copy.resolve().active_tools == ["a", "b"]
        # original untouched
        assert immutable.resolve().active_tools == ["b"]

    def test_inherits_mode_when_not_overridden(self) -> None:
        mutable = create_tool_set([a], mutable=True)
        clone = mutable.clone()
        after = clone.deactivate("a")
        assert after is clone
        # A clone is independent of its source.
        assert mutable.resolve().active_tools == ["a"]

    def test_clone_mutable_false_from_mutable_returns_new_instances(self) -> None:
        # Runtime half of upstream "preserves the exact source partition type
        # on clone() and clone({mutable: false})".
        mutable = create_tool_set([a, b], mutable=True)
        immutable = mutable.clone(mutable=False)
        after = immutable.deactivate("a")
        assert after is not immutable
        assert immutable.resolve().active_tools == ["a", "b"]
        assert after.resolve().active_tools == ["b"]


class TestMutableAliasingSoundness:
    def test_every_alias_of_a_mutable_instance_is_the_same_object(self) -> None:
        base = create_tool_set([a, b, c], mutable=True)
        alias_one = base
        alias_two = base

        after_activate = alias_one.activate("a")
        after_deactivate = alias_two.deactivate("b")
        after_activate_when = after_activate.activate_when("c", lambda _i: True)
        after_deactivate_when = after_deactivate.deactivate_when("c", lambda _i: False)

        assert after_activate is base
        assert after_deactivate is base
        assert after_activate_when is base
        assert after_deactivate_when is base

    def test_mutable_alias_observes_mutations_made_through_another_alias(self) -> None:
        mutable = create_tool_set([a, b], mutable=True)
        alias = mutable
        mutable.deactivate("a")
        assert alias.resolve().active_tools == ["b"]

    def test_keeps_the_immutable_chain_producing_distinct_instances(self) -> None:
        base = create_tool_set([a, b, c])
        after_deactivate = base.deactivate("b")
        after_activate_when = after_deactivate.activate_when("a", lambda _i: True)
        assert after_deactivate is not base
        assert after_activate_when is not after_deactivate


class TestResolveInputShapes:
    def test_handles_none_and_empty_input(self) -> None:
        ts = create_tool_set([a]).activate_when("a", lambda i: "state" not in i and "context" not in i)
        assert ts.resolve().active_tools == ["a"]
        assert ts.resolve({}).active_tools == ["a"]

    def test_passes_state_and_context_to_the_predicate(self) -> None:
        spy = PredicateSpy()
        ts = create_tool_set([a]).activate_when("a", spy)
        state = minimal_state(messages=[{"role": "user", "content": "hi"}])
        ts.resolve({"state": state, "context": {"foo": "bar"}})
        assert spy.calls == [{"state": state, "context": {"foo": "bar"}}]

    def test_keeps_infer_tools_as_a_back_compat_alias_of_resolve(self) -> None:
        ts = create_tool_set([a, b]).deactivate("a")
        via_resolve = ts.resolve()
        via_infer = ts.infer_tools()
        assert via_infer["tools"] == via_resolve.tools
        assert via_infer["active_tools"] == via_resolve.active_tools
        assert via_infer["enabled"] == via_resolve.enabled
        assert via_infer["disabled"] == via_resolve.disabled
        assert via_infer["status_by_tool"] == via_resolve.status_by_tool
        # Upstream marks the result with the TOOL_SET_SNAPSHOT symbol so
        # call_model can strip its metadata when the whole object is spread.
        assert via_infer[TOOL_SET_SNAPSHOT] is True
        assert {**via_infer}[TOOL_SET_SNAPSHOT] is True


class TestExhaustiveStatusByTool:
    def test_includes_every_id_with_reason_directive_predicate_metadata(self) -> None:
        ts = create_tool_set([a, b, c]).deactivate("b").activate_when("c", lambda _i: True)
        resolved = ts.resolve()
        assert sorted(resolved.status_by_tool) == ["a", "b", "c"]
        assert resolved.status_by_tool["a"] == {"enabled": True, "reason": "default"}
        assert resolved.status_by_tool["b"] == {"enabled": False, "reason": "deactivate", "directive": "deactivate"}
        assert resolved.status_by_tool["c"] == {
            "enabled": True,
            "reason": "activateWhen",
            "directive": "activateWhen",
            "predicate": True,
        }
        assert resolved.enabled == ["a", "c"]
        assert resolved.disabled == ["b"]

    def test_keeps_prototype_sensitive_ids_as_real_keys(self) -> None:
        # Upstream guards JS `__proto__` setter semantics; Python dicts have no
        # such hazard, but the invariant (every id is a real key, nothing
        # dropped) is still worth pinning.
        dunder_proto = make_tool("__proto__")
        ctor = make_tool("constructor")
        proto = make_tool("prototype")

        resolved = create_tool_set([a, dunder_proto, ctor, proto, b]).deactivate("constructor").resolve()

        assert sorted(resolved.status_by_tool) == ["__proto__", "a", "b", "constructor", "prototype"]
        assert resolved.status_by_tool["constructor"] == {
            "enabled": False,
            "reason": "deactivate",
            "directive": "deactivate",
        }
        assert resolved.status_by_tool["prototype"] == {"enabled": True, "reason": "default"}
        assert resolved.status_by_tool["__proto__"] == {"enabled": True, "reason": "default"}
        assert resolved.status_by_tool["a"] == {"enabled": True, "reason": "default"}
        assert resolved.status_by_tool["b"] == {"enabled": True, "reason": "default"}

        assert resolved.enabled == ["a", "__proto__", "prototype", "b"]
        assert resolved.disabled == ["constructor"]
        assert resolved.tools == [a, dunder_proto, proto, b]
        assert resolved.active_tools == ["a", "__proto__", "prototype", "b"]

    def test_resolves_exotic_ids_individually(self) -> None:
        dunder_proto = make_tool("__proto__")
        ctor = make_tool("constructor")
        proto = make_tool("prototype")

        resolved = (
            create_tool_set([dunder_proto, ctor, proto])
            .activate("__proto__")
            .deactivate("prototype")
            .activate_when("constructor", lambda _i: False)
            .resolve()
        )

        assert resolved.status_by_tool["__proto__"] == {"enabled": True, "reason": "activate", "directive": "activate"}
        assert resolved.status_by_tool["constructor"] == {
            "enabled": False,
            "reason": "activateWhen",
            "directive": "activateWhen",
            "predicate": True,
        }
        assert resolved.status_by_tool["prototype"] == {
            "enabled": False,
            "reason": "deactivate",
            "directive": "deactivate",
        }
        assert sorted(resolved.enabled) == ["__proto__"]
        assert sorted(resolved.disabled) == ["constructor", "prototype"]


class TestPartitionRuntime:
    def test_includes_a_conditional_id_in_disabled_when_predicate_resolves_false(self) -> None:
        snapshot = create_tool_set([a, b]).deactivate("b").activate_when("a", lambda _i: False).resolve()
        assert snapshot.disabled == ["a", "b"]

    def test_returns_the_active_tools_for_static_partitions(self) -> None:
        resolved = create_tool_set([a, b, c]).deactivate("b").resolve()
        assert resolved.tools == [a, c]
        assert resolved.active_tools == ["a", "c"]

    def test_predicate_must_return_exactly_true(self) -> None:
        # Upstream compares `predicate(input) === true`; truthy non-bools do not count.
        resolved = (
            create_tool_set([a, b]).activate_when("a", lambda _i: 1).deactivate_when("b", lambda _i: "yes").resolve()
        )
        assert resolved.enabled == ["b"]
        assert resolved.disabled == ["a"]


class TestServerTools:
    web_search = server_tool({"type": "web_search_2025_08_26"})
    datetime = server_tool({"type": "openrouter:datetime"})
    # Upstream: serverTool({type}, {id: 'server:public_search'}).
    public_search = {**server_tool({"type": "web_search_2025_08_26"}), "id": "server:public_search"}

    def test_assigns_default_server_ids_from_config_type(self) -> None:
        assert get_tool_id(self.web_search) == "server:web_search_2025_08_26"
        assert get_tool_id(self.datetime) == "server:openrouter:datetime"
        assert get_tool_id(self.public_search) == "server:public_search"

    def test_server_tool_stamps_default_id(self) -> None:
        assert self.web_search.get("id") == "server:web_search_2025_08_26"

    def test_preserves_server_tools_in_tools_in_construction_order(self) -> None:
        ts = create_tool_set([a, self.web_search, b, self.datetime])
        assert ts.tools == (a, self.web_search, b, self.datetime)
        assert ts.resolve().enabled == ["a", "server:web_search_2025_08_26", "b", "server:openrouter:datetime"]

    def test_includes_active_server_tools_in_tools_enabled_status_but_not_active_tools(self) -> None:
        resolved = create_tool_set([a, self.web_search, b]).deactivate("a").resolve()
        assert resolved.tools == [self.web_search, b]
        assert resolved.active_tools == ["b"]
        assert resolved.enabled == ["server:web_search_2025_08_26", "b"]
        assert resolved.status_by_tool["server:web_search_2025_08_26"] == {"enabled": True, "reason": "default"}

    def test_can_deactivate_server_tools_by_stable_id(self) -> None:
        resolved = create_tool_set([a, self.web_search, b]).deactivate("server:web_search_2025_08_26").resolve()
        assert resolved.tools == [a, b]
        assert resolved.enabled == ["a", "b"]
        assert resolved.disabled == ["server:web_search_2025_08_26"]
        assert resolved.status_by_tool["server:web_search_2025_08_26"] == {
            "enabled": False,
            "reason": "deactivate",
            "directive": "deactivate",
        }

    def test_supports_override_ids_and_rejects_duplicates(self) -> None:
        ts = create_tool_set([a, self.public_search])
        assert ts.resolve().enabled == ["a", "server:public_search"]

        with pytest.raises(ValueError, match='Duplicate tool ID: "server:web_search_2025_08_26"'):
            create_tool_set([self.web_search, server_tool({"type": "web_search_2025_08_26"})])

    def test_rejects_activate_on_raw_type_strings(self) -> None:
        ts = create_tool_set([a, self.web_search])
        with pytest.raises(ValueError, match="Unknown tool"):
            ts.activate("web_search_2025_08_26")

    def test_hand_written_server_tool_without_id_uses_synthesized_id(self) -> None:
        hand_written = {"_brand": "server-tool", "config": {"type": "web_search_2025_08_26"}}
        tid = "server:web_search_2025_08_26"
        resolved = create_tool_set([a, hand_written]).deactivate(tid).resolve()
        assert resolved.enabled == ["a"]
        assert resolved.disabled == [tid]
        assert resolved.status_by_tool[tid] == {"enabled": False, "reason": "deactivate", "directive": "deactivate"}
        assert resolved.tools == [a]

    def test_empty_explicit_id_falls_back_to_synthesized_id(self) -> None:
        # Upstream defaultServerId: only a non-empty string id wins.
        empty_id = {**server_tool({"type": "openrouter:datetime"}), "id": ""}
        assert create_tool_set([empty_id]).resolve().enabled == ["server:openrouter:datetime"]

    def test_custom_id_server_tool_accepts_the_real_runtime_id(self) -> None:
        resolved = create_tool_set([a, self.public_search]).deactivate("server:public_search").resolve()
        assert resolved.enabled == ["a"]
        assert resolved.disabled == ["server:public_search"]

    def test_custom_id_server_tool_rejects_the_synthesized_default_id(self) -> None:
        ts = create_tool_set([a, self.public_search])
        with pytest.raises(ValueError, match="Unknown tool"):
            ts.deactivate("server:web_search_2025_08_26")


class TestSituations:
    def test_overlays_static_enabled_disabled(self) -> None:
        ts = (
            create_tool_set([a, b, c])
            .deactivate("c")
            .define_situations(
                {
                    "guest": {"enabled": ["a"], "disabled": ["b", "c"]},
                    "full": {"enabled": ["a", "b", "c"]},
                }
            )
        )

        guest = ts.resolve_situation("guest")
        assert guest.tools == [a]
        assert guest.active_tools == ["a"]
        assert guest.enabled == ["a"]
        assert guest.disabled == ["b", "c"]
        assert guest.status_by_tool == {
            "a": {"enabled": True, "reason": "situation", "directive": "activate"},
            "b": {"enabled": False, "reason": "situation", "directive": "deactivate"},
            "c": {"enabled": False, "reason": "situation", "directive": "deactivate"},
        }

        full = ts.resolve_situation("full")
        assert full.tools == [a, b, c]

    def test_supports_conditional_situation_rules_with_runtime_exact_status(self) -> None:
        ts = create_tool_set([a, b, c]).define_situations(
            {
                "authed": {
                    "enabled": ["a"],
                    "disabled": ["b"],
                    "conditional": {"c": lambda i: ctx(i).get("admin") is True},
                }
            }
        )

        denied = ts.resolve_situation("authed", {"context": {"admin": False}})
        assert denied.tools == [a]
        assert denied.enabled == ["a"]
        assert denied.disabled == ["b", "c"]
        assert denied.status_by_tool["c"] == {
            "enabled": False,
            "reason": "situation",
            "directive": "activateWhen",
            "predicate": True,
        }

        allowed = ts.resolve_situation("authed", {"context": {"admin": True}})
        assert allowed.tools == [a, c]
        assert allowed.enabled == ["a", "c"]

    def test_supports_deactivate_when_situation_rules(self) -> None:
        ts = create_tool_set([a]).define_situations(
            {
                "guarded": {
                    "conditional": {
                        "a": {"mode": "deactivateWhen", "predicate": lambda i: ctx(i).get("blocked") is True}
                    }
                }
            }
        )

        assert ts.resolve_situation("guarded").enabled == ["a"]
        blocked = ts.resolve_situation("guarded", {"context": {"blocked": True}})
        assert blocked.disabled == ["a"]
        assert blocked.status_by_tool["a"] == {
            "enabled": False,
            "reason": "situation",
            "directive": "deactivateWhen",
            "predicate": True,
        }

    def test_validates_unknown_duplicate_conflicting_ids(self) -> None:
        base = create_tool_set([a, b])

        with pytest.raises(ValueError, match='Unknown tool: "nope"'):
            base.define_situations({"bad": {"enabled": ["nope"]}})

        with pytest.raises(ValueError, match='lists tool "a" more than once'):
            base.define_situations({"bad": {"enabled": ["a"], "disabled": ["a"]}})

        with pytest.raises(ValueError, match='lists tool "a" more than once'):
            base.define_situations({"bad": {"enabled": ["a"], "conditional": {"a": lambda _i: True}}})

    def test_ignores_none_conditional_entries(self) -> None:
        ts = create_tool_set([a, b]).define_situations(
            {"optional": {"conditional": {"a": None, "b": lambda _i: False}}}
        )
        assert ts.resolve_situation("optional").active_tools == ["a"]

    def test_rejects_malformed_conditional_rules_at_definition_time(self) -> None:
        base = create_tool_set([a])
        expected = (
            'Situation "checkout": conditional rule for tool "a" must be a function or { mode, predicate } object'
        )
        with pytest.raises(ValueError) as excinfo:
            base.define_situations({"checkout": {"conditional": {"a": "invalid"}}})  # type: ignore[dict-item]
        assert str(excinfo.value) == expected

    def test_rejects_a_missing_conditional_predicate_at_definition_time(self) -> None:
        base = create_tool_set([a])
        expected = (
            'Situation "checkout": conditional rule for tool "a" must be a function or { mode, predicate } object'
        )
        with pytest.raises(ValueError) as excinfo:
            base.define_situations({"checkout": {"conditional": {"a": {"mode": "activateWhen"}}}})
        assert str(excinfo.value) == expected

    def test_rejects_an_unknown_conditional_mode(self) -> None:
        base = create_tool_set([a])
        with pytest.raises(ValueError, match='Situation "checkout": conditional rule for tool "a"'):
            base.define_situations(
                {"checkout": {"conditional": {"a": {"mode": "sometimes", "predicate": lambda _i: True}}}}  # type: ignore[typeddict-item, dict-item]
            )

    def test_throws_on_unknown_situation_names_at_resolve_time(self) -> None:
        ts = create_tool_set([a]).define_situations({"guest": {"enabled": ["a"]}})
        with pytest.raises(ValueError, match='Unknown situation: "missing"'):
            ts.resolve_situation("missing")

    def test_leaves_unmentioned_ids_on_the_base_partition(self) -> None:
        ts = create_tool_set([a, b, c]).deactivate("c").define_situations({"onlyB": {"disabled": ["b"]}})
        snapshot = ts.resolve_situation("onlyB")
        # a stays default-enabled, b disabled by situation, c disabled by base
        assert snapshot.enabled == ["a"]
        assert snapshot.disabled == ["b", "c"]

    def test_define_situations_replaces_the_registry_and_respects_mutability(self) -> None:
        # Upstream docstring: "Replaces any previously defined situations".
        immutable = create_tool_set([a]).define_situations({"one": {"disabled": ["a"]}})
        replaced = immutable.define_situations({"two": {"enabled": ["a"]}})
        assert replaced is not immutable
        assert immutable.resolve_situation("one").disabled == ["a"]
        with pytest.raises(ValueError, match='Unknown situation: "one"'):
            replaced.resolve_situation("one")

        mutable = create_tool_set([a], mutable=True)
        assert mutable.define_situations({"one": {"disabled": ["a"]}}) is mutable
        assert mutable.resolve_situation("one").disabled == ["a"]
        mutable.define_situations({"two": {"enabled": ["a"]}})
        with pytest.raises(ValueError, match='Unknown situation: "one"'):
            mutable.resolve_situation("one")


class TestSharedContext:
    def test_predicate_context_drives_activation(self) -> None:
        def is_authenticated(inp: ActivationInput) -> bool:
            context = inp.get("context")
            if not context:
                return False
            return bool(context["isAuthenticated"])

        ts = create_tool_set([a]).activate_when("a", is_authenticated)
        assert ts.resolve({"context": {"isAuthenticated": True, "userId": "u1"}}).active_tools == ["a"]
        assert ts.resolve({"context": {"isAuthenticated": False, "userId": "u1"}}).active_tools == []

    def test_untyped_context_default(self) -> None:
        ts = create_tool_set([a]).activate_when("a", lambda i: ctx(i).get("enabled") is True)
        assert ts.resolve({"context": {"enabled": True}}).active_tools == ["a"]


class TestCallModelSpreadShape:
    def test_produces_tools_and_active_tools_suitable_for_call_model_spread(self) -> None:
        snapshot = create_tool_set([a, b, c]).deactivate("b").resolve()
        for_call_model = snapshot.call_model
        assert for_call_model == {"tools": [a, c], "active_tools": ["a", "c"]}
        # Exactly these two keys: snapshot metadata never leaks into the request.
        assert list(for_call_model) == ["tools", "active_tools"]

    def test_tool_set_create_classmethod_matches_factory(self) -> None:
        ts = ToolSet.create(tools=[a, b], mutable=True)
        assert ts.mutable is True
        assert ts.deactivate("a") is ts
        assert ts.resolve().active_tools == ["b"]


class TestArgumentValidation:
    def test_activate_when_requires_a_predicate_with_a_name(self) -> None:
        with pytest.raises(ValueError, match="requires a predicate"):
            create_tool_set([a]).activate_when("a")

    def test_activate_when_rejects_non_mapping_non_string(self) -> None:
        with pytest.raises(ValueError, match="requires a name\\+predicate or predicate map"):
            create_tool_set([a]).deactivate_when(["a"])  # type: ignore[arg-type]

    def test_predicate_map_skips_non_callable_entries(self) -> None:
        resolved = create_tool_set([a, b]).activate_when({"a": None, "b": lambda _i: False}).resolve()
        assert resolved.active_tools == ["a"]
        assert resolved.status_by_tool["a"] == {"enabled": True, "reason": "default"}
