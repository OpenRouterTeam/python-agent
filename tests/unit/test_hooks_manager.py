from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from openrouter_agent import HookEntry, HookName, HooksManager
from openrouter_agent.hooks_types import AsyncOutput


async def test_emit_validates_payload_and_invokes_matching_handlers() -> None:
    manager = HooksManager()
    seen = []

    def handler(payload, ctx):
        seen.append((payload, ctx))
        return {"block": False}

    manager.on(HookName.PreToolUse.value, HookEntry(handler=handler))

    result = await manager.emit(HookName.PreToolUse.value, {"tool_name": "search", "tool_input": {"q": "x"}})

    assert result.results == [{"mutated_input": None, "block": False}]
    assert seen[0][0] == {"tool_name": "search", "tool_input": {"q": "x"}}
    assert seen[0][1].session_id == ""
    assert seen[0][1].hook_name == HookName.PreToolUse.value


async def test_emit_invalid_payload_warns_and_skips_handlers_by_default() -> None:
    manager = HooksManager()
    calls = []
    manager.on(HookName.PreToolUse.value, HookEntry(handler=lambda payload, ctx: calls.append(payload)))

    with pytest.warns(UserWarning):
        result = await manager.emit(HookName.PreToolUse.value, {"tool_name": "search"})  # missing tool_input

    assert calls == []
    assert result.results == []


async def test_emit_invalid_payload_raises_when_throw_on_handler_error() -> None:
    manager = HooksManager(throw_on_handler_error=True)
    manager.on(HookName.PreToolUse.value, HookEntry(handler=lambda payload, ctx: None))

    with pytest.raises(RuntimeError):
        await manager.emit(HookName.PreToolUse.value, {"tool_name": "search"})


async def test_pre_tool_use_block_short_circuits_and_mutation_pipes_into_payload() -> None:
    manager = HooksManager()

    def mutate(payload, ctx):
        return {"mutated_input": {"q": "mutated"}}

    def block(payload, ctx):
        assert payload["tool_input"] == {"q": "mutated"}
        return {"block": "not allowed"}

    never_called = []
    manager.on(HookName.PreToolUse.value, HookEntry(handler=mutate))
    manager.on(HookName.PreToolUse.value, HookEntry(handler=block))
    manager.on(HookName.PreToolUse.value, HookEntry(handler=lambda p, c: never_called.append(p)))

    result = await manager.emit(HookName.PreToolUse.value, {"tool_name": "search", "tool_input": {"q": "orig"}})

    assert result.blocked is True
    assert result.mutated is True
    assert result.final_payload["tool_input"] == {"q": "mutated"}
    assert never_called == []


async def test_tool_matcher_filters_by_tool_name() -> None:
    manager = HooksManager()
    calls = []
    manager.on(
        HookName.PreToolUse.value,
        HookEntry(handler=lambda p, c: calls.append(p["tool_name"]), matcher="search"),
    )

    await manager.emit(HookName.PreToolUse.value, {"tool_name": "other", "tool_input": {}}, tool_name="other")
    await manager.emit(HookName.PreToolUse.value, {"tool_name": "search", "tool_input": {}}, tool_name="search")

    assert calls == ["search"]


async def test_regex_matcher_and_predicate_matcher() -> None:
    import re

    manager = HooksManager()
    seen = []
    manager.on(
        HookName.PreToolUse.value,
        HookEntry(handler=lambda p, c: seen.append("regex"), matcher=re.compile(r"^search_.*")),
    )
    manager.on(
        HookName.PreToolUse.value,
        HookEntry(handler=lambda p, c: seen.append("predicate"), matcher=lambda name: name == "search_docs"),
    )

    await manager.emit(HookName.PreToolUse.value, {"tool_name": "x", "tool_input": {}}, tool_name="search_docs")

    assert seen == ["regex", "predicate"]


async def test_off_removes_handler_and_remove_all_clears_hook() -> None:
    manager = HooksManager()
    calls = []

    def handler(payload, ctx):
        calls.append(payload)

    manager.on(HookName.PreToolUse.value, HookEntry(handler=handler))
    assert manager.off(HookName.PreToolUse.value, handler) is True
    await manager.emit(HookName.PreToolUse.value, {"tool_name": "x", "tool_input": {}})
    assert calls == []

    manager.on(HookName.PreToolUse.value, HookEntry(handler=handler))
    manager.remove_all(HookName.PreToolUse.value)
    assert manager.has_handlers(HookName.PreToolUse.value) is False


async def test_async_output_is_tracked_and_drained() -> None:
    manager = HooksManager()
    done = []

    async def background() -> None:
        await asyncio.sleep(0.01)
        done.append("bg")

    def handler(payload, ctx):
        return AsyncOutput(work=background())

    manager.on(HookName.PostToolUse.value, HookEntry(handler=handler))
    result = await manager.emit(
        HookName.PostToolUse.value,
        {"tool_name": "x", "tool_input": {}, "tool_output": 1, "duration_ms": 1.0},
    )

    assert len(result.pending) == 1
    assert done == []
    await manager.drain()
    assert done == ["bg"]


async def test_session_id_threads_per_emit_for_shared_manager() -> None:
    manager = HooksManager()
    seen = []
    manager.on(HookName.SessionStart.value, HookEntry(handler=lambda p, c: seen.append(c.session_id)))

    manager.set_session_id("default")
    await manager.emit(HookName.SessionStart.value, {}, session_id="run-a")
    await manager.emit(HookName.SessionStart.value, {}, session_id="run-b")
    await manager.emit(HookName.SessionStart.value, {})

    assert seen == ["run-a", "run-b", "default"]


async def test_custom_hook_registration_collides_with_built_in_name() -> None:
    from openrouter_agent.hooks_schemas import HookDefinition, PreToolUsePayload

    with pytest.raises(ValueError):
        HooksManager({HookName.PreToolUse.value: HookDefinition(payload=PreToolUsePayload, result=None)})


async def test_void_result_hook_accepts_arbitrary_handler_return_values() -> None:
    manager = HooksManager()
    manager.on(HookName.PostToolUse.value, HookEntry(handler=lambda p, c: "anything goes"))

    result = await manager.emit(
        HookName.PostToolUse.value,
        {"tool_name": "x", "tool_input": {}, "tool_output": 1, "duration_ms": 1.0},
    )

    assert result.results == ["anything goes"]


async def test_custom_void_result_hook_skips_validation_like_built_ins() -> None:
    from openrouter_agent.hooks_schemas import HookDefinition
    from pydantic import BaseModel

    class AuditPayload(BaseModel):
        pass

    manager = HooksManager(
        {"Audit": HookDefinition(payload=AuditPayload, result=None)},
        throw_on_handler_error=True,
    )
    manager.on("Audit", HookEntry(handler=lambda p, c: {"logged": True}))

    result = await manager.emit("Audit", {})

    assert result.results == [{"logged": True}]


def test_stop_result_pydantic_schema_validates_fields() -> None:
    from openrouter_agent.hooks_schemas import StopResult

    with pytest.raises(ValidationError):
        StopResult.model_validate({"force_resume": "not-a-bool"})
