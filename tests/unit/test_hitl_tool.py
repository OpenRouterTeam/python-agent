"""Ports tests/unit/hitl-tool.test.ts.

Upstream drives the integration half by constructing `ModelResult` directly
against a mocked `betaResponsesSend`; this port drives the same invariants
through `call_model` with the shared `QueuedClient` (contract, Test Parity:
prefer the public API).

Layering note: upstream's `executeHITLTool(tool, toolCall, turnContext)` builds
the execute context itself. This port's `execute_hitl_tool(tool, call, context)`
takes an already-built execute context and the context is built (and
`tool_call` synthesized) by the `execute_tool` dispatcher
(tool_executor.py `execute_tool`). The upstream `executeHITLTool` cases are
therefore ported through `execute_tool`, which reaches `execute_hitl_tool`
for every HITL tool; the direct-call divergence is pinned by an xfail below.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, TypeAdapter

from openrouter_agent import (
    ToolType,
    call_model,
    is_auto_resolvable_tool,
    is_hitl_tool,
    is_manual_tool,
    tool,
)
from openrouter_agent.tool_executor import apply_on_response_received_hooks, execute_hitl_tool, execute_tool
from openrouter_agent.tool_types import ConversationState, ParsedToolCall
from tests._fixtures import (
    MemoryStateAccessor,
    QueuedClient,
    function_call_item,
    make_response,
    message_item,
)

TURN_CONTEXT: Dict[str, Any] = {"number_of_turns": 1}


class AmountInput(BaseModel):
    amount: float


class XInput(BaseModel):
    x: float


class IdInput(BaseModel):
    id: str


class Ok(BaseModel):
    ok: bool


class Approved(BaseModel):
    approved: bool


class OkReviewed(BaseModel):
    ok: bool
    reviewedAt: float


class Recorder:
    """Records every invocation and returns a fixed value (or raises)."""

    def __init__(self, returns: Any = None, raises: Optional[BaseException] = None) -> None:
        self.calls: List[Any] = []
        self.contexts: List[Any] = []
        self._returns = returns
        self._raises = raises

    async def __call__(self, value: Any, context: Any = None) -> Any:
        self.calls.append(value)
        self.contexts.append(context)
        if self._raises is not None:
            raise self._raises
        return self._returns


def make_tool_call(name: str, call_id: str, args: Any) -> ParsedToolCall:
    return ParsedToolCall(id=call_id, name=name, arguments=args)


def output_item(call_id: str, output: Any) -> Dict[str, Any]:
    return {"type": "function_call_output", "id": f"output_{call_id}", "callId": call_id, "output": output}


def sent_function_call_outputs(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [item for item in request["input"] if isinstance(item, dict) and item.get("type") == "function_call_output"]


# -- tool() factory: HITL tools ---------------------------------------------


def test_creates_a_hitl_tool_when_on_tool_called_is_present() -> None:
    async def on_tool_called(params: Any, ctx: Any) -> Any:
        return {"ok": True} if params.amount < 100 else None

    async def on_response_received(raw: Any, ctx: Any) -> Any:
        return raw

    approve = tool(
        name="approve_payment",
        description="Approve a payment",
        input_schema=AmountInput,
        output_schema=Ok,
        on_tool_called=on_tool_called,
        on_response_received=on_response_received,
    )

    assert approve["type"] == ToolType.Function.value
    assert approve["function"]["name"] == "approve_payment"
    assert is_hitl_tool(approve) is True
    assert is_manual_tool(approve) is False
    assert is_auto_resolvable_tool(approve) is True
    assert "execute" not in approve["function"]


def test_omits_on_response_received_when_not_provided() -> None:
    t = tool(name="approve", input_schema=XInput, output_schema=Ok, on_tool_called=Recorder({"ok": True}))

    assert "on_response_received" not in t["function"]


def test_raises_when_a_hitl_tool_is_created_without_an_output_schema() -> None:
    with pytest.raises(ValueError, match="output_schema"):
        tool(name="approve", input_schema=XInput, on_tool_called=Recorder({"ok": True}))


def test_is_manual_tool_true_only_for_tools_with_neither_execute_nor_on_tool_called() -> None:
    manual = tool(name="manual", input_schema=XInput, execute=False)
    hitl = tool(name="hitl", input_schema=XInput, output_schema=Ok, on_tool_called=Recorder(None))
    regular = tool(name="regular", input_schema=XInput, execute=Recorder({"y": 1}))

    assert [is_manual_tool(t) for t in (manual, hitl, regular)] == [True, False, False]


# -- executeHITLTool (via the execute_tool dispatcher; see module docstring) --


async def test_returns_a_tool_execution_result_when_on_tool_called_returns_a_value() -> None:
    async def on_tool_called(params: Any, ctx: Any) -> Any:
        return {"ok": params.amount < 100}

    t = tool(name="approve", input_schema=AmountInput, output_schema=Ok, on_tool_called=on_tool_called)

    result = await execute_tool(t, make_tool_call("approve", "c1", {"amount": 5}), TURN_CONTEXT)

    assert result is not None
    assert result.get("error") is None
    assert result["tool_call_id"] == "c1"
    assert result["tool_name"] == "approve"
    assert result["result"].model_dump() == {"ok": True}


async def test_threads_the_executed_tool_call_into_the_hook_context() -> None:
    # Streaming-loop turn context: number_of_turns only, no tool_call.
    hook = Recorder(None)
    t = tool(name="approve", input_schema=AmountInput, output_schema=Ok, on_tool_called=hook)

    result = await execute_tool(t, make_tool_call("approve", "call_stream_1", {"amount": 5}), {"number_of_turns": 1})

    assert result is None
    assert len(hook.contexts) == 1
    assert hook.contexts[0]["tool_call"] == {
        "type": "function_call",
        "id": "call_stream_1",
        "callId": "call_stream_1",
        "name": "approve",
        "arguments": json.dumps({"amount": 5}, separators=(",", ":")),
    }
    # The synthesized item is wire-shaped: arguments is a JSON string that
    # round-trips to the executed call's arguments.
    assert json.loads(hook.contexts[0]["tool_call"]["arguments"]) == {"amount": 5}


async def test_prefers_a_caller_provided_turn_context_tool_call_over_the_synthesized_one() -> None:
    orchestrator_tool_call = {
        "type": "function_call",
        "id": "call_rich_1",
        "callId": "call_rich_1",
        "name": "approve",
        "arguments": '{"amount":5}',
        "status": "completed",
    }
    hook = Recorder(None)
    t = tool(name="approve", input_schema=AmountInput, output_schema=Ok, on_tool_called=hook)

    await execute_tool(
        t,
        make_tool_call("approve", "call_rich_1", {"amount": 5}),
        {"number_of_turns": 1, "tool_call": orchestrator_tool_call},
    )

    assert len(hook.contexts) == 1
    assert hook.contexts[0]["tool_call"] is orchestrator_tool_call


async def test_does_not_mutate_the_caller_turn_context_when_synthesizing_tool_call() -> None:
    hook = Recorder(None)
    t = tool(name="approve", input_schema=AmountInput, output_schema=Ok, on_tool_called=hook)
    turn_context: Dict[str, Any] = {"number_of_turns": 1}

    await execute_tool(t, make_tool_call("approve", "c1", {"amount": 5}), turn_context)

    assert turn_context == {"number_of_turns": 1}


async def test_direct_execute_hitl_tool_threads_the_executed_tool_call() -> None:
    hook = Recorder(None)
    t = tool(name="approve", input_schema=AmountInput, output_schema=Ok, on_tool_called=hook)

    await execute_hitl_tool(t, make_tool_call("approve", "call_direct", {"amount": 5}), {"number_of_turns": 1})

    assert len(hook.contexts) == 1
    assert hook.contexts[0]["tool_call"]["callId"] == "call_direct"


async def test_returns_none_when_on_tool_called_returns_none_pause() -> None:
    t = tool(name="approve", input_schema=AmountInput, output_schema=Ok, on_tool_called=Recorder(None))

    assert await execute_tool(t, make_tool_call("approve", "c2", {"amount": 5}), TURN_CONTEXT) is None


async def test_captures_thrown_errors_into_the_tool_execution_result() -> None:
    t = tool(
        name="approve",
        input_schema=AmountInput,
        output_schema=Ok,
        on_tool_called=Recorder(raises=RuntimeError("boom")),
    )

    result = await execute_tool(t, make_tool_call("approve", "c3", {"amount": 5}), TURN_CONTEXT)

    assert result is not None
    assert isinstance(result["error"], Exception)
    assert str(result["error"]) == "boom"
    assert result["result"] is None


async def test_validates_on_tool_called_return_value_against_output_schema() -> None:
    t = tool(name="approve", input_schema=AmountInput, output_schema=Ok, on_tool_called=Recorder({"ok": "not-a-bool"}))

    result = await execute_tool(t, make_tool_call("approve", "c4", {"amount": 5}), TURN_CONTEXT)

    assert result is not None
    assert isinstance(result["error"], Exception)
    assert result["result"] is None


# -- executeTool dispatcher with HITL tools ---------------------------------


async def test_routes_hitl_tools_through_execute_hitl_tool() -> None:
    t = tool(
        name="approve", input_schema=AmountInput, output_schema=Approved, on_tool_called=Recorder({"approved": True})
    )

    result = await execute_tool(t, make_tool_call("approve", "c1", {"amount": 5}), TURN_CONTEXT)

    assert result is not None
    assert result.get("error") is None
    assert result["result"].model_dump() == {"approved": True}


async def test_returns_none_for_hitl_pause_through_the_dispatcher() -> None:
    hook = Recorder(None)
    t = tool(name="approve", input_schema=AmountInput, output_schema=Ok, on_tool_called=hook)

    result = await execute_tool(t, make_tool_call("approve", "c1", {"amount": 5}), TURN_CONTEXT)

    assert result is None
    assert len(hook.calls) == 1


async def test_threads_the_executed_tool_call_into_execute_context_for_regular_tools_too() -> None:
    execute = Recorder({"y": 1})
    t = tool(name="regular", input_schema=XInput, execute=execute)

    await execute_tool(t, make_tool_call("regular", "call_reg_1", {"x": 2}), {"number_of_turns": 1})

    assert len(execute.contexts) == 1
    assert execute.contexts[0]["tool_call"] == {
        "type": "function_call",
        "id": "call_reg_1",
        "callId": "call_reg_1",
        "name": "regular",
        "arguments": '{"x":2}',
    }


# -- applyOnResponseReceivedHooks -------------------------------------------


def _call_item(call_id: str, name: str, args: str = "{}") -> Dict[str, Any]:
    return function_call_item(call_id, name, args)


async def test_transforms_a_function_call_output_item_when_its_tool_has_on_response_received() -> None:
    async def on_response_received(raw: Any, ctx: Any) -> Any:
        if not isinstance(raw, dict):
            return {"ok": False, "reviewedAt": 1234}
        return {**raw, "reviewedAt": 1234}

    t = tool(
        name="approve",
        input_schema=AmountInput,
        output_schema=OkReviewed,
        on_tool_called=Recorder(None),
        on_response_received=on_response_received,
    )
    input_items = [_call_item("c1", "approve"), output_item("c1", json.dumps({"ok": True}))]

    result = await apply_on_response_received_hooks(input_items, [t], TURN_CONTEXT)

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] is input_items[0]
    out = result[1]
    assert out["type"] == "function_call_output"
    assert out["callId"] == "c1"
    assert json.loads(out["output"]) == {"ok": True, "reviewedAt": 1234}
    # The caller's item is not mutated in place.
    assert json.loads(input_items[1]["output"]) == {"ok": True}


async def test_does_not_synthesize_a_tool_call_for_on_response_received() -> None:
    hook = Recorder({"ok": True})
    t = tool(
        name="approve",
        input_schema=AmountInput,
        output_schema=Ok,
        on_tool_called=Recorder(None),
        on_response_received=hook,
    )
    input_items = [_call_item("c1", "approve"), output_item("c1", json.dumps({"ok": True}))]

    # A streaming-shaped turn context: no caller-provided tool_call.
    await apply_on_response_received_hooks(input_items, [t], {"number_of_turns": 1})

    assert len(hook.calls) == 1
    assert "tool_call" not in hook.contexts[0]


async def test_on_response_received_still_sees_a_caller_provided_tool_call() -> None:
    # Upstream tool-executor.ts:937-942: no synthetic toolCall, but a
    # caller-provided turnContext.toolCall still flows through.
    caller_tool_call = {"type": "function_call", "id": "c1", "callId": "c1", "name": "approve", "arguments": "{}"}
    hook = Recorder({"ok": True})
    t = tool(
        name="approve",
        input_schema=AmountInput,
        output_schema=Ok,
        on_tool_called=Recorder(None),
        on_response_received=hook,
    )
    input_items = [_call_item("c1", "approve"), output_item("c1", json.dumps({"ok": True}))]

    await apply_on_response_received_hooks(input_items, [t], {"number_of_turns": 1, "tool_call": caller_tool_call})

    assert len(hook.contexts) == 1
    assert hook.contexts[0]["tool_call"] is caller_tool_call


async def test_leaves_output_unchanged_when_no_matching_tool_has_a_hook() -> None:
    t = tool(name="regular", input_schema=XInput, execute=Recorder({"y": 1}))
    input_items = [_call_item("c1", "regular"), output_item("c1", json.dumps({"y": 1}))]

    result = await apply_on_response_received_hooks(input_items, [t], TURN_CONTEXT)

    assert result is input_items


@pytest.mark.xfail(
    strict=True,
    reason=(
        "PORT GAP: upstream applyOnResponseReceivedHooks only considers HITL tools "
        "(buildHitlToolMap, tool-executor.ts:1018-1021); non-HITL outputs are never validated or "
        "rewritten. Python apply_on_response_received_hooks (tool_executor.py) matches ANY client "
        "tool by name and, with no hook, validates the caller output against that tool's "
        "output_schema, rewriting a regular tool's non-conforming output into an error wrapper."
    ),
)
async def test_leaves_non_hitl_tool_outputs_untouched_even_when_they_miss_the_output_schema() -> None:
    t = tool(name="regular", input_schema=XInput, output_schema=Ok, execute=Recorder({"ok": True}))
    input_items = [_call_item("c1", "regular"), output_item("c1", json.dumps({"ok": "not-a-bool"}))]

    result = await apply_on_response_received_hooks(input_items, [t], TURN_CONTEXT)

    assert result is input_items


async def test_replaces_output_with_an_error_object_preserving_the_original_when_the_hook_throws() -> None:
    t = tool(
        name="approve",
        input_schema=AmountInput,
        output_schema=Ok,
        on_tool_called=Recorder(None),
        on_response_received=Recorder(raises=RuntimeError("invalid result")),
    )
    original_payload = {"ok": True, "note": "carry-me-through"}
    input_items = [_call_item("c1", "approve"), output_item("c1", json.dumps(original_payload))]

    result = await apply_on_response_received_hooks(input_items, [t], TURN_CONTEXT)

    parsed = json.loads(result[1]["output"])
    assert parsed == {"error": "invalid result", "originalOutput": original_payload}


async def test_preserves_the_raw_string_as_original_output_when_not_json_and_the_hook_throws() -> None:
    t = tool(
        name="approve",
        input_schema=AmountInput,
        output_schema=Ok,
        on_tool_called=Recorder(None),
        on_response_received=Recorder(raises=RuntimeError("invalid result")),
    )
    input_items = [_call_item("c1", "approve"), output_item("c1", "not-json")]

    result = await apply_on_response_received_hooks(input_items, [t], TURN_CONTEXT)

    parsed = json.loads(result[1]["output"])
    assert parsed == {"error": "invalid result", "originalOutput": "not-json"}


async def test_passes_the_parsed_raw_result_not_the_raw_string_to_the_hook() -> None:
    spy = Recorder({"ok": True, "note": "hi"})
    t = tool(
        name="approve",
        input_schema=AmountInput,
        output_schema=TypeAdapter(Dict[str, Any]),
        on_tool_called=Recorder(None),
        on_response_received=spy,
    )
    payload = {"ok": True, "note": "hi"}
    input_items = [_call_item("c1", "approve"), output_item("c1", json.dumps(payload))]

    await apply_on_response_received_hooks(input_items, [t], TURN_CONTEXT)

    assert len(spy.calls) == 1
    assert spy.calls[0] == payload


async def test_passes_the_raw_string_through_to_the_hook_when_output_is_not_json() -> None:
    spy = Recorder("not-json")
    t = tool(
        name="approve",
        input_schema=AmountInput,
        output_schema=TypeAdapter(str),
        on_tool_called=Recorder(None),
        on_response_received=spy,
    )
    input_items = [_call_item("c1", "approve"), output_item("c1", "not-json")]

    await apply_on_response_received_hooks(input_items, [t], TURN_CONTEXT)

    assert spy.calls == ["not-json"]


async def test_leaves_outputs_whose_call_id_has_no_matching_function_call_untouched() -> None:
    spy = Recorder({"tagged": True})
    t = tool(
        name="approve",
        input_schema=AmountInput,
        output_schema=TypeAdapter(Dict[str, Any]),
        on_tool_called=Recorder(None),
        on_response_received=spy,
    )
    # No function_call in the input at all -- just an orphan output.
    input_items = [output_item("orphan", json.dumps({"ok": True}))]

    result = await apply_on_response_received_hooks(input_items, [t], TURN_CONTEXT)

    assert result is input_items
    assert spy.calls == []


async def test_validates_raw_output_against_output_schema_without_a_hook_passes() -> None:
    t = tool(name="approve", input_schema=AmountInput, output_schema=Ok, on_tool_called=Recorder(None))
    input_items = [_call_item("c1", "approve"), output_item("c1", json.dumps({"ok": True}))]

    result = await apply_on_response_received_hooks(input_items, [t], TURN_CONTEXT)

    assert result is input_items


async def test_replaces_output_with_an_error_object_when_raw_misses_output_schema_and_no_hook() -> None:
    t = tool(name="approve", input_schema=AmountInput, output_schema=Ok, on_tool_called=Recorder(None))
    bad_payload = {"ok": "nope"}
    input_items = [_call_item("c1", "approve"), output_item("c1", json.dumps(bad_payload))]

    result = await apply_on_response_received_hooks(input_items, [t], TURN_CONTEXT)

    assert result is not input_items
    parsed = json.loads(result[1]["output"])
    assert isinstance(parsed["error"], str)
    assert len(parsed["error"]) > 0
    assert parsed["originalOutput"] == bad_payload


async def test_replaces_output_with_an_error_object_when_the_hook_return_misses_output_schema() -> None:
    t = tool(
        name="approve",
        input_schema=AmountInput,
        output_schema=Ok,
        on_tool_called=Recorder(None),
        on_response_received=Recorder({"ok": "not-a-bool"}),
    )
    original_payload = {"ok": True}
    input_items = [_call_item("c1", "approve"), output_item("c1", json.dumps(original_payload))]

    result = await apply_on_response_received_hooks(input_items, [t], TURN_CONTEXT)

    parsed = json.loads(result[1]["output"])
    assert isinstance(parsed["error"], str)
    assert len(parsed["error"]) > 0
    assert parsed["originalOutput"] == original_payload


async def test_preserves_a_content_array_hook_return_verbatim() -> None:
    content_array = [
        {"type": "input_text", "text": "hello"},
        {"type": "input_image", "detail": "auto", "imageUrl": "https://example.com/img.png"},
    ]
    t = tool(
        name="describe",
        input_schema=IdInput,
        output_schema=TypeAdapter(List[Dict[str, Any]]),
        on_tool_called=Recorder(None),
        on_response_received=Recorder(content_array),
    )
    input_items = [_call_item("c1", "describe"), output_item("c1", json.dumps({"ok": True}))]

    result = await apply_on_response_received_hooks(input_items, [t], TURN_CONTEXT)

    # The output is the content array itself, not a JSON string.
    assert isinstance(result[1]["output"], list)
    assert result[1]["output"] == content_array


async def test_preserves_a_caller_supplied_content_array_output_across_no_hook_validation() -> None:
    content_array = [{"type": "input_text", "text": "plain text"}]
    t = tool(
        name="describe",
        input_schema=IdInput,
        output_schema=TypeAdapter(List[Dict[str, Any]]),
        on_tool_called=Recorder(None),
    )
    input_items = [_call_item("c1", "describe"), output_item("c1", content_array)]

    result = await apply_on_response_received_hooks(input_items, [t], TURN_CONTEXT)

    assert result is input_items


@pytest.mark.xfail(
    strict=True,
    reason=(
        "PORT GAP: when a hook fails on a caller-supplied content-array output, upstream "
        "formatHookError (tool-executor.ts:849-868) emits a content array: an input_text block "
        "carrying JSON {error} followed by the original blocks. Python "
        "apply_on_response_received_hooks (tool_executor.py, the `except` branch) always emits "
        'a JSON string {"error", "originalOutput"}, embedding the content array as JSON.'
    ),
)
async def test_preserves_a_caller_supplied_content_array_in_original_output_when_the_hook_throws() -> None:
    content_array = [{"type": "input_text", "text": "caller payload"}]
    t = tool(
        name="describe",
        input_schema=IdInput,
        output_schema=Ok,
        on_tool_called=Recorder(None),
        on_response_received=Recorder(raises=RuntimeError("bad hook")),
    )
    input_items = [_call_item("c1", "describe"), output_item("c1", content_array)]

    result = await apply_on_response_received_hooks(input_items, [t], TURN_CONTEXT)

    out = result[1]["output"]
    assert isinstance(out, list)
    assert out[0]["type"] == "input_text"
    assert json.loads(out[0]["text"]) == {"error": "bad hook"}
    assert out[1:] == content_array


async def test_ignores_manual_tools_they_are_not_hitl() -> None:
    t = tool(name="manual", input_schema=XInput, execute=False)
    input_items = [_call_item("c1", "manual"), output_item("c1", json.dumps({"y": 1}))]

    result = await apply_on_response_received_hooks(input_items, [t], TURN_CONTEXT)

    assert result is input_items


# -- HITL flow through call_model (integration) -----------------------------


async def test_auto_resolve_on_tool_called_value_feeds_the_model_on_the_next_turn() -> None:
    client = QueuedClient(
        [
            make_response("resp_turn_0", [function_call_item("c1", "approve", json.dumps({"amount": 5}))]),
            make_response("resp_turn_1", [message_item("approved", item_id="msg_1")]),
        ]
    )
    hook = Recorder({"ok": True})
    approve = tool(name="approve", input_schema=AmountInput, output_schema=Ok, on_tool_called=hook)

    final = await call_model(client, {"model": "test-model", "input": "approve 5", "tools": [approve]}).get_response()

    # Both turns fired: no pause because on_tool_called returned a value.
    assert len(client.requests) == 2
    assert len(hook.calls) == 1
    assert final["id"] == "resp_turn_1"
    outputs = sent_function_call_outputs(client.requests[1])
    assert len(outputs) == 1
    # Upstream asserts `callId`. This port emits snake_case `call_id`: _send
    # normalizes at the transport boundary (model_result.py `_send`).
    # Deliberate -- do not change back to callId.
    assert outputs[0]["call_id"] == "c1"
    assert isinstance(outputs[0]["output"], str)
    assert json.loads(outputs[0]["output"]) == {"ok": True}


async def test_pause_on_tool_called_returning_none_breaks_the_loop_cleanly() -> None:
    client = QueuedClient(
        [make_response("resp_turn_0", [function_call_item("c1", "approve", json.dumps({"amount": 5}))])]
    )
    approve = tool(name="approve", input_schema=AmountInput, output_schema=Ok, on_tool_called=Recorder(None))

    result = call_model(client, {"model": "test-model", "input": "approve 5", "tools": [approve]})
    final = await result.get_response()

    # No follow-up request: the caller resumes with its own function_call_output.
    assert len(client.requests) == 1
    tool_calls = await result.get_tool_calls()
    assert [(c.id, c.name) for c in tool_calls] == [("c1", "approve")]
    pending_calls = [item for item in final["output"] if item.get("type") == "function_call"]
    assert len(pending_calls) == 1
    assert pending_calls[0]["callId"] == "c1"


async def test_resume_caller_output_is_transformed_by_on_response_received_before_the_model_sees_it() -> None:
    client = QueuedClient([make_response("resp_resumed", [message_item("approved after review", item_id="msg_1")])])
    on_tool_called = Recorder(None)

    async def on_response_received(raw: Any, ctx: Any) -> Any:
        if not isinstance(raw, dict):
            return {"ok": False, "reviewedAt": 1234}
        return {"ok": raw.get("ok") is True, "reviewedAt": 1234}

    approve = tool(
        name="approve",
        input_schema=AmountInput,
        output_schema=OkReviewed,
        on_tool_called=on_tool_called,
        on_response_received=on_response_received,
    )
    resume_input = [
        function_call_item("c1", "approve", json.dumps({"amount": 5})),
        output_item("c1", json.dumps({"ok": True})),
    ]

    text = await call_model(client, {"model": "test-model", "input": resume_input, "tools": [approve]}).get_text()

    # Single turn: the caller pre-baked the function_call_output.
    assert len(client.requests) == 1
    # on_tool_called does not fire on resume.
    assert on_tool_called.calls == []
    outputs = sent_function_call_outputs(client.requests[0])
    assert len(outputs) == 1
    assert isinstance(outputs[0]["output"], str)
    assert json.loads(outputs[0]["output"]) == {"ok": True, "reviewedAt": 1234}
    assert text == "approved after review"


# -- state machine: HITL pause + resume -------------------------------------


async def test_hitl_pause_sets_status_awaiting_hitl_and_populates_pending_tool_calls() -> None:
    client = QueuedClient(
        [make_response("resp_turn_0", [function_call_item("c1", "approve", json.dumps({"amount": 5}))])]
    )
    accessor = MemoryStateAccessor()
    approve = tool(name="approve", input_schema=AmountInput, output_schema=Ok, on_tool_called=Recorder(None))

    result = call_model(client, {"model": "test-model", "input": "approve 5", "tools": [approve], "state": accessor})
    await result.get_response()

    assert len(client.requests) == 1
    saved = accessor.stored
    assert saved is not None
    assert saved.status == "awaiting_hitl"
    assert [c.id for c in saved.pending_tool_calls or []] == ["c1"]
    pending = await result.get_pending_tool_calls()
    assert [c.id for c in pending] == ["c1"]
    assert await result.requires_approval() is True


def _awaiting_approval_state() -> ConversationState:
    return ConversationState(
        id="conv_test",
        messages=[function_call_item("c1", "approve", json.dumps({"amount": 5}))],
        status="awaiting_approval",
        created_at=0,
        updated_at=0,
        pending_tool_calls=[ParsedToolCall(id="c1", name="approve", arguments={"amount": 5})],
    )


async def test_resume_approved_hitl_returning_none_stays_pending_with_status_awaiting_hitl() -> None:
    client = QueuedClient([])
    accessor = MemoryStateAccessor()
    accessor.stored = _awaiting_approval_state()
    on_tool_called = Recorder(None)
    approve = tool(
        name="approve",
        input_schema=AmountInput,
        output_schema=Ok,
        require_approval=True,
        on_tool_called=on_tool_called,
    )

    result = call_model(
        client,
        {
            "model": "test-model",
            "input": "approve 5",
            "tools": [approve],
            "state": accessor,
            "approve_tool_calls": ["c1"],
        },
    )
    paused_calls = await result.get_pending_tool_calls()

    assert len(on_tool_called.calls) == 1
    # Paused before any follow-up turn.
    assert len(client.requests) == 0
    assert [c.id for c in paused_calls] == ["c1"]
    saved = accessor.stored
    assert saved.status == "awaiting_hitl"
    assert [c.id for c in saved.pending_tool_calls or []] == ["c1"]


async def test_hitl_only_tool_calls_invoke_on_tool_called_instead_of_being_classified_manual() -> None:
    client = QueuedClient(
        [
            make_response("resp_turn_0", [function_call_item("c1", "approve", json.dumps({"amount": 5}))]),
            make_response("resp_turn_1", [message_item("approved", item_id="msg_1")]),
        ]
    )
    on_tool_called = Recorder({"ok": True})
    approve = tool(name="approve", input_schema=AmountInput, output_schema=Ok, on_tool_called=on_tool_called)

    final = await call_model(client, {"model": "test-model", "input": "approve 5", "tools": [approve]}).get_response()

    assert len(client.requests) == 2
    assert len(on_tool_called.calls) == 1
    assert final["id"] == "resp_turn_1"


async def test_on_response_received_does_not_fire_for_sdk_generated_outputs_on_approval_resume() -> None:
    client = QueuedClient([make_response("resp_followup", [message_item("done", item_id="msg_1")])])
    accessor = MemoryStateAccessor()
    accessor.stored = _awaiting_approval_state()
    on_tool_called = Recorder({"ok": True})
    on_response_received = Recorder({"ok": True, "hooked": True})
    approve = tool(
        name="approve",
        input_schema=AmountInput,
        output_schema=Ok,
        require_approval=True,
        on_tool_called=on_tool_called,
        on_response_received=on_response_received,
    )

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "approve 5",
            "tools": [approve],
            "state": accessor,
            "approve_tool_calls": ["c1"],
        },
    ).get_response()

    assert len(on_tool_called.calls) == 1
    # SDK-generated outputs are appended verbatim; the hook is reserved for
    # caller-supplied outputs.
    assert on_response_received.calls == []
    assert len(client.requests) == 1
    outputs = sent_function_call_outputs(client.requests[0])
    assert len(outputs) == 1
    assert isinstance(outputs[0]["output"], str)
    assert json.loads(outputs[0]["output"]) == {"ok": True}


def _historical_state() -> ConversationState:
    return ConversationState(
        id="conv_test",
        messages=[
            function_call_item("c0", "approve", json.dumps({"amount": 1})),
            output_item("c0", json.dumps({"ok": True, "hooked": True})),
        ],
        status="in_progress",
        created_at=0,
        updated_at=0,
    )


def _fresh_input() -> List[Dict[str, Any]]:
    return [
        function_call_item("c1", "approve", json.dumps({"amount": 2})),
        output_item("c1", json.dumps({"ok": True})),
    ]


async def _hook_tagging(raw: Any, ctx: Any) -> Any:
    return {**raw, "hooked": True} if isinstance(raw, dict) else {"hooked": True}


async def test_on_response_received_fires_only_for_fresh_outputs_not_historical_ones_from_state() -> None:
    client = QueuedClient([make_response("resp_reply", [message_item("noted", item_id="msg_1")])])
    accessor = MemoryStateAccessor()
    accessor.stored = _historical_state()
    hook_calls: List[Any] = []

    async def on_response_received(raw: Any, ctx: Any) -> Any:
        hook_calls.append(raw)
        return await _hook_tagging(raw, ctx)

    approve = tool(
        name="approve",
        input_schema=AmountInput,
        output_schema=Ok,
        on_tool_called=Recorder(None),
        on_response_received=on_response_received,
    )

    await call_model(
        client, {"model": "test-model", "input": _fresh_input(), "tools": [approve], "state": accessor}
    ).get_response()

    # Fired exactly once -- for the fresh c1 output, with the fresh payload.
    assert hook_calls == [{"ok": True}]
    assert len(client.requests) == 1
    outputs = sent_function_call_outputs(client.requests[0])
    assert [o["call_id"] for o in outputs] == ["c0", "c1"]
    # c0 preserved verbatim from history -- NOT re-hooked.
    assert json.loads(outputs[0]["output"]) == {"ok": True, "hooked": True}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "PORT GAP: upstream invokeOnResponseReceived validates the hook return against outputSchema "
        "but serializes the hook's RAW return (tool-executor.ts:944-949 "
        "`toFunctionCallOutputValue(hookResult)`), so keys outside the schema survive. Python "
        "apply_on_response_received_hooks (tool_executor.py) serializes the VALIDATED value, so a "
        "pydantic output_schema strips the extra `hooked` key: c1 is sent as {ok: true}."
    ),
)
async def test_fresh_output_carries_the_raw_hook_return_not_the_schema_stripped_value() -> None:
    client = QueuedClient([make_response("resp_reply", [message_item("noted", item_id="msg_1")])])
    accessor = MemoryStateAccessor()
    accessor.stored = _historical_state()
    approve = tool(
        name="approve",
        input_schema=AmountInput,
        output_schema=Ok,
        on_tool_called=Recorder(None),
        on_response_received=_hook_tagging,
    )

    await call_model(
        client, {"model": "test-model", "input": _fresh_input(), "tools": [approve], "state": accessor}
    ).get_response()

    outputs = sent_function_call_outputs(client.requests[0])
    c1 = next(o for o in outputs if o["call_id"] == "c1")
    assert json.loads(c1["output"]) == {"ok": True, "hooked": True}
