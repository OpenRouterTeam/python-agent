"""A round mixing auto and manual tool calls must not send an orphaned function_call.

Ports `packages/agent/tests/unit/mixed-manual-tool-round.test.ts`.

The guard: when one round returns both an auto-executable call and a manual
(no-`execute`) call, the loop must stop and surface the response so the caller can
resolve the manual call. Sending a follow-up would put `exec_command`'s
`function_call` in the input with no matching `function_call_output`, which
providers reject outright:

    400 "No tool output found for function call call_manual_1"

The failure is a hard provider error on real traffic, and nothing in the
deterministic suite would catch it — which is exactly why the request *count*, not
just the final text, is the assertion that matters here.

Relationship to `test_manual_tool_pending_state.py`: that file covers the stop
behavior plus state persistence for pending manual calls. It does **not** cover
the all-resolve follow-up pairing below, and it does not assert request counts for
the mixed round. Both files are kept so the upstream↔port test mapping stays 1:1
(`mixed-manual-tool-round.test.ts` → this file); do not delete either as redundant.
"""

from __future__ import annotations

from typing import Any, Dict, List

from openrouter_agent import call_model, tool
from tests._fixtures import QueuedClient, function_call_item, make_response, text_response

# `execute=False` marks a manual tool. `execute=None` raises instead
# (`tool.py:47-48`), so the literal False is required.
manual_tool = tool(name="exec_command", input_schema=dict, output_schema=dict, execute=False)
auto_tool = tool(
    name="auto_search",
    input_schema=dict,
    output_schema=dict,
    execute=lambda params, ctx: {"result": "found it"},
)


async def test_stops_instead_of_sending_a_follow_up_with_an_orphaned_function_call() -> None:
    mixed_round = make_response(
        "resp_mixed",
        [
            function_call_item("call_auto_1", "auto_search", '{"query":"docs"}'),
            function_call_item("call_manual_1", "exec_command", '{"command":"ls"}'),
        ],
    )
    client = QueuedClient([mixed_round])

    result = call_model(
        client,
        {"model": "test-model", "input": "do both things", "tools": [auto_tool, manual_tool]},
    )
    response = await result.get_response()

    # The response carrying the unresolved manual call is surfaced, so the caller
    # can execute it and continue.
    assert response["id"] == "resp_mixed"
    # And crucially: no follow-up was sent. Only the queued response was consumed.
    assert len(client.requests) == 1, (
        "a follow-up request was sent for a round with an unresolved manual call; "
        "its input would carry an orphaned function_call and the provider would 400"
    )


async def test_still_loops_when_every_tool_call_in_the_round_resolves() -> None:
    """The stop above must not over-trigger: an all-auto round still continues."""
    client = QueuedClient(
        [
            make_response("resp_auto", [function_call_item("call_auto_1", "auto_search", '{"query":"docs"}')]),
            text_response("resp_final", "All done."),
        ]
    )

    result = call_model(
        client,
        {"model": "test-model", "input": "search the docs", "tools": [auto_tool, manual_tool]},
    )
    text = await result.get_text()

    assert text == "All done."
    assert len(client.requests) == 2

    follow_up_input: List[Any] = client.requests[1]["input"]
    assert isinstance(follow_up_input, list)
    outputs: List[Dict[str, Any]] = [
        item for item in follow_up_input if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    assert len(outputs) == 1, f"expected exactly one function_call_output, got {len(outputs)}"

    # Upstream asserts `callId` here. This port emits snake_case `call_id`:
    # `ModelResult._send` normalizes camelCase to the generated SDK's snake_case at
    # the transport boundary (`model_result.py:148-156`), and this assertion reads
    # the outgoing request. This is a deliberate divergence — do not "fix" it back
    # to callId, it will fail.
    assert outputs[0]["call_id"] == "call_auto_1"
    assert "found it" in outputs[0]["output"]
