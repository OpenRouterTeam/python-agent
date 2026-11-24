"""Ported from upstream tests/unit/get-usage.test.ts (#97).

`ModelResult.get_usage()` aggregates token/cost usage across every model call a
run made, independent of the hook system, and never raises.

Not ported: the streaming-resume and `get_items_stream` interleaving cases,
which exercise upstream's lazy stream initialization; this port drives the
whole loop in one task, so every getter waits for the same completion.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

from openrouter_agent import HookEntry, HookName, HooksManager, call_model, tool
from tests._fixtures import MemoryStateAccessor, QueuedClient, text_response, tool_call_response


def _usage(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "input_tokens_details": {"cached_tokens": 25},
        "output_tokens_details": {"reasoning_tokens": 10},
        "cost": 0.002,
    }
    base.update(overrides)
    return base


_echo = tool(name="echo", input_schema=dict, execute=lambda _p, _c: {"ok": True})


async def test_sums_usage_across_every_round_of_a_multi_round_tool_loop() -> None:
    client = QueuedClient(
        [
            tool_call_response("r1", "echo", usage=_usage()),
            text_response(
                "r2",
                "done",
                usage=_usage(
                    input_tokens=200,
                    output_tokens=100,
                    total_tokens=300,
                    cost=0.003,
                    input_tokens_details={"cached_tokens": 25},
                ),
            ),
        ]
    )
    result = call_model(client, {"model": "test-model", "input": "hi", "tools": [_echo]})

    final = await result.get_response()
    usage = await result.get_usage()

    assert usage == {
        "model_calls": 2,
        "input_tokens": 300,
        "output_tokens": 150,
        "total_tokens": 450,
        "cached_tokens": 50,
        "reasoning_tokens": 20,
        "cost": 0.005,
    }
    # The gap this closes: the final response alone reports only round 2.
    assert final["usage"]["total_tokens"] == 300


async def test_awaits_run_completion_when_called_without_awaiting_get_response_first() -> None:
    client = QueuedClient(
        [tool_call_response("r1", "echo", usage=_usage()), text_response("r2", "done", usage=_usage())]
    )
    result = call_model(client, {"model": "test-model", "input": "hi", "tools": [_echo]})

    usage = await result.get_usage()

    assert usage["model_calls"] == 2
    assert usage["total_tokens"] == 300


async def test_accumulates_with_no_hooks_configured() -> None:
    result = call_model(
        QueuedClient([text_response("r1", "hi", usage=_usage())]), {"model": "test-model", "input": "hi"}
    )
    await result.get_text()

    assert await result.get_usage() == {
        "model_calls": 1,
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "cached_tokens": 25,
        "reasoning_tokens": 10,
        "cost": 0.002,
    }


async def test_agrees_with_the_session_end_total_usage_hook_payload() -> None:
    session_end: List[Any] = []
    hooks = HooksManager()
    hooks.on(HookName.SessionEnd.value, HookEntry(handler=lambda payload, _ctx: session_end.append(payload)))
    client = QueuedClient(
        [tool_call_response("r1", "echo", usage=_usage()), text_response("r2", "done", usage=_usage())]
    )
    result = call_model(client, {"model": "test-model", "input": "hi", "tools": [_echo], "hooks": hooks})

    usage = await result.get_usage()

    assert len(session_end) == 1
    assert session_end[0]["total_usage"] == usage


async def test_counts_usage_less_responses_in_model_calls_and_omits_cost() -> None:
    result = call_model(QueuedClient([text_response("r1", "hi")]), {"model": "test-model", "input": "hi"})
    await result.get_text()

    assert await result.get_usage() == {
        "model_calls": 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }


async def test_returns_zeroed_totals_when_no_model_call_completed_warning_with_the_cause() -> None:
    # Error injection on the shared fake client (no bespoke client class).
    client = QueuedClient([])

    async def fail(**_kwargs: Any) -> Any:
        raise RuntimeError("api down")

    client.responses.send_async = fail  # type: ignore[method-assign]
    result = call_model(client, {"model": "test-model", "input": "hi"})

    try:
        await result.get_text()
    except RuntimeError as error:
        assert str(error) == "api down"
    else:  # pragma: no cover - the run must fail
        raise AssertionError("expected the run to fail")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        usage = await result.get_usage()

    assert usage["model_calls"] == 0
    assert usage["total_tokens"] == 0
    assert "cost" not in usage
    # Never-raises must not mean silent.
    assert [str(w.message) for w in caught] == ["[get_usage] run failed; reporting totals accrued so far: api down"]


async def test_reports_only_the_model_calls_that_completed_before_an_approval_pause() -> None:
    risky = tool(name="risky", input_schema=dict, execute=lambda _p, _c: {"ok": True}, require_approval=True)
    state = MemoryStateAccessor()
    client = QueuedClient([tool_call_response("r1", "risky", usage=_usage())])
    result = call_model(client, {"model": "test-model", "input": "do it", "tools": [risky], "state": state})

    await result.get_pending_tool_calls()
    first = await result.get_usage()
    second = await result.get_usage()

    assert state.stored is not None and state.stored.status == "awaiting_approval"
    assert len(client.requests) == 1
    assert (
        first
        == second
        == {
            "model_calls": 1,
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cached_tokens": 25,
            "reasoning_tokens": 10,
            "cost": 0.002,
        }
    )


async def test_scopes_totals_per_model_result_across_an_approval_resume() -> None:
    risky = tool(name="risky", input_schema=dict, execute=lambda _p, _c: {"ok": True}, require_approval=True)
    state = MemoryStateAccessor()
    first = call_model(
        QueuedClient([tool_call_response("r1", "risky", call_id="call_r", usage=_usage())]),
        {"model": "test-model", "input": "do it", "tools": [risky], "state": state},
    )
    await first.get_pending_tool_calls()

    resumed = call_model(
        QueuedClient([text_response("r2", "done", usage=_usage(total_tokens=7, cost=0.001))]),
        {"model": "test-model", "input": [], "tools": [risky], "state": state, "approve_tool_calls": ["call_r"]},
    )
    assert await resumed.get_text() == "done"

    assert (await first.get_usage())["model_calls"] == 1
    usage = await resumed.get_usage()
    assert usage["model_calls"] == 1
    assert usage["total_tokens"] == 7
    assert usage["cost"] == 0.001
