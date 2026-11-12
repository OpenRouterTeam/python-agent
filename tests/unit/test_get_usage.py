"""Port of upstream `tests/unit/get-usage.test.ts` -- the pull-based usage
accessor `await result.get_usage()`.

Python totals are snake_case: `model_calls`, `input_tokens`, `output_tokens`,
`total_tokens`, `cached_tokens`, `reasoning_tokens`, and `cost` (present only
when a response reported it). Usage blocks are snake_case too
(`_extract_model_call_usage` in model_result.py reads `input_tokens`,
`input_tokens_details.cached_tokens`, `output_tokens_details.reasoning_tokens`,
`cost`).
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, cast

import pytest
from pydantic import BaseModel

from openrouter_agent import HookEntry, HookName, HooksManager, call_model, tool
from tests._fixtures import (
    MemoryStateAccessor,
    QueuedClient,
    QueuedResponses,
    text_response,
    tool_call_response,
    usage_block,
)


def usage(**overrides: Any) -> Dict[str, Any]:
    """Upstream's usageBlock: 100/50/150 tokens, 25 cached, 10 reasoning, cost 0.002."""
    base: Dict[str, Any] = {
        "input_tokens_details": {"cached_tokens": 25},
        "output_tokens_details": {"reasoning_tokens": 10},
    }
    base.update(overrides)
    return usage_block(**base)


ROUND_TWO_USAGE = {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300, "cost": 0.003}


class EchoInput(BaseModel):
    model_config = {"extra": "allow"}


echo_tool = tool(name="echo", description="echo", input_schema=EchoInput, execute=lambda params, ctx: {"ok": True})


class RiskyInput(BaseModel):
    target: str


class RiskyOutput(BaseModel):
    done: bool


approval_tool = tool(
    name="risky",
    description="Does something that needs a human to sign off.",
    input_schema=RiskyInput,
    output_schema=RiskyOutput,
    require_approval=True,
    execute=lambda params, ctx: {"done": True},
)


def echo_call(response_id: str, **usage_overrides: Any) -> Dict[str, Any]:
    return tool_call_response(response_id, "echo", usage=usage(**usage_overrides))


def risky_call(response_id: str) -> Dict[str, Any]:
    return tool_call_response(response_id, "risky", arguments=json.dumps({"target": "prod"}), usage=usage())


def text(response_id: str, **usage_overrides: Any) -> Dict[str, Any]:
    return text_response(response_id, "hello back", usage=usage(**usage_overrides))


class FailingResponses(QueuedResponses):
    """Transport failure on every send (upstream `{ ok: false, error }`)."""

    async def send_async(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        raise RuntimeError("api down")


class FailingClient(QueuedClient):
    def __init__(self) -> None:
        super().__init__([])
        self.responses = FailingResponses([])
        self.beta = type("Beta", (), {"responses": self.responses})()


async def _pause_on_risky(client: QueuedClient, accessor: MemoryStateAccessor) -> Any:
    first = call_model(
        client, {"model": "test-model", "input": "do the risky thing", "tools": [approval_tool], "state": accessor}
    )
    await first.get_pending_tool_calls()
    assert accessor.stored is not None
    assert accessor.stored.status == "awaiting_approval"
    return first


def _paused_call_id(accessor: MemoryStateAccessor) -> str:
    assert accessor.stored is not None
    pending = accessor.stored.pending_tool_calls or []
    assert len(pending) == 1
    return pending[0].id


async def test_sums_usage_across_every_round_of_a_multi_round_tool_loop() -> None:
    client = QueuedClient([echo_call("r1"), text("r2", **ROUND_TWO_USAGE)])

    result = call_model(client, {"model": "test-model", "input": "hi", "tools": [echo_tool]})
    await result.get_response()
    totals = await result.get_usage()

    assert totals == {
        "model_calls": 2,
        "input_tokens": 300,
        "output_tokens": 150,
        "total_tokens": 450,
        "cached_tokens": 50,
        "reasoning_tokens": 20,
        "cost": pytest.approx(0.005),
    }
    assert len(client.requests) == 2
    # The gap this closes: the final response alone reports only round 2.
    final_response = await result.get_response()
    assert final_response["usage"]["total_tokens"] == 300


async def test_awaits_run_completion_when_called_without_awaiting_get_response_first() -> None:
    client = QueuedClient([echo_call("r1"), text("r2")])

    result = call_model(client, {"model": "test-model", "input": "hi", "tools": [echo_tool]})

    # No prior await -- get_usage() must gate on completion itself.
    totals = await result.get_usage()
    assert totals["model_calls"] == 2
    assert totals["total_tokens"] == 300
    assert len(client.requests) == 2


async def test_accumulates_with_no_hooks_configured() -> None:
    client = QueuedClient([text("r1")])

    result = call_model(client, {"model": "test-model", "input": "hi"})
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


async def test_reports_complete_totals_after_consuming_get_items_stream_on_a_tool_loop() -> None:
    client = QueuedClient([echo_call("r1"), text("r2", **ROUND_TWO_USAGE)])

    result = call_model(client, {"model": "test-model", "input": "hi", "tools": [echo_tool]})
    async for _item in result.get_items_stream():
        pass

    assert await result.get_usage() == {
        "model_calls": 2,
        "input_tokens": 300,
        "output_tokens": 150,
        "total_tokens": 450,
        "cached_tokens": 50,
        "reasoning_tokens": 20,
        "cost": pytest.approx(0.005),
    }


async def test_reports_complete_totals_after_consuming_the_no_tools_get_items_stream() -> None:
    client = QueuedClient([text("r1")])

    result = call_model(client, {"model": "test-model", "input": "hi"})
    async for _item in result.get_items_stream():
        pass

    totals = await result.get_usage()
    assert totals["model_calls"] == 1
    assert totals["total_tokens"] == 150


async def test_agrees_with_the_session_end_total_usage_hook_payload() -> None:
    client = QueuedClient([echo_call("r1"), text("r2")])
    hooks = HooksManager()
    ends: List[Dict[str, Any]] = []

    def on_end(payload: Dict[str, Any], ctx: Any) -> None:
        ends.append(payload)

    hooks.on(HookName.SessionEnd.value, HookEntry(handler=on_end))

    result = call_model(client, {"model": "test-model", "input": "hi", "tools": [echo_tool], "hooks": hooks})
    await result.get_text()

    assert len(ends) == 1
    assert ends[0]["total_usage"]["model_calls"] == 2
    assert await result.get_usage() == ends[0]["total_usage"]


async def test_counts_usage_less_responses_in_model_calls_and_omits_cost() -> None:
    client = QueuedClient([text_response("r1", "hello back")])

    result = call_model(client, {"model": "test-model", "input": "hi"})
    await result.get_text()

    totals = await result.get_usage()
    assert totals == {
        "model_calls": 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
    assert "cost" not in totals


async def test_returns_zeroed_totals_when_no_model_call_completed_warning_with_the_swallowed_cause() -> None:
    client = FailingClient()

    result = call_model(client, {"model": "test-model", "input": "hi"})
    with pytest.raises(RuntimeError, match="api down"):
        await result.get_text()

    # Never-raises must not mean silent. Upstream logs
    # console.warn('[getUsage] run failed; reporting totals accrued so far:', err);
    # the port emits a UserWarning with the snake_case accessor name.
    with pytest.warns(UserWarning, match=r"^\[get_usage\] run failed; reporting totals accrued so far: api down$"):
        totals = await result.get_usage()

    assert totals["model_calls"] == 0
    assert totals["total_tokens"] == 0
    assert "cost" not in totals
    assert len(client.requests) == 1


# -- paused at awaiting_approval ---------------------------------------------


async def test_paused_reports_only_the_model_calls_that_completed_before_the_pause() -> None:
    accessor = MemoryStateAccessor()
    client = QueuedClient([risky_call("r1")])

    result = await _pause_on_risky(client, accessor)
    assert len(client.requests) == 1

    assert await result.get_usage() == {
        "model_calls": 1,
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "cached_tokens": 25,
        "reasoning_tokens": 10,
        "cost": 0.002,
    }


async def test_paused_is_idempotent_and_drives_no_further_model_requests() -> None:
    accessor = MemoryStateAccessor()
    # The second response is queued precisely so an accidental dispatch would
    # show up as both a request-count bump and inflated totals.
    client = QueuedClient([risky_call("r1"), text("should-not-be-consumed")])

    result = await _pause_on_risky(client, accessor)

    first = await result.get_usage()
    assert first["model_calls"] == 1
    assert await result.get_usage() == first
    assert len(client.requests) == 1
    assert accessor.stored is not None
    assert accessor.stored.status == "awaiting_approval"


async def test_scopes_totals_per_model_result_across_an_approval_resume() -> None:
    accessor = MemoryStateAccessor()
    client = QueuedClient([risky_call("r1"), text("r2", **ROUND_TWO_USAGE)])

    first = await _pause_on_risky(client, accessor)
    paused_call_id = _paused_call_id(accessor)

    resumed = call_model(
        client,
        {"model": "test-model", "tools": [approval_tool], "state": accessor, "approve_tool_calls": [paused_call_id]},
    )
    await resumed.get_pending_tool_calls()

    assert len(client.requests) == 2
    # Per-ModelResult, not per-conversation: the resume reports only its own
    # generation (r2); r1 stays on `first`.
    resumed_totals = await resumed.get_usage()
    assert resumed_totals["model_calls"] == 1
    assert resumed_totals["input_tokens"] == 200
    assert resumed_totals["output_tokens"] == 100
    assert resumed_totals["total_tokens"] == 300
    assert resumed_totals["cost"] == 0.003

    first_totals = await first.get_usage()
    assert first_totals["model_calls"] == 1
    assert first_totals["input_tokens"] == 100
    assert first_totals["total_tokens"] == 150
    assert first_totals["cost"] == 0.002


@pytest.mark.xfail(
    strict=True,
    reason=(
        "PORT GAP: upstream getUsage() on an approval-resumed run is observation-only "
        "(src/lib/model-result.ts:7704-7712, `isResumingFromApproval` guard skips executeToolsIfNeeded), "
        "so after a bare getUsage() the resume generation is counted but state stays "
        "'in_progress'. Python get_usage() (model_result.py:1487-1499, `await "
        "self._ensure_task()`) drives the whole `_run` task, which settles the conversation "
        "to 'complete' as a side effect of reading usage."
    ),
)
async def test_does_not_advance_an_approval_resumed_run_when_read_before_the_loop() -> None:
    accessor = MemoryStateAccessor()
    client = QueuedClient([risky_call("r1"), text("r2")])

    await _pause_on_risky(client, accessor)
    paused_call_id = _paused_call_id(accessor)

    resumed = call_model(
        client,
        {"model": "test-model", "tools": [approval_tool], "state": accessor, "approve_tool_calls": [paused_call_id]},
    )

    # get_usage() is the FIRST call on the resumed run.
    assert (await resumed.get_usage())["model_calls"] == 1
    assert accessor.stored is not None
    assert accessor.stored.status == "in_progress"


async def test_counts_a_streaming_resume_generation() -> None:
    """Upstream 'counts a STREAMING resume generation without advancing the
    loop'. The usage half is ported here; the "without advancing" half (status
    stays 'in_progress') is the PORT GAP pinned by the xfail above."""
    accessor = MemoryStateAccessor()
    resume_response = text("r2_streamed", **ROUND_TWO_USAGE)

    async def resume_stream() -> AsyncIterator[Dict[str, Any]]:
        yield {"type": "response.completed", "response": resume_response, "sequence_number": 0}

    # Bespoke transport: the resume dispatch returns a real event stream.
    # QueuedResponses returns whatever is queued; the payload is still built
    # by the shared builders.
    client = QueuedClient([risky_call("r1"), cast(Dict[str, Any], resume_stream())])

    await _pause_on_risky(client, accessor)
    paused_call_id = _paused_call_id(accessor)

    resumed = call_model(
        client,
        {"model": "test-model", "tools": [approval_tool], "state": accessor, "approve_tool_calls": [paused_call_id]},
    )

    totals = await resumed.get_usage()
    assert totals["model_calls"] == 1
    assert totals["input_tokens"] == 200
    assert totals["output_tokens"] == 100
    assert totals["total_tokens"] == 300
    assert totals["cost"] == 0.003
    assert len(client.requests) == 2
