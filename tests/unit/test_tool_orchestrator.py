"""Ports tests/unit/tool-orchestrator.test.ts.

Upstream `lib/tool-orchestrator.ts` exports a standalone non-streaming loop
(`executeToolLoop`) plus result helpers (`toolResultsToMap`,
`summarizeToolExecutions`, `hasToolExecutionErrors`, `getToolExecutionErrors`).
This port's `tool_orchestrator.py` is a re-export shim only (see its module
docstring): none of those functions exist. Every upstream case is therefore
ported as a strict xfail against the snake_case API the naming map implies
(`execute_tool_loop`, `tool_results_to_map`, `summarize_tool_executions`,
`has_tool_execution_errors`, `get_tool_execution_errors`), so porting the module
flips them to XPASS.

The loop invariants that also hold for the inlined `ModelResult._run` loop are
additionally asserted through `call_model` (contract, Test Parity: prefer the
public API) and pass today.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import pytest
from pydantic import BaseModel

from openrouter_agent import call_model, mark_mcp, tool, tool_orchestrator
from openrouter_agent.conversation_state import partition_tool_calls
from openrouter_agent.model_result import ModelResult
from openrouter_agent.tool_executor import execute_tool
from tests._fixtures import QueuedClient, function_call_item, make_response, text_response

# Typed as Any so the missing attributes are a runtime (xfail) failure, not a
# mypy error.
orch: Any = tool_orchestrator

MISSING_LOOP = (
    "PORT GAP: upstream tool-orchestrator.ts:44 exports executeToolLoop(sendRequest, initialInput, "
    "initialRequest, tools, apiTools, options) returning {finalResponse, allResponses, "
    "toolExecutionResults, conversationInput}. Python tool_orchestrator.py only re-exports "
    "ModelResult/execute_tool/partition_tool_calls; execute_tool_loop does not exist."
)
MISSING_HELPERS = (
    "PORT GAP: upstream tool-orchestrator.ts:185-235 exports toolResultsToMap, "
    "summarizeToolExecutions, hasToolExecutionErrors, getToolExecutionErrors. Python "
    "tool_orchestrator.py has none of them."
)


class AddIn(BaseModel):
    a: float
    b: float


class Empty(BaseModel):
    pass


class ValueIn(BaseModel):
    value: str


class Question(BaseModel):
    question: str


class Pct(BaseModel):
    pct: float


class Done(BaseModel):
    done: bool


class ApprovedOut(BaseModel):
    approved: bool


def _dump(value: Any) -> Any:
    return value.model_dump() if hasattr(value, "model_dump") else value


def _call(call_id: str, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return function_call_item(call_id, name, json.dumps(args))


def _response_with(*items: Dict[str, Any], response_id: str = "resp") -> Dict[str, Any]:
    return make_response(response_id, list(items))


def _terminal(response_id: str = "resp_end") -> Dict[str, Any]:
    return text_response(response_id, "")


INITIAL_INPUT: List[Dict[str, Any]] = [{"role": "user", "content": "hi"}]
INITIAL_REQUEST: Dict[str, Any] = {"model": "openai/gpt-4", "input": INITIAL_INPUT}


class QueueSendRequest:
    """`send_request(input, tools)` stub returning queued responses in order."""

    def __init__(self, *responses: Dict[str, Any]) -> None:
        self._responses = list(responses)
        self.calls: List[Tuple[Any, Any]] = []

    async def __call__(self, input_items: Any, tools: Any) -> Dict[str, Any]:
        self.calls.append((input_items, tools))
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]


async def _add(params: Any, ctx: Any) -> Dict[str, Any]:
    return {"sum": params.a + params.b}


async def _explode(params: Any, ctx: Any) -> Any:
    raise RuntimeError("kaboom")


async def _progress(params: Any, ctx: Any) -> AsyncIterator[Dict[str, Any]]:
    yield {"pct": 50}
    yield {"done": True}


# -- module surface that exists ---------------------------------------------


def test_module_re_exports_the_loop_building_blocks() -> None:
    assert tool_orchestrator.ModelResult is ModelResult
    assert tool_orchestrator.execute_tool is execute_tool
    assert tool_orchestrator.partition_tool_calls is partition_tool_calls
    assert sorted(tool_orchestrator.__all__) == ["ModelResult", "execute_tool", "partition_tool_calls"]


# -- executeToolLoop (PORT GAP) ---------------------------------------------


@pytest.mark.xfail(strict=True, reason=MISSING_LOOP)
async def test_execute_tool_loop_returns_immediately_without_tool_calls() -> None:
    response = _terminal()
    send = QueueSendRequest(response)

    result = await orch.execute_tool_loop(send, INITIAL_INPUT, INITIAL_REQUEST, [], [])

    assert result["final_response"] is response
    assert result["all_responses"] == [response]
    assert result["tool_execution_results"] == []
    assert result["conversation_input"] is INITIAL_INPUT
    assert len(send.calls) == 1


@pytest.mark.xfail(strict=True, reason=MISSING_LOOP)
async def test_execute_tool_loop_executes_a_tool_call_and_loops_until_done() -> None:
    add_tool = tool(name="add", input_schema=AddIn, execute=_add)
    first = _response_with(_call("call_1", "add", {"a": 2, "b": 3}))
    second = _terminal()
    send = QueueSendRequest(first, second)

    result = await orch.execute_tool_loop(send, INITIAL_INPUT, INITIAL_REQUEST, [add_tool], [])

    assert len(send.calls) == 2
    assert result["all_responses"] == [first, second]
    assert result["final_response"] is second
    assert len(result["tool_execution_results"]) == 1
    only = result["tool_execution_results"][0]
    assert (only["tool_call_id"], only["tool_name"], only["source"]) == ("call_1", "add", "client")
    assert _dump(only["result"]) == {"sum": 5}


@pytest.mark.xfail(strict=True, reason=MISSING_LOOP)
async def test_execute_tool_loop_supports_multiple_rounds() -> None:
    ping = tool(name="ping", input_schema=Empty, execute=lambda params, ctx: "pong")
    send = QueueSendRequest(
        _response_with(_call("call_1", "ping", {})), _response_with(_call("call_2", "ping", {})), _terminal()
    )

    result = await orch.execute_tool_loop(send, INITIAL_INPUT, INITIAL_REQUEST, [ping], [])

    assert len(send.calls) == 3
    assert len(result["all_responses"]) == 3
    assert [r["tool_call_id"] for r in result["tool_execution_results"]] == ["call_1", "call_2"]


@pytest.mark.xfail(strict=True, reason=MISSING_LOOP)
async def test_execute_tool_loop_executes_multiple_calls_in_one_round() -> None:
    echo = tool(name="echo", input_schema=ValueIn, execute=lambda params, ctx: params.value)
    send = QueueSendRequest(
        _response_with(_call("call_a", "echo", {"value": "one"}), _call("call_b", "echo", {"value": "two"})),
        _terminal(),
    )

    result = await orch.execute_tool_loop(send, INITIAL_INPUT, INITIAL_REQUEST, [echo], [])

    assert [r["result"] for r in result["tool_execution_results"]] == ["one", "two"]


@pytest.mark.xfail(strict=True, reason=MISSING_LOOP)
async def test_execute_tool_loop_returns_an_error_result_for_unknown_tools() -> None:
    known = tool(name="known", input_schema=Empty, execute=lambda params, ctx: "ok")
    send = QueueSendRequest(
        _response_with(_call("call_1", "known", {}), _call("call_2", "ghost", {})),
        _terminal(),
    )

    result = await orch.execute_tool_loop(send, INITIAL_INPUT, INITIAL_REQUEST, [known], [])

    by_id = {r["tool_call_id"]: r for r in result["tool_execution_results"]}
    assert str(by_id["call_2"]["error"]) == 'Tool "ghost" not found in tool definitions'
    assert by_id["call_2"]["source"] == "client"
    assert by_id["call_1"].get("error") is None


@pytest.mark.xfail(strict=True, reason=MISSING_LOOP)
async def test_execute_tool_loop_stops_without_executing_manual_tools() -> None:
    manual = tool(name="needs_human", input_schema=Question, execute=False)
    first = _response_with(_call("call_1", "needs_human", {"question": "?"}))
    send = QueueSendRequest(first)

    result = await orch.execute_tool_loop(send, INITIAL_INPUT, INITIAL_REQUEST, [manual], [])

    assert len(send.calls) == 1
    assert result["tool_execution_results"] == []
    assert result["final_response"] is first


@pytest.mark.xfail(strict=True, reason=MISSING_LOOP)
async def test_execute_tool_loop_captures_execute_failures_and_keeps_looping() -> None:
    fail = tool(name="explodes", input_schema=Empty, execute=_explode)
    send = QueueSendRequest(_response_with(_call("call_1", "explodes", {})), _terminal())

    result = await orch.execute_tool_loop(send, INITIAL_INPUT, INITIAL_REQUEST, [fail], [])

    assert len(result["tool_execution_results"]) == 1
    assert str(result["tool_execution_results"][0]["error"]) == "kaboom"
    assert len(send.calls) == 2


@pytest.mark.xfail(strict=True, reason=MISSING_LOOP)
async def test_execute_tool_loop_marks_mcp_failures_with_source_mcp() -> None:
    mcp = mark_mcp(tool(name="mcp_tool", input_schema=Empty, execute=_explode))
    send = QueueSendRequest(_response_with(_call("call_1", "mcp_tool", {})), _terminal())

    result = await orch.execute_tool_loop(send, INITIAL_INPUT, INITIAL_REQUEST, [mcp], [])

    only = result["tool_execution_results"][0]
    assert only["source"] == "mcp"
    assert only["result"] is None
    assert str(only["error"]) == "kaboom"


@pytest.mark.xfail(strict=True, reason=MISSING_LOOP)
async def test_execute_tool_loop_filters_hitl_null_results_but_continues() -> None:
    hitl = tool(name="approve", input_schema=Empty, output_schema=ApprovedOut, on_tool_called=lambda p, c: None)
    send = QueueSendRequest(_response_with(_call("call_1", "approve", {})), _terminal())

    result = await orch.execute_tool_loop(send, INITIAL_INPUT, INITIAL_REQUEST, [hitl], [])

    assert result["tool_execution_results"] == []
    assert len(send.calls) == 2


@pytest.mark.skip(reason="Out of scope: background/deferred lifecycle (async) tools are not ported.")
def test_execute_tool_loop_rejects_async_lifecycle_tools_without_invoking_run() -> None:
    raise AssertionError("not ported")


@pytest.mark.xfail(strict=True, reason=MISSING_LOOP)
async def test_execute_tool_loop_forwards_generator_preliminary_results() -> None:
    progress = tool(name="progress", input_schema=Empty, event_schema=Pct, output_schema=Done, execute=_progress)
    send = QueueSendRequest(_response_with(_call("call_1", "progress", {})), _terminal())
    seen: List[Tuple[str, Any]] = []

    result = await orch.execute_tool_loop(
        send,
        INITIAL_INPUT,
        INITIAL_REQUEST,
        [progress],
        [],
        {"on_preliminary_result": lambda call_id, value: seen.append((call_id, _dump(value)))},
    )

    assert seen == [("call_1", {"pct": 50})]
    assert _dump(result["tool_execution_results"][0]["result"]) == {"done": True}
    assert "preliminary_results" not in result["tool_execution_results"][0]


@pytest.mark.xfail(strict=True, reason=MISSING_LOOP)
async def test_execute_tool_loop_applies_next_turn_params_input_changes() -> None:
    new_input = [{"role": "user", "content": "rewritten"}]
    steer = tool(
        name="steer",
        input_schema=Empty,
        execute=lambda params, ctx: "ok",
        next_turn_params={"input": lambda params, ctx: new_input},
    )
    send = QueueSendRequest(_response_with(_call("call_1", "steer", {})), _terminal())

    result = await orch.execute_tool_loop(send, INITIAL_INPUT, INITIAL_REQUEST, [steer], [])

    assert result["conversation_input"] == new_input
    assert send.calls[-1] == (new_input, [])


@pytest.mark.xfail(strict=True, reason=MISSING_LOOP)
async def test_execute_tool_loop_keeps_original_input_when_next_turn_params_change_nothing() -> None:
    plain = tool(name="plain", input_schema=Empty, execute=lambda params, ctx: "ok")
    send = QueueSendRequest(_response_with(_call("call_1", "plain", {})), _terminal())

    result = await orch.execute_tool_loop(send, INITIAL_INPUT, INITIAL_REQUEST, [plain], [])

    assert result["conversation_input"] is INITIAL_INPUT


@pytest.mark.xfail(strict=True, reason=MISSING_LOOP)
async def test_execute_tool_loop_keeps_input_aligned_with_request_when_params_were_computed() -> None:
    tune = tool(
        name="tune",
        input_schema=Empty,
        execute=lambda params, ctx: "ok",
        next_turn_params={"temperature": lambda params, ctx: 0.2},
    )
    send = QueueSendRequest(_response_with(_call("call_1", "tune", {})), _terminal())

    result = await orch.execute_tool_loop(send, INITIAL_INPUT, INITIAL_REQUEST, [tune], [])

    assert result["conversation_input"] == INITIAL_REQUEST["input"]


# -- result helpers (PORT GAP) ------------------------------------------------


def _result(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {"tool_call_id": "call_1", "tool_name": "some_tool", "source": "client", "result": "ok"}
    base.update(overrides)
    return base


@pytest.mark.xfail(strict=True, reason=MISSING_HELPERS)
def test_tool_results_to_map_maps_results_by_tool_call_id() -> None:
    mapping = orch.tool_results_to_map([_result(tool_call_id="a", result=1), _result(tool_call_id="b", result=2)])

    assert mapping == {"a": {"result": 1}, "b": {"result": 2}}
    assert len(orch.tool_results_to_map([])) == 0


@pytest.mark.xfail(strict=True, reason=MISSING_HELPERS)
def test_summarize_tool_executions_summarizes_successes_and_errors() -> None:
    summary = orch.summarize_tool_executions(
        [
            _result(tool_name="good", tool_call_id="a"),
            _result(tool_name="chatty", tool_call_id="b"),
            _result(tool_name="bad", tool_call_id="c", result=None, error=RuntimeError("nope")),
        ]
    )

    assert summary == "\n".join(
        ["\u2705 good (a): SUCCESS", "\u2705 chatty (b): SUCCESS", "\u274c bad (c): ERROR - nope"]
    )
    assert orch.summarize_tool_executions([]) == ""


@pytest.mark.xfail(strict=True, reason=MISSING_HELPERS)
def test_has_tool_execution_errors() -> None:
    assert orch.has_tool_execution_errors([_result()]) is False
    assert orch.has_tool_execution_errors([]) is False
    assert orch.has_tool_execution_errors([_result(), _result(error=RuntimeError("x"), result=None)]) is True


@pytest.mark.xfail(strict=True, reason=MISSING_HELPERS)
def test_get_tool_execution_errors_returns_only_errors_in_order() -> None:
    boom = RuntimeError("boom")
    errors = orch.get_tool_execution_errors(
        [
            _result(),
            _result(tool_call_id="b", error=boom, result=None),
            _result(tool_call_id="c", error=RuntimeError("pow"), result=None),
        ]
    )

    assert len(errors) == 2
    assert errors[0] is boom
    assert [str(e) for e in errors] == ["boom", "pow"]
    assert orch.get_tool_execution_errors([_result()]) == []


# -- the same loop invariants through call_model (pass today) ----------------


def _outputs(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [i for i in request["input"] if isinstance(i, dict) and i.get("type") == "function_call_output"]


async def _tool_results(result: Any) -> List[Dict[str, Any]]:
    return [e for e in [ev async for ev in result.get_full_responses_stream()] if _type(e) == "tool.result"]


def _type(event: Any) -> Optional[str]:
    return event.get("type") if isinstance(event, dict) else getattr(event, "type", None)


async def test_loop_returns_after_one_request_without_tool_calls() -> None:
    client = QueuedClient([text_response("resp_1", "hello")])

    text = await call_model(client, {"model": "m", "input": INITIAL_INPUT, "tools": []}).get_text()

    assert text == "hello"
    assert len(client.requests) == 1


async def test_loop_executes_a_call_and_sends_its_output_on_the_follow_up() -> None:
    add_tool = tool(name="add", input_schema=AddIn, execute=_add)
    client = QueuedClient([_response_with(_call("call_1", "add", {"a": 2, "b": 3})), text_response("r2", "5")])

    result = call_model(client, {"model": "m", "input": INITIAL_INPUT, "tools": [add_tool]})
    tool_results = await _tool_results(result)

    assert len(client.requests) == 2
    assert [(e["toolCallId"], e["source"], e["result"]) for e in tool_results] == [("call_1", "client", {"sum": 5})]
    outputs = _outputs(client.requests[1])
    # snake_case `call_id`: _send normalizes at the transport boundary
    # (model_result.py `_send`); upstream's wire item uses `callId`.
    assert [(o["call_id"], json.loads(o["output"])) for o in outputs] == [("call_1", {"sum": 5})]


async def test_loop_supports_multiple_rounds_executing_each_call_once() -> None:
    calls: List[str] = []

    def ping(params: Any, ctx: Any) -> str:
        calls.append(ctx["tool_call"]["callId"])
        return "pong"

    ping_tool = tool(name="ping", input_schema=Empty, execute=ping)
    client = QueuedClient(
        [
            _response_with(_call("call_1", "ping", {}), response_id="r1"),
            _response_with(_call("call_2", "ping", {}), response_id="r2"),
            text_response("r3", "end"),
        ]
    )

    await call_model(client, {"model": "m", "input": INITIAL_INPUT, "tools": [ping_tool]}).get_text()

    assert len(client.requests) == 3
    assert calls == ["call_1", "call_2"]


async def test_loop_executes_multiple_calls_in_one_round_in_order() -> None:
    echo = tool(name="echo", input_schema=ValueIn, execute=lambda params, ctx: params.value)
    client = QueuedClient(
        [
            _response_with(_call("call_a", "echo", {"value": "one"}), _call("call_b", "echo", {"value": "two"})),
            text_response("r2", "end"),
        ]
    )

    await call_model(client, {"model": "m", "input": INITIAL_INPUT, "tools": [echo]}).get_text()

    assert len(client.requests) == 2
    outputs = _outputs(client.requests[1])
    assert [(o["call_id"], json.loads(o["output"])) for o in outputs] == [("call_a", "one"), ("call_b", "two")]


async def test_loop_stops_without_executing_manual_tools() -> None:
    manual = tool(name="needs_human", input_schema=Question, execute=False)
    first = _response_with(_call("call_1", "needs_human", {"question": "?"}))
    client = QueuedClient([first])

    result = call_model(client, {"model": "m", "input": INITIAL_INPUT, "tools": [manual]})
    final = await result.get_response()

    assert len(client.requests) == 1
    assert final is first
    assert await _tool_results(result) == []


async def test_loop_captures_execute_failures_and_keeps_looping() -> None:
    fail = tool(name="explodes", input_schema=Empty, execute=_explode)
    client = QueuedClient([_response_with(_call("call_1", "explodes", {})), text_response("r2", "end")])

    result = call_model(client, {"model": "m", "input": INITIAL_INPUT, "tools": [fail]})
    tool_results = await _tool_results(result)

    assert [e["result"] for e in tool_results] == [{"error": "kaboom"}]
    assert len(client.requests) == 2
    outputs = _outputs(client.requests[1])
    assert [json.loads(o["output"]) for o in outputs] == [{"error": "kaboom"}]


async def test_loop_marks_mcp_failures_with_source_mcp() -> None:
    mcp = mark_mcp(tool(name="mcp_tool", input_schema=Empty, execute=_explode))
    client = QueuedClient([_response_with(_call("call_1", "mcp_tool", {})), text_response("r2", "end")])

    result = call_model(client, {"model": "m", "input": INITIAL_INPUT, "tools": [mcp]})
    tool_results = await _tool_results(result)

    assert [(e["source"], e["result"]) for e in tool_results] == [("mcp", {"error": "kaboom"})]


async def test_loop_forwards_generator_preliminary_results_before_the_result() -> None:
    progress = tool(name="progress", input_schema=Empty, event_schema=Pct, output_schema=Done, execute=_progress)
    client = QueuedClient([_response_with(_call("call_1", "progress", {})), text_response("r2", "end")])

    result = call_model(client, {"model": "m", "input": INITIAL_INPUT, "tools": [progress]})
    events = [
        e
        for e in [ev async for ev in result.get_full_responses_stream()]
        if _type(e) in {"tool.preliminary_result", "tool.result"}
    ]

    assert [_type(e) for e in events] == ["tool.preliminary_result", "tool.result"]
    assert _dump(events[0]["result"]) == {"pct": 50}
    assert _dump(events[1]["result"]) == {"done": True}
    assert "preliminary_results" not in events[1]
