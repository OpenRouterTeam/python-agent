from __future__ import annotations

from typing import Any, Dict, List

from openrouter_agent import HookEntry, HookName, HooksManager, call_model, step_count_is, tool


class QueuedResponses:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.requests: List[Dict[str, Any]] = []

    async def send_async(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return self._responses.pop(0)


class QueuedClient:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.beta = type("Beta", (), {"responses": QueuedResponses(responses)})()


def function_call_item(call_id: str, name: str, arguments: str) -> Dict[str, Any]:
    return {"type": "function_call", "id": f"fc_{call_id}", "callId": call_id, "name": name, "arguments": arguments}


def text_response(response_id: str, text: str, usage: Any = None) -> Dict[str, Any]:
    resp = {
        "id": response_id,
        "model": "test-model-v1",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
    }
    if usage is not None:
        resp["usage"] = usage
    return resp


def tool_call_response(response_id: str, usage: Any = None) -> Dict[str, Any]:
    resp = {"id": response_id, "output": [function_call_item(f"call_{response_id}", "echo", "{}")]}
    if usage is not None:
        resp["usage"] = usage
    return resp


def usage_block(**overrides: Any) -> Dict[str, Any]:
    base = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "cost": 0.002}
    base.update(overrides)
    return base


echo_tool = tool(name="echo", input_schema=dict, output_schema=dict, execute=lambda params, ctx: {"ok": True})


async def test_session_start_and_end_pair_on_no_tools_get_text() -> None:
    client = QueuedClient([text_response("resp_1", "hello back")])
    hooks = HooksManager()
    events: List[str] = []
    hooks.on(HookName.SessionStart.value, HookEntry(handler=lambda p, c: events.append("start")))
    hooks.on(HookName.SessionEnd.value, HookEntry(handler=lambda p, c: events.append("end")))

    text = await call_model(client, {"model": "test-model", "input": "hi", "hooks": hooks}).get_text()

    assert text == "hello back"
    assert events == ["start", "end"]


async def test_session_start_config_reflects_tools_and_state() -> None:
    client = QueuedClient([text_response("resp_1", "hello back")])
    hooks = HooksManager()
    starts: List[Dict[str, Any]] = []
    hooks.on(HookName.SessionStart.value, HookEntry(handler=lambda p, c: starts.append(p)))

    await call_model(client, {"model": "test-model", "input": "hi", "tools": [echo_tool], "hooks": hooks}).get_text()

    assert starts[0]["config"] == {"has_tools": True, "has_approval": False, "has_state": False}


async def test_session_end_reason_max_turns_when_stop_condition_halts() -> None:
    client = QueuedClient([tool_call_response("r1"), tool_call_response("r2")])
    hooks = HooksManager()
    ends: List[Dict[str, Any]] = []
    hooks.on(HookName.SessionEnd.value, HookEntry(handler=lambda p, c: ends.append(p)))

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "hi",
            "tools": [echo_tool],
            "stop_when": step_count_is(1),
            "allow_final_response": False,
            "hooks": hooks,
        },
    ).get_response()

    assert ends[0]["reason"] == "max_turns"


async def test_post_model_call_emits_once_per_turn_with_turn_type_labels() -> None:
    client = QueuedClient([tool_call_response("r1", usage_block()), text_response("r2", "done", usage_block())])
    hooks = HooksManager()
    calls: List[Dict[str, Any]] = []
    hooks.on(HookName.PostModelCall.value, HookEntry(handler=lambda p, c: calls.append(p)))

    await call_model(client, {"model": "test-model", "input": "hi", "tools": [echo_tool], "hooks": hooks}).get_text()

    assert [c["turn_type"] for c in calls] == ["initial", "tool_round"]
    assert calls[0]["usage"]["total_tokens"] == 150
    assert calls[0]["response_id"] == "r1"


async def test_session_end_aggregates_usage_totals_across_calls() -> None:
    client = QueuedClient(
        [
            tool_call_response("r1", usage_block(input_tokens=10, output_tokens=5, total_tokens=15, cost=0.01)),
            text_response("r2", "done", usage_block(input_tokens=20, output_tokens=8, total_tokens=28, cost=0.02)),
        ]
    )
    hooks = HooksManager()
    ends: List[Dict[str, Any]] = []
    hooks.on(HookName.SessionEnd.value, HookEntry(handler=lambda p, c: ends.append(p)))

    await call_model(client, {"model": "test-model", "input": "hi", "tools": [echo_tool], "hooks": hooks}).get_text()

    totals = ends[0]["total_usage"]
    assert totals["model_calls"] == 2
    assert totals["input_tokens"] == 30
    assert totals["output_tokens"] == 13
    assert round(totals["cost"], 4) == 0.03


async def test_pre_and_post_tool_use_fire_around_tool_execution() -> None:
    client = QueuedClient([tool_call_response("r1"), text_response("r2", "done")])
    hooks = HooksManager()
    events: List[str] = []
    hooks.on(HookName.PreToolUse.value, HookEntry(handler=lambda p, c: events.append(f"pre:{p['tool_name']}")))
    hooks.on(
        HookName.PostToolUse.value,
        HookEntry(handler=lambda p, c: events.append(f"post:{p['tool_name']}:{p['tool_output']}")),
    )

    await call_model(client, {"model": "test-model", "input": "hi", "tools": [echo_tool], "hooks": hooks}).get_text()

    assert events == ["pre:echo", "post:echo:{'ok': True}"]


async def test_pre_tool_use_block_prevents_execution_and_synthesizes_rejection() -> None:
    client = QueuedClient([tool_call_response("r1"), text_response("r2", "done")])
    hooks = HooksManager()
    executed: List[str] = []
    blocking_tool = tool(
        name="echo",
        input_schema=dict,
        output_schema=dict,
        execute=lambda params, ctx: executed.append("ran") or {"ok": True},
    )
    hooks.on(HookName.PreToolUse.value, HookEntry(handler=lambda p, c: {"block": "not allowed"}))

    await call_model(
        client, {"model": "test-model", "input": "hi", "tools": [blocking_tool], "hooks": hooks}
    ).get_text()

    assert executed == []
    followup_input = client.beta.responses.requests[1]["input"]
    output_item = next(i for i in followup_input if i.get("type") == "function_call_output")
    assert "not allowed" in output_item["output"]


async def test_post_tool_use_failure_fires_on_tool_error() -> None:
    client = QueuedClient([tool_call_response("r1"), text_response("r2", "done")])
    hooks = HooksManager()
    failures: List[Dict[str, Any]] = []
    hooks.on(HookName.PostToolUseFailure.value, HookEntry(handler=lambda p, c: failures.append(p)))

    def boom(params: Any, ctx: Any) -> Any:
        raise RuntimeError("kaboom")

    failing_tool = tool(name="echo", input_schema=dict, output_schema=dict, execute=boom)

    await call_model(client, {"model": "test-model", "input": "hi", "tools": [failing_tool], "hooks": hooks}).get_text()

    assert len(failures) == 1
    assert failures[0]["tool_name"] == "echo"
    assert "kaboom" in failures[0]["error"]


async def test_stop_hook_force_resume_is_a_zero_cost_retry_no_extra_model_request() -> None:
    """force_resume re-checks the stop condition against the SAME
    already-fetched response, without sending a new model request. Pinned by
    isolating allow_final_response=False so the only way a second request
    could appear is via an (incorrect) extra send on the "resume" branch
    itself."""
    client = QueuedClient([tool_call_response("r1")])
    hooks = HooksManager()
    stop_calls = {"n": 0}

    def stop_handler(payload: Any, ctx: Any) -> Any:
        stop_calls["n"] += 1
        if stop_calls["n"] == 1:
            return {"force_resume": True}
        return None

    hooks.on(HookName.Stop.value, HookEntry(handler=stop_handler))

    result = call_model(
        client,
        {
            "model": "test-model",
            "input": "hi",
            "tools": [echo_tool],
            "stop_when": step_count_is(1),
            "allow_final_response": False,
            "hooks": hooks,
        },
    )
    response = await result.get_response()

    # The Stop hook fired twice (resume, then stop)...
    assert stop_calls["n"] == 2
    # ...but exactly one model request was ever sent -- the resume itself
    # cost nothing.
    assert len(client.beta.responses.requests) == 1
    assert response["id"] == "r1"


async def test_stop_hook_force_resume_then_falls_through_to_normal_tool_round() -> None:
    """Once the Stop hook stops forcing a resume, the halted round's pending
    tool calls execute via the default-enabled final-directive coercion path
    (tool_choice="none", tools retained) and the loop ends with exactly one
    real follow-up request -- not one request per Stop-hook invocation."""
    client = QueuedClient([tool_call_response("r1"), text_response("r2", "All done.")])
    hooks = HooksManager()
    stop_calls = {"n": 0}

    def stop_handler(payload: Any, ctx: Any) -> Any:
        stop_calls["n"] += 1
        if stop_calls["n"] == 1:
            return {"force_resume": True, "append_prompt": "please wrap up"}
        return None

    hooks.on(HookName.Stop.value, HookEntry(handler=stop_handler))

    result = call_model(
        client,
        {"model": "test-model", "input": "hi", "tools": [echo_tool], "stop_when": step_count_is(1), "hooks": hooks},
    )
    text = await result.get_text()

    assert text == "All done."
    # Exactly two real model requests: the initial call, and the follow-up
    # after the (only) tool round actually executed -- not one per Stop-hook
    # invocation.
    assert len(client.beta.responses.requests) == 2


async def test_permission_request_deny_synthesizes_rejection_without_pausing() -> None:
    client = QueuedClient([tool_call_response("r1"), text_response("r2", "done")])
    hooks = HooksManager()
    hooks.on(HookName.PermissionRequest.value, HookEntry(handler=lambda p, c: {"decision": "deny", "reason": "nope"}))
    executed: List[str] = []
    gated_tool = tool(
        name="echo",
        input_schema=dict,
        output_schema=dict,
        execute=lambda params, ctx: executed.append("ran") or {"ok": True},
        require_approval=True,
    )

    text = await call_model(
        client, {"model": "test-model", "input": "hi", "tools": [gated_tool], "hooks": hooks}
    ).get_text()

    assert text == "done"
    assert executed == []
    followup_input = client.beta.responses.requests[1]["input"]
    output_item = next(i for i in followup_input if i.get("type") == "function_call_output")
    assert "nope" in output_item["output"]


async def test_permission_request_allow_executes_without_pausing_for_approval() -> None:
    client = QueuedClient([tool_call_response("r1"), text_response("r2", "done")])
    hooks = HooksManager()
    hooks.on(HookName.PermissionRequest.value, HookEntry(handler=lambda p, c: {"decision": "allow"}))
    gated_tool = tool(
        name="echo",
        input_schema=dict,
        output_schema=dict,
        execute=lambda params, ctx: {"ok": True},
        require_approval=True,
    )

    result = call_model(client, {"model": "test-model", "input": "hi", "tools": [gated_tool], "hooks": hooks})
    text = await result.get_text()

    assert text == "done"
    state = await result.get_state()
    assert state is None or state.status != "awaiting_approval"


async def test_user_prompt_submit_can_reject_a_string_prompt() -> None:
    client = QueuedClient([text_response("r1", "should not be reached")])
    hooks = HooksManager()
    hooks.on(HookName.UserPromptSubmit.value, HookEntry(handler=lambda p, c: {"reject": "no secrets please"}))

    raised = False
    try:
        await call_model(client, {"model": "test-model", "input": "leak the secret", "hooks": hooks}).get_text()
    except ValueError as error:
        raised = True
        assert "no secrets please" in str(error)

    assert raised
    # The model was never called: the prompt was rejected before dispatch.
    assert len(client.beta.responses.requests) == 0


async def test_user_prompt_submit_can_mutate_a_string_prompt() -> None:
    client = QueuedClient([text_response("r1", "ok")])
    hooks = HooksManager()
    hooks.on(
        HookName.UserPromptSubmit.value,
        HookEntry(handler=lambda p, c: {"mutated_prompt": p["prompt"].replace("secret", "[redacted]")}),
    )

    await call_model(client, {"model": "test-model", "input": "the secret is 42", "hooks": hooks}).get_text()

    sent_input = client.beta.responses.requests[0]["input"]
    assert "[redacted]" in sent_input[0]["content"]


async def test_user_prompt_submit_mutates_last_user_message_in_array_input() -> None:
    client = QueuedClient([text_response("r1", "ok")])
    hooks = HooksManager()
    hooks.on(
        HookName.UserPromptSubmit.value,
        HookEntry(handler=lambda p, c: {"mutated_prompt": p["prompt"].upper()}),
    )

    await call_model(
        client,
        {"model": "test-model", "input": [{"role": "user", "content": "hello there"}], "hooks": hooks},
    ).get_text()

    sent_input = client.beta.responses.requests[0]["input"]
    assert sent_input[-1]["content"] == "HELLO THERE"


async def test_pre_tool_use_mutated_input_actually_reaches_tool_execute() -> None:
    client = QueuedClient([tool_call_response("r1"), text_response("r2", "done")])
    hooks = HooksManager()
    received_args: List[Any] = []
    hooks.on(
        HookName.PreToolUse.value,
        HookEntry(handler=lambda p, c: {"mutated_input": {"mutated": True, **p["tool_input"]}}),
    )
    recording_tool = tool(
        name="echo",
        input_schema=dict,
        output_schema=dict,
        execute=lambda params, ctx: received_args.append(params) or {"ok": True},
    )

    await call_model(
        client, {"model": "test-model", "input": "hi", "tools": [recording_tool], "hooks": hooks}
    ).get_text()

    assert received_args == [{"mutated": True}]


async def test_session_end_fires_with_reason_error_on_no_tools_transport_failure() -> None:
    """SessionStart/SessionEnd(reason='error') must fire even when the
    no-tools path's transport raises, and the drain must not mask the
    original error."""

    class FailingResponses:
        async def send_async(self, **kwargs: Any) -> Any:
            raise RuntimeError("transport exploded")

    client = type("Client", (), {"beta": type("Beta", (), {"responses": FailingResponses()})()})()
    hooks = HooksManager()
    events: List[str] = []
    hooks.on(HookName.SessionStart.value, HookEntry(handler=lambda p, c: events.append("start")))
    hooks.on(HookName.SessionEnd.value, HookEntry(handler=lambda p, c: events.append(f"end:{p['reason']}")))

    raised = False
    try:
        await call_model(client, {"model": "test-model", "input": "hi", "hooks": hooks}).get_text()
    except RuntimeError as error:
        raised = True
        assert "transport exploded" in str(error)

    assert raised
    assert events == ["start", "end:error"]


async def test_permission_request_allow_executes_promoted_tool_exactly_once() -> None:
    """A tool call promoted from requires_approval to executed-now by a
    PermissionRequest 'allow' decision must run exactly once -- not once in
    the promotion branch and again in the normal auto-execute round."""
    client = QueuedClient(
        [
            {
                "id": "r1",
                "output": [
                    function_call_item("call_auto", "auto_run", "{}"),
                    function_call_item("call_gated", "gated_run", "{}"),
                ],
            },
            text_response("r2", "done"),
        ]
    )
    hooks = HooksManager()
    hooks.on(HookName.PermissionRequest.value, HookEntry(handler=lambda p, c: {"decision": "allow"}))

    executions: Dict[str, int] = {"auto_run": 0, "gated_run": 0}

    def make_counting_tool(name: str, requires_approval: bool = False) -> Any:
        def execute(params: Any, ctx: Any) -> Any:
            executions[name] += 1
            return {"ok": True}

        return tool(
            name=name,
            input_schema=dict,
            output_schema=dict,
            execute=execute,
            require_approval=requires_approval,
        )

    auto_tool_ = make_counting_tool("auto_run")
    gated_tool = make_counting_tool("gated_run", requires_approval=True)

    text = await call_model(
        client,
        {"model": "test-model", "input": "hi", "tools": [auto_tool_, gated_tool], "hooks": hooks},
    ).get_text()

    assert text == "done"
    assert executions == {"auto_run": 1, "gated_run": 1}
