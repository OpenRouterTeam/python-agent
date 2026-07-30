from __future__ import annotations

from typing import Any, Dict

from openrouter_agent import call_model, tool
from tests._fixtures import QueuedClient, function_call_item, make_response, text_response


def empty_response(response_id: str = "resp_empty") -> Dict[str, Any]:
    """A response with no output items -- the empty-final case under test."""
    return make_response(response_id, [])


auto_tool = tool(
    name="auto_search", input_schema=dict, output_schema=dict, execute=lambda params, ctx: {"result": "found it"}
)
manual_tool = tool(name="exec_command", input_schema=dict, execute=False)
post_comment_tool = tool(
    name="post_comment", input_schema=dict, output_schema=dict, execute=lambda params, ctx: {"ok": True}
)


async def test_stops_loop_instead_of_orphaned_function_call_followup() -> None:
    client = QueuedClient(
        [
            make_response(
                "resp_mixed",
                [
                    function_call_item("call_auto_1", "auto_search", '{"query":"docs"}'),
                    function_call_item("call_manual_1", "exec_command", '{"command":"ls"}'),
                ],
            )
        ]
    )

    result = call_model(client, {"model": "test-model", "input": "do both things", "tools": [auto_tool, manual_tool]})
    response = await result.get_response()

    assert response["id"] == "resp_mixed"
    assert len(client.requests) == 1


async def test_still_loops_when_every_call_in_the_round_resolves() -> None:
    client = QueuedClient(
        [
            make_response("resp_auto", [function_call_item("call_auto_1", "auto_search", '{"query":"docs"}')]),
            text_response("resp_final", "All done."),
        ]
    )

    result = call_model(client, {"model": "test-model", "input": "search the docs", "tools": [auto_tool, manual_tool]})
    text = await result.get_text()

    assert text == "All done."
    assert len(client.requests) == 2
    followup_input = client.requests[1]["input"]
    fn_call_output = next((i for i in followup_input if i.get("type") == "function_call_output"), None)
    assert fn_call_output is not None
    # On the wire the SDK's snake_case spelling is used (internal items keep
    # upstream's camelCase callId; _send converts at the transport boundary).
    assert fn_call_output["call_id"] == "call_auto_1"
    assert "found it" in fn_call_output["output"]


async def test_retries_once_then_accepts_empty_final_after_a_completed_tool_round() -> None:
    client = QueuedClient(
        [
            make_response("resp_tool_call", [function_call_item("call_abc", "post_comment", '{"body":"lgtm"}')]),
            empty_response("resp_empty_1"),
            empty_response("resp_empty_2"),
        ]
    )

    result = call_model(client, {"model": "test-model", "input": "review", "tools": [post_comment_tool]})

    text = await result.get_text()
    assert text == ""
    assert len(client.requests) == 3

    response = await result.get_response()
    assert response["id"] == "resp_empty_2"
    assert response["output"] == []


async def test_returns_text_when_the_empty_final_retry_succeeds() -> None:
    client = QueuedClient(
        [
            make_response("resp_tool_call", [function_call_item("call_abc", "post_comment", '{"body":"lgtm"}')]),
            empty_response("resp_empty"),
            text_response("resp_retry_text", "Done posting."),
        ]
    )

    text = await call_model(client, {"model": "test-model", "input": "review", "tools": [post_comment_tool]}).get_text()

    assert text == "Done posting."
    assert len(client.requests) == 3


async def test_retry_forces_tool_choice_none_while_keeping_tools_in_request() -> None:
    client = QueuedClient(
        [
            make_response("resp_tool_call", [function_call_item("call_abc", "post_comment", '{"body":"lgtm"}')]),
            empty_response("resp_empty"),
            text_response("resp_retry_text", "Done posting."),
        ]
    )

    await call_model(client, {"model": "test-model", "input": "review", "tools": [post_comment_tool]}).get_text()

    followup_request = client.requests[1]
    assert "tools" in followup_request
    assert followup_request.get("tool_choice") != "none"

    retry_request = client.requests[2]
    assert "tools" in retry_request
    assert retry_request["tool_choice"] == "none"
    assert retry_request["input"] == followup_request["input"]


async def test_throws_on_empty_final_when_strict_final_response_is_true() -> None:
    client = QueuedClient(
        [
            make_response("resp_tool_call", [function_call_item("call_abc", "post_comment", '{"body":"lgtm"}')]),
            empty_response(),
        ]
    )

    raised = False
    try:
        await call_model(
            client,
            {
                "model": "test-model",
                "input": "review",
                "tools": [post_comment_tool],
                "strict_final_response": True,
            },
        ).get_text()
    except ValueError as error:
        raised = True
        assert "Invalid final response: empty or invalid output" in str(error)

    assert raised
    assert len(client.requests) == 2


async def test_still_throws_on_empty_output_when_no_tool_rounds_completed() -> None:
    client = QueuedClient([empty_response()])

    raised = False
    try:
        await call_model(client, {"model": "test-model", "input": "hello"}).get_text()
    except ValueError:
        raised = True

    assert raised
    assert len(client.requests) == 1


async def test_does_not_send_client_only_fields_to_the_api() -> None:
    client = QueuedClient([text_response("resp_text", "hi")])

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "hello",
            "strict_final_response": True,
            "allow_final_response": True,
        },
    ).get_text()

    request = client.requests[0]
    for key in (
        "strict_final_response",
        "allow_final_response",
        "stop_when",
        "shared_context_schema",
        "on_turn_start",
        "on_turn_end",
        "hooks",
    ):
        assert key not in request
