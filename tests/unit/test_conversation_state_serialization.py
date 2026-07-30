from __future__ import annotations

import dataclasses
import json

import pytest

from openrouter_agent import (
    CONVERSATION_STATE_VERSION,
    InvalidStateError,
    UnsupportedStateVersionError,
    create_initial_state,
    deserialize_conversation_state,
    serialize_conversation_state,
)
from openrouter_agent.tool_types import ParsedToolCall, UnsentToolResult


def test_round_trips_a_fresh_initial_state_with_version_1() -> None:
    state = create_initial_state("conv_fresh")
    assert state.version == 1

    restored = deserialize_conversation_state(serialize_conversation_state(state))

    assert restored == state
    assert restored.version == 1


def test_round_trips_a_rich_awaiting_client_tools_state() -> None:
    base = create_initial_state("conv_manual_pause")
    rich = dataclasses.replace(
        base,
        status="awaiting_client_tools",
        previous_response_id="resp_manual",
        messages=[
            {"type": "message", "role": "user", "content": "run ls"},
            {
                "type": "function_call",
                "id": "fc_call_manual_1",
                "callId": "call_manual_1",
                "name": "exec_command",
                "arguments": '{"command":"ls"}',
                "status": "completed",
            },
        ],
        pending_tool_calls=[ParsedToolCall(id="call_manual_1", name="exec_command", arguments={"command": "ls"})],
        unsent_tool_results=[
            UnsentToolResult(call_id="call_auto_1", name="auto_search", output={"result": "found it"})
        ],
    )

    restored = deserialize_conversation_state(serialize_conversation_state(rich))

    assert restored == rich
    assert restored.status == "awaiting_client_tools"
    assert restored.pending_tool_calls == [
        ParsedToolCall(id="call_manual_1", name="exec_command", arguments={"command": "ls"})
    ]
    assert restored.unsent_tool_results is not None
    assert restored.unsent_tool_results[0].call_id == "call_auto_1"


def test_deserializes_version_less_legacy_json_and_normalizes_to_version_1() -> None:
    legacy_json = (
        '{"id": "conv_legacy", "messages": [{"type": "message", "role": "user", "content": "hi"}], '
        '"status": "complete", "created_at": 1600000000000, "updated_at": 1600000000100, '
        '"previous_response_id": "resp_legacy"}'
    )

    restored = deserialize_conversation_state(legacy_json)

    assert restored.version == 1
    assert restored.id == "conv_legacy"
    assert restored.status == "complete"
    assert len(restored.messages) == 1
    assert restored.previous_response_id == "resp_legacy"


def test_raises_unsupported_state_version_error_for_a_future_version() -> None:
    future_json = (
        '{"version": 2, "id": "conv_future", "messages": [], "status": "in_progress", "created_at": 1, "updated_at": 1}'
    )

    with pytest.raises(UnsupportedStateVersionError) as exc_info:
        deserialize_conversation_state(future_json)

    error = exc_info.value
    assert error.found == 2
    assert error.supported == [1]
    assert error.name == "UnsupportedStateVersionError"


def test_raises_invalid_state_error_for_malformed_json() -> None:
    with pytest.raises(InvalidStateError):
        deserialize_conversation_state("not json")


@pytest.mark.parametrize(
    "raw_json",
    [
        '{"messages": [], "status": "in_progress", "created_at": 1, "updated_at": 1}',  # missing id
        '{"id": "x", "status": "in_progress", "created_at": 1, "updated_at": 1}',  # missing messages
        '{"id": "x", "messages": "not-an-array", "status": "in_progress", "created_at": 1, "updated_at": 1}',
        '{"id": "x", "messages": [], "created_at": 1, "updated_at": 1}',  # missing status
        '{"id": "x", "messages": [], "status": "in_progress", "updated_at": 1}',  # missing created_at
    ],
)
def test_raises_invalid_state_error_for_missing_required_fields(raw_json: str) -> None:
    with pytest.raises(InvalidStateError):
        deserialize_conversation_state(raw_json)


def test_serialize_injects_version_when_absent() -> None:
    state = create_initial_state("conv_no_version")
    stateless = dataclasses.replace(state, version=None)

    json_str = serialize_conversation_state(stateless)

    assert '"version":1' in json_str.replace(" ", "")


def test_conversation_state_version_constant() -> None:
    assert CONVERSATION_STATE_VERSION == 1


def test_serialize_dumps_sdk_pydantic_items_to_json() -> None:
    """Live states hold SDK pydantic response items (e.g. OutputFunctionCallItem),
    which dataclasses.asdict passes through untouched. serialize must emit valid
    JSON for them (json.dumps default=dump), not raise TypeError. Regression
    guard for the fix e2e found — this unit test runs on every PR, while the
    e2e test needs OPENROUTER_API_KEY."""
    from pydantic import BaseModel

    class FakeSDKItem(BaseModel):
        type: str = "function_call"
        callId: str = "call_1"
        name: str = "t"
        arguments: str = "{}"

    state = dataclasses.replace(create_initial_state("conv_sdk_items"), messages=[FakeSDKItem()])

    raw = serialize_conversation_state(state)

    parsed = json.loads(raw)
    assert parsed["messages"][0]["callId"] == "call_1"
