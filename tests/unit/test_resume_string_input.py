from __future__ import annotations

from openrouter_agent import call_model
from tests._fixtures import MemoryStateAccessor, QueuedClient, text_response


async def test_normalizes_a_bare_string_input_when_resuming_loaded_history() -> None:
    accessor = MemoryStateAccessor()
    client = QueuedClient([text_response("resp_1", "First answer."), text_response("resp_2", "Second answer.")])

    await call_model(client, {"model": "test-model", "input": "First question", "state": accessor}).get_text()
    assert accessor.stored is not None
    # Assert *which* messages persisted, not merely that some did: a bare
    # `len(...) > 0` passes whether the port stored the user turn, the assistant
    # turn, both, or forty.
    roles = [message["role"] for message in accessor.stored.messages if isinstance(message, dict) and "role" in message]
    assert "user" in roles, f"user turn was not persisted; roles={roles}"
    stored_text = str(accessor.stored.messages)
    assert "First question" in stored_text
    assert "First answer." in stored_text

    await call_model(client, {"model": "test-model", "input": "Follow-up question", "state": accessor}).get_text()

    request = client.requests[1]
    assert isinstance(request["input"], list)
    for item in request["input"]:
        assert not isinstance(item, str)
    last = request["input"][-1]
    assert last["role"] == "user"
    assert last["content"] == "Follow-up question"


async def test_still_accepts_array_input_when_resuming_loaded_history() -> None:
    accessor = MemoryStateAccessor()
    client = QueuedClient([text_response("resp_1", "First answer."), text_response("resp_2", "Second answer.")])

    await call_model(client, {"model": "test-model", "input": "First question", "state": accessor}).get_text()
    await call_model(
        client,
        {"model": "test-model", "input": [{"role": "user", "content": "Follow-up question"}], "state": accessor},
    ).get_text()

    request = client.requests[1]
    last = request["input"][-1]
    assert last["role"] == "user"
    assert last["content"] == "Follow-up question"
