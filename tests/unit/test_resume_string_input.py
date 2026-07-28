from __future__ import annotations

from typing import Any, Dict, List, Optional

from openrouter_agent import call_model
from openrouter_agent.tool_types import ConversationState


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


class MemoryStateAccessor:
    def __init__(self) -> None:
        self.stored: Optional[ConversationState] = None

    async def load(self) -> Optional[ConversationState]:
        return self.stored

    async def save(self, state: ConversationState) -> None:
        self.stored = state


def text_response(response_id: str, text: str) -> Dict[str, Any]:
    return {
        "id": response_id,
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
    }


async def test_normalizes_a_bare_string_input_when_resuming_loaded_history() -> None:
    accessor = MemoryStateAccessor()
    client = QueuedClient([text_response("resp_1", "First answer."), text_response("resp_2", "Second answer.")])

    await call_model(client, {"model": "test-model", "input": "First question", "state": accessor}).get_text()
    assert accessor.stored is not None
    assert len(accessor.stored.messages) > 0

    await call_model(client, {"model": "test-model", "input": "Follow-up question", "state": accessor}).get_text()

    request = client.beta.responses.requests[1]
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

    request = client.beta.responses.requests[1]
    last = request["input"][-1]
    assert last["role"] == "user"
    assert last["content"] == "Follow-up question"
