"""Shared test fixtures: canonical fake client and Responses-API payload builders.

Why this module exists
----------------------
Before it, 11 of 14 test files hand-copied their own `QueuedResponses` /
`QueuedClient` stubs (6 of them byte-identical) plus their own
`function_call_item` / `text_response` helpers. Those copies populated only `id`
and `output`, omitting every other field a real Responses API result carries
(`status`, `model`, `object`, `created_at`, `error`, `tool_choice`, ...). A port
that mishandled — or silently depended on the absence of — any omitted field
passed the whole suite.

`make_response` populates the complete required field set instead, so a stub can
no longer be more forgiving than the real API.

Casing: builders emit upstream's **camelCase** wire shape (`callId`, `createdAt`)
because that is what the port's internals consume; `ModelResult._send` normalizes
to the generated SDK's snake_case at the transport boundary only
(`model_result.py:148-156`). Building these from the SDK's own models would emit
snake_case and exercise only the fallback path — the opposite of what these tests
need to pin. `assert_matches_sdk_response_shape` bridges the two: it converts and
validates, so the *field set* still breaks loudly if the SDK changes.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional

from openrouter_agent.tool_types import ConversationState

# Every field `openrouter.components.OpenResponsesResult` marks required, with
# neutral defaults. Keep in sync with assert_matches_sdk_response_shape below —
# that helper is what fails when the SDK's required set drifts from this one.
_REQUIRED_RESPONSE_DEFAULTS: Dict[str, Any] = {
    "object": "response",
    "createdAt": 0,
    "completedAt": 0,
    "status": "completed",
    "model": "test-model-v1",
    "error": None,
    "incompleteDetails": None,
    "instructions": None,
    "metadata": None,
    "frequencyPenalty": None,
    "presencePenalty": None,
    "temperature": None,
    "topP": None,
    "toolChoice": "auto",
    "tools": [],
    "parallelToolCalls": False,
}


def make_response(
    response_id: str,
    output: List[Dict[str, Any]],
    *,
    usage: Any = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a complete Responses-API result.

    Pass `**overrides` to express a non-default field (e.g. `status="incomplete"`)
    rather than hand-rolling a partial dict.
    """
    response: Dict[str, Any] = {"id": response_id, "output": list(output)}
    response.update(_REQUIRED_RESPONSE_DEFAULTS)
    if usage is not None:
        response["usage"] = usage
    response.update(overrides)
    return response


def function_call_item(call_id: str, name: str, arguments: str = "{}") -> Dict[str, Any]:
    """A `function_call` output item, in upstream's camelCase `callId` shape."""
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "callId": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def message_item(text: str, *, role: str = "assistant", item_id: str = "msg_1") -> Dict[str, Any]:
    return {
        "type": "message",
        "id": item_id,
        "role": role,
        "status": "completed",
        "content": [{"type": "output_text", "text": text}],
    }


def text_response(response_id: str, text: str, usage: Any = None, **overrides: Any) -> Dict[str, Any]:
    """A terminal assistant text response."""
    return make_response(response_id, [message_item(text, item_id=f"msg_{response_id}")], usage=usage, **overrides)


def tool_call_response(
    response_id: str,
    name: str = "echo",
    *,
    call_id: Optional[str] = None,
    arguments: str = "{}",
    usage: Any = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """A response whose single output item is one function call."""
    return make_response(
        response_id,
        [function_call_item(call_id or f"call_{response_id}", name, arguments)],
        usage=usage,
        **overrides,
    )


def usage_block(**overrides: Any) -> Dict[str, Any]:
    base = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "cost": 0.002}
    base.update(overrides)
    return base


class QueuedResponses:
    """Fake `client.beta.responses` returning queued payloads in order.

    Records every request in `.requests` so tests can assert the request *count*
    and the exact follow-up input — several upstream invariants are literally
    "no follow-up request was made".
    """

    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.requests: List[Dict[str, Any]] = []

    async def send_async(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if not self._responses:
            # A bare IndexError here reads as a fixture bug. It is usually the
            # actual finding: the port sent a turn the test did not expect.
            raise AssertionError(
                f"the port sent {len(self.requests)} request(s) but only "
                f"{len(self.requests) - 1} response(s) were queued; "
                "an unexpected extra turn was requested"
            )
        return self._responses.pop(0)


class QueuedClient:
    """Minimal stand-in for `OpenRouter` exposing `.beta.responses.send_async`."""

    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = QueuedResponses(responses)
        self.beta = type("Beta", (), {"responses": self.responses})()

    @property
    def requests(self) -> List[Dict[str, Any]]:
        """Requests the port sent, so tests need not reach through `.beta`."""
        return self.responses.requests


class MemoryStateAccessor:
    """In-memory `state` accessor. `saved` keeps every version for ordering checks."""

    def __init__(self) -> None:
        self.stored: Optional[ConversationState] = None
        self.saved: List[ConversationState] = []

    async def load(self) -> Optional[ConversationState]:
        return self.stored

    async def save(self, state: ConversationState) -> None:
        self.stored = state
        self.saved.append(state)


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def assert_matches_sdk_response_shape(response: Mapping[str, Any]) -> None:
    """Validate a builder payload against the generated SDK's own response model.

    The builders emit camelCase (what the port consumes); the SDK model is
    snake_case. This converts and validates, so if the SDK adds or renames a
    required field, `_REQUIRED_RESPONSE_DEFAULTS` fails loudly here instead of
    tests quietly drifting onto a payload shape the real API never produces.
    """
    from openrouter.components import OpenResponsesResult

    payload = _to_snake_deep(response)
    payload.pop("usage", None)  # optional, and its own nested model
    OpenResponsesResult.model_validate(payload)


def _to_snake_deep(value: Any) -> Any:
    """Recursively snake_case every key, so nested output items convert too."""
    if isinstance(value, Mapping):
        return {_camel_to_snake(key): _to_snake_deep(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_snake_deep(item) for item in value]
    return value
