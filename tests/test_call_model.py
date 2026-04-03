"""Tests for state management and approval workflow in call_model."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from openrouter_agent import (
    ConversationState,
    ConversationStatus,
    call_model,
    create_initial_state,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockStateAccessor:
    """In-memory state accessor for tests."""

    def __init__(self, initial_state: ConversationState | None = None) -> None:
        self.saved_state: ConversationState | None = None
        self._state = initial_state

    async def load(self) -> ConversationState | None:
        return self._state

    async def save(self, state: ConversationState) -> None:
        self.saved_state = state


def _make_client(responses: list[dict[str, Any]] | None = None) -> Any:
    """Return a mock client whose beta.responses.send_async returns canned responses."""
    if responses is None:
        responses = [
            {
                "id": "resp-1",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Hello"}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                "finish_reason": "stop",
            }
        ]

    call_count = 0

    async def _send_async(**kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        idx = min(call_count, len(responses) - 1)
        call_count += 1
        return responses[idx]

    client = MagicMock()
    client.beta.responses.send_async = _send_async
    return client


# ---------------------------------------------------------------------------
# Tests: state is loaded and saved during execution
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_state_loaded_and_saved() -> None:
    """When a StateAccessor is provided, load() and save() are called."""
    state_accessor = MockStateAccessor()
    client = _make_client()

    result = await call_model(
        client,
        {
            "model": "test-model",
            "input": "Hi",
            "state": state_accessor,
        },
    )

    assert result.get_text() == "Hello"
    # State should have been saved
    assert state_accessor.saved_state is not None
    assert state_accessor.saved_state.status == ConversationStatus.COMPLETE


@pytest.mark.anyio
async def test_state_not_used_when_absent() -> None:
    """Without a StateAccessor the result state is None."""
    client = _make_client()

    result = await call_model(
        client,
        {
            "model": "test-model",
            "input": "Hi",
        },
    )

    assert result.get_text() == "Hello"
    assert result.get_state() is None


@pytest.mark.anyio
async def test_state_preserves_previous_response_id() -> None:
    """The saved state should carry the response id from the API."""
    state_accessor = MockStateAccessor()
    client = _make_client()

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "Hi",
            "state": state_accessor,
        },
    )

    assert state_accessor.saved_state is not None
    assert state_accessor.saved_state.previous_response_id == "resp-1"


# ---------------------------------------------------------------------------
# Tests: approval workflow pauses execution
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_approval_pauses_on_tool_call() -> None:
    """When require_approval returns True, the loop pauses with awaiting_approval."""
    tool_call_response = {
        "id": "resp-2",
        "output": [
            {
                "type": "function_call",
                "id": "call-abc",
                "name": "dangerous_tool",
                "arguments": "{}",
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "finish_reason": "tool_calls",
    }

    client = _make_client([tool_call_response])
    state_accessor = MockStateAccessor()

    # require_approval always returns True
    def _always_approve(tc: Any, ctx: Any) -> bool:
        return True

    from pydantic import BaseModel

    from openrouter_agent._tool_factory import tool

    class EmptyInput(BaseModel):
        pass

    dangerous = tool(
        name="dangerous_tool",
        description="A dangerous tool",
        input_schema=EmptyInput,
        execute=lambda args, ctx: "boom",
    )

    result = await call_model(
        client,
        {
            "model": "test-model",
            "input": "Do something dangerous",
            "tools": [dangerous],
            "require_approval": _always_approve,
            "state": state_accessor,
        },
    )

    # The loop should have paused
    assert state_accessor.saved_state is not None
    assert state_accessor.saved_state.status == ConversationStatus.AWAITING_APPROVAL
    assert state_accessor.saved_state.pending_tool_calls is not None
    assert len(state_accessor.saved_state.pending_tool_calls) == 1
    assert state_accessor.saved_state.pending_tool_calls[0].id == "call-abc"

    # ModelResult should expose pending calls
    pending = result.get_pending_tool_calls()
    assert len(pending) == 1
    assert pending[0].name == "dangerous_tool"


@pytest.mark.anyio
async def test_reject_pending_tool_calls() -> None:
    """Rejecting a pending tool call adds a rejected result to input."""
    from openrouter_agent import ParsedToolCall

    pending = [ParsedToolCall(id="call-abc", name="dangerous_tool", arguments={})]
    initial_state = create_initial_state()
    initial_state = initial_state.model_copy(update={
        "status": ConversationStatus.AWAITING_APPROVAL,
        "pending_tool_calls": pending,
    })

    state_accessor = MockStateAccessor(initial_state)
    client = _make_client()

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "Continue",
            "state": state_accessor,
            "reject_tool_calls": ["call-abc"],
        },
    )

    # Should have completed (the model returned text, no tool calls)
    assert state_accessor.saved_state is not None
    assert state_accessor.saved_state.status == ConversationStatus.COMPLETE
    # The rejection result should appear in the saved messages
    messages = state_accessor.saved_state.messages
    rejection_items = [
        m for m in messages
        if isinstance(m, dict)
        and m.get("type") == "function_call_output"
        and m.get("call_id") == "call-abc"
    ]
    assert len(rejection_items) == 1
    output_text = rejection_items[0]["output"].lower()
    assert "rejected" in output_text or "error" in output_text
