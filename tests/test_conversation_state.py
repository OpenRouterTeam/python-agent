"""Tests for conversation state management."""


from openrouter_agent import (
    ConversationStatus,
    append_to_messages,
    create_initial_state,
    create_rejected_result,
    create_unsent_result,
    generate_conversation_id,
    update_state,
)


def test_generate_conversation_id():
    id1 = generate_conversation_id()
    id2 = generate_conversation_id()
    assert id1 != id2
    assert len(id1) > 10


def test_create_initial_state():
    state = create_initial_state()
    assert state.status == ConversationStatus.IN_PROGRESS
    assert state.messages == []
    assert state.created_at > 0


def test_create_initial_state_with_id():
    state = create_initial_state("my-id")
    assert state.id == "my-id"


def test_update_state():
    state = create_initial_state()
    updated = update_state(state, {"status": ConversationStatus.COMPLETE})
    assert updated.status == ConversationStatus.COMPLETE
    assert updated.id == state.id
    assert updated.updated_at >= state.updated_at


def test_append_to_messages():
    current = [{"role": "user", "content": "hello"}]
    new = [{"role": "assistant", "content": "hi"}]
    result = append_to_messages(current, new)
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    # Original unchanged
    assert len(current) == 1


def test_create_unsent_result():
    r = create_unsent_result("call_1", "search", {"results": ["a", "b"]})
    assert r.call_id == "call_1"
    assert r.name == "search"
    assert r.output == {"results": ["a", "b"]}
    assert r.error is None


def test_create_rejected_result():
    r = create_rejected_result("call_1", "dangerous", "Not allowed")
    assert r.call_id == "call_1"
    assert r.error == "Not allowed"
