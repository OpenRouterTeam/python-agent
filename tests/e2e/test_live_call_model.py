"""Live end-to-end tests against the real OpenRouter API.

These exercise the load-bearing loop the same way upstream's
`packages/agent/tests/e2e` suite does: real streaming, a real tool round,
approval pause/resume across two `call_model` calls, lifecycle hooks firing
on live traffic, and state serialization surviving a round trip.

Skipped entirely without OPENROUTER_API_KEY. Uses a small, cheap model —
these tests assert behavior (a tool ran, a hook fired, state advanced),
never model quality, so prompts pin outputs as hard as possible.
"""

from __future__ import annotations

import json
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY is required for OpenRouter e2e tests",
)

MODEL = os.getenv("OPENROUTER_E2E_MODEL", "anthropic/claude-haiku-4.5")


def _client(**kwargs):
    from openrouter_agent import OpenRouter

    return OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"], **kwargs)


class MemoryState:
    def __init__(self):
        self.current = None
        self.saved = []

    async def load(self):
        return self.current

    async def save(self, new_state):
        self.current = new_state
        self.saved.append(new_state)


async def test_live_text_and_stream_agree() -> None:
    """Basic call: streamed deltas concatenate to the same final text."""
    from openrouter_agent import call_model

    result = call_model(
        _client(),
        {"model": MODEL, "input": "Reply with exactly the word: pong"},
    )
    chunks = [chunk async for chunk in result.get_text_stream()]
    text = await result.get_text()

    assert "pong" in text.lower()
    assert "".join(chunks) == text


async def test_live_tool_loop_executes_and_feeds_result_back() -> None:
    """The model calls our tool, and the tool's output shapes the final answer."""
    from openrouter_agent import call_model, tool

    calls = []

    def lookup(params, ctx):
        calls.append(params)
        return {"secret": "BANANA-42"}

    secret_tool = tool(
        name="get_secret",
        description="Returns the secret code. Call this to answer any question about the secret code.",
        input_schema=dict,
        execute=lookup,
    )

    result = call_model(
        _client(),
        {
            "model": MODEL,
            "input": "What is the secret code? Use the get_secret tool, then repeat the code back verbatim.",
            "tools": [secret_tool],
            # No tool_choice="required" here: it persists across follow-up
            # turns (matching upstream), which forces tool calls forever and
            # trips the 20-turn safety limit. The approval tests can use it
            # because they pause after the first turn.
        },
    )
    text = await result.get_text()

    assert len(calls) >= 1, "model never called the tool"
    assert "BANANA-42" in text
    tool_calls = await result.get_tool_calls()
    assert "get_secret" in [c.name for c in tool_calls]


async def test_live_approval_pause_and_resume_across_calls() -> None:
    """require_approval pauses the run with state persisted; a second
    call_model with approve_tool_calls resumes, executes, and completes.

    This is the mixed approval/HITL turn-ordering surface the port review
    flagged as the thing to watch — run against the real API.
    """
    from openrouter_agent import call_model, tool

    executed = []

    delete_tool = tool(
        name="delete_record",
        description="Deletes the record. Requires approval.",
        input_schema=dict,
        output_schema=dict,
        execute=lambda params, ctx: executed.append(params) or {"deleted": True},
        require_approval=True,
    )

    state = MemoryState()
    first = call_model(
        _client(),
        {
            "model": MODEL,
            "input": "Delete the record with id 7 using the delete_record tool.",
            "tools": [delete_tool],
            "state": state,
            # Force the tool call: this test asserts the approval pause, not
            # the model's willingness to use tools. Without this the model
            # occasionally answers in prose and the run legitimately completes.
            "tool_choice": "required",
        },
    )
    await first.get_response()

    paused = state.current
    assert paused is not None, "no state was saved"
    assert paused.status == "awaiting_approval"
    assert executed == [], "tool must not run before approval"

    pending = await first.get_pending_tool_calls()
    assert len(pending) == 1
    call_id = pending[0].id

    resumed = call_model(
        _client(),
        {
            "model": MODEL,
            "input": [],
            "tools": [delete_tool],
            "state": state,
            "approve_tool_calls": [call_id],
        },
    )
    text = await resumed.get_text()

    assert len(executed) == 1, "approved tool did not execute exactly once"
    assert state.current.status == "complete"
    assert isinstance(text, str) and text.strip()


async def test_live_hooks_fire_on_real_traffic() -> None:
    """PreToolUse / PostToolUse / SessionStart / SessionEnd / PostModelCall
    all fire during a live tool round, and SessionEnd reports real usage."""
    from openrouter_agent import HookEntry, HookName, HooksManager, call_model, tool

    fired = []
    usage_totals = {}

    manager = HooksManager()
    for hook_name in (
        HookName.SessionStart,
        HookName.PreToolUse,
        HookName.PostToolUse,
        HookName.PostModelCall,
    ):
        manager.on(
            hook_name.value,
            HookEntry(handler=lambda payload, ctx, _n=hook_name.value: fired.append(_n) or {}),
        )

    def session_end(payload, ctx):
        fired.append(HookName.SessionEnd.value)
        usage_totals.update(payload.get("total_usage") or {})
        return {}

    manager.on(HookName.SessionEnd.value, HookEntry(handler=session_end))

    echo = tool(
        name="echo",
        description="Echoes back the given text.",
        input_schema=dict,
        execute=lambda params, ctx: {"echoed": params.get("text", "")},
    )

    result = call_model(
        _client(),
        {
            "model": MODEL,
            "input": "Use the echo tool with text 'hi', then say done.",
            "tools": [echo],
            "hooks": manager,
        },
    )
    await result.get_text()

    assert fired[0] == HookName.SessionStart.value
    assert fired[-1] == HookName.SessionEnd.value
    assert HookName.PreToolUse.value in fired
    assert HookName.PostToolUse.value in fired
    assert HookName.PostModelCall.value in fired
    # SessionEnd carries aggregated real usage — a live call must cost tokens.
    assert any(v for v in usage_totals.values() if isinstance(v, (int, float)) and v > 0), (
        f"SessionEnd usage totals empty: {usage_totals}"
    )


async def test_live_state_serialization_round_trip_resumes() -> None:
    """A live paused state survives serialize -> JSON -> deserialize and the
    deserialized state resumes correctly — the durable-storage story works
    against real response ids, not just fixtures."""
    from openrouter_agent import (
        call_model,
        deserialize_conversation_state,
        serialize_conversation_state,
        tool,
    )

    executed = []
    approve_tool = tool(
        name="launch",
        description="Launches the rocket. Requires approval.",
        input_schema=dict,
        output_schema=dict,
        execute=lambda params, ctx: executed.append(1) or {"launched": True},
        require_approval=True,
    )

    state = MemoryState()
    first = call_model(
        _client(),
        {
            "model": MODEL,
            "input": "Launch the rocket using the launch tool.",
            "tools": [approve_tool],
            "state": state,
            "tool_choice": "required",  # see approval test: pin the tool call
        },
    )
    await first.get_response()
    assert state.current.status == "awaiting_approval"
    pending = await first.get_pending_tool_calls()

    # Round-trip through the wire format, as a durable store would.
    raw = serialize_conversation_state(state.current)
    json.loads(raw)  # must be valid JSON, not repr()
    restored = MemoryState()
    restored.current = deserialize_conversation_state(raw)

    resumed = call_model(
        _client(),
        {
            "model": MODEL,
            "input": [],
            "tools": [approve_tool],
            "state": restored,
            "approve_tool_calls": [pending[0].id],
        },
    )
    await resumed.get_text()

    assert executed == [1]
    assert restored.current.status == "complete"
