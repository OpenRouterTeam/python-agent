"""Integration tests for the doom-loop remediation pass.

Ports `packages/agent/tests/unit/doom-loop-remediation.test.ts`. Every
scenario simulates the LLM with scripted `QueuedClient` responses.

Coverage map (finding -> test group):
- B1/D7  state integrity + session reason on text-verdict stops
- B3     parallel same-round duplicates are one piece of evidence
- D1     no final-response request after a doom stop
- D2/D3  approval-resume gating + verdict persistence/clearing
- D4     steer survives pauses via persisted pending_steer; in-loop steer
- D5     loop_key returning nothing -> no colliding constant identity
- H1     DOCUMENTED MISS: nonce-varying args evade the fingerprint
- H2     server-tool repetition detected at the step checkpoint
- API1   loop_key forms (computed function, False)
- config ON->OFF / ON->ON lifecycle across resumes

Port notes: state/hook keys are snake_case (`pending_steer`, `stop_verdict`,
`consumed_forced_tool_choice_key`); request input items carry `call_id`
(normalized at the transport boundary by `ModelResult._send`), while persisted
state messages keep upstream's `callId`.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from typing import Any, Dict, Iterator, List, Mapping

import pytest

from openrouter_agent import HookEntry, HooksManager, call_model, tool
from tests._fixtures import MemoryStateAccessor, QueuedClient, function_call_item, make_response, message_item

pytestmark = pytest.mark.filterwarnings("ignore:\\[DoomLoop\\]:UserWarning")


class Script:
    def __init__(self) -> None:
        self.responses = 0
        self.calls = 0

    def turn(self, *items: Dict[str, Any]) -> Dict[str, Any]:
        self.responses += 1
        return make_response(f"resp_{self.responses}", list(items))

    def call(self, name: str, args: Any) -> Dict[str, Any]:
        self.calls += 1
        return function_call_item(f"call_{self.calls}", name, json.dumps(args))

    def message(self, text: str) -> Dict[str, Any]:
        self.calls += 1
        return message_item(text, item_id=f"msg_{self.calls}")

    def tool_call_turn(self, name: str, args: Any) -> Dict[str, Any]:
        return self.turn(self.call(name, args))

    def text_turn(self, text: str) -> Dict[str, Any]:
        return self.turn(self.message(text))


@pytest.fixture()
def script() -> Iterator[Script]:
    yield Script()


def _call_id(item: Mapping[str, Any]) -> Any:
    return item.get("callId", item.get("call_id"))


def assert_well_formed_history(messages: Any) -> None:
    """Every function_call in `messages` has a matching output."""
    items = messages if isinstance(messages, list) else []
    call_ids = {_call_id(i) for i in items if isinstance(i, Mapping) and i.get("type") == "function_call"}
    output_ids = {_call_id(i) for i in items if isinstance(i, Mapping) and i.get("type") == "function_call_output"}
    missing = call_ids - output_ids
    assert not missing, f"function_call(s) without a matching output: {sorted(missing)}"


def outputs_in(messages: Any) -> List[Mapping[str, Any]]:
    return [m for m in messages or [] if isinstance(m, Mapping) and m.get("type") == "function_call_output"]


def collect(hooks: HooksManager, hook_name: str = "DoomLoopDetected") -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []

    def handler(payload: Dict[str, Any], ctx: Any) -> None:
        collected.append(payload)

    hooks.on(hook_name, HookEntry(handler=handler))
    return collected


def actions_and_streaks(detections: List[Dict[str, Any]]) -> List[List[Any]]:
    return [[d["action"], d["streak"]] for d in detections]


class Counter:
    def __init__(self, result: Any) -> None:
        self.count = 0
        self.result = result

    def __call__(self, params: Any, ctx: Any) -> Any:
        self.count += 1
        return self.result


search_tool = tool(
    name="web_search",
    description="Search the web.",
    input_schema=dict,
    output_schema=dict,
    loop_key=lambda args: args["query"].strip().lower(),
    execute=lambda p, c: {"results": [f"result for {p['query']}"]},
)


def note_tool() -> Dict[str, Any]:
    return tool(name="note", input_schema=dict, output_schema=dict, execute=lambda p, c: {"ok": True})


# ---------------------------------------------------------------------------
# B3 -- parallel same-round duplicates
# ---------------------------------------------------------------------------


async def test_a_6_wide_identical_fan_out_executes_fully_with_zero_detections(script: Script) -> None:
    execute = Counter({"results": ["r"]})
    fanout_tool = tool(name="web_search", input_schema=dict, output_schema=dict, execute=execute)
    hooks = HooksManager()
    detections = collect(hooks)
    args = {"query": "same"}
    client = QueuedClient(
        [script.turn(*[script.call("web_search", args) for _ in range(6)]), script.text_turn("All six ran.")]
    )

    text = await call_model(
        client, {"model": "test-model", "input": "Fan out.", "tools": [fanout_tool], "doom_loop": True, "hooks": hooks}
    ).get_text()

    assert text == "All six ran."
    # One round = one increment (streak 1): everything executes, no verdicts.
    assert execute.count == 6
    assert detections == []


async def test_duplicates_of_a_call_blocked_this_round_share_the_block_decision(script: Script) -> None:
    execute = Counter({"ok": True})
    noisy = tool(name="noisy", input_schema=dict, output_schema=dict, execute=execute)
    hooks = HooksManager()
    detections = collect(hooks)
    client = QueuedClient(
        [
            script.tool_call_turn("noisy", {}),
            script.tool_call_turn("noisy", {}),
            script.turn(script.call("noisy", {}), script.call("noisy", {}), script.call("noisy", {})),
            script.text_turn("done"),
        ]
    )

    text = await call_model(
        client, {"model": "test-model", "input": "go", "tools": [noisy], "doom_loop": True, "hooks": hooks}
    ).get_text()

    assert text == "done"
    # Rounds 1+2 executed one call each; round 3's three duplicates all blocked.
    assert execute.count == 2
    assert actions_and_streaks(detections) == [["observe", 2], ["block", 3]]
    # Every duplicate received the shared block decision's output.
    round_3_outputs = outputs_in(client.requests[3]["input"])[-3:]
    assert [o["call_id"] for o in round_3_outputs] == ["call_3", "call_4", "call_5"]
    assert all("Doom loop suspected" in o["output"] for o in round_3_outputs)


# ---------------------------------------------------------------------------
# B1 + D7 -- state integrity and session reason on text-verdict stops
# ---------------------------------------------------------------------------


async def test_text_stop_on_a_response_that_also_carries_tool_calls_leaves_no_dangling_function_call(
    script: Script,
) -> None:
    accessor = MemoryStateAccessor()
    hooks = HooksManager()
    session_ends = collect(hooks, "SessionEnd")
    doom_text = " ".join(["I am stuck."] * 20)
    client = QueuedClient([script.turn(script.message(doom_text), script.call("web_search", {"query": "never runs"}))])

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "go",
            "tools": [search_tool],
            "doom_loop": True,
            "state": accessor,
            "hooks": hooks,
        },
    ).get_response()

    # Only the initial request was made -- the loop halted before executing.
    assert len(client.requests) == 1
    assert len(session_ends) == 1
    assert session_ends[0]["reason"] == "doom_loop"

    state = accessor.stored
    assert state is not None
    assert_well_formed_history(state.messages)
    outputs = outputs_in(state.messages)
    assert len(outputs) == 1
    assert "halted by doom-loop detection" in outputs[0]["output"]


async def test_a_resumed_conversation_after_a_sealed_text_stop_sends_well_formed_input(script: Script) -> None:
    accessor = MemoryStateAccessor()
    doom_text = " ".join(["I am stuck."] * 20)
    client = QueuedClient(
        [
            script.turn(script.message(doom_text), script.call("web_search", {"query": "never runs"})),
            script.text_turn("resumed fine"),
        ]
    )
    await call_model(
        client, {"model": "test-model", "input": "go", "tools": [search_tool], "doom_loop": True, "state": accessor}
    ).get_response()

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": "continue please",
            "tools": [search_tool],
            "doom_loop": True,
            "state": accessor,
        },
    ).get_text()

    assert text == "resumed fine"
    assert len(client.requests) == 2
    assert_well_formed_history(client.requests[1]["input"])


async def test_text_only_no_tools_stop_reports_session_end_reason_doom_loop(script: Script) -> None:
    hooks = HooksManager()
    session_ends = collect(hooks, "SessionEnd")
    doom_text = " ".join(["I am stuck."] * 20)
    client = QueuedClient([script.text_turn(doom_text)])

    result = call_model(client, {"model": "test-model", "input": "Summarize.", "doom_loop": True, "hooks": hooks})
    text = await result.get_text()

    assert text == doom_text
    assert len(session_ends) == 1
    assert session_ends[0]["reason"] == "doom_loop"
    verdict = await result.get_doom_loop_verdict()
    assert verdict is not None
    assert (verdict["detector"], verdict["action"]) == ("text-repetition", "stop")


# ---------------------------------------------------------------------------
# D1 -- allow_final_response gating
# ---------------------------------------------------------------------------


async def test_stop_when_plus_doom_stop_during_the_halted_turn_skips_the_final_response_request(
    script: Script,
) -> None:
    client = QueuedClient(
        [
            script.tool_call_turn("web_search", {"query": "same"}),
            script.tool_call_turn("web_search", {"query": "same"}),
            # A third scripted turn exists; it must NEVER be requested.
            script.text_turn("should never be fetched"),
        ]
    )

    result = call_model(
        client,
        {
            "model": "test-model",
            "input": "go",
            "tools": [search_tool],
            "doom_loop": {"ladder": {"observe": 2, "block": 2, "stop": 2}},
            "stop_when": lambda ctx: len(ctx["steps"]) >= 1,
            "allow_final_response": True,
        },
    )
    await result.get_response()

    # Initial + the tool-round follow-up that stop_when then halts. The final
    # text-coercion request is SKIPPED because the doom stop armed while
    # executing the halted turn's calls.
    assert len(client.requests) == 2
    verdict = await result.get_doom_loop_verdict()
    assert verdict is not None
    assert (verdict["action"], verdict["streak"]) == ("stop", 2)


# ---------------------------------------------------------------------------
# D2 + D3 -- approval resume gating, verdict persistence and clearing
# ---------------------------------------------------------------------------

approval_tool = tool(
    name="risky",
    input_schema=dict,
    output_schema=dict,
    require_approval=True,
    execute=lambda p, c: {"done": True},
)
APPROVAL_LADDER: Dict[str, Any] = {"ladder": {"observe": 1, "block": False, "stop": 2}}


async def test_doom_stop_armed_on_the_approval_continuation_makes_no_further_request(script: Script) -> None:
    accessor = MemoryStateAccessor()
    client = QueuedClient(
        [
            script.tool_call_turn("risky", {"target": "same"}),
            script.tool_call_turn("risky", {"target": "same"}),
        ]
    )

    # Run 1: risky(same) -> pauses for approval. The checkpoint runs BEFORE
    # approval partitioning, so the paused call is already recorded (streak 1).
    run1 = call_model(
        client,
        {
            "model": "test-model",
            "input": "do the risky thing",
            "tools": [approval_tool],
            "doom_loop": APPROVAL_LADDER,
            "tool_choice": "required",
            "state": accessor,
        },
    )
    await run1.get_response()
    paused = accessor.stored
    assert paused is not None
    assert paused.status == "awaiting_approval"
    assert paused.pending_tool_calls is not None
    paused_call_id = paused.pending_tool_calls[0].id
    assert len(client.requests) == 1
    assert paused.doom_loop is not None
    assert paused.doom_loop["tools"]["risky"]["streak"] == 1

    # Resume: approve -> executes, unsent-results request runs, the model
    # calls the same tool again -> the checkpoint crosses the stop rung before
    # a second approval pause or PreToolUse can begin.
    resume1 = call_model(
        client,
        {
            "model": "test-model",
            "tools": [approval_tool],
            "doom_loop": APPROVAL_LADDER,
            "tool_choice": "required",
            "state": accessor,
            "approve_tool_calls": [paused_call_id],
        },
    )
    requests_before_resume = len(client.requests)
    try:
        await resume1.get_response()
    except Exception:  # noqa: BLE001 - upstream `.catch(() => undefined)`
        pass

    verdict = await resume1.get_doom_loop_verdict()
    assert verdict is not None
    assert (verdict["action"], verdict["streak"], verdict["tool_name"]) == ("stop", 2, "risky")
    # No additional model request after the doom stop.
    assert len(client.requests) == requests_before_resume + 1

    final_state = accessor.stored
    assert final_state is not None
    assert final_state.doom_loop is not None
    stop_verdict = final_state.doom_loop["stop_verdict"]
    assert (stop_verdict["action"], stop_verdict["streak"]) == ("stop", 2)
    # Doom stop is terminal: a later fresh run must not inherit the consumed
    # forced-choice key from the approval continuation.
    assert final_state.consumed_forced_tool_choice_key is None
    # The condemned call was never paused for approval again.
    assert final_state.status == "complete"
    assert_well_formed_history(final_state.messages)


async def test_a_doom_blocked_sibling_of_an_approval_pause_is_persisted_as_a_rejected_unsent_result(
    script: Script,
) -> None:
    """Port-added: pins upstream handleApprovalCheck's `executableNow = [...,
    ...blockedCalls]` -- a call refused by the doom checkpoint in a response
    that pauses for approval is neither executed nor dropped: it is persisted
    as a rejected unsent result carrying the verdict message, so the resume
    sends a well-formed history."""
    accessor = MemoryStateAccessor()
    execute = Counter({"results": ["r"]})
    counted_search = tool(name="web_search", input_schema=dict, output_schema=dict, execute=execute)
    client = QueuedClient(
        [
            script.tool_call_turn("web_search", {"query": "same"}),
            script.turn(script.call("web_search", {"query": "same"}), script.call("risky", {"target": "t"})),
        ]
    )

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "go",
            "tools": [counted_search, approval_tool],
            "doom_loop": {"ladder": {"observe": 1, "block": 2, "stop": 6}},
            "state": accessor,
        },
    ).get_response()

    state = accessor.stored
    assert state is not None
    assert state.status == "awaiting_approval"
    assert state.pending_tool_calls is not None
    assert [c.name for c in state.pending_tool_calls] == ["risky"]
    # Round 1 executed; the round-2 repeat was blocked, never executed.
    assert execute.count == 1
    assert state.unsent_tool_results is not None
    assert [(r.call_id, r.name) for r in state.unsent_tool_results] == [("call_2", "web_search")]
    assert "Doom loop suspected" in str(state.unsent_tool_results[0].error)
    assert len(client.requests) == 2


async def test_a_decision_only_resume_of_a_condemned_run_halts_without_a_model_request(script: Script) -> None:
    """Condemnation rule, decision-resume half (upstream processApprovalDecisions
    doom gate): a persisted stop verdict is re-armed on approve/reject resumes,
    so the approved call runs but no unsent-results request is made."""
    accessor = MemoryStateAccessor()
    hooks = HooksManager()
    session_ends = collect(hooks, "SessionEnd")
    client = QueuedClient([script.tool_call_turn("risky", {"target": "same"})])
    await call_model(
        client,
        {
            "model": "test-model",
            "input": "do it",
            "tools": [approval_tool],
            "doom_loop": True,
            "state": accessor,
        },
    ).get_response()
    paused = accessor.stored
    assert paused is not None
    assert paused.pending_tool_calls is not None
    paused_call_id = paused.pending_tool_calls[0].id
    condemned = {
        "detector": "tool-fingerprint",
        "action": "stop",
        "streak": 6,
        "fingerprint": "f" * 64,
        "tool_name": "risky",
        "message": "condemned",
    }
    accessor.stored = dataclasses.replace(paused, doom_loop={**(paused.doom_loop or {}), "stop_verdict": condemned})

    execute_calls = Counter({"done": True})
    counted = tool(name="risky", input_schema=dict, output_schema=dict, require_approval=True, execute=execute_calls)
    result = call_model(
        client,
        {
            "model": "test-model",
            "tools": [counted],
            "doom_loop": True,
            "state": accessor,
            "approve_tool_calls": [paused_call_id],
            "hooks": hooks,
        },
    )
    assert await result.get_doom_loop_verdict() == condemned
    with pytest.raises(RuntimeError, match="Response not available"):
        await result.get_response()

    # The approved call executed exactly once; no model request followed.
    assert execute_calls.count == 1
    assert len(client.requests) == 1
    state = accessor.stored
    assert state is not None
    assert state.status == "in_progress"
    assert state.unsent_tool_results is not None
    assert [r.call_id for r in state.unsent_tool_results] == [paused_call_id]
    assert state.doom_loop is not None
    assert state.doom_loop["stop_verdict"] == condemned
    assert [e["reason"] for e in session_ends] == ["doom_loop"]


async def test_a_fresh_conversational_turn_clears_the_persisted_stop_verdict_streaks_survive(
    script: Script,
) -> None:
    accessor = MemoryStateAccessor()
    client = QueuedClient(
        [
            script.tool_call_turn("web_search", {"query": "x"}),
            script.text_turn("paused here"),
            script.text_turn("recovered"),
        ]
    )
    await call_model(
        client, {"model": "test-model", "input": "go", "tools": [search_tool], "doom_loop": True, "state": accessor}
    ).get_text()
    base = accessor.stored
    assert base is not None
    # Inject a stop verdict as if the run had been condemned.
    accessor.stored = dataclasses.replace(
        base,
        doom_loop={
            **(base.doom_loop or {"tools": {}}),
            "stop_verdict": {
                "detector": "tool-fingerprint",
                "action": "stop",
                "streak": 6,
                "fingerprint": "f" * 64,
                "tool_name": "web_search",
                "message": "condemned",
            },
        },
    )

    # Fresh conversational turn (plain input, no decisions): the verdict clears.
    result = call_model(
        client,
        {
            "model": "test-model",
            "input": "try something new",
            "tools": [search_tool],
            "doom_loop": True,
            "state": accessor,
        },
    )
    assert await result.get_text() == "recovered"
    assert await result.get_doom_loop_verdict() is None
    state = accessor.stored
    assert state is not None
    assert state.doom_loop is not None
    # Cleared from persistence too (the next save omits it).
    assert "stop_verdict" not in state.doom_loop
    # Streaks survived the clear.
    assert state.doom_loop["tools"]["web_search"]["streak"] == 1


# ---------------------------------------------------------------------------
# D4 -- steer delivery
# ---------------------------------------------------------------------------

STEER_LADDER: Dict[str, Any] = {"ladder": {"observe": 2, "steer": 2, "block": False, "stop": 6}}


def _steer_messages(items: Any) -> List[Mapping[str, Any]]:
    return [
        item
        for item in items or []
        if isinstance(item, Mapping)
        and item.get("role") == "user"
        and isinstance(item.get("content"), str)
        and "Doom loop suspected" in item["content"]
    ]


async def test_injects_corrective_guidance_before_the_next_model_request(script: Script) -> None:
    client = QueuedClient(
        [
            script.tool_call_turn("web_search", {"query": "same"}),
            script.tool_call_turn("web_search", {"query": "same"}),
            script.text_turn("adjusted course"),
        ]
    )

    text = await call_model(
        client, {"model": "test-model", "input": "go", "tools": [search_tool], "doom_loop": STEER_LADDER}
    ).get_text()

    assert text == "adjusted course"
    # The steer verdict fired on round 2; the guidance is a user message in
    # the THIRD request's input -- and in no earlier one.
    assert len(_steer_messages(client.requests[2]["input"])) == 1
    assert _steer_messages(client.requests[1]["input"]) == []


async def test_steer_queued_at_a_hitl_pause_persists_in_pending_steer_and_delivers_on_resume(
    script: Script,
) -> None:
    accessor = MemoryStateAccessor()
    hitl_tool = tool(name="ask_human", input_schema=dict, output_schema=dict, on_tool_called=lambda p, c: None)
    steer_tool = tool(name="lookup", input_schema=dict, output_schema=dict, execute=lambda p, c: {"value": "v"})
    client = QueuedClient(
        [
            script.tool_call_turn("lookup", {"key": "same"}),
            script.turn(script.call("lookup", {"key": "same"}), script.call("ask_human", {"question": "help?"})),
            script.text_turn("thanks"),
        ]
    )

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "go",
            "tools": [steer_tool, hitl_tool],
            "doom_loop": STEER_LADDER,
            "state": accessor,
        },
    ).get_response()

    paused = accessor.stored
    assert paused is not None
    assert paused.status == "awaiting_hitl"
    assert len(client.requests) == 2
    # Not injected mid-pause (invalid history) but persisted for the resume.
    assert paused.doom_loop is not None
    pending = paused.doom_loop["pending_steer"]
    assert len(pending) == 1
    assert "Doom loop suspected" in pending[0]
    assert _steer_messages(paused.messages) == []

    # Upstream stops at the persisted assertion; this port also pins the
    # restore half: the queue is rehydrated on resume (and still persisted,
    # because the resumed run has no safe flush point before it completes).
    assert paused.pending_tool_calls is not None
    hitl_call_id = paused.pending_tool_calls[0].id
    resumed = call_model(
        client,
        {
            "model": "test-model",
            "input": [{"type": "function_call_output", "callId": hitl_call_id, "output": '{"answer": "yes"}'}],
            "tools": [steer_tool, hitl_tool],
            "doom_loop": STEER_LADDER,
            "state": accessor,
        },
    )
    assert await resumed.get_text() == "thanks"
    assert accessor.stored is not None
    assert accessor.stored.doom_loop is not None
    assert accessor.stored.doom_loop["pending_steer"] == pending


# ---------------------------------------------------------------------------
# D5 + API1 -- loop_key declaration forms
# ---------------------------------------------------------------------------


async def test_subset_loop_key_function_same_command_in_a_different_cwd_is_not_a_loop(script: Script) -> None:
    runs: List[str] = []

    def run(params: Any, ctx: Any) -> Dict[str, Any]:
        runs.append(f"{params['cwd']}:{params['command']}")
        return {"stdout": "ok"}

    bash_tool = tool(
        name="bash",
        input_schema=dict,
        output_schema=dict,
        loop_key=lambda args: {"command": args["command"], "cwd": args["cwd"]},
        execute=run,
    )
    client = QueuedClient(
        [
            script.tool_call_turn("bash", {"command": "ls", "cwd": "/a"}),
            script.tool_call_turn("bash", {"command": "ls", "cwd": "/b"}),
            script.text_turn("both listed"),
        ]
    )

    text = await call_model(
        client, {"model": "test-model", "input": "list", "tools": [bash_tool], "doom_loop": {"ladder": {"block": 2}}}
    ).get_text()

    assert text == "both listed"
    assert runs == ["/a:ls", "/b:ls"]


async def test_function_loop_key_collapses_excluded_field_variation(script: Script) -> None:
    bash_tool = tool(
        name="bash",
        input_schema=dict,
        output_schema=dict,
        loop_key=lambda args: {"command": args["command"], "cwd": args["cwd"]},
        execute=lambda p, c: {"stdout": "ok"},
    )
    hooks = HooksManager()
    detections = collect(hooks)
    client = QueuedClient(
        [
            script.tool_call_turn("bash", {"command": "ls", "cwd": "/a", "verbose": False}),
            script.tool_call_turn("bash", {"command": "ls", "cwd": "/a", "verbose": True}),
            script.text_turn("done"),
        ]
    )

    await call_model(
        client, {"model": "test-model", "input": "list", "tools": [bash_tool], "doom_loop": True, "hooks": hooks}
    ).get_text()

    assert actions_and_streaks(detections) == [["observe", 2]]


async def test_loop_key_false_statically_exempts_a_tool(script: Script) -> None:
    poll_tool = tool(
        name="check_status",
        input_schema=dict,
        output_schema=dict,
        loop_key=False,
        execute=lambda p, c: {"done": False},
    )
    hooks = HooksManager()
    detections = collect(hooks)
    client = QueuedClient(
        [
            *[script.tool_call_turn("check_status", {"jobId": "job-1"}) for _ in range(5)],
            script.text_turn("still running"),
        ]
    )

    text = await call_model(
        client, {"model": "test-model", "input": "wait", "tools": [poll_tool], "doom_loop": True, "hooks": hooks}
    ).get_text()

    assert text == "still running"
    assert detections == []
    assert len(client.requests) == 6


async def test_d5_loop_key_returning_nothing_never_collides_varying_args_onto_one_identity(script: Script) -> None:
    """Upstream: `loopKey: () => undefined` falls back to full args with a
    "returned undefined" warning. Python has no `undefined`: a bare `return`
    yields `None`, which this port defines as "exempt this call" (upstream's
    `null`; doom_loop.py module docstring). The invariant ported is the one
    the finding is about -- varying args never collide onto a constant
    identity, so no false streak -- not the warning text."""
    buggy_tool = tool(
        name="buggy",
        input_schema=dict,
        output_schema=dict,
        loop_key=lambda args: None,
        execute=lambda p, c: {"ok": True},
    )
    hooks = HooksManager()
    detections = collect(hooks)
    client = QueuedClient(
        [
            script.tool_call_turn("buggy", {"value": "a"}),
            script.tool_call_turn("buggy", {"value": "b"}),
            script.tool_call_turn("buggy", {"value": "c"}),
            script.text_turn("done"),
        ]
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        text = await call_model(
            client, {"model": "test-model", "input": "go", "tools": [buggy_tool], "doom_loop": True, "hooks": hooks}
        ).get_text()

    assert text == "done"
    assert detections == []
    # Divergence: exempt, so no fallback warning is emitted.
    assert [str(w.message) for w in caught if "DoomLoop" in str(w.message)] == []


# ---------------------------------------------------------------------------
# H1 -- documented miss: nonce evasion
# ---------------------------------------------------------------------------


async def test_a_model_inventing_a_fresh_nonce_each_turn_is_not_detected_without_a_loop_key(script: Script) -> None:
    nonce_tool = tool(
        name="lookup", input_schema=dict, output_schema=dict, execute=lambda p, c: {"value": "same answer every time"}
    )
    hooks = HooksManager()
    detections = collect(hooks)
    client = QueuedClient(
        [
            *[script.tool_call_turn("lookup", {"key": "same", "request_id": f"r{i}"}) for i in range(6)],
            script.text_turn("gave up"),
        ]
    )

    text = await call_model(
        client, {"model": "test-model", "input": "go", "tools": [nonce_tool], "doom_loop": True, "hooks": hooks}
    ).get_text()

    # The documented limit: locked so the gap is visible, not silent.
    assert text == "gave up"
    assert detections == []


async def test_the_same_loop_is_caught_when_the_tool_declares_a_loop_key_over_the_meaningful_fields(
    script: Script,
) -> None:
    nonce_tool = tool(
        name="lookup",
        input_schema=dict,
        output_schema=dict,
        loop_key=lambda args: {"key": args["key"]},
        execute=lambda p, c: {"value": "same"},
    )
    hooks = HooksManager()
    detections = collect(hooks)
    client = QueuedClient(
        [
            *[script.tool_call_turn("lookup", {"key": "same", "request_id": f"r{i}"}) for i in range(3)],
            script.text_turn("done"),
        ]
    )

    await call_model(
        client, {"model": "test-model", "input": "go", "tools": [nonce_tool], "doom_loop": True, "hooks": hooks}
    ).get_text()

    assert actions_and_streaks(detections) == [["observe", 2], ["block", 3]]


# ---------------------------------------------------------------------------
# H2 -- server-tool fingerprinting
# ---------------------------------------------------------------------------


def server_search_item(script: Script, query: str) -> Dict[str, Any]:
    script.calls += 1
    return {
        "type": "web_search_call",
        "id": f"ws_{script.calls}",
        "status": "completed",
        "action": {"type": "search", "query": query},
    }


async def test_repeated_identical_server_tool_calls_build_a_streak_and_can_stop_the_run(script: Script) -> None:
    hooks = HooksManager()
    detections = collect(hooks)
    client = QueuedClient(
        [
            *[script.turn(server_search_item(script, "same query"), script.call("note", {"n": n})) for n in (1, 2, 3)],
            script.text_turn("done"),
        ]
    )

    result = call_model(
        client,
        {
            "model": "test-model",
            "input": "search",
            "tools": [note_tool()],
            # block is meaningless for server tools -- downgrades to observe.
            "doom_loop": {"ladder": {"observe": 2, "block": 3, "stop": 3}},
            "hooks": hooks,
        },
    )
    await result.get_response()

    # Server streak: 1, 2 (observe), 3 (stop) -- exactly 3 requests.
    assert len(client.requests) == 3
    assert [d["detector"] for d in detections] == ["server-tool-fingerprint", "server-tool-fingerprint"]
    assert [d["action"] for d in detections] == ["observe", "stop"]
    verdict = await result.get_doom_loop_verdict()
    assert verdict is not None
    assert (verdict["detector"], verdict["action"], verdict["streak"], verdict["tool_name"]) == (
        "server-tool-fingerprint",
        "stop",
        3,
        "server:web_search_call",
    )


async def test_distinct_server_tool_queries_do_not_build_a_streak(script: Script) -> None:
    hooks = HooksManager()
    detections = collect(hooks)
    client = QueuedClient(
        [
            script.turn(server_search_item(script, "first"), script.call("note", {"n": 1})),
            script.turn(server_search_item(script, "second"), script.call("note", {"n": 2})),
            script.text_turn("done"),
        ]
    )

    text = await call_model(
        client, {"model": "test-model", "input": "search", "tools": [note_tool()], "doom_loop": True, "hooks": hooks}
    ).get_text()

    assert text == "done"
    assert detections == []


async def test_a_proto_field_on_a_server_tool_item_participates_in_the_fingerprint(script: Script) -> None:
    """Upstream guards a JS prototype-setter hazard. Python dict keys are inert
    data, so the invariant (an own `__proto__` field is part of the identity)
    holds structurally; ported to keep it pinned."""
    hooks = HooksManager()
    detections = collect(hooks)

    def proto_item(marker: str) -> Dict[str, Any]:
        script.calls += 1
        return {
            **json.loads(f'{{"__proto__":"{marker}"}}'),
            "type": "web_search_call",
            "id": f"ws_{script.calls}",
            "status": "completed",
            "action": {"type": "search", "query": "same query"},
        }

    client = QueuedClient(
        [
            *[script.turn(proto_item(marker), script.call("note", {"n": n})) for n, marker in enumerate("abc", 1)],
            script.text_turn("done"),
        ]
    )

    text = await call_model(
        client, {"model": "test-model", "input": "search", "tools": [note_tool()], "doom_loop": True, "hooks": hooks}
    ).get_text()

    assert text == "done"
    assert detections == []


# ---------------------------------------------------------------------------
# Config lifecycle across resumes
# ---------------------------------------------------------------------------


async def test_on_to_off_stale_doom_loop_blob_is_inert_off_run_repeats_freely(script: Script) -> None:
    accessor = MemoryStateAccessor()
    client = QueuedClient(
        [
            script.tool_call_turn("web_search", {"query": "x"}),
            script.tool_call_turn("web_search", {"query": "x"}),
            script.text_turn("run 1 done"),
            script.tool_call_turn("web_search", {"query": "x"}),
            script.tool_call_turn("web_search", {"query": "x"}),
            script.text_turn("run 2 done"),
        ]
    )
    await call_model(
        client, {"model": "test-model", "input": "go", "tools": [search_tool], "doom_loop": True, "state": accessor}
    ).get_text()
    assert accessor.stored is not None
    assert accessor.stored.doom_loop is not None
    assert accessor.stored.doom_loop["tools"]["web_search"]["streak"] == 2

    hooks = HooksManager()
    detections = collect(hooks)
    text = await call_model(
        client, {"model": "test-model", "input": "more", "tools": [search_tool], "state": accessor, "hooks": hooks}
    ).get_text()

    assert text == "run 2 done"
    assert detections == []
    assert len(client.requests) == 6
    for request_index in (4, 5):
        assert "Doom loop" not in outputs_in(client.requests[request_index]["input"])[-1]["output"]


async def test_on_to_on_streaks_carry_across_the_resume(script: Script) -> None:
    accessor = MemoryStateAccessor()
    client = QueuedClient(
        [
            script.tool_call_turn("web_search", {"query": "persistent"}),
            script.tool_call_turn("web_search", {"query": "persistent"}),
            script.text_turn("pausing"),
            script.tool_call_turn("web_search", {"query": "persistent"}),
            script.text_turn("recovered"),
        ]
    )
    await call_model(
        client, {"model": "test-model", "input": "go", "tools": [search_tool], "doom_loop": True, "state": accessor}
    ).get_text()

    hooks = HooksManager()
    detections = collect(hooks)
    await call_model(
        client,
        {
            "model": "test-model",
            "input": "continue",
            "tools": [search_tool],
            "doom_loop": True,
            "state": accessor,
            "hooks": hooks,
        },
    ).get_text()

    assert actions_and_streaks(detections) == [["block", 3]]
