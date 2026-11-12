"""Integration tests for doom-loop detection in the call_model tool loop.

Ports `packages/agent/tests/unit/doom-loop-integration.test.ts`.

No live model anywhere: the transport is a `QueuedClient` playing scripted
responses that SIMULATE an LLM stuck in a doom loop -- the same tool call with
identical arguments turn after turn, repeated EMPTY tool calls, repeated
unparseable (invalid-JSON) calls, and degenerate repeated-token text. Detection
is deterministic, so each scenario asserts the exact turn on which the ladder
fires and exactly how the engine responds (hook emission, blocked tool output,
halted loop, persisted state).

Port notes:
- Hook payloads are snake_case (`tool_name`, `tool_input`), per hooks_schemas.py.
- Outgoing `function_call_output` items carry snake_case `call_id`:
  `ModelResult._send` normalizes at the transport boundary. Deliberate.
- Python has no `undefined`: a `loop_key` callable returning `None` exempts the
  call (upstream's `null`), see doom_loop.py's module docstring.
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Dict, Iterator, List, Mapping

import pytest

from openrouter_agent import HookEntry, HooksManager, call_model, resolve_loop_key_material, tool
from tests._fixtures import MemoryStateAccessor, QueuedClient, function_call_item, make_response, message_item

#: Expected detector diagnostics (ladder hazards, unhashable key material) are
#: asserted where they matter; silence them elsewhere.
pytestmark = pytest.mark.filterwarnings("ignore:\\[DoomLoop\\]:UserWarning")

# ---------------------------------------------------------------------------
# Scripted-response fixtures (the "simulated LLM"), built on tests/_fixtures.py
# ---------------------------------------------------------------------------


class Script:
    """Numbers responses `resp_N` and calls `call_N` like upstream's counters."""

    def __init__(self) -> None:
        self.responses = 0
        self.calls = 0

    def response(self, output: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.responses += 1
        return make_response(f"resp_{self.responses}", output)

    def call(self, name: str, args: Any) -> Dict[str, Any]:
        self.calls += 1
        return function_call_item(f"call_{self.calls}", name, json.dumps(args))

    def message(self, text: str) -> Dict[str, Any]:
        self.calls += 1
        return message_item(text, item_id=f"msg_{self.calls}")

    def tool_call_turn(self, name: str, args: Any) -> Dict[str, Any]:
        return self.response([self.call(name, args)])

    def invalid_json_turn(self, name: str, raw_arguments: str) -> Dict[str, Any]:
        self.calls += 1
        return self.response([function_call_item(f"call_{self.calls}", name, raw_arguments)])

    def text_turn(self, text: str) -> Dict[str, Any]:
        return self.response([self.message(text)])

    def fan_out_turn(self, name: str, args_list: List[Any]) -> Dict[str, Any]:
        return self.response([self.call(name, args) for args in args_list])


def sent_tool_outputs(client: QueuedClient, request_index: int) -> List[Mapping[str, Any]]:
    """The function_call_output items the engine sent in request `request_index`."""
    request = client.requests[request_index]
    items = request.get("input") if isinstance(request.get("input"), list) else []
    return [item for item in items or [] if isinstance(item, Mapping) and item.get("type") == "function_call_output"]


def collect(hooks: HooksManager, hook_name: str) -> List[Dict[str, Any]]:
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

    def __call__(self, params: Any, ctx: Any = None) -> Any:
        self.count += 1
        return self.result


def _search_execute(params: Any, ctx: Any) -> Dict[str, Any]:
    return {"results": [f"result for {params['query']}"]}


# Doom-loop identity: the query only -- "a web-search tool hashes its query".
search_tool = tool(
    name="web_search",
    description="Search the web.",
    input_schema=dict,
    output_schema=dict,
    loop_key=lambda args: args["query"].strip().lower(),
    execute=_search_execute,
)


@pytest.fixture()
def script() -> Iterator[Script]:
    yield Script()


# ---------------------------------------------------------------------------
# Scenario 1: identical tool calls, turn after turn
# ---------------------------------------------------------------------------


async def test_fan_out_escalates_observe_to_block_through_call_model_and_stops_the_whole_fan_out(
    script: Script,
) -> None:
    executed: List[str] = []

    def read(params: Any, ctx: Any) -> Dict[str, Any]:
        executed.append(params["path"])
        return {"content": f"contents of {params['path']}"}

    read_tool = tool(name="read", input_schema=dict, output_schema=dict, execute=read)
    hooks = HooksManager()
    detections = collect(hooks, "DoomLoopDetected")

    paths = [{"path": "a"}, {"path": "b"}, {"path": "c"}]
    client = QueuedClient(
        [
            script.fan_out_turn("read", paths),
            script.fan_out_turn("read", paths),
            script.fan_out_turn("read", paths),
            script.fan_out_turn("read", paths),
            script.text_turn("recovered"),
        ]
    )

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": "Summarize these files.",
            "tools": [read_tool],
            "doom_loop": True,
            "hooks": hooks,
        },
    ).get_text()

    assert text == "recovered"
    # Round 2 observes all three calls; rounds 3-4 block all three. The block
    # rung stops the WHOLE fan-out spending: only rounds 1-2 execute.
    assert actions_and_streaks(detections) == [
        ["observe", 2],
        ["observe", 2],
        ["observe", 2],
        ["block", 3],
        ["block", 3],
        ["block", 3],
        ["block", 4],
        ["block", 4],
        ["block", 4],
    ]
    assert executed == ["a", "b", "c", "a", "b", "c"]
    assert len(client.requests) == 5
    # Round verdicts (not per-call) name the set: pins the round declaration.
    for detection in detections:
        assert "same set of 3 parallel calls" in detection["message"]


async def test_a_repeat_sharing_its_id_with_a_varying_sibling_is_still_detected(script: Script) -> None:
    read_tool = tool(name="read", input_schema=dict, output_schema=dict, execute=lambda p, c: {"content": p["path"]})
    hooks = HooksManager()
    detections = collect(hooks, "DoomLoopDetected")

    def dup_id_turn(round_index: int) -> Dict[str, Any]:
        items = []
        for index, args in enumerate([{"path": "a"}, {"path": f"b{round_index}"}]):
            item = function_call_item("call_X", "read", json.dumps(args))
            item["id"] = f"fc_{round_index}_{index}"
            items.append(item)
        return script.response(items)

    client = QueuedClient([dup_id_turn(0), dup_id_turn(1), dup_id_turn(2), dup_id_turn(3), script.text_turn("done")])

    try:
        await call_model(
            client, {"model": "test-model", "input": "go", "tools": [read_tool], "doom_loop": True, "hooks": hooks}
        ).get_text()
    except Exception:  # noqa: BLE001 - upstream `.catch(() => undefined)`: only detection is asserted
        pass

    # The repeated `a` accumulates despite the id collision (upstream asserts
    # membership of 'observe' and 'block'; the exact sequence is pinned here):
    # only `a` repeats, so it observes at 2 and blocks at 3 and 4.
    assert [d["action"] for d in detections] == ["observe", "block", "block"]
    assert all(d["tool_input"] == {"path": "a"} for d in detections)


async def test_observes_at_2_blocks_at_3_with_an_explanatory_tool_error_and_lets_the_model_recover(
    script: Script,
) -> None:
    execute = Counter(None)

    def spied(params: Any, ctx: Any) -> Dict[str, Any]:
        execute(params)
        return {"results": [f"result for {params['query']}"]}

    spied_search = tool(
        name="web_search",
        input_schema=dict,
        output_schema=dict,
        loop_key=lambda args: args["query"].strip().lower(),
        execute=spied,
    )
    hooks = HooksManager()
    detections = collect(hooks, "DoomLoopDetected")

    # Three identical searches (case/whitespace vary -- loop_key collapses them).
    client = QueuedClient(
        [
            script.tool_call_turn("web_search", {"query": "openrouter sdk"}),
            script.tool_call_turn("web_search", {"query": "OpenRouter SDK"}),
            script.tool_call_turn("web_search", {"query": "  openrouter sdk  "}),
            script.text_turn("Giving up on repeating; here is my answer."),
        ]
    )

    text = await call_model(
        client,
        {"model": "test-model", "input": "Find the SDK.", "tools": [spied_search], "doom_loop": True, "hooks": hooks},
    ).get_text()

    assert text == "Giving up on repeating; here is my answer."
    # Calls 1 and 2 executed; call 3 was BLOCKED before execution.
    assert execute.count == 2
    assert actions_and_streaks(detections) == [["observe", 2], ["block", 3]]
    assert detections[0]["detector"] == "tool-fingerprint"
    assert detections[0]["tool_name"] == "web_search"

    # Request 4 carries the third call's (blocked) function_call_output.
    outputs = sent_tool_outputs(client, 3)
    blocked = outputs[-1]
    assert blocked["call_id"] == "call_3"
    assert "Doom loop suspected" in blocked["output"]
    assert "web_search" in blocked["output"]
    assert "3 consecutive rounds" in blocked["output"]


async def test_stops_the_run_at_the_stop_threshold_instead_of_spending_forever(script: Script) -> None:
    client = QueuedClient([script.tool_call_turn("web_search", {"query": "same query"}) for _ in range(10)])

    result = call_model(
        client,
        {
            "model": "test-model",
            "input": "Find it.",
            "tools": [search_tool],
            "doom_loop": {"ladder": {"observe": 2, "block": 3, "stop": 4}},
        },
    )
    response = await result.get_response()

    # Streaks: 1 (executes), 2 (observe), 3 (block), 4 (stop). The loop halts
    # before requesting a 5th turn.
    assert len(client.requests) == 4
    verdict = await result.get_doom_loop_verdict()
    assert verdict is not None
    assert verdict["action"] == "stop"
    assert verdict["detector"] == "tool-fingerprint"
    assert verdict["streak"] == 4
    assert verdict["tool_name"] == "web_search"
    # The run surfaces the last response rather than throwing.
    assert response["id"] == "resp_4"


async def test_emits_session_end_with_reason_doom_loop_when_stopped(script: Script) -> None:
    hooks = HooksManager()
    session_ends = collect(hooks, "SessionEnd")
    client = QueuedClient([script.tool_call_turn("web_search", {"query": "x"}) for _ in range(6)])

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "go",
            "tools": [search_tool],
            "doom_loop": {"ladder": {"stop": 3}},
            "hooks": hooks,
        },
    ).get_response()

    assert len(session_ends) == 1
    assert session_ends[0]["reason"] == "doom_loop"
    assert len(client.requests) == 3


async def test_does_not_interfere_with_a_healthy_run(script: Script) -> None:
    client = QueuedClient(
        [
            script.tool_call_turn("web_search", {"query": "first"}),
            script.tool_call_turn("web_search", {"query": "second"}),
            script.tool_call_turn("web_search", {"query": "third"}),
            script.text_turn("All three found."),
        ]
    )
    hooks = HooksManager()
    detections = collect(hooks, "DoomLoopDetected")

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": "Search three things.",
            "tools": [search_tool],
            "doom_loop": True,
            "hooks": hooks,
        },
    ).get_text()

    assert text == "All three found."
    assert detections == []
    assert len(client.requests) == 4


async def test_is_fully_inert_when_doom_loop_is_not_configured(script: Script) -> None:
    client = QueuedClient(
        [
            script.tool_call_turn("web_search", {"query": "same"}),
            script.tool_call_turn("web_search", {"query": "same"}),
            script.tool_call_turn("web_search", {"query": "same"}),
            script.tool_call_turn("web_search", {"query": "same"}),
            script.text_turn("done"),
        ]
    )
    accessor = MemoryStateAccessor()

    text = await call_model(
        client, {"model": "test-model", "input": "go", "tools": [search_tool], "state": accessor}
    ).get_text()

    # All four identical calls executed; nothing blocked, nothing stopped.
    assert text == "done"
    for request_index in (1, 2, 3, 4):
        outputs = sent_tool_outputs(client, request_index)
        assert "Doom loop" not in outputs[-1]["output"]
    # And nothing is persisted for a detector that is off.
    assert accessor.stored is not None
    assert accessor.stored.doom_loop is None


# ---------------------------------------------------------------------------
# Scenario 2: repeated EMPTY tool calls
# ---------------------------------------------------------------------------


async def test_treats_repeated_empty_calls_as_a_streak_and_blocks_at_the_threshold(script: Script) -> None:
    list_tool = tool(
        name="list_tasks",
        description="List everything.",
        input_schema=dict,
        output_schema=dict,
        execute=lambda p, c: {"tasks": ["task-1"]},
    )
    hooks = HooksManager()
    detections = collect(hooks, "DoomLoopDetected")
    client = QueuedClient(
        [
            script.tool_call_turn("list_tasks", {}),
            script.tool_call_turn("list_tasks", {}),
            script.tool_call_turn("list_tasks", {}),
            script.text_turn("The list has not changed; stopping."),
        ]
    )

    text = await call_model(
        client,
        {"model": "test-model", "input": "What tasks exist?", "tools": [list_tool], "doom_loop": True, "hooks": hooks},
    ).get_text()

    assert text == "The list has not changed; stopping."
    assert actions_and_streaks(detections) == [["observe", 2], ["block", 3]]
    assert "Doom loop suspected" in sent_tool_outputs(client, 3)[-1]["output"]


# ---------------------------------------------------------------------------
# Scenario 3: repeated malformed (invalid JSON) tool calls
# ---------------------------------------------------------------------------


async def test_counts_parse_error_repeats_and_stops_instead_of_bouncing_forever(script: Script) -> None:
    hooks = HooksManager()
    detections = collect(hooks, "DoomLoopDetected")
    client = QueuedClient([script.invalid_json_turn("web_search", '{"query": broken') for _ in range(4)])

    result = call_model(
        client,
        {
            "model": "test-model",
            "input": "Search.",
            "tools": [search_tool],
            "doom_loop": {"ladder": {"observe": 2, "block": 3, "stop": 3}},
            "hooks": hooks,
        },
    )
    await result.get_response()

    # Streak 2 observes; streak 3 crosses stop (stop wins over block).
    assert actions_and_streaks(detections) == [["observe", 2], ["stop", 3]]
    # Raw-string arguments are not a record: no tool_input on the payload.
    assert all(d["tool_input"] is None for d in detections)
    # Halted after the 3rd scripted turn -- the 4th was never requested.
    assert len(client.requests) == 3
    verdict = await result.get_doom_loop_verdict()
    assert verdict is not None
    assert (verdict["action"], verdict["streak"]) == ("stop", 3)


# ---------------------------------------------------------------------------
# Scenario 4: doom-loop TEXT tokens
# ---------------------------------------------------------------------------


async def test_stops_a_run_whose_response_is_one_degenerate_repeated_token_loop(script: Script) -> None:
    hooks = HooksManager()
    detections = collect(hooks, "DoomLoopDetected")
    doom_text = " ".join(["I need to check the file again."] * 40)
    client = QueuedClient([script.text_turn(doom_text)])

    result = call_model(client, {"model": "test-model", "input": "Summarize.", "doom_loop": True, "hooks": hooks})
    text = await result.get_text()

    # The tokens were already emitted -- the response is surfaced as-is -- but
    # the verdict fired deterministically with the repeat count.
    assert text == doom_text
    assert len(detections) == 1
    assert detections[0]["detector"] == "text-repetition"
    assert detections[0]["action"] == "stop"
    assert detections[0]["streak"] == 40


async def test_halts_a_tool_loop_when_every_turn_repeats_identical_filler_text(script: Script) -> None:
    filler = "Hmm, let me look somewhere else."

    def text_and_call_turn(query: str) -> Dict[str, Any]:
        return script.response([script.message(filler), script.call("web_search", {"query": query})])

    client = QueuedClient([text_and_call_turn(f"q{i}") for i in range(1, 6)])

    result = call_model(
        client,
        {
            "model": "test-model",
            "input": "Find it.",
            "tools": [search_tool],
            "doom_loop": {"ladder": {"observe": 2, "block": 3, "stop": 3}, "text": True},
        },
    )
    await result.get_response()

    # Text streak: turn1=1, turn2=2 (observe), turn3=3 (stop): exactly 3 requests.
    assert len(client.requests) == 3
    verdict = await result.get_doom_loop_verdict()
    assert verdict is not None
    assert (verdict["detector"], verdict["action"], verdict["streak"]) == ("text-streak", "stop", 3)


# ---------------------------------------------------------------------------
# loop_key contract
# ---------------------------------------------------------------------------


async def test_bash_style_loop_key_same_command_in_a_different_cwd_is_not_a_loop(script: Script) -> None:
    runs: List[str] = []

    def run(params: Any, ctx: Any) -> Dict[str, Any]:
        runs.append(f"{params['cwd']}:{params['command']}")
        return {"stdout": f"ran {params['command']} in {params['cwd']}"}

    bash_tool = tool(
        name="bash",
        description="Run a command.",
        input_schema=dict,
        output_schema=dict,
        loop_key=lambda args: {"command": args["command"].strip(), "cwd": args["cwd"]},
        execute=run,
    )
    client = QueuedClient(
        [
            script.tool_call_turn("bash", {"command": "ls", "cwd": "/a"}),
            script.tool_call_turn("bash", {"command": "ls", "cwd": "/b"}),
            script.tool_call_turn("bash", {"command": "ls", "cwd": "/c"}),
            script.text_turn("Directory listings acquired."),
        ]
    )

    text = await call_model(
        client,
        {
            "model": "test-model",
            "input": "List all three directories.",
            "tools": [bash_tool],
            "doom_loop": {"ladder": {"block": 2}},  # aggressive: ANY repeat would block
        },
    ).get_text()

    assert text == "Directory listings acquired."
    assert runs == ["/a:ls", "/b:ls", "/c:ls"]


async def test_loop_key_returning_none_exempts_a_legitimately_repetitive_tool(script: Script) -> None:
    poll_tool = tool(
        name="check_status",
        description="Poll job status.",
        input_schema=dict,
        output_schema=dict,
        loop_key=lambda args: None,  # repetition is this tool's job
        execute=lambda p, c: {"done": False},
    )
    client = QueuedClient(
        [
            *[script.tool_call_turn("check_status", {"jobId": "job-1"}) for _ in range(5)],
            script.text_turn("Job still running after 5 polls."),
        ]
    )
    hooks = HooksManager()
    detections = collect(hooks, "DoomLoopDetected")

    text = await call_model(
        client,
        {"model": "test-model", "input": "Wait for the job.", "tools": [poll_tool], "doom_loop": True, "hooks": hooks},
    ).get_text()

    assert text == "Job still running after 5 polls."
    assert detections == []
    assert len(client.requests) == 6


async def test_a_raising_loop_key_falls_back_to_full_arguments_instead_of_failing_the_run(script: Script) -> None:
    def broken_key(args: Any) -> Any:
        raise RuntimeError("loopKey bug")

    broken_key_tool = tool(
        name="flaky", input_schema=dict, output_schema=dict, loop_key=broken_key, execute=lambda p, c: {"ok": True}
    )
    client = QueuedClient(
        [
            script.tool_call_turn("flaky", {"value": "same"}),
            script.tool_call_turn("flaky", {"value": "same"}),
            script.tool_call_turn("flaky", {"value": "same"}),
            script.text_turn("done"),
        ]
    )
    hooks = HooksManager()
    detections = collect(hooks, "DoomLoopDetected")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        text = await call_model(
            client,
            {"model": "test-model", "input": "go", "tools": [broken_key_tool], "doom_loop": True, "hooks": hooks},
        ).get_text()

    # Detection still worked via the fallback identity.
    assert text == "done"
    assert actions_and_streaks(detections) == [["observe", 2], ["block", 3]]
    fallback_warnings = [
        str(w.message)
        for w in caught
        if "loop_key threw (loopKey bug); falling back to full arguments" in str(w.message)
    ]
    # One per checkpointed call (the round declaration does not warn).
    assert len(fallback_warnings) == 3


async def test_resolve_loop_key_material_can_raise_so_its_callers_must_guard_it() -> None:
    """The field-list form reads the arguments (`field in args`, `args[field]`),
    so a hostile mapping escapes it uncaught. Pinned so a refactor cannot
    quietly remove the guards in `_begin_doom_loop_round` and the checkpoint.
    Python analog of upstream's throwing getter / proxy `has` trap."""

    class ThrowingGetter(Dict[str, Any]):
        def __getitem__(self, key: str) -> Any:
            raise RuntimeError("getter boom")

    throwing_getter = ThrowingGetter(command="ls")
    with pytest.raises(RuntimeError, match="getter boom"):
        resolve_loop_key_material(["command"], throwing_getter)

    class HostileContains(Dict[str, Any]):
        def __contains__(self, key: object) -> bool:
            raise RuntimeError("has boom")

    with pytest.raises(RuntimeError, match="has boom"):
        resolve_loop_key_material(["command"], HostileContains())


# ---------------------------------------------------------------------------
# DoomLoopDetected hook override_action
# ---------------------------------------------------------------------------


async def test_a_handler_can_de_escalate_a_block_to_observe(script: Script) -> None:
    execute = Counter({"ok": True})
    noisy = tool(name="noisy", input_schema=dict, output_schema=dict, execute=execute)
    hooks = HooksManager()
    hooks.on("DoomLoopDetected", HookEntry(handler=lambda p, c: {"override_action": "observe"}))
    client = QueuedClient(
        [
            script.tool_call_turn("noisy", {}),
            script.tool_call_turn("noisy", {}),
            script.tool_call_turn("noisy", {}),
            script.text_turn("done"),
        ]
    )

    text = await call_model(
        client, {"model": "test-model", "input": "go", "tools": [noisy], "doom_loop": True, "hooks": hooks}
    ).get_text()

    assert text == "done"
    # Without the override the 3rd call would be blocked.
    assert execute.count == 3


async def test_a_handler_can_escalate_an_observe_to_stop(script: Script) -> None:
    hooks = HooksManager()
    hooks.on("DoomLoopDetected", HookEntry(handler=lambda p, c: {"override_action": "stop"}))
    client = QueuedClient([script.tool_call_turn("web_search", {"query": "x"}) for _ in range(3)])

    result = call_model(
        client, {"model": "test-model", "input": "go", "tools": [search_tool], "doom_loop": True, "hooks": hooks}
    )
    await result.get_response()

    # Streak 2 (observe) escalated to stop: the loop halts after the 2nd turn.
    assert len(client.requests) == 2
    verdict = await result.get_doom_loop_verdict()
    assert verdict is not None
    assert verdict["streak"] == 2


# ---------------------------------------------------------------------------
# State persistence: a resumed doom loop is still a doom loop
# ---------------------------------------------------------------------------


async def test_persists_streaks_into_conversation_state_and_keeps_counting_across_a_resume(script: Script) -> None:
    accessor = MemoryStateAccessor()
    client = QueuedClient(
        [
            script.tool_call_turn("web_search", {"query": "persistent"}),
            script.tool_call_turn("web_search", {"query": "persistent"}),
            script.text_turn("pausing here"),
        ]
    )
    await call_model(
        client, {"model": "test-model", "input": "go", "tools": [search_tool], "doom_loop": True, "state": accessor}
    ).get_text()

    assert accessor.stored is not None
    assert accessor.stored.doom_loop is not None
    assert accessor.stored.doom_loop["tools"]["web_search"]["streak"] == 2

    # Run 2: ONE more identical call crosses the block rung -- the streak
    # survived the round-trip.
    hooks = HooksManager()
    detections = collect(hooks, "DoomLoopDetected")
    client2 = QueuedClient(
        [script.tool_call_turn("web_search", {"query": "persistent"}), script.text_turn("recovered")]
    )
    await call_model(
        client2,
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


async def test_doom_loop_state_round_trips_through_json_serialization(script: Script) -> None:
    from openrouter_agent import deserialize_conversation_state, serialize_conversation_state

    accessor = MemoryStateAccessor()
    client = QueuedClient(
        [
            script.tool_call_turn("web_search", {"query": "x"}),
            script.tool_call_turn("web_search", {"query": "x"}),
            script.text_turn("done"),
        ]
    )
    await call_model(
        client, {"model": "test-model", "input": "go", "tools": [search_tool], "doom_loop": True, "state": accessor}
    ).get_text()

    state = accessor.stored
    assert state is not None
    assert state.doom_loop is not None
    # Plain JSON: survives dumps/loads identically, and through the versioned
    # serialization contract.
    assert json.loads(json.dumps(state.doom_loop)) == state.doom_loop
    assert deserialize_conversation_state(serialize_conversation_state(state)).doom_loop == state.doom_loop


async def test_invokes_a_tool_supplied_loop_key_exactly_once_per_call(script: Script) -> None:
    calls: List[str] = []

    def loop_key(args: Any) -> str:
        calls.append(args["query"])
        return str(args["query"]).strip().lower()

    counted_search = tool(
        name="web_search", input_schema=dict, output_schema=dict, loop_key=loop_key, execute=_search_execute
    )
    client = QueuedClient(
        [
            script.tool_call_turn("web_search", {"query": "first"}),
            script.tool_call_turn("web_search", {"query": "second"}),
            script.text_turn("Done."),
        ]
    )

    await call_model(
        client, {"model": "test-model", "input": "Search twice.", "tools": [counted_search], "doom_loop": True}
    ).get_text()

    # Two calls issued, so exactly two invocations -- not four.
    assert calls == ["first", "second"]


async def test_never_invokes_loop_key_for_a_manual_call_the_detector_will_not_check(script: Script) -> None:
    manual_calls: List[Any] = []
    executable_calls: List[Any] = []

    def manual_key(args: Any) -> Any:
        manual_calls.append(args)
        return args["query"]

    def executable_key(args: Any) -> Any:
        executable_calls.append(args)
        return args["query"]

    manual_tool = tool(name="ask_human", input_schema=dict, output_schema=dict, loop_key=manual_key, execute=False)
    executable_tool = tool(
        name="web_search",
        input_schema=dict,
        output_schema=dict,
        loop_key=executable_key,
        execute=lambda p, c: {"results": [p["query"]]},
    )
    # MIXED batch: an all-manual round never reaches the doom-loop round.
    client = QueuedClient(
        [
            script.response(
                [script.call("web_search", {"query": "go"}), script.call("ask_human", {"query": "what now?"})]
            )
        ]
    )

    await call_model(
        client,
        {
            "model": "test-model",
            "input": "Search, and ask me something.",
            "tools": [executable_tool, manual_tool],
            "doom_loop": True,
        },
    ).get_text()

    assert len(executable_calls) == 1
    assert manual_calls == []
    assert len(client.requests) == 1
