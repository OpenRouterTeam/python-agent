"""Ported from upstream tests/unit/to-model-output-parity.test.ts (#155).

`to_model_output` receives the tool input after `input_schema` parsing (the
same value `execute` gets), and it runs for results produced on the
approval-resume path too.
"""

from __future__ import annotations

import json
from typing import Any, List, Mapping

from pydantic import BaseModel, field_validator

from openrouter_agent import call_model, tool
from tests._fixtures import (
    recording,
    MemoryStateAccessor,
    QueuedClient,
    function_call_item,
    make_response,
    text_response,
    tool_call_response,
)

SECRET = "sk-live-secret"
MASK = "[redacted]"


class _Lookup(BaseModel):
    access: str
    verbose: bool = False

    @field_validator("access")
    @classmethod
    def _lower(cls, value: str) -> str:
        return value.lower()


def _sent_tool_outputs(client: QueuedClient) -> List[Any]:
    # Outgoing items use snake_case `call_id` (ModelResult._send normalizes at
    # the transport boundary); only the `output` payload matters here.
    return [
        item["output"]
        for request in client.requests
        for item in request["input"]
        if isinstance(item, Mapping) and item.get("type") == "function_call_output"
    ]


async def test_passes_input_schema_normalized_arguments_to_to_model_output() -> None:
    execute_inputs: List[Any] = []
    mapper_inputs: List[Any] = []

    def execute(params: _Lookup, _ctx: Any) -> Any:
        execute_inputs.append(params)
        return {"secret": SECRET}

    def to_model_output(payload: Mapping[str, Any]) -> Any:
        mapper_inputs.append(payload["input"])
        text = MASK if payload["input"].access == "private" else payload["output"]["secret"]
        return {"type": "content", "value": [{"type": "input_text", "text": text}]}

    lookup = tool(name="lookup", input_schema=_Lookup, execute=execute, to_model_output=to_model_output)
    client = QueuedClient(
        [
            tool_call_response("resp_1", "lookup", call_id="call_1", arguments='{"access":"PRIVATE"}'),
            text_response("resp_2", "done"),
        ]
    )

    await call_model(client, {"model": "test-model", "input": "look", "tools": [lookup]}).get_response()

    parsed = _Lookup(access="private", verbose=False)
    assert execute_inputs == [parsed]
    # The mapper sees the same parsed value execute got, not the raw wire args.
    assert mapper_inputs == [parsed]
    assert SECRET not in json.dumps(_sent_tool_outputs(client))


async def test_maps_the_output_of_an_approved_call_before_sending_it_to_the_model() -> None:
    mapper_calls: List[Any] = []
    executed: List[Any] = []

    def to_model_output(payload: Mapping[str, Any]) -> Any:
        mapper_calls.append(payload)
        return {"type": "content", "value": [{"type": "input_text", "text": MASK}]}

    lookup = tool(
        name="lookup",
        input_schema=_Lookup,
        require_approval=True,
        execute=recording(executed, {"secret": SECRET}),
        to_model_output=to_model_output,
    )
    state = MemoryStateAccessor()
    await call_model(
        QueuedClient([tool_call_response("resp_1", "lookup", call_id="call_1", arguments='{"access":"PRIVATE"}')]),
        {"model": "test-model", "input": "look", "tools": [lookup], "state": state},
    ).get_response()
    assert state.stored is not None and state.stored.status == "awaiting_approval"
    assert executed == []

    client = QueuedClient([text_response("resp_2", "done")])
    await call_model(
        client,
        {"model": "test-model", "input": [], "tools": [lookup], "state": state, "approve_tool_calls": ["call_1"]},
    ).get_response()

    assert len(executed) == 1
    assert len(mapper_calls) == 1
    assert mapper_calls[0]["input"] == _Lookup(access="private")
    assert mapper_calls[0]["output"] == {"secret": SECRET}
    outputs = _sent_tool_outputs(client)
    assert outputs == [[{"type": "input_text", "text": MASK}]]
    assert SECRET not in json.dumps(outputs)


async def test_maps_auto_executed_siblings_persisted_before_an_approval_pause() -> None:
    # Unsent results recorded before a pause go through the same mapper, so
    # the resume request carries the mapped output, not the raw result.
    lookup = tool(
        name="lookup",
        input_schema=_Lookup,
        execute=lambda _params, _ctx: {"secret": SECRET},
        to_model_output=lambda _payload: {"type": "content", "value": [{"type": "input_text", "text": MASK}]},
    )
    gated = tool(name="gated", input_schema=dict, require_approval=True, execute=lambda _p, _c: {"ok": True})
    state = MemoryStateAccessor()
    await call_model(
        QueuedClient(
            [
                make_response(
                    "resp_1",
                    [
                        function_call_item("call_lookup", "lookup", '{"access":"x"}'),
                        function_call_item("call_gated", "gated", "{}"),
                    ],
                )
            ]
        ),
        {"model": "test-model", "input": "go", "tools": [lookup, gated], "state": state},
    ).get_response()
    assert state.stored is not None
    assert [(r.call_id, r.output) for r in state.stored.unsent_tool_results or []] == [
        ("call_lookup", [{"type": "input_text", "text": MASK}])
    ]
