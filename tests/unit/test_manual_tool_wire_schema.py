"""Ports tests/unit/manual-tool-wire-schema.test.ts.

`tool(..., wire_input_schema=...)` is a caller-owned JSON Schema sent to the
model verbatim (minus `~`-prefixed metadata keys) in place of the schema
generated from `input_schema`; `input_schema` keeps validating the call.

The type-level companion (`manual-tool-wire-schema.test-d.ts`) has no runtime
counterpart: Python cannot forbid `wire_input_schema` on executable tools at the
type level (contract, Idiomatic Divergences #4).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from pydantic import BaseModel

from openrouter_agent import call_model, tool
from openrouter_agent.tool_executor import convert_tools_to_api_format, execute_tool
from openrouter_agent.tool_types import ParsedToolCall
from tests._fixtures import QueuedClient, text_response


class ChoiceInput(BaseModel):
    choice: str


class ValueInput(BaseModel):
    value: str


class CountInput(BaseModel):
    count: float


def test_serializes_any_of_and_one_of_while_sanitizing_tilde_prefixed_keys() -> None:
    wire_input_schema: Dict[str, Any] = {
        "type": "object",
        "~rootMetadata": "remove me",
        "properties": {
            "choice": {
                "~propertyMetadata": "remove me",
                "anyOf": [{"type": "string"}, {"type": "number"}],
                "oneOf": [{"const": "first"}, {"const": "second"}],
            },
        },
        "required": ["choice"],
    }
    manual_tool = tool(
        name="choose_value",
        description="Choose a value",
        input_schema=ChoiceInput,
        wire_input_schema=wire_input_schema,
        execute=False,
        strict=True,
    )

    api = convert_tools_to_api_format([manual_tool])

    assert len(api) == 1
    emitted = api[0]
    assert emitted["type"] == "function"
    assert emitted["name"] == "choose_value"
    assert emitted["description"] == "Choose a value"
    assert emitted["strict"] is True
    assert emitted["parameters"] == {
        "type": "object",
        "properties": {
            "choice": {
                "anyOf": [{"type": "string"}, {"type": "number"}],
                "oneOf": [{"const": "first"}, {"const": "second"}],
            },
        },
        "required": ["choice"],
    }


def test_does_not_mutate_the_caller_owned_schema_and_emits_a_copy() -> None:
    wire_input_schema: Dict[str, Any] = {
        "type": "object",
        "~rootMetadata": True,
        "properties": {"value": {"type": "string", "~nestedMetadata": True}},
    }
    original_schema = copy.deepcopy(wire_input_schema)
    manual_tool = tool(
        name="copy_schema",
        input_schema=ValueInput,
        wire_input_schema=wire_input_schema,
        execute=False,
    )

    api = convert_tools_to_api_format([manual_tool])
    parameters = api[0]["parameters"]

    assert wire_input_schema == original_schema
    assert parameters is not wire_input_schema
    assert parameters["properties"] is not wire_input_schema["properties"]
    assert parameters == {"type": "object", "properties": {"value": {"type": "string"}}}


def test_falls_back_to_the_model_derived_schema_without_wire_input_schema() -> None:
    manual_tool = tool(name="fallback_schema", input_schema=ValueInput, execute=False)

    parameters = convert_tools_to_api_format([manual_tool])[0]["parameters"]

    assert parameters["type"] == "object"
    assert parameters["properties"]["value"]["type"] == "string"
    assert parameters["required"] == ["value"]


def test_uses_the_model_derived_schema_for_executable_shared_context_tools() -> None:
    def execute(params: Any, context: Any) -> float:
        _ = context["shared"].get("sessionId")
        return float(params.count)

    executable_tool = tool(name="shared_context_tool", input_schema=CountInput, execute=execute)

    parameters = convert_tools_to_api_format([executable_tool])[0]["parameters"]

    assert parameters["type"] == "object"
    assert parameters["properties"]["count"]["type"] == "number"


async def test_input_schema_still_validates_the_call_when_a_wire_schema_is_set() -> None:
    # Python-side invariant from the task contract: the wire schema replaces
    # only what the model sees; execution still validates against
    # `input_schema`. (Upstream's types forbid wireInputSchema on executable
    # tools, but the runtime attaches it whenever provided -- tool.ts:687.)
    calls: List[Any] = []

    def execute(params: Any, context: Any) -> str:
        calls.append(params)
        return params.value

    wired = tool(
        name="wired",
        input_schema=ValueInput,
        wire_input_schema={"type": "object", "properties": {"anything": {}}},
        execute=execute,
    )

    bad = await execute_tool(wired, ParsedToolCall(id="c1", name="wired", arguments={"anything": 1}))
    good = await execute_tool(wired, ParsedToolCall(id="c2", name="wired", arguments={"value": "ok"}))

    assert bad is not None and good is not None
    assert bad["result"] is None
    assert bad["error"] is not None
    assert good.get("error") is None
    assert good["result"] == "ok"
    assert len(calls) == 1


async def test_call_model_sends_the_sanitized_wire_schema_on_the_request() -> None:
    client = QueuedClient([text_response("resp_1", "done")])
    manual_tool = tool(
        name="choose_value",
        input_schema=ChoiceInput,
        wire_input_schema={"type": "object", "~meta": 1, "properties": {"choice": {"type": "string"}}},
        execute=False,
    )

    await call_model(client, {"model": "m", "input": "hi", "tools": [manual_tool]}).get_text()

    assert len(client.requests) == 1
    sent_tools = client.requests[0]["tools"]
    assert len(sent_tools) == 1
    assert sent_tools[0]["parameters"] == {"type": "object", "properties": {"choice": {"type": "string"}}}
