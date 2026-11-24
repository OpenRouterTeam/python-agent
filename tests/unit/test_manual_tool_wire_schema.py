"""Ported from upstream tests/unit/manual-tool-wire-schema.test.ts (#118) plus
the MCP wire-schema rule from #146 (`mark_mcp(..., wire_input_schema=...)`)."""

from __future__ import annotations

import copy

import pytest
from pydantic import BaseModel

from openrouter_agent import tool
from openrouter_agent.tool import mark_mcp
from openrouter_agent.tool_executor import convert_tools_to_api_format


class _Choice(BaseModel):
    choice: str


class _Value(BaseModel):
    value: str


def test_serializes_any_of_and_one_of_while_sanitizing_tilde_prefixed_keys() -> None:
    wire_input_schema = {
        "type": "object",
        "~rootMetadata": "remove me",
        "properties": {
            "choice": {
                "~propertyMetadata": "remove me",
                "anyOf": [{"type": "string"}, {"type": "number"}],
                "oneOf": [{"const": "first"}, {"const": "second"}],
            }
        },
        "required": ["choice"],
    }
    manual = tool(
        name="choose_value",
        description="Choose a value",
        input_schema=_Choice,
        wire_input_schema=wire_input_schema,
        execute=False,
        strict=True,
    )

    emitted = convert_tools_to_api_format([manual])[0]

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
            }
        },
        "required": ["choice"],
    }


def test_does_not_mutate_the_caller_owned_schema_and_emits_a_copy() -> None:
    wire_input_schema = {
        "type": "object",
        "~rootMetadata": True,
        "properties": {"value": {"type": "string", "~nestedMetadata": True}},
    }
    original = copy.deepcopy(wire_input_schema)
    manual = tool(name="copy_schema", input_schema=_Value, wire_input_schema=wire_input_schema, execute=False)

    emitted = convert_tools_to_api_format([manual])[0]

    assert wire_input_schema == original
    assert emitted["parameters"] is not wire_input_schema


def test_falls_back_to_the_generated_schema_without_wire_input_schema() -> None:
    emitted = convert_tools_to_api_format([tool(name="fallback_schema", input_schema=_Value, execute=False)])[0]

    assert emitted["parameters"]["type"] == "object"
    assert emitted["parameters"]["properties"]["value"]["type"] == "string"


def test_wire_input_schema_is_rejected_on_executable_tools() -> None:
    # Upstream rejects this at the type level (`wireInputSchema?: never`).
    with pytest.raises(ValueError, match="only supported on manual tools"):
        tool(name="exec", input_schema=_Value, wire_input_schema={"type": "object"}, execute=lambda p, _c: p)


def test_mark_mcp_sends_the_servers_own_object_schema() -> None:
    server_schema = {
        "type": "object",
        "properties": {"when": {"type": "string", "format": "date-time"}},
        "required": ["when"],
    }
    wrapped = tool(name="schedule", input_schema=dict, execute=lambda p, _c: p)

    marked = mark_mcp(wrapped, wire_input_schema=server_schema, annotations={"readOnlyHint": True})

    assert convert_tools_to_api_format([marked])[0]["parameters"] == server_schema
    assert marked["annotations"] == {"readOnlyHint": True}
    # Non-mutating: the unbranded tool keeps its generated schema.
    assert "wire_input_schema" not in wrapped["function"]


def test_mark_mcp_keeps_the_generated_schema_for_a_non_object_root() -> None:
    wrapped = tool(name="weird", input_schema=_Value, execute=lambda p, _c: p)

    marked = mark_mcp(wrapped, wire_input_schema={"type": "string"})

    assert convert_tools_to_api_format([marked])[0]["parameters"]["type"] == "object"
    assert "annotations" not in marked
