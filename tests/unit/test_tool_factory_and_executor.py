from __future__ import annotations

from pydantic import BaseModel

from openrouter_agent import (
    has_execute_function,
    is_generator_tool,
    is_hitl_tool,
    is_manual_tool,
    is_server_tool,
    server_tool,
    tool,
)
from openrouter_agent.tool_executor import convert_tools_to_api_format, execute_tool, sanitize_json_schema
from openrouter_agent.tool_types import ParsedToolCall


class Input(BaseModel):
    value: int


class Output(BaseModel):
    doubled: int


async def test_regular_tool_executes_and_validates() -> None:
    created = tool(
        name="double",
        input_schema=Input,
        output_schema=Output,
        execute=lambda params, ctx: {"doubled": params.value * 2},
    )
    assert has_execute_function(created)

    result = await execute_tool(created, ParsedToolCall(id="call_1", name="double", arguments={"value": 3}))

    assert result is not None
    assert result["error"] is None if "error" in result else True
    assert result["result"].doubled == 6


def test_manual_hitl_server_and_json_schema_shapes() -> None:
    manual = tool(name="manual", input_schema=Input, execute=False)
    hitl = tool(name="approve", input_schema=Input, output_schema=Output, on_tool_called=lambda params, ctx: None)
    server = server_tool({"type": "web_search_2025_08_26", "max_results": 3})

    assert is_manual_tool(manual)
    assert is_hitl_tool(hitl)
    assert is_server_tool(server)
    api = convert_tools_to_api_format([manual, server])
    assert api[0]["type"] == "function"
    assert api[0]["parameters"]["type"] == "object"
    assert api[1]["type"] == "web_search_2025_08_26"


def test_schema_sanitization_removes_standard_schema_metadata() -> None:
    schema = {"type": "object", "~standard": {"vendor": "x"}, "properties": {"x": {"~meta": True, "type": "string"}}}
    assert sanitize_json_schema(schema) == {"type": "object", "properties": {"x": {"type": "string"}}}


async def test_generator_tool_uses_final_yield_as_output() -> None:
    async def execute(params: Input, ctx):
        yield {"progress": 0.5}
        yield {"doubled": params.value * 2}

    generated = tool(name="gen", input_schema=Input, event_schema=dict, output_schema=Output, execute=execute)
    prelim = []
    result = await execute_tool(
        generated,
        ParsedToolCall(id="call_2", name="gen", arguments={"value": 4}),
        on_preliminary_result=lambda call_id, value: prelim.append((call_id, value)),
    )

    assert is_generator_tool(generated)
    assert prelim == [("call_2", {"progress": 0.5})]
    assert result is not None
    assert result["result"].doubled == 8
    assert result["preliminary_results"] == [{"progress": 0.5}]


async def test_generator_tool_with_overlapping_dict_schemas_keeps_last_yield_as_output() -> None:
    async def execute(params, ctx):
        yield {"progress": 0.5}
        yield {"answer": params["value"] * 2}

    generated = tool(name="gen_dict", input_schema=dict, event_schema=dict, output_schema=dict, execute=execute)
    prelim = []

    result = await execute_tool(
        generated,
        ParsedToolCall(id="call_3", name="gen_dict", arguments={"value": 5}),
        on_preliminary_result=lambda call_id, value: prelim.append((call_id, value)),
    )

    assert prelim == [("call_3", {"progress": 0.5})]
    assert result is not None
    assert result["result"] == {"answer": 10}
