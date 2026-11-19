from __future__ import annotations

import json
from typing import Any, Dict

from openrouter_agent import call_model, tool
from tests._fixtures import QueuedClient, text_response, tool_call_response

PRIVATE = "private-value-must-not-reach-model"


def _lookup_tool(to_model_output: Any) -> Any:
    def execute(params: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
        return {"public_status": "ok", "private_token": PRIVATE}

    return tool(
        name="lookup",
        input_schema=dict,
        output_schema=dict,
        execute=execute,
        to_model_output=to_model_output,
    )


async def _followup_output(to_model_output: Any) -> Any:
    client = QueuedClient([tool_call_response("r1", "lookup"), text_response("r2", "done")])
    await call_model(
        client, {"model": "test-model", "input": "hi", "tools": [_lookup_tool(to_model_output)]}
    ).get_text()
    followup_input = client.requests[1]["input"]
    return next(i for i in followup_input if i.get("type") == "function_call_output")["output"]


async def test_json_to_model_output_replaces_raw_tool_result() -> None:
    def redact(args: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": "json", "value": {"public_status": args["output"]["public_status"]}}

    output = await _followup_output(redact)

    assert PRIVATE not in output
    assert json.loads(output) == {"public_status": "ok"}


async def test_async_json_to_model_output_replaces_raw_tool_result() -> None:
    async def redact(args: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": "json", "value": {"redacted": True}}

    output = await _followup_output(redact)

    assert PRIVATE not in output
    assert json.loads(output) == {"redacted": True}


async def test_content_to_model_output_still_replaces_raw_tool_result() -> None:
    content = [{"type": "input_text", "text": "summary only"}]

    output = await _followup_output(lambda args: {"type": "content", "value": content})

    assert output == content
