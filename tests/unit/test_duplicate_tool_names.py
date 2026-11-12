from __future__ import annotations

from typing import Any, Dict, List

import pytest

from openrouter_agent import call_model, tool
from openrouter_agent.model_result import ModelResult
from tests._fixtures import MemoryStateAccessor, QueuedClient, text_response, tool_call_response


def _tool(name: str, hits: List[Dict[str, Any]], **kwargs: Any) -> Any:
    def execute(params: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
        hits.append(params)
        return {"ok": True}

    return tool(name=name, input_schema=dict, output_schema=dict, execute=execute, **kwargs)


def test_rejects_two_client_tools_with_the_same_name() -> None:
    first = _tool("read_value", [])
    second = _tool("read_value", [])

    with pytest.raises(ValueError, match='Duplicate tool name "read_value"'):
        ModelResult({"model": "test", "input": "hi", "tools": [first, second]})
    with pytest.raises(ValueError, match='Duplicate tool name "read_value"'):
        call_model(QueuedClient([]), {"model": "test", "input": "hi", "tools": [first, second]})


def test_allows_distinct_names() -> None:
    ModelResult({"model": "test", "input": "hi", "tools": [_tool("a", []), _tool("b", [])]})


async def test_approved_call_does_not_run_on_a_same_named_tool_added_before_resume() -> None:
    state = MemoryStateAccessor()
    approved_hits: List[Dict[str, Any]] = []
    other_hits: List[Dict[str, Any]] = []
    approved = _tool("read_value", approved_hits, require_approval=True)
    other = _tool("read_value", other_hits)

    first = call_model(
        QueuedClient([tool_call_response("resp_1", "read_value", call_id="call_1")]),
        {"model": "test", "input": "read it", "tools": [approved], "state": state},
    )
    await first.get_response()
    assert state.stored is not None
    assert state.stored.status == "awaiting_approval"

    with pytest.raises(ValueError, match="Duplicate tool name"):
        call_model(
            QueuedClient([text_response("resp_done", "done")]),
            {
                "model": "test",
                "input": [],
                "tools": [other, approved],
                "state": state,
                "approve_tool_calls": ["call_1"],
            },
        )

    assert other_hits == []
    assert approved_hits == []
    assert state.stored.status == "awaiting_approval"
