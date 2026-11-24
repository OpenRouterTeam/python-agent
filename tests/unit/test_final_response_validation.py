"""Ported from upstream tests/unit/final-response-validation.test.ts (#95).

Diagnostics only: the historical message prefixes are preserved and detail is
appended. Like upstream, these call the (pure) validator directly.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from openrouter_agent import call_model
from openrouter_agent.model_result import ModelResult
from tests._fixtures import QueuedClient, function_call_item, make_response, message_item


def _validator() -> Any:
    return ModelResult(
        {"client": QueuedClient([]), "request": {"model": "m", "input": "hello"}}
    )._validate_final_response


def _response(**overrides: Any) -> Dict[str, Any]:
    response: Dict[str, Any] = {"id": "resp_1", "object": "response", "model": "m", "status": "completed", "output": []}
    response.update(overrides)
    return response


def _message(response: Any, allow_empty: bool = False) -> str:
    with pytest.raises(ValueError) as caught:
        _validator()(response, allow_empty)
    return str(caught.value)


def test_reports_the_empty_array_its_length_and_the_response_id() -> None:
    message = _message(_response())

    assert message.startswith("Invalid final response: empty or invalid output")
    assert 'output array is empty (length 0) for response "resp_1"' in message
    assert "strict_final_response/allow_final_response" in message
    assert "The model returned no output items" in message
    assert "\n" not in message


def test_does_not_raise_when_empty_output_is_explicitly_allowed() -> None:
    _validator()(_response(), True)


def test_names_the_offending_runtime_type_for_a_non_array_output() -> None:
    assert 'output is not an array (got string) for response "resp_1"' in _message(_response(output="oops"))
    object_message = _message(_response(output={"type": "message"}))
    assert "(got object)" in object_message
    assert "is empty" not in object_message


def test_non_array_output_is_still_tolerated_under_allow_empty_output() -> None:
    _validator()(_response(output="oops"), True)


def test_names_the_missing_required_fields() -> None:
    assert _message(_response(id=None)) == "Invalid final response: missing required fields: id"
    assert _message(_response(output=None)) == "Invalid final response: missing required fields: output"
    assert _message({}) == "Invalid final response: missing required fields: id, output"
    assert _message(None) == "Invalid final response: missing required fields: id, output"


def test_accepts_a_tool_call_only_output_and_a_normal_message_output() -> None:
    _validator()(make_response("r1", [function_call_item("call_1", "t")]))
    _validator()(make_response("r2", [message_item("hi")]))


async def test_an_empty_final_turn_with_no_tool_rounds_raises_the_diagnostic_through_call_model() -> None:
    with pytest.raises(ValueError, match=r'output array is empty \(length 0\) for response "r_empty"'):
        await call_model(QueuedClient([make_response("r_empty", [])]), {"model": "m", "input": "hi"}).get_response()
