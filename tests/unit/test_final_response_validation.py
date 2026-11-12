"""Port of upstream `tests/unit/final-response-validation.test.ts` (issue #45).

Pins the specific diagnostics of the final-response validator so its message
cannot silently regress to something ambiguous. The validator is private and
the not-an-array / missing-field shapes are unreachable through the public
`call_model` path, so -- like upstream, which casts to reach
`validateFinalResponse` -- these drive `ModelResult._validate_final_response`
directly. No request is ever dispatched: the validator is pure.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict

import pytest

from openrouter_agent import call_model
from tests._fixtures import QueuedClient, function_call_item, make_response, message_item

Validator = Callable[..., None]


def validator() -> Validator:
    client = QueuedClient([])
    result = call_model(client, {"model": "test-model", "input": "hello"})
    validate: Validator = result._validate_final_response

    def checked(*args: Any) -> None:
        validate(*args)
        # The validator must never touch the transport.
        assert client.requests == []

    return checked


MISSING = object()


def response(response_id: Any = "resp_1", output: Any = MISSING) -> Dict[str, Any]:
    built = make_response("resp_1", [])
    if response_id is MISSING:
        del built["id"]
    else:
        built["id"] = response_id
    if output is not MISSING:
        built["output"] = output
    else:
        del built["output"]
    return built


def error_message(fn: Callable[[], None]) -> str:
    with pytest.raises(ValueError) as info:
        fn()
    return str(info.value)


# -- empty output array --------------------------------------------------------


async def test_empty_reports_the_empty_array_its_length_and_the_response_id() -> None:
    validate = validator()
    with pytest.raises(ValueError, match=r'output array is empty \(length 0\) for response "resp_empty"'):
        validate(response("resp_empty", []))


async def test_empty_points_at_the_strict_and_allow_final_response_escape_hatches() -> None:
    validate = validator()
    # Upstream names `strictFinalResponse/allowFinalResponse`; the port names its
    # own snake_case options (model_result.py `_validate_final_response`).
    with pytest.raises(ValueError, match=r"strict_final_response/allow_final_response"):
        validate(response(output=[]))


async def test_empty_explains_that_the_provider_returned_no_output_items() -> None:
    validate = validator()
    with pytest.raises(ValueError, match=r"The model returned no output items"):
        validate(response(output=[]))


async def test_empty_keeps_the_historical_message_prefix() -> None:
    validate = validator()
    message = error_message(lambda: validate(response(output=[])))
    assert re.match(r"^Invalid final response: empty or invalid output", message)
    assert message.startswith(
        'Invalid final response: empty or invalid output \u2014 output array is empty (length 0) for response "resp_1"'
    )


async def test_empty_stays_a_single_line() -> None:
    validate = validator()
    message = error_message(lambda: validate(response(output=[])))
    assert message != ""
    assert "\n" not in message


async def test_empty_does_not_throw_when_empty_output_is_explicitly_allowed() -> None:
    validate = validator()
    validate(response(output=[]), True)


# -- output present but not an array ---------------------------------------


async def test_non_array_names_the_offending_runtime_type() -> None:
    validate = validator()
    with pytest.raises(ValueError, match=r"output is not an array \(got string\)"):
        validate(response(output="not an array"))


async def test_non_array_distinguishes_an_object_from_an_array() -> None:
    validate = validator()
    with pytest.raises(ValueError, match=r"output is not an array \(got object\)"):
        validate(response(output={"type": "message"}))


async def test_non_array_does_not_claim_the_array_is_empty() -> None:
    validate = validator()
    message = error_message(lambda: validate(response(output=42)))
    assert re.search(r"output is not an array \(got number\)", message)
    assert "output array is empty" not in message


async def test_non_array_is_still_tolerated_under_allow_empty_output() -> None:
    validate = validator()
    # allow_empty_output short-circuits before the message is built, so a
    # non-array output is also swallowed on the tolerant path. Pre-existing
    # upstream behavior, pinned so a future logic fix is deliberate.
    validate(response(output="not an array"), True)


# -- missing required fields --------------------------------------------------


async def test_missing_names_id_when_only_the_id_is_missing() -> None:
    validate = validator()
    with pytest.raises(ValueError, match=r"missing required fields: id$"):
        validate(response(MISSING, [{"type": "message"}]))


async def test_missing_names_output_when_only_output_is_missing() -> None:
    validate = validator()
    with pytest.raises(ValueError, match=r"missing required fields: output$"):
        validate(response())


async def test_missing_names_both_fields_when_both_are_missing() -> None:
    validate = validator()
    with pytest.raises(ValueError, match=r"missing required fields: id, output$"):
        validate(response(MISSING))


async def test_missing_treats_a_none_response_as_missing_both_fields_rather_than_crashing() -> None:
    validate = validator()
    message = error_message(lambda: validate(None))
    assert message == "Invalid final response: missing required fields: id, output"


async def test_missing_keeps_the_historical_message_prefix() -> None:
    validate = validator()
    message = error_message(lambda: validate(response()))
    assert re.match(r"^Invalid final response: missing required fields", message)


# -- valid outputs --------------------------------------------------------------


async def test_valid_accepts_a_tool_call_only_output() -> None:
    validate = validator()
    validate(response(output=[function_call_item("call_1", "do_thing")]))


async def test_valid_accepts_a_normal_message_output() -> None:
    validate = validator()
    validate(response(output=[message_item("hi")]))
