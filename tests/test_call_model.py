"""Tests for call_model options pass-through."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from openrouter_agent import call_model


def _make_mock_client(response: dict[str, Any] | None = None) -> Any:
    """Build a mock client that records kwargs passed to send_async."""
    if response is None:
        response = {
            "id": "resp-1",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "hi"}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

    send_async = AsyncMock(return_value=response)

    class _Responses:
        send_async = None  # set below

    responses = _Responses()
    responses.send_async = send_async

    class _Beta:
        responses = None  # set below

    beta = _Beta()
    beta.responses = responses

    class _Client:
        beta = None  # set below

    client = _Client()
    client.beta = beta
    return client


@pytest.mark.anyio
async def test_options_headers_forwarded_to_api() -> None:
    """The x-openrouter-callmodel header should reach send_async as extra_headers."""
    client = _make_mock_client()

    await call_model(
        client=client,
        request={"model": "test/model", "input": "hello"},
    )

    send_async = client.beta.responses.send_async
    assert send_async.call_count == 1
    kwargs = send_async.call_args.kwargs
    assert "extra_headers" in kwargs
    assert kwargs["extra_headers"]["x-openrouter-callmodel"] == "true"


@pytest.mark.anyio
async def test_custom_headers_merged() -> None:
    """User-supplied headers in options should be merged with the SDK header."""
    client = _make_mock_client()

    await call_model(
        client=client,
        request={"model": "test/model", "input": "hello"},
        options={"headers": {"x-custom": "value"}},
    )

    kwargs = client.beta.responses.send_async.call_args.kwargs
    headers = kwargs["extra_headers"]
    assert headers["x-openrouter-callmodel"] == "true"
    assert headers["x-custom"] == "value"


@pytest.mark.anyio
async def test_options_timeout_forwarded() -> None:
    """Timeout in options should be forwarded to send_async."""
    client = _make_mock_client()

    await call_model(
        client=client,
        request={"model": "test/model", "input": "hello"},
        options={"timeout": 30},
    )

    kwargs = client.beta.responses.send_async.call_args.kwargs
    assert kwargs["timeout"] == 30


@pytest.mark.anyio
async def test_no_options_still_sends_callmodel_header() -> None:
    """Even without explicit options, the SDK header must be sent."""
    client = _make_mock_client()

    await call_model(
        client=client,
        request={"model": "test/model", "input": "hello"},
    )

    kwargs = client.beta.responses.send_async.call_args.kwargs
    assert kwargs["extra_headers"]["x-openrouter-callmodel"] == "true"
