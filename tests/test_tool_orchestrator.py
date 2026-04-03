"""Tests for _tool_orchestrator: _call_api streaming and non-streaming modes."""

from __future__ import annotations

from typing import Any

import pytest

from openrouter_agent._tool_context import ToolContextStore
from openrouter_agent._tool_event_broadcaster import ToolEventBroadcaster
from openrouter_agent._tool_orchestrator import _call_api, run_tool_loop
from openrouter_agent._types import ResponseStreamEvent

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

class _FakeNonStreamingResponse:
    """Mimics a non-streaming API response object with model_dump."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def model_dump(self) -> dict[str, Any]:
        return dict(self._data)


class _FakeSSEEvent:
    """Mimics a single SSE event from the streaming API."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self.type = data.get("type", "")

    def model_dump(self) -> dict[str, Any]:
        return dict(self._data)


class _FakeSSEStream:
    """Async iterable that yields SSE events."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = [_FakeSSEEvent(e) for e in events]

    def __aiter__(self) -> _FakeSSEStream:
        self._iter = iter(self._events)
        return self

    async def __anext__(self) -> _FakeSSEEvent:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _FakeResponses:
    """Fake ``client.beta.responses`` namespace."""

    def __init__(
        self,
        non_stream_response: Any = None,
        stream_events: list[dict[str, Any]] | None = None,
    ) -> None:
        self._non_stream_response = non_stream_response
        self._stream_events = stream_events or []
        self.last_kwargs: dict[str, Any] = {}

    async def send_async(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        if kwargs.get("stream"):
            return _FakeSSEStream(self._stream_events)
        return self._non_stream_response


class _FakeBeta:
    def __init__(self, responses: _FakeResponses) -> None:
        self.responses = responses


class _FakeClient:
    def __init__(self, responses: _FakeResponses) -> None:
        self.beta = _FakeBeta(responses)


# ---------------------------------------------------------------------------
# Non-streaming _call_api
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_call_api_non_streaming_returns_dict() -> None:
    payload = {
        "id": "resp_1",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}],
    }
    responses = _FakeResponses(non_stream_response=_FakeNonStreamingResponse(payload))
    client = _FakeClient(responses)

    result = await _call_api(client, {"model": "test"}, stream=False)

    assert result["id"] == "resp_1"
    assert responses.last_kwargs["stream"] is False


@pytest.mark.anyio
async def test_call_api_non_streaming_plain_dict() -> None:
    """Non-streaming response that is already a dict."""
    payload: dict[str, Any] = {"id": "resp_2", "output": []}
    responses = _FakeResponses(non_stream_response=payload)
    client = _FakeClient(responses)

    result = await _call_api(client, {"model": "test"}, stream=False)
    assert result["id"] == "resp_2"


# ---------------------------------------------------------------------------
# Streaming _call_api
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_call_api_streaming_completed_event() -> None:
    """When a response.completed event arrives, its payload is returned."""
    completed_response = {
        "id": "resp_3",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "hello"}]}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    events = [
        {"type": "response.created", "response": {"id": "resp_3"}},
        {"type": "response.output_text.delta", "delta": "hel", "output_index": 0},
        {"type": "response.output_text.delta", "delta": "lo", "output_index": 0},
        {"type": "response.completed", "response": completed_response},
    ]
    responses = _FakeResponses(stream_events=events)
    client = _FakeClient(responses)
    broadcaster: ToolEventBroadcaster[ResponseStreamEvent] = ToolEventBroadcaster()

    result = await _call_api(client, {"model": "test"}, stream=True, broadcaster=broadcaster)

    assert result["id"] == "resp_3"
    assert responses.last_kwargs["stream"] is True
    # Broadcaster should have received events
    assert len(broadcaster._buffer) >= 3


@pytest.mark.anyio
async def test_call_api_streaming_accumulates_text_without_completed() -> None:
    """When no response.completed event, text deltas are accumulated."""
    events = [
        {"type": "response.created", "response": {"id": "resp_4"}},
        {"type": "response.output_text.delta", "delta": "foo", "output_index": 0},
        {"type": "response.output_text.delta", "delta": "bar", "output_index": 0},
    ]
    responses = _FakeResponses(stream_events=events)
    client = _FakeClient(responses)
    broadcaster: ToolEventBroadcaster[ResponseStreamEvent] = ToolEventBroadcaster()

    result = await _call_api(client, {"model": "test"}, stream=True, broadcaster=broadcaster)

    assert result["id"] == "resp_4"
    # The accumulated output should contain the text
    output = result.get("output", [])
    assert len(output) >= 1
    text_block = output[0].get("content", [{}])[0]
    assert text_block.get("text") == "foobar"


@pytest.mark.anyio
async def test_call_api_streaming_accumulates_function_call() -> None:
    """Function call argument deltas are concatenated."""
    events = [
        {"type": "response.created", "response": {"id": "resp_5"}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "name": "get_weather", "id": "fc_1", "arguments": ""},
        },
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"ci'},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": 'ty":"SF"}'},
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "name": "get_weather",
                "id": "fc_1",
                "arguments": '{"city":"SF"}',
            },
        },
    ]
    responses = _FakeResponses(stream_events=events)
    client = _FakeClient(responses)
    broadcaster: ToolEventBroadcaster[ResponseStreamEvent] = ToolEventBroadcaster()

    result = await _call_api(client, {"model": "test"}, stream=True, broadcaster=broadcaster)

    output = result.get("output", [])
    assert len(output) == 1
    assert output[0]["type"] == "function_call"
    assert output[0]["name"] == "get_weather"
    assert output[0]["arguments"] == '{"city":"SF"}'


@pytest.mark.anyio
async def test_call_api_streaming_no_broadcaster() -> None:
    """Streaming works even when broadcaster is None (events are just not pushed)."""
    completed = {"id": "resp_6", "output": [], "usage": {}}
    events = [
        {"type": "response.created", "response": {"id": "resp_6"}},
        {"type": "response.completed", "response": completed},
    ]
    responses = _FakeResponses(stream_events=events)
    client = _FakeClient(responses)

    result = await _call_api(client, {"model": "test"}, stream=True, broadcaster=None)
    assert result["id"] == "resp_6"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_call_api_raises_runtime_error_on_failure() -> None:
    class _BrokenResponses:
        async def send_async(self, **kwargs: Any) -> Any:
            raise ConnectionError("boom")

    client = _FakeClient(_FakeResponses())  # type: ignore[arg-type]
    client.beta.responses = _BrokenResponses()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="API call failed"):
        await _call_api(client, {"model": "test"}, stream=False)


# ---------------------------------------------------------------------------
# run_tool_loop integration (non-streaming, no tools)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_run_tool_loop_non_streaming_no_tools() -> None:
    """run_tool_loop completes a single turn with no tool calls."""
    payload = {
        "id": "resp_loop_1",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}],
        "finish_reason": "stop",
    }
    responses = _FakeResponses(non_stream_response=_FakeNonStreamingResponse(payload))
    client = _FakeClient(responses)
    broadcaster: ToolEventBroadcaster[ResponseStreamEvent] = ToolEventBroadcaster()

    steps = await run_tool_loop(
        client=client,
        request_params={"model": "test", "input": "hello"},
        tools=[],
        stop_conditions=[],
        context_store=ToolContextStore(),
        broadcaster=broadcaster,
        stream=False,
    )

    assert len(steps) == 1
    assert steps[0].text == "done"
    assert steps[0].finish_reason == "stop"
    assert broadcaster.is_done


@pytest.mark.anyio
async def test_run_tool_loop_streaming_no_tools() -> None:
    """run_tool_loop works with streaming enabled (default)."""
    completed = {
        "id": "resp_loop_2",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "streamed"}]}],
        "finish_reason": "stop",
    }
    events = [
        {"type": "response.created", "response": {"id": "resp_loop_2"}},
        {"type": "response.output_text.delta", "delta": "streamed", "output_index": 0},
        {"type": "response.completed", "response": completed},
    ]
    responses = _FakeResponses(stream_events=events)
    client = _FakeClient(responses)
    broadcaster: ToolEventBroadcaster[ResponseStreamEvent] = ToolEventBroadcaster()

    steps = await run_tool_loop(
        client=client,
        request_params={"model": "test", "input": "hello"},
        tools=[],
        stop_conditions=[],
        context_store=ToolContextStore(),
        broadcaster=broadcaster,
        stream=True,
    )

    assert len(steps) == 1
    assert steps[0].text == "streamed"
    assert broadcaster.is_done
    # Streaming events should have been pushed to broadcaster
    assert len(broadcaster._buffer) > 0
