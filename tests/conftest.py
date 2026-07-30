"""Pytest configuration.

The builders live in `tests/_fixtures.py` rather than here because `conftest.py`
is not importable as a module, and several test files construct payloads and tools
at module scope. Import them directly:

    from tests._fixtures import QueuedClient, text_response, tool_call_response

They are re-exported below so `conftest` stays the single discovery point.
"""

from __future__ import annotations

from tests._fixtures import (  # noqa: F401  (re-exported for discoverability)
    MemoryStateAccessor,
    QueuedClient,
    QueuedResponses,
    assert_matches_sdk_response_shape,
    function_call_item,
    make_response,
    message_item,
    text_response,
    tool_call_response,
    usage_block,
)
