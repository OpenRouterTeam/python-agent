"""Ports the delta of tests/unit/hooks-session-lifecycle.test.ts.

The rest of the upstream file (SessionStart/SessionEnd pairing, max_turns,
error-path teardown) is ported in tests/unit/test_model_result_hooks.py; this
file carries the case changed at this sync so the 1:1 mapping exists.

Upstream constructs `ModelResult` directly (public export) with a request that
still carries client-only fields, on both request-resolution paths: the plain
path and the async-params path. Python `ModelResult._run` resolves every turn's
request through `resolve_async_functions`, which drops client-only keys
(async_params.py `_EXCLUDED`).
"""

from __future__ import annotations

from typing import Any, Dict, List

from openrouter_agent import HookEntry, HookName, HooksManager
from openrouter_agent.model_result import ModelResult
from tests._fixtures import QueuedClient, text_response


async def test_strips_hooks_and_active_tools_from_the_outgoing_request_on_both_resolution_paths() -> None:
    client = QueuedClient([text_response("resp_1", "hello back"), text_response("resp_2", "hello back")])
    hooks = HooksManager()
    events: List[str] = []
    hooks.on(HookName.SessionStart.value, HookEntry(handler=lambda p, c: events.append("start")))
    hooks.on(HookName.SessionEnd.value, HookEntry(handler=lambda p, c: events.append("end")))

    # Path 1: plain request, constructed directly so client-only fields are
    # still present on the request object. Upstream names the field
    # `activeTools`; this port's request key is `active_tools`.
    direct = ModelResult(
        {
            "client": client,
            "request": {"model": "test-model", "input": "hi", "hooks": hooks, "active_tools": ["echo"]},
            "hooks": hooks,
        }
    )
    assert await direct.get_text() == "hello back"

    # Path 2: request with an async (callable) param.
    async def temperature(turn: Dict[str, Any]) -> float:
        return 0.5

    with_async = ModelResult(
        {
            "client": client,
            "request": {
                "model": "test-model",
                "input": "hi",
                "temperature": temperature,
                "hooks": hooks,
                "active_tools": ["echo"],
            },
            "hooks": hooks,
        }
    )
    assert await with_async.get_text() == "hello back"

    assert len(client.requests) == 2
    for sent in client.requests:
        assert "hooks" not in sent
        assert "active_tools" not in sent
        assert "activeTools" not in sent
    # The async param was resolved, not sent as a callable.
    assert client.requests[1]["temperature"] == 0.5
    # The hooks manager still ran for both sessions, in pairs.
    assert events == ["start", "end", "start", "end"]
