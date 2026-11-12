"""Ports tests/unit/openrouter.test.ts.

Upstream inspects `openrouter._options` (the TS SDK's constructor options). The
generated Python SDK exposes the equivalent state on `sdk_configuration`
(`user_agent`, `server_url`, and the registered `SDKHooks` under `_hooks`);
this port additionally keeps the normalized hooks on `agent_hooks`
(openrouter.py). `OpenRouterCore` maps to the generated `openrouter.OpenRouter`.

Delta at this sync: the three user-agent cases, and the hook assertions moving
from `toHaveLength(1)` to `toContain(hook)` (the SDK may pre-register its own
hooks, so membership of the caller's hook -- not slot length -- is the
invariant).
"""

from __future__ import annotations

from typing import Any, Dict

import httpx
import pytest
from openrouter import OpenRouter as SDKOpenRouter
from openrouter._hooks import SDKHooks

from openrouter_agent import ModelResult, OpenRouter
from openrouter_agent.user_agent import AGENT_USER_AGENT

REQUEST: Dict[str, Any] = {"model": "openai/gpt-4o", "input": "Hello"}


class NoopBeforeRequest:
    def before_request(self, ctx: Any, request: Any) -> Any:
        return request


class NoopAfterSuccess:
    def after_success(self, ctx: Any, response: Any) -> Any:
        return response


class Combined(NoopBeforeRequest, NoopAfterSuccess):
    pass


def _client(**kwargs: Any) -> OpenRouter:
    """Construct the client with a caller-supplied async httpx client.

    The generated SDK's GC finalizer (openrouter/httpclient.py `close_clients`)
    calls `asyncio.run(async_client.aclose())` for an SDK-owned async client
    when collected outside a running loop. That swaps the event loop set on the
    policy at an arbitrary GC point and trips pytest-asyncio's unclosed-loop
    check in whichever async test runs next. A supplied client is never
    closed by that finalizer, so these sync tests cannot leak a loop.
    """
    kwargs.setdefault("async_client", httpx.AsyncClient())
    return OpenRouter(**kwargs)


def _sdk_hooks(client: OpenRouter) -> SDKHooks:
    hooks = client.sdk_configuration.__dict__["_hooks"]
    assert isinstance(hooks, SDKHooks)
    return hooks


def test_instantiates_without_options() -> None:
    assert isinstance(_client(), OpenRouter)


def test_instantiates_with_api_key() -> None:
    assert isinstance(_client(api_key="test-key"), OpenRouter)


def test_returns_a_model_result_from_call_model() -> None:
    client = _client(api_key="test-key")

    assert isinstance(client.call_model(REQUEST), ModelResult)


def test_preserves_binding_when_call_model_is_detached() -> None:
    client = _client(api_key="test-key")
    call_model = client.call_model

    result = call_model(REQUEST)

    assert isinstance(result, ModelResult)
    assert result.options["client"] is client


def test_accepts_sdk_options_like_server_url() -> None:
    assert isinstance(_client(api_key="test-key", server_url="https://custom.endpoint.com"), OpenRouter)


def test_defaults_to_the_agent_sdk_user_agent() -> None:
    assert _client(api_key="test-key").sdk_configuration.user_agent == AGENT_USER_AGENT


def test_preserves_an_explicitly_configured_user_agent() -> None:
    client = _client(api_key="test-key", user_agent="custom/1.0")

    assert client.sdk_configuration.user_agent == "custom/1.0"


async def test_sends_the_agent_sdk_user_agent_on_outgoing_requests() -> None:
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.setdefault("user_agents", []).append(request.headers.get("user-agent"))
        raise RuntimeError("captured request")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenRouter(api_key="test-key", async_client=async_client)
    try:
        with pytest.raises(RuntimeError, match="captured request"):
            await client.call_model(REQUEST).get_text()
    finally:
        await async_client.aclose()

    assert captured["user_agents"] == [AGENT_USER_AGENT]


def test_is_an_instance_of_the_generated_sdk_client() -> None:
    assert isinstance(_client(api_key="test-key"), SDKOpenRouter)


def test_passes_constructor_options_through_to_the_instance() -> None:
    client = _client(api_key="test-key", server_url="https://custom.endpoint.com")

    assert isinstance(client, SDKOpenRouter)
    assert client.sdk_configuration.server_url == "https://custom.endpoint.com"


def test_is_usable_as_both_agent_and_sdk_client() -> None:
    client = _client(api_key="test-key")

    assert isinstance(client, OpenRouter)
    assert isinstance(client, SDKOpenRouter)
    assert isinstance(client.call_model(REQUEST), ModelResult)


# -- hooks -------------------------------------------------------------------


def test_exposes_an_sdk_hooks_instance_by_default() -> None:
    client = _client(api_key="test-key")

    assert isinstance(_sdk_hooks(client), SDKHooks)
    # No agent-level hooks were supplied.
    assert client.agent_hooks is None


def test_uses_a_passed_sdk_hooks_instance_as_is() -> None:
    hook = NoopBeforeRequest()
    hooks = SDKHooks()
    hooks.register_before_request_hook(hook)

    client = _client(api_key="test-key", hooks=hooks)

    assert client.agent_hooks is hooks
    assert _sdk_hooks(client) is hooks
    assert hook in _sdk_hooks(client).before_request_hooks


def test_wraps_a_single_hook_object_into_an_sdk_hooks_instance() -> None:
    hook = NoopBeforeRequest()

    client = _client(api_key="test-key", user_agent="custom/1.0", hooks=hook)

    assert isinstance(_sdk_hooks(client), SDKHooks)
    assert hook in _sdk_hooks(client).before_request_hooks
    assert client.sdk_configuration.user_agent == "custom/1.0"


def test_wraps_a_list_of_hook_objects_into_an_sdk_hooks_instance() -> None:
    before = NoopBeforeRequest()
    after = NoopAfterSuccess()

    client = _client(api_key="test-key", hooks=[before, after])

    hooks = _sdk_hooks(client)
    assert before in hooks.before_request_hooks
    assert after in hooks.after_success_hooks
    assert after not in hooks.before_request_hooks
    assert before not in hooks.after_success_hooks


def test_registers_a_multi_interface_hook_under_every_matching_slot() -> None:
    combined = Combined()

    client = _client(api_key="test-key", hooks=combined)

    hooks = _sdk_hooks(client)
    assert hooks.before_request_hooks.count(combined) == 1
    assert hooks.after_success_hooks.count(combined) == 1
