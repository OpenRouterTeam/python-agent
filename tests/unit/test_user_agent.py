from __future__ import annotations

import importlib.metadata
import re

import httpx

from openrouter._version import __user_agent__
from openrouter.errors.responsevalidationerror import ResponseValidationError
from openrouter_agent import OpenRouter, call_model
from openrouter_agent.openrouter import _agent_user_agent_token, _append_agent_user_agent
from openrouter_agent.user_agent import AGENT_USER_AGENT, PACKAGE_VERSION, build_agent_user_agent


def test_agent_user_agent_composes_sdk_and_agent_package_versions() -> None:
    """Ports tests/unit/user-agent.test.ts.

    Upstream asserts ` @openrouter/agent/<semver>`; this port appends the PyPI
    distribution name instead (contract, Idiomatic Divergences #7). Do not
    "fix" this to the TypeScript literal.
    """
    assert PACKAGE_VERSION == importlib.metadata.version("openrouter-agent-sdk")
    assert AGENT_USER_AGENT == f"{__user_agent__} openrouter-agent-sdk/{PACKAGE_VERSION}"
    assert AGENT_USER_AGENT.startswith("speakeasy-sdk/python ")
    assert re.search(r" openrouter-agent-sdk/\d+\.\d+\.\d+$", AGENT_USER_AGENT)
    assert "@openrouter/agent" not in AGENT_USER_AGENT


def test_default_client_user_agent_equals_agent_user_agent() -> None:
    # Ports openrouter.test.ts "should default to the Agent SDK user agent".
    assert OpenRouter(api_key="test").sdk_configuration.user_agent == AGENT_USER_AGENT


def test_build_agent_user_agent_falls_back_to_bare_name_without_metadata() -> None:
    def missing(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    assert build_agent_user_agent(missing) == f"{__user_agent__} openrouter-agent-sdk"
    assert build_agent_user_agent(lambda _name: "9.9.9") == f"{__user_agent__} openrouter-agent-sdk/9.9.9"


def test_constructed_client_appends_agent_user_agent() -> None:
    client = OpenRouter(api_key="test")

    assert client.sdk_configuration.user_agent == (
        f"{__user_agent__} openrouter-agent-sdk/{importlib.metadata.version('openrouter-agent-sdk')}"
    )


def test_agent_user_agent_append_is_idempotent() -> None:
    client_one = OpenRouter(api_key="test")
    client_two = OpenRouter(api_key="test")
    expected = client_one.sdk_configuration.user_agent

    assert client_two.sdk_configuration.user_agent == expected
    assert _append_agent_user_agent(expected, _agent_user_agent_token()) == expected


def test_explicit_user_agent_is_preserved() -> None:
    class Hook:
        def before_request(self, ctx, request):
            return request

    client = OpenRouter(api_key="test", user_agent="custom/1.0", hooks=Hook())

    assert client.sdk_configuration.user_agent == "custom/1.0"
    assert client.agent_hooks is not None


def test_user_agent_changed_by_hook_is_preserved() -> None:
    class Hook:
        def sdk_init(self, configuration):
            configuration.user_agent = "hook/1.0"
            return configuration

    client = OpenRouter(api_key="test", hooks=Hook())

    assert client.sdk_configuration.user_agent == "hook/1.0"


def test_package_not_found_uses_bare_agent_package_name(monkeypatch) -> None:
    def package_version(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", package_version)

    client = OpenRouter(api_key="test")

    assert client.sdk_configuration.user_agent == f"{__user_agent__} openrouter-agent-sdk"


async def test_call_model_sends_agent_user_agent_and_call_model_marker() -> None:
    seen_headers: httpx.Headers | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_headers
        seen_headers = request.headers
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "test/model",
                "output": [],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            },
        )

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenRouter(api_key="test", server_url="https://mock.local", async_client=async_client)
    try:
        try:
            await call_model(client, {"model": "test/model", "input": "hello"}).get_response()
        except ResponseValidationError:
            pass
    finally:
        await async_client.aclose()

    assert seen_headers is not None
    assert seen_headers["user-agent"] == client.sdk_configuration.user_agent
    assert seen_headers["x-openrouter-callmodel"] == "true"
