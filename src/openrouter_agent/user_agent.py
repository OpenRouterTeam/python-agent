"""Agent SDK user-agent token.

Ports `packages/agent/src/lib/user-agent.ts`. Keeps the generated SDK's base
user-agent string byte-identical and appends one token identifying this
package for OpenRouter analytics.

Deliberate divergence (contract, Idiomatic Divergences #7): the token is the
PyPI distribution name ``openrouter-agent-sdk``, not upstream's npm name
``@openrouter/agent``. Format matches upstream: ``<sdk user agent> <name>/<version>``.
"""

from __future__ import annotations

from importlib import metadata
from typing import Callable, Optional

from openrouter._version import __user_agent__ as SDK_USER_AGENT

AGENT_PACKAGE_NAME = "openrouter-agent-sdk"


def get_package_version(version_getter: Optional[Callable[[str], str]] = None) -> Optional[str]:
    """This distribution's installed version, or ``None`` when no distribution
    metadata is available (e.g. running from a bare source checkout)."""
    try:
        return (version_getter or metadata.version)(AGENT_PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return None


def agent_user_agent_token(version_getter: Optional[Callable[[str], str]] = None) -> str:
    """``openrouter-agent-sdk/<version>``; the bare name if the version is unknown."""
    version = get_package_version(version_getter)
    if version is None:
        return AGENT_PACKAGE_NAME
    return f"{AGENT_PACKAGE_NAME}/{version}"


def append_agent_user_agent(user_agent: str, agent_token: str) -> str:
    """Append ``agent_token`` to ``user_agent`` once (idempotent)."""
    if agent_token in user_agent.split():
        return user_agent
    if not user_agent:
        return agent_token
    return f"{user_agent} {agent_token}"


def build_agent_user_agent(version_getter: Optional[Callable[[str], str]] = None) -> str:
    return append_agent_user_agent(SDK_USER_AGENT, agent_user_agent_token(version_getter))


PACKAGE_VERSION: Optional[str] = get_package_version()
"""This package's version as recorded in installed distribution metadata."""

AGENT_USER_AGENT: str = build_agent_user_agent()
"""The default user agent the `OpenRouter` client sends, e.g.
``speakeasy-sdk/python 1.x ... openrouter-agent-sdk/0.8.0``."""

__all__ = [
    "AGENT_PACKAGE_NAME",
    "AGENT_USER_AGENT",
    "PACKAGE_VERSION",
    "SDK_USER_AGENT",
    "agent_user_agent_token",
    "append_agent_user_agent",
    "build_agent_user_agent",
    "get_package_version",
]
