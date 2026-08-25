from __future__ import annotations

import inspect
from importlib import metadata
from typing import Any, Callable, Mapping, Optional, Protocol

from openrouter import OpenRouter as _SDKOpenRouter
from openrouter._hooks import SDKHooks
from openrouter._version import __user_agent__


from .call_model import call_model


class Hook(Protocol):
    def sdk_init(self, configuration: Any) -> Any: ...


OpenRouterOptions = Mapping[str, object]
SDKOptions = Mapping[str, object]

AGENT_PACKAGE_NAME = "openrouter-agent-sdk"


def _agent_user_agent_token(
    version_getter: Optional[Callable[[str], str]] = None,
) -> str:
    try:
        version = (version_getter or metadata.version)(AGENT_PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return AGENT_PACKAGE_NAME
    return f"{AGENT_PACKAGE_NAME}/{version}"


def _append_agent_user_agent(user_agent: str, agent_token: str) -> str:
    if agent_token in user_agent.split():
        return user_agent
    if not user_agent:
        return agent_token
    return f"{user_agent} {agent_token}"


class OpenRouter(_SDKOpenRouter):  # type: ignore[misc, valid-type]
    def __init__(self, *args: Any, hooks: Any = None, **kwargs: Any) -> None:
        explicit_user_agent = kwargs.pop("user_agent", None)
        normalized_hooks = self._normalize_hooks(hooks)
        sdk_accepts_hooks = "hooks" in inspect.signature(_SDKOpenRouter.__init__).parameters
        if normalized_hooks is not None and sdk_accepts_hooks:
            kwargs["hooks"] = normalized_hooks
        super().__init__(*args, **kwargs)
        if normalized_hooks is not None and not sdk_accepts_hooks and hasattr(self, "sdk_configuration"):
            # The current generated Python SDK has no constructor hook parameter.
            # Register on sdk_configuration immediately after construction, which
            # is the SDK-supported hook storage location used by requests.
            self.sdk_configuration.__dict__["_hooks"] = normalized_hooks
            self.sdk_configuration = normalized_hooks.sdk_init(self.sdk_configuration)
        self.agent_hooks = normalized_hooks
        if explicit_user_agent is not None:
            self.sdk_configuration.user_agent = explicit_user_agent
        elif self.sdk_configuration.user_agent == __user_agent__:
            self.sdk_configuration.user_agent = _append_agent_user_agent(
                self.sdk_configuration.user_agent,
                _agent_user_agent_token(),
            )

    @staticmethod
    def _normalize_hooks(hooks: Any) -> Any:
        if hooks is None:
            return None
        if isinstance(hooks, SDKHooks):
            return hooks
        normalized = SDKHooks()
        for hook in list(hooks) if isinstance(hooks, (list, tuple)) else [hooks]:
            if hasattr(hook, "sdk_init"):
                normalized.register_sdk_init_hook(hook)
            if hasattr(hook, "before_request"):
                normalized.register_before_request_hook(hook)
            if hasattr(hook, "after_success"):
                normalized.register_after_success_hook(hook)
            if hasattr(hook, "after_error"):
                normalized.register_after_error_hook(hook)
        return normalized

    def call_model(self, request: Mapping[str, Any], options: Optional[Mapping[str, Any]] = None):
        return call_model(self, request, options)
