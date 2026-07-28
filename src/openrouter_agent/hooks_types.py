"""Core types for the lifecycle hooks system.

Mirrors upstream `hooks-types.ts`, adapted to Python:

- Cancellation uses an `asyncio.Event` rather than `AbortSignal` (idiomatic
  divergence: this repo's cancellation is Python-native).
- The `AsyncOutput` fire-and-forget signal is detected via `isinstance`
  rather than schema validation -- Python has real classes, so there is no
  need for zod's structural-shape check.
- Payload/result validation failures raise `pydantic.ValidationError`
  instead of zod's `ZodError`; handled the same way (throw in strict mode,
  warn otherwise).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Pattern, Sequence, Union

from .hooks_schemas import HookName  # noqa: F401  (re-exported for import-surface parity)

#: Matcher for tool-scoped hooks. Filters handler invocation by tool name.
ToolMatcher = Union[str, Pattern[str], Callable[[str], bool]]

#: A hook handler receives the validated payload (dict) and context. May be
#: sync or async -- callers should invoke it through `maybe_await`.
HookHandler = Callable[[Dict[str, Any], "LifecycleHookContext"], Any]


@dataclass(frozen=True)
class LifecycleHookContext:
    """Context provided to every lifecycle-hook handler invocation.

    `cancel_event` is set if the manager's `abort_inflight()` is called
    while the emit is still running. Handlers that kick off background work
    via `AsyncOutput` should observe it for cancellation.

    `session_id` is the single source for session identity in handlers --
    payloads deliberately do not repeat it. The engine threads it per emit
    (safe for a manager shared across concurrent runs); direct `emit()`
    callers get the manager-level default from `set_session_id()` unless
    they pass a per-emit override.
    """

    cancel_event: asyncio.Event
    hook_name: str
    session_id: str


#: Default milliseconds before an async fire-and-forget handler is aborted.
DEFAULT_ASYNC_TIMEOUT_MS = 30_000


@dataclass(frozen=True)
class AsyncOutput:
    """Returned by a handler to signal fire-and-forget mode.

    The chain proceeds immediately without waiting for completion. Any
    background work the handler kicked off should be attached as `work` so
    the manager can track it for `drain()` and enforce `async_timeout_ms`.
    """

    work: Optional[Awaitable[Any]] = None
    async_timeout_ms: float = DEFAULT_ASYNC_TIMEOUT_MS


def is_async_output(value: Any) -> bool:
    """Type guard for an `AsyncOutput` fire-and-forget signal."""
    return isinstance(value, AsyncOutput)


@dataclass(frozen=True)
class HookEntry:
    """An entry registered for a specific hook."""

    handler: HookHandler
    matcher: Optional[ToolMatcher] = None
    filter: Optional[Callable[[Dict[str, Any]], bool]] = None


@dataclass(frozen=True)
class EmitResult:
    """Result of emitting a hook through the handler chain.

    INVARIANT: every entry in `results` passed the hook's result schema
    (invalid results are skipped or raised per the error policy, never
    collected), so consumers can rely on the shape without re-validating.
    For void-result hooks (no schema) entries are opaque dicts/values.
    """

    results: List[Any] = field(default_factory=list)
    #: Handles to detached async handler work (asyncio Tasks).
    pending: List["asyncio.Task[None]"] = field(default_factory=list)
    #: The payload after all mutation piping has been applied.
    final_payload: Dict[str, Any] = field(default_factory=dict)
    #: True if any handler triggered a block/reject short-circuit.
    blocked: bool = False
    #: True if any handler's result actually piped a mutation into the payload.
    mutated: bool = False


@dataclass(frozen=True)
class HookBehavior:
    """Per-hook chain behavior: which result fields pipe mutations back into
    the payload, and which result field short-circuits the chain."""

    mutations: Optional[Dict[str, str]] = None
    block_field: Optional[str] = None


#: Keyed by HookName value. Hooks absent from this table (all
#: observation-only hooks and every custom hook) collect results without
#: altering the payload or short-circuiting the chain.
HOOK_BEHAVIOR: Dict[str, HookBehavior] = {
    HookName.PreToolUse.value: HookBehavior(
        mutations={"mutated_input": "tool_input"},
        block_field="block",
    ),
    HookName.UserPromptSubmit.value: HookBehavior(
        mutations={"mutated_prompt": "prompt"},
        block_field="reject",
    ),
}

#: Inline hook config passed directly to call_model: `{hook_name: [entries]}`.
#: Only supports built-in hook names. For custom hooks, use a HooksManager.
InlineHookConfig = Dict[str, Sequence[HookEntry]]
