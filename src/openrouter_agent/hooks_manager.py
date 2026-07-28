"""Typed, extensible hook system for agent lifecycle events. Mirrors upstream
`hooks-manager.ts`.

Supports both built-in hooks (PreToolUse, PostToolUse, ...) and user-defined
custom hooks. Unlike the TypeScript reference, there is no internal-registrar
symbol trick: Python's `on()` is not statically constrained to known hook
names, so `resolve_hooks` can register inline-config entries directly.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import ValidationError

from .hooks_emit import execute_handler_chain
from .hooks_schemas import BUILT_IN_HOOK_NAMES, BUILT_IN_HOOKS, HookDefinition, HookRegistry
from .hooks_types import EmitResult, HookEntry, LifecycleHookContext


class HooksManager:
    """See module docstring. Register handlers with `on()`, fire them with
    `emit()`, and await outstanding fire-and-forget work with `drain()`."""

    def __init__(
        self,
        custom_hooks: Optional[HookRegistry] = None,
        *,
        throw_on_handler_error: bool = False,
    ) -> None:
        self._entries: Dict[str, List[HookEntry]] = {}
        self._pending_async: Set[Any] = set()
        self._inflight: Set[asyncio.Event] = set()
        self._throw_on_handler_error = throw_on_handler_error
        self._session_id = ""

        if custom_hooks:
            for name in custom_hooks:
                if name == "":
                    raise ValueError("Custom hook names must be non-empty strings.")
                if name in BUILT_IN_HOOK_NAMES:
                    raise ValueError(
                        f'Custom hook name "{name}" collides with a built-in hook. Choose a different name.'
                    )
            self._custom_hooks: HookRegistry = dict(custom_hooks)
        else:
            self._custom_hooks = {}

    def set_session_id(self, session_id: str) -> None:
        """Set the manager-level default session ID exposed as
        `context.session_id` to handler invocations.

        This is a single mutable default on the manager instance: when one
        manager is shared by concurrent runs, callers MUST pass `session_id`
        to `emit()` instead (as `ModelResult` does), otherwise the last
        `set_session_id()` call wins and concurrent emits observe the wrong id.
        """
        self._session_id = session_id

    def on(self, hook_name: str, entry: HookEntry) -> Callable[[], None]:
        """Register a handler for a hook. Returns an unsubscribe function."""
        return self._register(hook_name, entry)

    def off(self, hook_name: str, handler: Callable[..., Any]) -> bool:
        """Remove a specific handler function from a hook."""
        entries = self._entries.get(hook_name)
        if not entries:
            return False
        for index, entry in enumerate(entries):
            if entry.handler == handler:
                entries.pop(index)
                if not entries:
                    del self._entries[hook_name]
                return True
        return False

    def remove_all(self, hook_name: Optional[str] = None) -> None:
        """Remove all handlers for a specific hook, or all handlers if omitted."""
        if hook_name is not None:
            self._entries.pop(hook_name, None)
        else:
            self._entries.clear()

    async def emit(
        self,
        hook_name: str,
        payload: Dict[str, Any],
        *,
        tool_name: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> EmitResult:
        """Validate the payload (and each handler's result) against the
        registered schemas, invoke matching handlers, and return results.

        Payload validation failure is handled per `throw_on_handler_error`:
        strict mode re-raises, default mode warns and returns an empty result
        without invoking any handlers.
        """
        entries = list(self._entries.get(hook_name, []))
        definition = self._definition_for(hook_name)

        chain_payload: Dict[str, Any] = payload
        if definition is not None:
            try:
                parsed = definition.payload.model_validate(payload)
            except ValidationError as error:
                message = f'[HooksManager] Invalid payload for hook "{hook_name}": {error}'
                if self._throw_on_handler_error:
                    raise RuntimeError(message) from error
                warnings.warn(message, stacklevel=2)
                return EmitResult(results=[], pending=[], final_payload=chain_payload, blocked=False, mutated=False)
            chain_payload = parsed.model_dump()

        cancel_event = asyncio.Event()
        self._inflight.add(cancel_event)

        context = LifecycleHookContext(
            cancel_event=cancel_event,
            hook_name=hook_name,
            session_id=session_id if session_id is not None else self._session_id,
        )

        has_detached_work = False
        try:
            result_schema = definition.result if definition is not None else None
            result = await execute_handler_chain(
                entries,
                chain_payload,
                context,
                hook_name=hook_name,
                throw_on_handler_error=self._throw_on_handler_error,
                tool_name=tool_name,
                result_schema=result_schema,
                on_async_timeout=lambda _name: cancel_event.set(),
            )

            has_detached_work = len(result.pending) > 0
            if has_detached_work:
                remaining = len(result.pending)

                def _make_cleanup(task: Any) -> Callable[[Any], None]:
                    def _cleanup(_done: Any) -> None:
                        nonlocal remaining
                        self._pending_async.discard(task)
                        remaining -= 1
                        if remaining == 0:
                            self._inflight.discard(cancel_event)

                    return _cleanup

                for task in result.pending:
                    self._pending_async.add(task)
                    task.add_done_callback(_make_cleanup(task))

            return result
        finally:
            if not has_detached_work:
                self._inflight.discard(cancel_event)

    async def drain(self) -> None:
        """Await all in-flight async handler work. Used for graceful shutdown."""
        while self._pending_async:
            snapshot = list(self._pending_async)
            await asyncio.gather(*snapshot, return_exceptions=True)

    def abort_inflight(self) -> None:
        """Signal cancellation to every in-flight `emit()`. Does not remove
        pending async work -- call `drain()` afterward to wait it out."""
        for event in self._inflight:
            event.set()

    def has_handlers(self, hook_name: str) -> bool:
        """Check if any handlers are registered for a given hook."""
        entries = self._entries.get(hook_name)
        return entries is not None and len(entries) > 0

    def _register(self, hook_name: str, entry: HookEntry) -> Callable[[], None]:
        entries = self._entries.setdefault(hook_name, [])
        entries.append(entry)

        def _unsubscribe() -> None:
            current = self._entries.get(hook_name)
            if not current:
                return
            try:
                current.remove(entry)
            except ValueError:
                pass

        return _unsubscribe

    def _definition_for(self, hook_name: str) -> Optional[HookDefinition]:
        built_in = BUILT_IN_HOOKS.get(hook_name)
        if built_in is not None:
            return built_in
        return self._custom_hooks.get(hook_name)
