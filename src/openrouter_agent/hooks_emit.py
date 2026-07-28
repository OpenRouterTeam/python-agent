"""Sequential hook handler-chain execution. Mirrors upstream `hooks-emit.ts`.

Supports:
- ToolMatcher and filter-based skipping (matcher fails closed: a handler with
  a matcher and no `tool_name` for this emit is skipped)
- Sync results validated against the hook's result schema and collected
- Async fire-and-forget via a returned `AsyncOutput` -- its `work` is tracked
  without being awaited; the manager drains/times it out
- Per-hook mutation piping (driven by `HOOK_BEHAVIOR`)
- Short-circuit on block/reject fields (non-empty string or `True`)
- Cooperative cancellation via `context.cancel_event`
"""

from __future__ import annotations

import asyncio
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

from pydantic import BaseModel, ValidationError

from ._utils import maybe_await
from .hooks_matchers import matches_tool
from .hooks_types import (
    DEFAULT_ASYNC_TIMEOUT_MS,
    EmitResult,
    HOOK_BEHAVIOR,
    HookEntry,
    LifecycleHookContext,
    is_async_output,
)


@dataclass(frozen=True)
class _ChainOptions:
    hook_name: str
    throw_on_handler_error: bool
    tool_name: Optional[str] = None
    result_schema: Optional[Type[BaseModel]] = None
    on_async_timeout: Optional[Callable[[str], None]] = None


async def execute_handler_chain(
    entries: Sequence[HookEntry],
    initial_payload: Dict[str, Any],
    context: LifecycleHookContext,
    *,
    hook_name: str,
    throw_on_handler_error: bool,
    tool_name: Optional[str] = None,
    result_schema: Optional[Type[BaseModel]] = None,
    on_async_timeout: Optional[Callable[[str], None]] = None,
) -> EmitResult:
    options = _ChainOptions(
        hook_name=hook_name,
        throw_on_handler_error=throw_on_handler_error,
        tool_name=tool_name,
        result_schema=result_schema,
        on_async_timeout=on_async_timeout,
    )
    results: List[Any] = []
    pending: List["asyncio.Task[None]"] = []
    current_payload: Dict[str, Any] = dict(initial_payload) if isinstance(initial_payload, dict) else initial_payload
    blocked = False
    mutated = False

    behavior = HOOK_BEHAVIOR.get(hook_name)

    for index, entry in enumerate(entries):
        if context.cancel_event.is_set():
            break

        gate = _evaluate_entry_gate(entry, current_payload, index, options)
        if gate == "skip":
            continue

        try:
            return_value = await maybe_await(entry.handler(current_payload, context))
            outcome = _classify_handler_return(return_value, index, options)

            if outcome[0] == "async":
                tracked = outcome[1]
                if tracked is not None:
                    pending.append(tracked)
                continue
            if outcome[0] == "skip":
                continue

            result = outcome[1]
            results.append(result)

            if behavior and behavior.mutations:
                applied = _apply_mutations(current_payload, result, behavior.mutations)
                if applied is not current_payload:
                    current_payload = applied
                    mutated = True

            if behavior and behavior.block_field and _is_block_triggered(result, behavior.block_field):
                blocked = True
                break
        except Exception as error:  # noqa: BLE001 - policy decides re-raise vs. warn
            if throw_on_handler_error:
                raise
            warnings.warn(f'[HooksManager] Handler {index} for hook "{hook_name}" threw: {error}', stacklevel=2)

    return EmitResult(
        results=results,
        pending=pending,
        final_payload=current_payload,
        blocked=blocked,
        mutated=mutated,
    )


def _evaluate_entry_gate(entry: HookEntry, payload: Dict[str, Any], index: int, options: _ChainOptions) -> str:
    try:
        matcher_passes = entry.matcher is None or (
            options.tool_name is not None and matches_tool(entry.matcher, options.tool_name)
        )
        if not matcher_passes:
            return "skip"
        return "run" if not entry.filter or bool(entry.filter(payload)) else "skip"
    except Exception as error:  # noqa: BLE001
        if options.throw_on_handler_error:
            raise
        warnings.warn(
            f'[HooksManager] Matcher/filter for handler {index} of hook "{options.hook_name}" threw: {error}',
            stacklevel=2,
        )
        return "skip"


def _classify_handler_return(return_value: Any, index: int, options: _ChainOptions) -> Any:
    if is_async_output(return_value):
        return ("async", _track_async_work(return_value, options.hook_name, options.on_async_timeout))
    if return_value is None:
        return ("skip", None)
    if options.result_schema is None:
        return ("result", return_value)
    try:
        validated = options.result_schema.model_validate(return_value)
    except ValidationError as error:
        message = f'[HooksManager] Handler {index} for hook "{options.hook_name}" returned an invalid result: {error}'
        if options.throw_on_handler_error:
            raise RuntimeError(message) from error
        warnings.warn(message, stacklevel=2)
        return ("skip", None)
    return ("result", validated.model_dump())


def _track_async_work(
    output: Any,
    hook_name: str,
    on_timeout: Optional[Callable[[str], None]],
) -> Optional["asyncio.Task[None]"]:
    if output.work is None:
        return None
    timeout_s = (output.async_timeout_ms or DEFAULT_ASYNC_TIMEOUT_MS) / 1000

    async def _wait() -> None:
        try:
            await asyncio.wait_for(output.work, timeout=timeout_s)
        except asyncio.TimeoutError:
            warnings.warn(
                f'[HooksManager] Async work for hook "{hook_name}" exceeded its timeout; abandoning wait.',
                stacklevel=2,
            )
            if on_timeout is not None:
                on_timeout(hook_name)
        except Exception as error:  # noqa: BLE001
            warnings.warn(f'[HooksManager] Async work for hook "{hook_name}" rejected: {error}', stacklevel=2)

    return asyncio.ensure_future(_wait())


def _apply_mutations(payload: Dict[str, Any], result: Any, mutation_map: Dict[str, str]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return payload
    mutated = payload
    for result_field, payload_field in mutation_map.items():
        if result_field in result:
            value = result[result_field]
            if value is not None:
                mutated = {**mutated, payload_field: value}
    return mutated


def _is_block_triggered(result: Any, block_field: str) -> bool:
    if not isinstance(result, dict):
        return False
    value = result.get(block_field)
    if value is True:
        return True
    return isinstance(value, str) and len(value) > 0
