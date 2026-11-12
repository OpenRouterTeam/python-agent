from __future__ import annotations

import asyncio
from typing import Any, Callable, Mapping, Sequence

from ._utils import get_field, maybe_await


def _steps(options: Mapping[str, Any]) -> Sequence[Any]:
    return options.get("steps", [])


def step_count_is(count: int) -> Callable[[Mapping[str, Any]], bool]:
    return lambda options: len(_steps(options)) >= count


def has_tool_call(name: str) -> Callable[[Mapping[str, Any]], bool]:
    def check(options: Mapping[str, Any]) -> bool:
        return any(
            getattr(call, "name", None) == name or (isinstance(call, Mapping) and call.get("name") == name)
            for step in _steps(options)
            for call in (get_field(step, "tool_calls", None) or get_field(step, "toolCalls", None) or [])
        )

    return check


def max_tokens_used(limit: int) -> Callable[[Mapping[str, Any]], bool]:
    def check(options: Mapping[str, Any]) -> bool:
        total = 0
        for step in _steps(options):
            usage = get_field(step, "usage", None) or get_field(get_field(step, "response", {}), "usage", {}) or {}
            # Match upstream: usage.totalTokens / total_tokens is already the cumulative
            # input+output count for a step, so adding input/output again would double-count.
            total += int(get_field(usage, "total_tokens", get_field(usage, "totalTokens", 0)) or 0)
        return total >= limit

    return check


def max_cost(limit: float) -> Callable[[Mapping[str, Any]], bool]:
    def check(options: Mapping[str, Any]) -> bool:
        total = 0.0
        for step in _steps(options):
            usage = get_field(step, "usage", None) or get_field(get_field(step, "response", {}), "usage", {}) or {}
            total += float(
                get_field(usage, "cost", 0) or get_field(get_field(usage, "cost_details", {}), "total", 0) or 0
            )
        return total >= limit

    return check


def finish_reason_is(reason: str) -> Callable[[Mapping[str, Any]], bool]:
    return lambda options: any(
        get_field(step, "finish_reason", get_field(step, "finishReason", None)) == reason for step in _steps(options)
    )


async def is_stop_condition_met(
    stop_conditions: Sequence[Callable[[Mapping[str, Any]], Any]], steps: Sequence[Any]
) -> bool:
    """Evaluate every stop condition (OR logic).

    Matches upstream `isStopConditionMet`: all conditions are evaluated
    concurrently (no short-circuit), a failing condition propagates its error,
    and only a result that is exactly ``True`` stops — ``None`` or other truthy
    values do not.
    """
    results = await asyncio.gather(*(maybe_await(condition({"steps": steps})) for condition in stop_conditions))
    return any(result is True for result in results)
