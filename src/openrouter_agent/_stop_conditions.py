"""Stop condition factories for controlling the tool execution loop."""

from __future__ import annotations

import inspect

from ._types import StepResult, StopCondition


def step_count_is(count: int) -> StopCondition:
    """Stop after a specific number of steps."""
    def check(steps: list[StepResult]) -> bool:
        return len(steps) >= count
    return check


def has_tool_call(tool_name: str) -> StopCondition:
    """Stop when a specific tool has been called."""
    def check(steps: list[StepResult]) -> bool:
        for step in steps:
            for tc in step.tool_calls:
                if tc.name == tool_name:
                    return True
        return False
    return check


def max_tokens_used(max_tokens: int) -> StopCondition:
    """Stop when total tokens exceed a threshold."""
    def check(steps: list[StepResult]) -> bool:
        total = sum(
            (step.usage.total_tokens if step.usage else 0) for step in steps
        )
        return total >= max_tokens
    return check


def max_cost(max_cost_dollars: float) -> StopCondition:
    """Stop when estimated cost exceeds a threshold."""
    def check(steps: list[StepResult]) -> bool:
        # Cost estimation based on token usage
        # This is approximate; actual pricing depends on model
        total_tokens = sum(
            (step.usage.total_tokens if step.usage else 0) for step in steps
        )
        # Use a rough estimate of $0.01 per 1K tokens as a fallback
        estimated_cost = total_tokens * 0.00001
        return estimated_cost >= max_cost_dollars
    return check


def finish_reason_is(reason: str) -> StopCondition:
    """Stop when the finish reason matches."""
    def check(steps: list[StepResult]) -> bool:
        if steps:
            return steps[-1].finish_reason == reason
        return False
    return check


async def is_stop_condition_met(
    stop_conditions: list[StopCondition],
    steps: list[StepResult],
) -> bool:
    """Evaluate all stop conditions (OR logic). Returns True if any are met."""
    for condition in stop_conditions:
        result = condition(steps)
        if inspect.isawaitable(result):
            result = await result
        if result:
            return True
    return False
