"""A tool executes exactly once per round — never twice, never zero times.

Ports `packages/agent/tests/unit/tool-execution-once.test.ts`.

Upstream regression this guards: a revision where `handleApprovalCheck`
pre-executed auto-approve tools on every response and the main loop then ran them
again, producing double (and in one path triple) execution. For a tool with side
effects — a payment, a write, an email — that is the difference between correct
and catastrophic, and it is invisible to any assertion that only checks the final
text.

These four cases already hold on this port, so this file *pins* behavior rather
than exposing a bug. That is deliberate: nothing else in the deterministic suite
asserts an execution count, so a re-introduced double-execution would otherwise
reach `main` with CI green. The only other exactly-once assertion lives in
`tests/e2e/`, which is skipped without an API key.

Divergence from upstream's test structure: upstream casts `ModelResult` to an
internal type and calls `executeToolsIfNeeded()` directly. This port has no such
method — the tool loop is inlined in `ModelResult._run` (`model_result.py:825-882`)
— so these drive the public `call_model` path instead. Same invariant, no reliance
on private shape that the next sync could rename.
"""

from __future__ import annotations

from typing import Any, Dict, List

from openrouter_agent import HookEntry, HookName, HooksManager, call_model, tool
from tests._fixtures import QueuedClient, text_response, tool_call_response


class ExecutionCounter:
    """Counts executions.

    A class rather than `lambda p, c: calls.append(...) or {...}`: `list.append`
    returns None, so the lambda form is both a mypy `func-returns-value` error and
    a silent way to return the wrong tool output.
    """

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, params: Any, ctx: Any) -> Dict[str, Any]:
        self.count += 1
        return {"ok": True}


def _client() -> QueuedClient:
    """One tool round, then a terminal text turn."""
    return QueuedClient([tool_call_response("r1", "counted"), text_response("r2", "done")])


async def test_auto_tool_executes_exactly_once_without_hooks() -> None:
    execute = ExecutionCounter()
    counted = tool(name="counted", input_schema=dict, output_schema=dict, execute=execute)

    await call_model(_client(), {"model": "test-model", "input": "go", "tools": [counted]}).get_text()

    assert execute.count == 1


async def test_auto_tool_executes_exactly_once_with_pre_tool_use_hook() -> None:
    """Attaching hooks must not add an execution — the original regression's shape."""
    execute = ExecutionCounter()
    pre_tool_use_calls: List[Any] = []
    hooks = HooksManager()
    hooks.on(
        HookName.PreToolUse.value,
        HookEntry(handler=lambda payload, ctx: pre_tool_use_calls.append(payload)),
    )
    counted = tool(name="counted", input_schema=dict, output_schema=dict, execute=execute)

    await call_model(
        _client(),
        {"model": "test-model", "input": "go", "tools": [counted], "hooks": hooks},
    ).get_text()

    assert execute.count == 1
    assert len(pre_tool_use_calls) == 1


async def test_gated_tool_executes_exactly_once_when_permission_request_allows() -> None:
    execute = ExecutionCounter()
    hooks = HooksManager()
    hooks.on(
        HookName.PermissionRequest.value,
        HookEntry(handler=lambda payload, ctx: {"decision": "allow"}),
    )
    # `require_approval=True` is consulted only when hooks are present
    # (model_result.py:764); without hooks and without a state accessor the run
    # raises instead of gating, so the HooksManager is load-bearing here.
    gated = tool(name="counted", input_schema=dict, output_schema=dict, execute=execute, require_approval=True)

    await call_model(
        _client(),
        {"model": "test-model", "input": "go", "tools": [gated], "hooks": hooks},
    ).get_text()

    assert execute.count == 1


async def test_gated_tool_never_executes_when_permission_request_denies() -> None:
    """A denied tool must run zero times — the assertion that makes 'gating' mean anything."""
    execute = ExecutionCounter()
    hooks = HooksManager()
    hooks.on(
        HookName.PermissionRequest.value,
        HookEntry(handler=lambda payload, ctx: {"decision": "deny", "reason": "policy"}),
    )
    gated = tool(name="counted", input_schema=dict, output_schema=dict, execute=execute, require_approval=True)

    await call_model(
        _client(),
        {"model": "test-model", "input": "go", "tools": [gated], "hooks": hooks},
    ).get_text()

    assert execute.count == 0
