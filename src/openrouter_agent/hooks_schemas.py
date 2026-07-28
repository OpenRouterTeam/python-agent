"""Pydantic v2 payload/result schemas for the lifecycle hooks system.

Mirrors upstream `hooks-schemas.ts`: HookName, HookDefinition, and the
built-in hook registry live here (next to the schemas they describe) so the
runtime validation and the static shape can never drift apart -- the same
"single source of truth" discipline the TypeScript reference applies with Zod.

Field names are idiomatic Python snake_case (e.g. ``tool_name``,
``duration_ms``) rather than the upstream camelCase -- these payloads are an
internal Python-facing API for hook handlers, not wire types serialized to
the OpenRouter API, so there is no interop reason to keep camelCase.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel
from typing_extensions import Literal


class HookName(str, Enum):
    PreToolUse = "PreToolUse"
    PostToolUse = "PostToolUse"
    PostToolUseFailure = "PostToolUseFailure"
    UserPromptSubmit = "UserPromptSubmit"
    Stop = "Stop"
    PermissionRequest = "PermissionRequest"
    SessionStart = "SessionStart"
    SessionEnd = "SessionEnd"
    PostModelCall = "PostModelCall"


@dataclass(frozen=True)
class HookDefinition:
    """A hook definition pairs a payload schema with an optional result schema.

    ``result=None`` marks an observation-only hook: handlers may return
    anything and it is collected as an opaque result without validation
    (mirrors upstream's `z.void()` result schema, minus the schema-shape
    introspection -- Python just uses ``None`` directly).
    """

    payload: Type[BaseModel]
    result: Optional[Type[BaseModel]]


#: A registry maps hook names to their definitions. Used for both the
#: built-in hooks (below) and a HooksManager's custom hook registry.
HookRegistry = Dict[str, HookDefinition]


# ---------------------------------------------------------------------------
# Payload schemas
# ---------------------------------------------------------------------------


class PreToolUsePayload(BaseModel):
    tool_name: str
    tool_input: Dict[str, Any]


class PostToolUsePayload(BaseModel):
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Any = None
    duration_ms: float


class PostToolUseFailurePayload(BaseModel):
    """Fired when a tool EXECUTION throws or returns an error.

    Deliberately NOT fired when a tool never ran: a PermissionRequest 'deny',
    a user rejection on approval resume, or a PreToolUse block all synthesize
    a rejected result without execution, so no failure event is emitted.
    Observe those outcomes via the PermissionRequest / PreToolUse hooks
    themselves.
    """

    tool_name: str
    tool_input: Dict[str, Any]
    error: Any = None


class StopPayload(BaseModel):
    reason: Literal["max_turns"]


class PermissionRequestPayload(BaseModel):
    tool_name: str
    tool_input: Dict[str, Any]
    risk_level: Literal["low", "medium", "high"]


class UserPromptSubmitPayload(BaseModel):
    prompt: str


class SessionStartPayload(BaseModel):
    config: Optional[Dict[str, Any]] = None


class ModelCallUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    cost: Optional[float] = None


class SessionUsageTotals(ModelCallUsage):
    model_calls: int


class SessionEndPayload(BaseModel):
    reason: Literal["user", "error", "max_turns", "complete"]
    total_usage: Optional[SessionUsageTotals] = None


class PostModelCallPayload(BaseModel):
    session_id: str
    response_id: str
    model: str
    duration_ms: float
    turn_type: Literal["initial", "resume", "tool_round", "final", "retry"]
    turn_number: int
    usage: Optional[ModelCallUsage] = None


# ---------------------------------------------------------------------------
# Result schemas
# ---------------------------------------------------------------------------


class PreToolUseResult(BaseModel):
    mutated_input: Optional[Dict[str, Any]] = None
    block: Optional[Union[bool, str]] = None


class StopResult(BaseModel):
    """Result of a Stop hook handler.

    ``force_resume=True`` alone does NOT change any state: the stop
    condition (e.g. ``step_count_is``) will typically fire again immediately
    on the next iteration, so a bare force_resume burns through the
    consecutive-override cap in rapid succession and then stops. Pair it
    with ``append_prompt`` (injects a user message, advancing the
    conversation) to make resumption useful.
    """

    force_resume: Optional[bool] = None
    append_prompt: Optional[str] = None


class PermissionRequestResult(BaseModel):
    decision: Literal["allow", "deny", "ask_user"]
    reason: Optional[str] = None


class UserPromptSubmitResult(BaseModel):
    mutated_prompt: Optional[str] = None
    reject: Optional[Union[bool, str]] = None


# ---------------------------------------------------------------------------
# Built-in hook registry
# ---------------------------------------------------------------------------

BUILT_IN_HOOKS: HookRegistry = {
    HookName.PreToolUse.value: HookDefinition(payload=PreToolUsePayload, result=PreToolUseResult),
    HookName.PostToolUse.value: HookDefinition(payload=PostToolUsePayload, result=None),
    HookName.PostToolUseFailure.value: HookDefinition(payload=PostToolUseFailurePayload, result=None),
    HookName.UserPromptSubmit.value: HookDefinition(payload=UserPromptSubmitPayload, result=UserPromptSubmitResult),
    HookName.Stop.value: HookDefinition(payload=StopPayload, result=StopResult),
    HookName.PermissionRequest.value: HookDefinition(payload=PermissionRequestPayload, result=PermissionRequestResult),
    HookName.SessionStart.value: HookDefinition(payload=SessionStartPayload, result=None),
    HookName.SessionEnd.value: HookDefinition(payload=SessionEndPayload, result=None),
    HookName.PostModelCall.value: HookDefinition(payload=PostModelCallPayload, result=None),
}

BUILT_IN_HOOK_NAMES = frozenset(BUILT_IN_HOOKS.keys())
