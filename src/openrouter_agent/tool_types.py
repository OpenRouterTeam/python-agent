from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Union

from typing_extensions import Literal, TypeAlias, TypedDict

SHARED_CONTEXT_KEY = "shared"


class ToolType(str, Enum):
    Function = "function"


class StateAccessor(Protocol):
    async def load(self) -> Optional["ConversationState"]: ...

    async def save(self, state: "ConversationState") -> None: ...


@dataclass(frozen=True)
class ParsedToolCall:
    id: str
    name: str
    arguments: Any
    #: Set (True) on a persisted pending call when PreToolUse already ran and
    #: `arguments` holds its effective input, so a resumed run does not apply
    #: the hook twice (upstream `PendingToolCall.preToolUseApplied`). Legacy
    #: state without the marker keeps its prior behavior.
    pre_tool_use_applied: Optional[bool] = None


#: A persisted pending call (upstream `PendingToolCall`) -- a ParsedToolCall
#: that may carry `pre_tool_use_applied`.
PendingToolCall = ParsedToolCall


@dataclass(frozen=True)
class UnsentToolResult:
    call_id: str
    name: str
    output: Any
    error: Optional[str] = None


@dataclass(frozen=True)
class PartialResponse:
    text: Optional[str] = None
    tool_calls: Optional[List[ParsedToolCall]] = None


@dataclass(frozen=True)
class ConversationState:
    id: str
    messages: Any
    status: str
    created_at: int
    updated_at: int
    previous_response_id: Optional[str] = None
    pending_tool_calls: Optional[List[ParsedToolCall]] = None
    unsent_tool_results: Optional[List[UnsentToolResult]] = None
    partial_response: Optional[PartialResponse] = None
    interrupted_by: Optional[str] = None
    #: Serialization-contract version for this state blob (see
    #: `serialize_conversation_state` / `deserialize_conversation_state`).
    #: Optional so legacy (pre-version-field) states remain constructible;
    #: absence is treated as version 1 by `deserialize_conversation_state`.
    version: Optional[int] = None
    #: RFC 8785 canonical key of the forced `tool_choice` most recently
    #: consumed by the active logical run (upstream
    #: `consumedForcedToolChoiceKey`). Persisted across pauses so an unchanged
    #: forced choice stays relaxed after resume.
    consumed_forced_tool_choice_key: Optional[str] = None
    #: Doom-loop detector state (see `doom_loop` on `call_model`). Bounded
    #: plain JSON; additive within version 1.
    doom_loop: Optional[Dict[str, Any]] = None
    #: Async tool tasks whose placeholder output was sent to the model but
    #: whose real result has not been delivered yet. Additive within v1.
    pending_async_tools: Optional[List["PendingAsyncTool"]] = None
    #: Call ids whose async result has already been delivered -- the
    #: at-most-once guard against replayed resolutions. Additive within v1.
    settled_async_call_ids: Optional[List[str]] = None


@dataclass(frozen=True)
class StepResult:
    step_type: str
    text: str
    tool_calls: List[ParsedToolCall]
    tool_results: List[Any]
    response: Any
    server_tool_results: Optional[List[Any]] = None
    usage: Any = None
    finish_reason: Optional[str] = None
    warnings: Optional[List[Any]] = None
    experimental_provider_metadata: Optional[Mapping[str, Any]] = None


StopCondition = Callable[[Dict[str, Sequence[StepResult]]], Any]
StopWhen = Union[StopCondition, Sequence[StopCondition]]


class ToModelOutputResult(TypedDict, total=False):
    type: Literal["content", "json"]
    value: object


ToModelOutputFunction = Callable[[Dict[str, Any]], Any]
ToolApprovalCheck = Callable[[Any, Mapping[str, Any]], Any]
ToolHasApproval = bool


class ToolFunction(TypedDict, total=False):
    name: str
    description: str
    input_schema: object
    output_schema: object
    event_schema: object
    context_schema: object
    require_approval: Union[bool, ToolApprovalCheck]
    execute: Union[Callable[..., object], Literal[False]]
    on_tool_called: Callable[..., object]
    on_response_received: Callable[..., object]
    to_model_output: ToModelOutputFunction
    next_turn_params: Mapping[str, Callable[..., object]]


class ClientTool(TypedDict, total=False):
    type: Literal["function"]
    function: Dict[str, Any]


ServerToolConfig: TypeAlias = Mapping[str, object]


class ServerTool(TypedDict):
    _brand: Literal["server-tool"]
    config: Dict[str, Any]


Tool: TypeAlias = Union[ClientTool, ServerTool]
ManualTool: TypeAlias = ClientTool
HITLTool: TypeAlias = ClientTool
HITLToolFunction: TypeAlias = ToolFunction
ToolWithExecute: TypeAlias = ClientTool
ToolWithGenerator: TypeAlias = ClientTool


class ToolExecutionResult(TypedDict, total=False):
    tool_call_id: str
    tool_name: str
    #: Origin of the tool: "mcp" for tools branded via `mark_mcp` (wrapped
    #: from a remote MCP server, whose result is untyped), "client" for
    #: locally-defined tools. See `is_mcp_tool` / `mark_mcp`.
    source: Literal["client", "mcp"]
    result: object
    preliminary_results: List[object]
    error: BaseException


ToolExecutionResultUnion: TypeAlias = ToolExecutionResult


class ToolOutputContentItem(TypedDict, total=False):
    type: str
    text: str
    image_url: str
    file_id: str


class ToolPreliminaryResultEvent(TypedDict, total=False):
    type: Literal["tool.preliminary_result", "preliminary_result"]
    toolCallId: str
    tool_call_id: str
    result: object
    timestamp: int


class ToolResultEvent(TypedDict, total=False):
    type: Literal["tool.result", "tool_result"]
    toolCallId: str
    tool_call_id: str
    #: Origin of the tool result: "mcp" for MCP-branded tools, "client"
    #: otherwise. See `is_mcp_tool` / `mark_mcp`.
    source: Literal["client", "mcp"]
    result: object
    preliminaryResults: List[object]
    preliminary_results: List[object]
    timestamp: int


class ToolCallOutputEvent(TypedDict, total=False):
    type: Literal["tool.call_output", "tool_call_output"]
    output: object
    timestamp: int


class TurnStartEvent(TypedDict, total=False):
    type: Literal["turn.start"]
    turnNumber: int
    turn_number: int
    timestamp: int


class TurnEndEvent(TypedDict, total=False):
    type: Literal["turn.end"]
    turnNumber: int
    turn_number: int
    timestamp: int


ToolStreamEvent: TypeAlias = Union[
    ToolPreliminaryResultEvent, ToolResultEvent, ToolCallOutputEvent, TurnStartEvent, TurnEndEvent
]
ResponseStreamEvent: TypeAlias = Union[ToolStreamEvent, Mapping[str, object]]
EnhancedResponseStreamEvent = ResponseStreamEvent
ChatStreamEvent: TypeAlias = Mapping[str, object]
TurnContext: TypeAlias = Dict[str, Any]
ContextInput = Union[Mapping[str, Any], Callable[[TurnContext], Any]]
NextTurnParamsContext: TypeAlias = Dict[str, Any]
NextTurnParamsFunctions = Mapping[str, Callable[[Any, NextTurnParamsContext], Any]]
ServerToolType = str
ServerToolResultItem: TypeAlias = Mapping[str, object]
ToolResultItem: TypeAlias = Mapping[str, object]
Warning: TypeAlias = Mapping[str, object]


class Hook(Protocol):
    def sdk_init(self, configuration: object) -> object: ...


OpenRouterOptions: TypeAlias = Dict[str, Any]
SDKOptions: TypeAlias = Dict[str, Any]


class CallModelInput(TypedDict, total=False):
    model: str
    input: object
    tools: Sequence[Tool]
    stop_when: StopWhen
    state: StateAccessorLike
    require_approval: ToolApprovalCheck
    approve_tool_calls: Sequence[str]
    reject_tool_calls: Sequence[str]
    context: ContextInput
    shared_context_schema: object
    allow_final_response: Union[bool, str]


CallModelInputWithState = CallModelInput


class ResolvedCallModelInput(TypedDict, total=False):
    model: str
    input: object
    tools: Sequence[Mapping[str, object]]
    stream: bool


GetResponseOptions: TypeAlias = Dict[str, Any]
StreamableOutputItem: TypeAlias = Mapping[str, object]
StateAccessorLike = StateAccessor
HasApprovalTools = bool
InferToolEvent: TypeAlias = object
InferToolEventsUnion: TypeAlias = object
InferToolInput: TypeAlias = object
InferToolOutput: TypeAlias = object
InferToolOutputsUnion: TypeAlias = object
TypedToolCall = ParsedToolCall
TypedToolCallUnion = ParsedToolCall
ConversationStatus = str
ToModelOutputFunctionType = ToModelOutputFunction


def is_server_tool(tool: Mapping[str, Any]) -> bool:
    return tool.get("_brand") == "server-tool"


def is_client_tool(tool: Mapping[str, Any]) -> bool:
    return not is_server_tool(tool)


#: A client tool additionally branded as originating from an MCP server (see
#: `mark_mcp`). Purely a structural marker: it does not change execution or
#: wire serialization, only how `is_mcp_tool` and the `source` field on tool
#: results discriminate it from a precisely-typed client tool.
McpBranded = Mapping[str, Any]


def is_mcp_tool(tool: Mapping[str, Any]) -> bool:
    """Type guard: true if the tool carries the additive MCP brand (see
    `mark_mcp`)."""
    return bool(isinstance(tool, Mapping) and tool.get("_mcp") is True)


def get_tool_function(tool: Mapping[str, Any]) -> Dict[str, Any]:
    value = tool.get("function", {})
    return value if isinstance(value, dict) else {}


def _has_run(tool: Mapping[str, Any]) -> bool:
    return is_client_tool(tool) and callable(get_tool_function(tool).get("run"))


def has_execute_function(tool: Mapping[str, Any]) -> bool:
    """Regular or generator `execute` tool. Unified `run` tools are excluded --
    they have their own dispatch (upstream `hasExecuteFunction`)."""
    fn = get_tool_function(tool)
    return is_client_tool(tool) and not _has_run(tool) and callable(fn.get("execute"))


def is_generator_tool(tool: Mapping[str, Any]) -> bool:
    """Legacy generator tool (has event_schema). Unified tools may declare an
    `event_schema` for run yields but are not legacy generator tools."""
    fn = get_tool_function(tool)
    return is_client_tool(tool) and not _has_run(tool) and "event_schema" in fn


def is_regular_execute_tool(tool: Mapping[str, Any]) -> bool:
    return has_execute_function(tool) and not is_generator_tool(tool)


def is_hitl_tool(tool: Mapping[str, Any]) -> bool:
    fn = get_tool_function(tool)
    return is_client_tool(tool) and callable(fn.get("on_tool_called"))


def is_manual_tool(tool: Mapping[str, Any]) -> bool:
    """No execute, no on_tool_called, no run."""
    fn = get_tool_function(tool)
    return (
        is_client_tool(tool)
        and not callable(fn.get("execute"))
        and not callable(fn.get("on_tool_called"))
        and not callable(fn.get("run"))
    )


def is_auto_resolvable_tool(tool: Mapping[str, Any]) -> bool:
    """Auto-resolvable within a turn: execute/generator, HITL on_tool_called,
    or a unified `run` (which always produces at least a placeholder output)."""
    return has_execute_function(tool) or is_hitl_tool(tool) or _has_run(tool)


def tool_has_approval_configured(tool: Mapping[str, Any]) -> bool:
    if is_server_tool(tool):
        return False
    value = get_tool_function(tool).get("require_approval")
    return value is True or callable(value)


def has_approval_required_tools(tools: Sequence[Mapping[str, Any]]) -> bool:
    return any(tool_has_approval_configured(item) for item in tools)


def is_tool_preliminary_result_event(event: Mapping[str, Any]) -> bool:
    return event.get("type") == "tool.preliminary_result"


def is_tool_result_event(event: Mapping[str, Any]) -> bool:
    return event.get("type") == "tool.result"


def is_tool_call_output_event(event: Mapping[str, Any]) -> bool:
    return event.get("type") == "tool.call_output"


def is_turn_start_event(event: Mapping[str, Any]) -> bool:
    return event.get("type") == "turn.start"


def is_turn_end_event(event: Mapping[str, Any]) -> bool:
    return event.get("type") == "turn.end"


def is_tool_async_started_event(event: Mapping[str, Any]) -> bool:
    return event.get("type") == "tool.async_started"


def is_tool_async_settled_event(event: Mapping[str, Any]) -> bool:
    return event.get("type") == "tool.async_settled"


# ---------------------------------------------------------------------------
# Async tools (upstream `tool-types.ts`: unified `run` tools, PendingAsyncTool)
# ---------------------------------------------------------------------------

#: How a unified tool's execution relates to the tool round
#: (upstream `ToolLifecycle`).
ToolLifecycle = Literal["sync", "background", "deferred"]


def is_unified_tool(tool: Mapping[str, Any]) -> bool:
    """True if the tool is a unified `run`-based tool (callable `function.run`).

    Mirrors upstream `isUnifiedTool` (`tool-types.ts`): server tools are never
    unified; a client tool is unified exactly when its `run` is callable.
    """
    if is_server_tool(tool):
        return False
    return callable(get_tool_function(tool).get("run"))


def is_long_running_tool(tool: Mapping[str, Any]) -> bool:
    """True when the tool can outlive its round: a unified tool whose
    `lifecycle` is not `'sync'` (upstream `isLongRunningTool`).

    Note upstream compares `lifecycle !== 'sync'` — a unified tool built
    outside `tool()` with no `lifecycle` key therefore counts as long-running,
    exactly as upstream (`tool()` always stamps a lifecycle, default `'sync'`).
    """
    return is_unified_tool(tool) and get_tool_function(tool).get("lifecycle") != "sync"


def is_agent_tool(tool: Mapping[str, Any]) -> bool:
    """True for agent tools (`tool.agent()`): unified tools whose run drives a
    child conversation.

    Upstream marks these with `function.kind == 'agent'`; the Python port also
    accepts an `"agent"` key on the function dict (the run-spec factory the
    Python `tool.agent()` builder carries).
    """
    if not is_unified_tool(tool):
        return False
    fn = get_tool_function(tool)
    return fn.get("kind") == "agent" or "agent" in fn


def is_deferred_handle(value: Any) -> bool:
    """Runtime guard for a `ctx.defer()` handle (upstream `isDeferredHandle`).

    Structural: a mapping (or object) whose `__deferred` marker is exactly
    `True` and whose `task_id` (upstream `taskId`, also accepted) is a string.
    """
    if value is None:
        return False
    if isinstance(value, Mapping):
        marker = value.get("__deferred")
        task_id = value.get("task_id", value.get("taskId"))
    else:
        marker = getattr(value, "__deferred", None)
        task_id = getattr(value, "task_id", None)
    return marker is True and isinstance(task_id, str)


@dataclass(frozen=True)
class PendingAsyncToolLastLog:
    """Most recent log entry of an async task, truncated (~200 chars)."""

    at: int
    text: str


@dataclass(frozen=True)
class PendingAsyncTool:
    """A pending (or settled) async tool task tracked on
    `ConversationState.pending_async_tools` (upstream `PendingAsyncTool`).

    One entry per background / deferred call that produced a placeholder
    output. Optional upstream fields that are *absent* are `None` here.
    """

    #: The originating `function_call`'s call id.
    call_id: str
    #: Durable task id (deferred: caller-supplied; background/agent: generated).
    task_id: str
    #: The tool's name.
    name: str
    #: How the task escapes the round: "background" | "defer" | "agent".
    mode: str
    #: "working" | "input_required" | "completed" | "failed" | "cancelled".
    status: str
    #: Unix ms when the task started.
    started_at: int
    #: Unix ms after which the task is considered expired.
    expires_at: Optional[int] = None
    #: Poll-interval hint surfaced to the model and external pollers.
    poll_after_ms: Optional[int] = None
    #: Set (True) on a background task left running under `on_run_end: 'detach'`.
    orphaned: Optional[bool] = None
    #: The most recent log entry — the one piece of progress surviving a restart.
    last_log: Optional[PendingAsyncToolLastLog] = None
