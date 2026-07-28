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


def has_execute_function(tool: Mapping[str, Any]) -> bool:
    fn = get_tool_function(tool)
    return is_client_tool(tool) and callable(fn.get("execute"))


def is_generator_tool(tool: Mapping[str, Any]) -> bool:
    fn = get_tool_function(tool)
    return is_client_tool(tool) and "event_schema" in fn


def is_regular_execute_tool(tool: Mapping[str, Any]) -> bool:
    return has_execute_function(tool) and not is_generator_tool(tool)


def is_hitl_tool(tool: Mapping[str, Any]) -> bool:
    fn = get_tool_function(tool)
    return is_client_tool(tool) and callable(fn.get("on_tool_called"))


def is_manual_tool(tool: Mapping[str, Any]) -> bool:
    fn = get_tool_function(tool)
    return is_client_tool(tool) and not callable(fn.get("execute")) and not callable(fn.get("on_tool_called"))


def is_auto_resolvable_tool(tool: Mapping[str, Any]) -> bool:
    return has_execute_function(tool) or is_hitl_tool(tool)


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
