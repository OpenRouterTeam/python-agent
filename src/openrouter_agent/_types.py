"""Core type definitions for the OpenRouter Agent SDK."""

from __future__ import annotations

import enum
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import (
    Any,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ToolType(str, enum.Enum):
    FUNCTION = "function"


class ConversationStatus(str, enum.Enum):
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"
    AWAITING_APPROVAL = "awaiting_approval"
    IN_PROGRESS = "in_progress"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARED_CONTEXT_KEY: str = "shared"

CLAUDE_CONTENT_BLOCK_TYPE = {
    "Text": "text",
    "Image": "image",
    "ToolUse": "tool_use",
    "ToolResult": "tool_result",
}

NON_CLAUDE_MESSAGE_ROLE = {
    "System": "system",
    "Developer": "developer",
    "Tool": "tool",
}


# ---------------------------------------------------------------------------
# Generic type vars
# ---------------------------------------------------------------------------

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)
TEvent = TypeVar("TEvent", bound=BaseModel)
TContext = TypeVar("TContext", bound=dict[str, Any])
TShared = TypeVar("TShared", bound=dict[str, Any])
TName = TypeVar("TName", bound=str)
T = TypeVar("T")


# ---------------------------------------------------------------------------
# TurnContext
# ---------------------------------------------------------------------------

class FunctionCallItem(BaseModel):
    """Represents a function call from the model."""

    id: str
    name: str
    arguments: str
    call_id: str = ""

    model_config = {"extra": "allow"}


class TurnContext(BaseModel):
    """Context passed to tool execute functions and async parameter resolution."""

    tool_call: FunctionCallItem | None = None
    number_of_turns: int = 0
    turn_request: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


class ToolExecuteContext(Generic[TContext, TShared]):
    """Flat context passed as second arg to tool execute functions."""

    def __init__(
        self,
        *,
        tool_call: FunctionCallItem | None = None,
        number_of_turns: int = 0,
        turn_request: dict[str, Any] | None = None,
        tool_name: str = "",
        local: dict[str, Any] | None = None,
        shared: dict[str, Any] | None = None,
        set_context_fn: Callable[[dict[str, Any]], None] | None = None,
        set_shared_context_fn: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.tool_call = tool_call
        self.number_of_turns = number_of_turns
        self.turn_request = turn_request
        self.tool_name = tool_name
        self.local: dict[str, Any] = local or {}
        self.shared: dict[str, Any] = shared or {}
        self._set_context_fn = set_context_fn
        self._set_shared_context_fn = set_shared_context_fn

    def set_context(self, partial: dict[str, Any]) -> None:
        if self._set_context_fn:
            self._set_context_fn(partial)

    def set_shared_context(self, partial: dict[str, Any]) -> None:
        if self._set_shared_context_fn:
            self._set_shared_context_fn(partial)


# ---------------------------------------------------------------------------
# Tool function types
# ---------------------------------------------------------------------------

class BaseToolFunction(BaseModel):
    """Base interface for all tool definitions."""

    name: str
    description: str | None = None
    input_schema: type[BaseModel]
    context_schema: type[BaseModel] | None = None
    next_turn_params: dict[str, Callable[..., Any]] | None = None
    require_approval: bool | Callable[..., bool | Awaitable[bool]] = False

    model_config = {"arbitrary_types_allowed": True}


class ToolFunctionWithExecute(BaseToolFunction):
    """Regular tool with sync/async execute."""

    output_schema: type[BaseModel] | None = None
    execute: Callable[..., Any] = Field(...)

    model_config = {"arbitrary_types_allowed": True}


class ToolFunctionWithGenerator(BaseToolFunction):
    """Generator tool with streaming events."""

    event_schema: type[BaseModel]
    output_schema: type[BaseModel]
    execute: Callable[..., AsyncGenerator[Any, None]] = Field(...)

    model_config = {"arbitrary_types_allowed": True}


class ManualToolFunction(BaseToolFunction):
    """Tool without execute (human-in-the-loop)."""

    output_schema: type[BaseModel] | None = None


# ---------------------------------------------------------------------------
# Tool wrapper types
# ---------------------------------------------------------------------------

class ToolWithExecute(BaseModel):
    type: ToolType = ToolType.FUNCTION
    function: ToolFunctionWithExecute

    model_config = {"arbitrary_types_allowed": True}


class ToolWithGenerator(BaseModel):
    type: ToolType = ToolType.FUNCTION
    function: ToolFunctionWithGenerator

    model_config = {"arbitrary_types_allowed": True}


class ManualTool(BaseModel):
    type: ToolType = ToolType.FUNCTION
    function: ManualToolFunction

    model_config = {"arbitrary_types_allowed": True}


Tool = ToolWithExecute | ToolWithGenerator | ManualTool


# ---------------------------------------------------------------------------
# Parsed types
# ---------------------------------------------------------------------------

class ParsedToolCall(BaseModel):
    """A parsed tool call with typed arguments."""

    id: str
    name: str
    arguments: dict[str, Any]

    model_config = {"extra": "allow"}


class ToolExecutionResult(BaseModel):
    """Result from executing a tool."""

    tool_call_id: str
    tool_name: str
    result: Any = None
    preliminary_results: list[Any] | None = None
    error: str | None = None


class Warning(BaseModel):
    type: str
    message: str


# ---------------------------------------------------------------------------
# Step and Stop Condition types
# ---------------------------------------------------------------------------

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    model_config = {"extra": "allow"}


class StepResult(BaseModel):
    """Result from one step of the tool execution loop."""

    step_type: str  # "initial" | "continue"
    text: str = ""
    tool_calls: list[ParsedToolCall] = Field(default_factory=list)
    tool_results: list[ToolExecutionResult] = Field(default_factory=list)
    response: dict[str, Any] = Field(default_factory=dict)
    usage: Usage | None = None
    finish_reason: str | None = None
    warnings: list[Warning] | None = None

    model_config = {"extra": "allow"}


StopCondition = Callable[[list[StepResult]], bool | Awaitable[bool]]
StopWhen = StopCondition | list[StopCondition]


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class ToolPreliminaryResultEvent(BaseModel):
    type: str = "tool.preliminary_result"
    tool_call_id: str
    result: Any
    timestamp: float


class ToolResultEvent(BaseModel):
    type: str = "tool.result"
    tool_call_id: str
    result: Any
    timestamp: float
    preliminary_results: list[Any] | None = None


class ToolCallOutputEvent(BaseModel):
    type: str = "tool.call_output"
    output: dict[str, Any]
    timestamp: float


class TurnStartEvent(BaseModel):
    type: str = "turn.start"
    turn_number: int
    timestamp: float


class TurnEndEvent(BaseModel):
    type: str = "turn.end"
    turn_number: int
    timestamp: float


ResponseStreamEvent = (
    ToolPreliminaryResultEvent
    | ToolResultEvent
    | ToolCallOutputEvent
    | TurnStartEvent
    | TurnEndEvent
    | dict[str, Any]  # generic stream events from API
)


class ToolStreamEvent(BaseModel):
    type: str  # "delta" | "preliminary_result"
    content: str | None = None
    tool_call_id: str | None = None
    result: Any = None


class ChatStreamEvent(BaseModel):
    type: str  # "content.delta" | "message.complete" | "tool.preliminary_result" | other
    delta: str | None = None
    response: dict[str, Any] | None = None
    tool_call_id: str | None = None
    result: Any = None
    event: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Conversation State types
# ---------------------------------------------------------------------------

class UnsentToolResult(BaseModel):
    call_id: str
    name: str
    output: Any = None
    error: str | None = None


class PartialResponse(BaseModel):
    text: str | None = None
    tool_calls: list[ParsedToolCall] | None = None


class ConversationState(BaseModel):
    id: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    previous_response_id: str | None = None
    pending_tool_calls: list[ParsedToolCall] | None = None
    unsent_tool_results: list[UnsentToolResult] | None = None
    partial_response: PartialResponse | None = None
    interrupted_by: str | None = None
    status: ConversationStatus = ConversationStatus.IN_PROGRESS
    created_at: float = 0.0
    updated_at: float = 0.0


@runtime_checkable
class StateAccessor(Protocol):
    """Protocol for loading/saving conversation state."""

    async def load(self) -> ConversationState | None: ...
    async def save(self, state: ConversationState) -> None: ...


# ---------------------------------------------------------------------------
# Parameter types
# ---------------------------------------------------------------------------

class NextTurnParamsContext(BaseModel):
    input: list[dict[str, Any]] | str = ""
    model: str = ""
    models: list[str] = Field(default_factory=list)
    temperature: float | None = None
    max_output_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    instructions: str | None = None


FieldOrAsyncFunction = T | Callable[[TurnContext], T | Awaitable[T]]
ContextInput = T | Callable[[TurnContext], T] | Callable[[TurnContext], Awaitable[T]]


# ---------------------------------------------------------------------------
# API Tool Format (wire format)
# ---------------------------------------------------------------------------

class APITool(BaseModel):
    """Wire format sent to the OpenRouter API."""

    type: str = "function"
    name: str
    description: str | None = None
    strict: bool | None = None
    parameters: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Typed request / options dicts
# ---------------------------------------------------------------------------


class RequestOptions(TypedDict, total=False):
    """Options passed alongside the model request (headers, timeout, etc.)."""

    headers: dict[str, str]
    timeout: float | None


class CallModelInput(TypedDict, total=False):
    """Typed dict describing the *request* parameter accepted by ``call_model``.

    All keys are optional (``total=False``) so callers only need to specify
    the fields they care about.  The type remains backward-compatible with
    ``dict[str, Any]`` — any extra keys are silently forwarded to the API.
    """

    # Standard API fields
    model: str
    input: list[dict[str, Any]] | str
    instructions: str
    temperature: float
    max_output_tokens: int
    top_p: float
    top_k: int
    previous_response_id: str

    # SDK-specific fields
    tools: list[Tool]
    stop_when: StopWhen
    context: Any
    state: StateAccessor
    require_approval: Any
    on_turn_start: Any
    on_turn_end: Any
    approve_tool_calls: list[str]
    reject_tool_calls: list[str]


# ---------------------------------------------------------------------------
# Type guards
# ---------------------------------------------------------------------------

def has_execute_function(t: Tool) -> bool:
    return isinstance(t, (ToolWithExecute, ToolWithGenerator))


def is_generator_tool(t: Tool) -> bool:
    return isinstance(t, ToolWithGenerator)


def is_regular_execute_tool(t: Tool) -> bool:
    return isinstance(t, ToolWithExecute)


def is_manual_tool(t: Tool) -> bool:
    return isinstance(t, ManualTool)


def is_tool_preliminary_result_event(event: Any) -> bool:
    return isinstance(event, ToolPreliminaryResultEvent) or (
        isinstance(event, dict) and event.get("type") == "tool.preliminary_result"
    )


def is_tool_result_event(event: Any) -> bool:
    return isinstance(event, ToolResultEvent) or (
        isinstance(event, dict) and event.get("type") == "tool.result"
    )


def is_tool_call_output_event(event: Any) -> bool:
    return isinstance(event, ToolCallOutputEvent) or (
        isinstance(event, dict) and event.get("type") == "tool.call_output"
    )


def is_turn_start_event(event: Any) -> bool:
    return isinstance(event, TurnStartEvent) or (
        isinstance(event, dict) and event.get("type") == "turn.start"
    )


def is_turn_end_event(event: Any) -> bool:
    return isinstance(event, TurnEndEvent) or (
        isinstance(event, dict) and event.get("type") == "turn.end"
    )


def is_claude_style_messages(input_data: Any) -> bool:
    """Detect if input is in Claude message format."""
    if not isinstance(input_data, list):
        return False
    for msg in input_data:
        if isinstance(msg, dict) and msg.get("role") in ("user", "assistant"):
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in (
                        "text",
                        "image",
                        "tool_use",
                        "tool_result",
                    ):
                        return True
    return False


def tool_has_approval_configured(t: Tool) -> bool:
    return t.function.require_approval is not False


def has_approval_required_tools(tools: list[Tool]) -> bool:
    return any(tool_has_approval_configured(t) for t in tools)
