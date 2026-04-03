"""OpenRouter Agent SDK for Python."""

# Result type
# Format compatibility
from ._anthropic_compat import from_claude_messages, to_claude_message

# Async params
from ._async_params import has_async_functions, resolve_async_functions

# Core loop
from ._call_model import call_model
from ._chat_compat import from_chat_messages, to_chat_message

# Conversation state
from ._conversation_state import (
    append_to_messages,
    create_initial_state,
    create_rejected_result,
    create_unsent_result,
    generate_conversation_id,
    partition_tool_calls,
    tool_requires_approval,
    unsent_results_to_api_format,
    update_state,
)
from ._model_result import ModelResult

# Next turn params
from ._next_turn_params import (
    apply_next_turn_params_to_request,
    build_next_turn_params_context,
    execute_next_turn_params_functions,
)
from ._result import Err, Ok, Result

# Streaming
from ._reusable_stream import ReusableAsyncStream

# Stop conditions
from ._stop_conditions import (
    finish_reason_is,
    has_tool_call,
    is_stop_condition_met,
    max_cost,
    max_tokens_used,
    step_count_is,
)
from ._stream_transformers import (
    extract_reasoning_deltas,
    extract_text_deltas,
    extract_tool_stream_events,
    extract_unsupported_content,
    get_unsupported_content_summary,
    has_unsupported_content,
)

# Tool context
from ._tool_context import ToolContextStore, build_tool_execute_context, resolve_context
from ._tool_event_broadcaster import ToolEventBroadcaster

# Tool executor
from ._tool_executor import (
    execute_generator_tool,
    execute_regular_tool,
    execute_tool,
    find_tool_by_name,
    parse_tool_call_arguments,
    tool_to_api_format,
)

# Tool factory
from ._tool_factory import tool

# Turn context
from ._turn_context import build_turn_context, normalize_input_to_array

# Core types
from ._types import (
    CLAUDE_CONTENT_BLOCK_TYPE,
    NON_CLAUDE_MESSAGE_ROLE,
    SHARED_CONTEXT_KEY,
    APITool,
    BaseToolFunction,
    CallModelInput,
    ChatStreamEvent,
    ConversationState,
    ConversationStatus,
    FunctionCallItem,
    ManualTool,
    ManualToolFunction,
    NextTurnParamsContext,
    ParsedToolCall,
    PartialResponse,
    RequestOptions,
    ResponseStreamEvent,
    StepResult,
    StopCondition,
    StopWhen,
    Tool,
    ToolCallOutputEvent,
    ToolExecuteContext,
    ToolExecutionResult,
    ToolFunctionWithExecute,
    ToolFunctionWithGenerator,
    ToolPreliminaryResultEvent,
    ToolResultEvent,
    ToolStreamEvent,
    ToolType,
    ToolWithExecute,
    ToolWithGenerator,
    TurnContext,
    TurnEndEvent,
    TurnStartEvent,
    UnsentToolResult,
    Usage,
    Warning,
    # Type guards
    has_approval_required_tools,
    has_execute_function,
    is_claude_style_messages,
    is_generator_tool,
    is_manual_tool,
    is_regular_execute_tool,
    is_tool_call_output_event,
    is_tool_preliminary_result_event,
    is_tool_result_event,
    is_turn_end_event,
    is_turn_start_event,
    tool_has_approval_configured,
)

__all__ = [
    # Result
    "Ok",
    "Err",
    "Result",
    # Enums
    "ToolType",
    "ConversationStatus",
    # Constants
    "SHARED_CONTEXT_KEY",
    "CLAUDE_CONTENT_BLOCK_TYPE",
    "NON_CLAUDE_MESSAGE_ROLE",
    # Core types
    "FunctionCallItem",
    "TurnContext",
    "ToolExecuteContext",
    "BaseToolFunction",
    "ToolFunctionWithExecute",
    "ToolFunctionWithGenerator",
    "ManualToolFunction",
    "ToolWithExecute",
    "ToolWithGenerator",
    "ManualTool",
    "Tool",
    "ParsedToolCall",
    "ToolExecutionResult",
    "Warning",
    "Usage",
    "StepResult",
    "StopCondition",
    "StopWhen",
    "APITool",
    "CallModelInput",
    "RequestOptions",
    "NextTurnParamsContext",
    "UnsentToolResult",
    "PartialResponse",
    "ConversationState",
    # Event types
    "ToolPreliminaryResultEvent",
    "ToolResultEvent",
    "ToolCallOutputEvent",
    "TurnStartEvent",
    "TurnEndEvent",
    "ResponseStreamEvent",
    "ToolStreamEvent",
    "ChatStreamEvent",
    # Type guards
    "has_execute_function",
    "is_generator_tool",
    "is_regular_execute_tool",
    "is_manual_tool",
    "is_tool_preliminary_result_event",
    "is_tool_result_event",
    "is_tool_call_output_event",
    "is_turn_start_event",
    "is_turn_end_event",
    "is_claude_style_messages",
    "tool_has_approval_configured",
    "has_approval_required_tools",
    # Tool factory
    "tool",
    # Tool context
    "ToolContextStore",
    "build_tool_execute_context",
    "resolve_context",
    # Tool executor
    "tool_to_api_format",
    "parse_tool_call_arguments",
    "execute_regular_tool",
    "execute_generator_tool",
    "execute_tool",
    "find_tool_by_name",
    # Streaming
    "ReusableAsyncStream",
    "ToolEventBroadcaster",
    "extract_text_deltas",
    "extract_reasoning_deltas",
    "extract_tool_stream_events",
    "extract_unsupported_content",
    "has_unsupported_content",
    "get_unsupported_content_summary",
    # Core loop
    "call_model",
    "ModelResult",
    # Async params
    "resolve_async_functions",
    "has_async_functions",
    # Stop conditions
    "step_count_is",
    "has_tool_call",
    "max_tokens_used",
    "max_cost",
    "finish_reason_is",
    "is_stop_condition_met",
    # Turn context
    "build_turn_context",
    "normalize_input_to_array",
    # Next turn params
    "build_next_turn_params_context",
    "execute_next_turn_params_functions",
    "apply_next_turn_params_to_request",
    # Conversation state
    "generate_conversation_id",
    "create_initial_state",
    "update_state",
    "append_to_messages",
    "tool_requires_approval",
    "partition_tool_calls",
    "create_unsent_result",
    "create_rejected_result",
    "unsent_results_to_api_format",
    # Format compatibility
    "from_claude_messages",
    "to_claude_message",
    "from_chat_messages",
    "to_chat_message",
]
