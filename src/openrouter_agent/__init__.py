"""OpenRouter Agent SDK for Python."""

# Format compatibility
from ._anthropic_compat import from_claude_messages, to_claude_message

# Async params (internal, kept for submodule use)
from ._async_params import has_async_functions as has_async_functions  # noqa: F401
from ._async_params import resolve_async_functions as resolve_async_functions  # noqa: F401

# Core loop
from ._call_model import call_model
from ._chat_compat import from_chat_messages, to_chat_message

# Conversation state (internal, kept for submodule use)
from ._conversation_state import append_to_messages as append_to_messages  # noqa: F401
from ._conversation_state import create_initial_state as create_initial_state  # noqa: F401
from ._conversation_state import create_rejected_result as create_rejected_result  # noqa: F401
from ._conversation_state import create_unsent_result as create_unsent_result  # noqa: F401
from ._conversation_state import generate_conversation_id as generate_conversation_id  # noqa: F401
from ._conversation_state import partition_tool_calls as partition_tool_calls  # noqa: F401
from ._conversation_state import tool_requires_approval as tool_requires_approval  # noqa: F401
from ._conversation_state import (
    unsent_results_to_api_format as unsent_results_to_api_format,  # noqa: F401
)
from ._conversation_state import update_state as update_state  # noqa: F401
from ._model_result import ModelResult

# Next turn params (internal, kept for submodule use)
from ._next_turn_params import (
    apply_next_turn_params_to_request as apply_next_turn_params_to_request,  # noqa: F401
)
from ._next_turn_params import (
    build_next_turn_params_context as build_next_turn_params_context,  # noqa: F401
)
from ._next_turn_params import (
    execute_next_turn_params_functions as execute_next_turn_params_functions,  # noqa: F401
)
from ._result import Err, Ok, Result

# Streaming (internal, kept for submodule use)
from ._reusable_stream import ReusableAsyncStream as ReusableAsyncStream  # noqa: F401

# Stop conditions
from ._stop_conditions import (
    finish_reason_is,
    has_tool_call,
    max_cost,
    max_tokens_used,
    step_count_is,
)
from ._stop_conditions import is_stop_condition_met as is_stop_condition_met  # noqa: F401
from ._stream_transformers import extract_reasoning_deltas as extract_reasoning_deltas  # noqa: F401
from ._stream_transformers import extract_text_deltas as extract_text_deltas  # noqa: F401
from ._stream_transformers import (
    extract_tool_stream_events as extract_tool_stream_events,  # noqa: F401
)
from ._stream_transformers import (
    extract_unsupported_content as extract_unsupported_content,  # noqa: F401
)
from ._stream_transformers import (
    get_unsupported_content_summary as get_unsupported_content_summary,  # noqa: F401
)
from ._stream_transformers import has_unsupported_content as has_unsupported_content  # noqa: F401

# Tool context (internal, kept for submodule use)
from ._tool_context import ToolContextStore as ToolContextStore  # noqa: F401
from ._tool_context import build_tool_execute_context as build_tool_execute_context  # noqa: F401
from ._tool_context import resolve_context as resolve_context  # noqa: F401
from ._tool_event_broadcaster import ToolEventBroadcaster as ToolEventBroadcaster  # noqa: F401

# Tool executor (internal, kept for submodule use)
from ._tool_executor import execute_generator_tool as execute_generator_tool  # noqa: F401
from ._tool_executor import execute_regular_tool as execute_regular_tool  # noqa: F401
from ._tool_executor import execute_tool as execute_tool  # noqa: F401
from ._tool_executor import find_tool_by_name as find_tool_by_name  # noqa: F401
from ._tool_executor import parse_tool_call_arguments as parse_tool_call_arguments  # noqa: F401
from ._tool_executor import tool_to_api_format as tool_to_api_format  # noqa: F401

# Tool factory
from ._tool_factory import tool

# Turn context (internal, kept for submodule use)
from ._turn_context import build_turn_context as build_turn_context  # noqa: F401
from ._turn_context import normalize_input_to_array as normalize_input_to_array  # noqa: F401

# Core types
from ._types import (
    CLAUDE_CONTENT_BLOCK_TYPE,
    NON_CLAUDE_MESSAGE_ROLE,
    SHARED_CONTEXT_KEY,
    APITool,
    ChatStreamEvent,
    ConversationState,
    ConversationStatus,
    FunctionCallItem,
    ManualTool,
    NextTurnParamsContext,
    OpenRouterClient,
    ParsedToolCall,
    PartialResponse,
    ResponseStreamEvent,
    StepResult,
    StopCondition,
    StopWhen,
    Tool,
    ToolCallOutputEvent,
    ToolExecuteContext,
    ToolExecutionResult,
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

# Internal types re-exported for backward compat (not in __all__)
from ._types import BaseToolFunction as BaseToolFunction  # noqa: F401
from ._types import ManualToolFunction as ManualToolFunction  # noqa: F401
from ._types import ToolFunctionWithExecute as ToolFunctionWithExecute  # noqa: F401
from ._types import ToolFunctionWithGenerator as ToolFunctionWithGenerator  # noqa: F401

__all__ = [
    # Result
    "Ok",
    "Err",
    "Result",
    # Core loop
    "call_model",
    "ModelResult",
    # Tool factory
    "tool",
    # Client protocol
    "OpenRouterClient",
    # Enums
    "ToolType",
    "ConversationStatus",
    # Constants
    "SHARED_CONTEXT_KEY",
    "CLAUDE_CONTENT_BLOCK_TYPE",
    "NON_CLAUDE_MESSAGE_ROLE",
    # Core types
    "Tool",
    "ToolWithExecute",
    "ToolWithGenerator",
    "ManualTool",
    "FunctionCallItem",
    "TurnContext",
    "ToolExecuteContext",
    "ParsedToolCall",
    "ToolExecutionResult",
    "Warning",
    "Usage",
    "StepResult",
    "StopCondition",
    "StopWhen",
    "APITool",
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
    # Stop conditions
    "step_count_is",
    "has_tool_call",
    "max_tokens_used",
    "max_cost",
    "finish_reason_is",
    # Format compatibility
    "from_claude_messages",
    "to_claude_message",
    "from_chat_messages",
    "to_chat_message",
]
