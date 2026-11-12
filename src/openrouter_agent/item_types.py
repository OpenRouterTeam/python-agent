from __future__ import annotations

from typing import Any, Dict

AssistantMessageItem = Dict[str, Any]
CallFileSearchItem = Dict[str, Any]
CallFunctionToolItem = Dict[str, Any]
CallImageGenerationItem = Dict[str, Any]
CallWebSearchItem = Dict[str, Any]
DeveloperMessageItem = Dict[str, Any]
ErrorItem = Dict[str, Any]
FunctionProgressItem = Dict[str, Any]
FunctionResultItem = Dict[str, Any]
Item = Dict[str, Any]
NewAssistantMessageItem = Dict[str, Any]
"""A new assistant message for input (not yet persisted, no id).

Distinct from `AssistantMessageItem` (the model's output message, which has an
id and structured content). Exists so caller-supplied history, e.g. the output
of `from_chat_messages` / `from_claude_messages`, is a valid `Item`."""
NewDeveloperMessageItem = Dict[str, Any]
NewSystemMessageItem = Dict[str, Any]
"""A new system message for input (not yet persisted, no id)."""
NewUserMessageItem = Dict[str, Any]
ReasoningItem = Dict[str, Any]
SystemMessageItem = Dict[str, Any]
UserMessageItem = Dict[str, Any]
