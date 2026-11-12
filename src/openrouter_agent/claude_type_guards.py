"""Ports `packages/agent/src/lib/claude-type-guards.ts`."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .claude_constants import ClaudeContentBlockType, NonClaudeMessageRole

_NON_CLAUDE_ROLES = frozenset(role.value for role in NonClaudeMessageRole)


def _is_record(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_non_claude_role(role: Any) -> bool:
    return isinstance(role, str) and role in _NON_CLAUDE_ROLES


def _is_claude_tool_result_block(block: Any) -> bool:
    return _is_record(block) and block.get("type") == ClaudeContentBlockType.ToolResult.value


def _is_claude_image_block_with_source(block: Any) -> bool:
    return (
        _is_record(block)
        and block.get("type") == ClaudeContentBlockType.Image.value
        and "source" in block
        and _is_record(block["source"])
    )


def _is_claude_tool_use_block_with_id(block: Any) -> bool:
    return (
        _is_record(block)
        and block.get("type") == ClaudeContentBlockType.ToolUse.value
        and isinstance(block.get("id"), str)
    )


def _has_claude_specific_blocks(content: Sequence[Any]) -> bool:
    return any(
        _is_claude_tool_result_block(block)
        or _is_claude_image_block_with_source(block)
        or _is_claude_tool_use_block_with_id(block)
        for block in content
    )


def _is_array(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def is_claude_style_messages(value: Any) -> bool:
    """Whether ``value`` is in Claude message format.

    Structural analysis, matching upstream: plain ``{role, content: str}``
    messages are ambiguous (also valid chat/Responses input) and do NOT count.
    Returns True only once a message carries a Claude-specific content block
    (``tool_result``, ``image`` with a ``source`` object, or ``tool_use`` with a
    string ``id``). Any ``system`` / ``developer`` / ``tool`` role seen before
    that makes it non-Claude. Non-mapping entries, role-less entries, and
    entries with a top-level ``type`` (Responses items) are skipped.
    """
    if not _is_array(value) or len(value) == 0:
        return False

    for message in value:
        if not _is_record(message):
            continue
        if "role" not in message:
            continue
        if "type" in message:
            continue  # Claude messages don't have a top-level "type"

        if _is_non_claude_role(message["role"]):
            return False

        content = message.get("content")
        if _is_array(content) and _has_claude_specific_blocks(content):
            return True

    return False
