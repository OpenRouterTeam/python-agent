"""Port of upstream `lib/agent-tool.ts` -- ``tool.agent()`` subagent tools.

An agent tool is a long-running (``lifecycle="background"``) tool whose work
IS a child `call_model` conversation: the parent loop keeps going, each child
turn becomes a task log entry, the child's conversation is the check-in
transcript, and its final answer (via the ``result`` mapper, default
``{"text": await child.get_text()}``) is delivered like any background
result. Steering messages are injected into the child as user messages at its
next turn boundary; cancelling the task (or the parent run) cancels the child.

Children run in-memory and do not inherit parent hooks (pass child hooks in
the run spec explicitly).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional

from ._utils import dump, maybe_await
from .stream_transformers import extract_text_from_response
from .tool_task import truncate_transcript_tail
from .tool_types import SHARED_CONTEXT_KEY, ConversationState, ToolType

_TASK_TOOL_NAME = "task"
_TEXT_PREVIEW_CHARS = 200
_ARGS_PREVIEW_CHARS = 80

#: Paused child statuses an in-memory agent child cannot recover from.
_CHILD_PAUSE_STATUSES = frozenset(
    {"awaiting_approval", "awaiting_hitl", "awaiting_client_tools", "awaiting_async_tools"}
)

#: Run-spec keys an agent child may NOT set: the engine supplies ``state``
#: (internal in-memory accessor) and ``signal``; approval decisions belong to
#: a durable session (upstream `AgentRunSpec`).
_EXCLUDED_SPEC_KEYS = ("state", "approve_tool_calls", "reject_tool_calls", "signal")


def _preview(value: Any, max_chars: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=dump)
        except (TypeError, ValueError):
            text = str(value)
    return f"{text[:max_chars]}…" if len(text) > max_chars else text


class AgentTranscriptSource:
    """Live transcript over an agent child's conversation, rendered from the
    child's in-memory conversation state."""

    def __init__(self, read_state: Callable[[], Optional[ConversationState]]) -> None:
        self._read_state = read_state
        self._turns_started = 0
        self._turns_ended = 0
        self._current_activity = "starting"

    def note_turn_start(self) -> None:
        self._turns_started += 1
        self._current_activity = f"turn {self._turns_started} in progress"

    def note_turn_end(self, last_text: str) -> None:
        self._turns_ended += 1
        self._current_activity = f"responded: {_preview(last_text, 80)}" if last_text else "thinking"

    def set_activity(self, activity: str) -> None:
        self._current_activity = activity

    def status_extras(self) -> Dict[str, Any]:
        return {
            "turns_completed": self._turns_ended,
            "turns_started": self._turns_started,
            "current_activity": self._current_activity,
        }

    def render(self, max_chars: int) -> str:
        state = self._read_state()
        if state is None or not isinstance(state.messages, list):
            return ""
        lines: List[str] = []
        for raw in state.messages:
            item = raw if isinstance(raw, Mapping) else dump(raw)
            if not isinstance(item, Mapping):
                continue
            typ = item.get("type")
            role = item.get("role")
            if role == "user" and isinstance(item.get("content"), str):
                lines.append(f"user: {_preview(item['content'], _TEXT_PREVIEW_CHARS)}")
            elif typ == "message" and role == "assistant":
                content = item.get("content")
                text = (
                    "".join(str(c.get("text", "")) if isinstance(c, Mapping) else "" for c in content).strip()
                    if isinstance(content, list)
                    else ""
                )
                if text:
                    lines.append(f"assistant: {_preview(text, _TEXT_PREVIEW_CHARS)}")
            elif typ == "function_call":
                lines.append(f"→ {item.get('name')}({_preview(item.get('arguments'), _ARGS_PREVIEW_CHARS)})")
            elif typ == "function_call_output":
                lines.append(f"  ⇒ {_preview(item.get('output'), _ARGS_PREVIEW_CHARS)}")
        return truncate_transcript_tail("\n".join(lines), max_chars)


class _ChildStateAccessor:
    def __init__(self) -> None:
        self.state: Optional[ConversationState] = None

    async def load(self) -> Optional[ConversationState]:
        return self.state

    async def save(self, state: ConversationState) -> None:
        self.state = state


def agent_tool(
    *,
    name: str,
    input_schema: Any,
    output_schema: Any,
    agent: Callable[..., Any],
    result: Optional[Callable[[Any], Any]] = None,
    description: Optional[str] = None,
    strict: Optional[bool] = None,
    grace_ms: Optional[float] = None,
    timeout_ms: Optional[float] = None,
    max_concurrency: Optional[int] = None,
    ack: Any = None,
    check: Any = None,
    context_schema: Any = None,
    next_turn_params: Any = None,
    require_approval: Any = None,
    loop_key: Any = None,
) -> Dict[str, Any]:
    """Create an agent tool (``tool.agent(...)``; upstream `agentToolBuilder`).

    ``agent(params, ctx)`` builds the child run spec (a `call_model` request
    dict, minus ``state`` / ``signal`` / approval decisions). ``result(child)``
    maps the finished child `ModelResult` to this tool's output (default
    ``{"text": await child.get_text()}``), validated against
    ``output_schema``.
    """
    if name == SHARED_CONTEXT_KEY:
        raise ValueError('Tool name "shared" is reserved for shared context. Choose a different name.')
    if name == _TASK_TOOL_NAME:
        raise ValueError(
            f'Tool name "{_TASK_TOOL_NAME}" is reserved for the built-in task-interaction tool. '
            "Choose a different name."
        )
    if output_schema is None:
        raise ValueError(
            f'Agent tool "{name}" must declare an output_schema. The child\'s mapped result is validated '
            "when it settles."
        )

    async def default_result(child: Any) -> Dict[str, Any]:
        return {"text": await child.get_text()}

    map_result = result or default_result

    async def run(params: Any, ctx: Optional[Mapping[str, Any]] = None) -> Any:
        ctx = ctx or {}
        client = ctx.get("client")
        if client is None:
            raise RuntimeError(
                f'Agent tool "{name}": no client available on the run context. Agent tools must execute '
                "inside a call_model run."
            )
        spec = dict(await maybe_await(agent(params, ctx)) or {})
        for key in _EXCLUDED_SPEC_KEYS:
            spec.pop(key, None)

        accessor = _ChildStateAccessor()
        transcript = AgentTranscriptSource(lambda: accessor.state)
        task_transcript = ctx.get("task_transcript")
        if task_transcript is not None:
            task_transcript["transcript_source"] = transcript

        from .call_model import call_model

        user_on_turn_start = spec.get("on_turn_start")
        user_on_turn_end = spec.get("on_turn_end")
        log = ctx.get("log")

        async def on_turn_start(turn_context: Any) -> None:
            transcript.note_turn_start()
            if user_on_turn_start is not None:
                await maybe_await(user_on_turn_start(turn_context))

        async def on_turn_end(turn_context: Any, response: Any) -> None:
            text = extract_text_from_response(response)
            transcript.note_turn_end(text)
            if callable(log):
                log(
                    {
                        "turn": turn_context.get("number_of_turns"),
                        "text_preview": _preview(text.strip(), _TEXT_PREVIEW_CHARS),
                    }
                )
            if user_on_turn_end is not None:
                await maybe_await(user_on_turn_end(turn_context, response))

        child_request: Dict[str, Any] = {
            **spec,
            "state": accessor,
            "on_turn_start": on_turn_start,
            "on_turn_end": on_turn_end,
        }
        if ctx.get("signal") is not None:
            child_request["signal"] = ctx["signal"]
        child = call_model(client, child_request)

        on_message = ctx.get("on_message")
        if callable(on_message):

            def forward(message: Any) -> None:
                child.queue_user_message(
                    message if isinstance(message, str) else json.dumps(message, separators=(",", ":"), default=dump)
                )

            on_message(forward)

        await child.get_response()
        transcript.set_activity("finished")

        final_status = accessor.state.status if accessor.state is not None else None
        if final_status is not None and final_status in _CHILD_PAUSE_STATUSES:
            raise RuntimeError(
                f"Agent tool \"{name}\": the child run paused with status '{final_status}'. Agent children "
                "run in-memory and cannot pause — avoid HITL/manual/deferred/approval tools inside agents "
                "(use lifecycle: 'deferred' on the parent tool instead)."
            )
        return await maybe_await(map_result(child))

    fn: Dict[str, Any] = {
        "lifecycle": "background",
        "kind": "agent",
        "name": name,
        "input_schema": input_schema,
        "output_schema": output_schema,
        "run": run,
    }
    for key, value in (
        ("description", description),
        ("strict", strict),
        ("context_schema", context_schema),
        ("next_turn_params", next_turn_params),
        ("require_approval", require_approval),
        ("loop_key", loop_key),
        ("timeout_ms", timeout_ms),
        ("max_concurrency", max_concurrency),
        ("ack", ack),
        ("grace_ms", grace_ms),
        ("check", check),
    ):
        if value is not None:
            fn[key] = value
    return {"type": ToolType.Function.value, "function": fn}
