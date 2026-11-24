from __future__ import annotations

import asyncio
import dataclasses
import json
import time
import warnings
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Sequence, Tuple

from ._utils import dump, get_field, is_async_iterable, json_dumps, maybe_await, sdk_request_kwargs, validate_schema
from .async_params import resolve_async_functions
from .conversation_state import (
    append_to_messages,
    create_initial_state,
    create_rejected_result,
    create_unsent_result,
    generate_conversation_id,
    _parse_tool_arguments,
    partition_tool_calls,
    tool_requires_approval,
    unsent_results_to_api_format,
    update_state,
)
from .next_turn_params import apply_next_turn_params_to_request, execute_next_turn_params_functions
from .stop_conditions import is_stop_condition_met
from .stream_transformers import (
    assert_streamed_tool_calls_match_response,
    extract_text_from_response,
    extract_tool_calls_from_response,
)
from .tool_context import ToolContextStore, resolve_context
from .tool_executor import apply_on_response_received_hooks, execute_tool
from .tool_types import (
    ParsedToolCall,
    StepResult,
    Tool,
    UnsentToolResult,
    assert_unique_client_tool_names,
    get_tool_function,
    is_auto_resolvable_tool,
    is_client_tool,
    is_manual_tool,
    is_mcp_tool,
    tool_has_approval_configured,
)
from .turn_context import normalize_input_to_array

GetResponseOptions = Dict[str, Any]

#: Default directive appended as a final user message on the forced final
#: turn (`allow_final_response` defaulting to on, or explicitly `True`).
#: Forbidding tools via `tool_choice: "none"` alone is not enough: models
#: that emit tool-call syntax as text will attempt another call and leak it
#: into `content` as unparsed text unless they are told this is the final
#: turn. Pass a non-empty string to `allow_final_response` to override the
#: wording, or `""` to append no message at all (legacy behavior).
DEFAULT_FINAL_RESPONSE_DIRECTIVE = (
    "You have reached the tool-use limit, and tools are no longer available. "
    "Do not attempt to call any more tools. Using the information you already have, "
    "write your final answer now."
)

#: Safety cap on turns so a misbehaving loop cannot spin forever.
_MAX_TURNS = 20

#: Cap consecutive Stop-hook force_resume overrides so a misbehaving handler
#: cannot spin the loop forever.
_MAX_FORCE_RESUME_OVERRIDES = 3


def _now_ms() -> int:
    return int(time.time() * 1000)


def _extract_model_call_usage(usage: Any) -> Optional[Dict[str, Any]]:
    """Map the response's usage block onto the hook-facing ModelCallUsage
    shape. Returns None when the response carried no usage accounting."""
    if usage is None:
        return None
    input_details = get_field(usage, "input_tokens_details")
    output_details = get_field(usage, "output_tokens_details")
    result: Dict[str, Any] = {
        "input_tokens": int(get_field(usage, "input_tokens", 0) or 0),
        "output_tokens": int(get_field(usage, "output_tokens", 0) or 0),
        "total_tokens": int(get_field(usage, "total_tokens", 0) or 0),
        "cached_tokens": int(get_field(input_details, "cached_tokens", 0) or 0) if input_details is not None else 0,
        "reasoning_tokens": int(get_field(output_details, "reasoning_tokens", 0) or 0)
        if output_details is not None
        else 0,
    }
    cost = get_field(usage, "cost", None)
    if cost is not None:
        result["cost"] = cost
    return result


def _is_user_string_message(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("role") == "user" and isinstance(value.get("content"), str)


def _describe_non_record(value: Any) -> str:
    """Name a value's type the way upstream's `describeNonRecord` does, so
    diagnostics read the same across ports."""
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return "object"


#: Most pending call IDs listed in a "conversation is paused" error. A state
#: can hold any number of pending calls; the message stays readable and the
#: full list is available from `get_pending_tool_calls()`.
_MAX_PENDING_IDS_IN_ERROR = 10


def _relax_forced_tool_choice(tool_choice: Any) -> Any:
    """Relax a tool choice that *forces* a tool call, for follow-up turns after
    a tool round has executed (upstream DEV-785).

    - `"required"` -> `"auto"`
    - a specific-tool object (`{"type": "function", "name": ...}`, ...) -> `"auto"`
    - `allowed_tools` with `mode: "required"` -> same set with `mode: "auto"`
      (the constraint on *which* tools may be used still applies; only the
      force-a-call part is spent)
    - `"auto"`, `"none"`, `allowed_tools` with `mode: "auto"`, and `None` pass
      through unchanged.
    """
    if tool_choice is None or isinstance(tool_choice, str):
        return "auto" if tool_choice == "required" else tool_choice
    if get_field(tool_choice, "type") == "allowed_tools":
        if get_field(tool_choice, "mode") == "required":
            return {**dict(dump(tool_choice)), "mode": "auto"}
        return tool_choice
    return "auto"


def _forced_tool_choice_key(tool_choice: Any) -> Optional[str]:
    """Stable identity for a concrete forced choice; `None` means tools are not
    forced."""
    if tool_choice is None or isinstance(tool_choice, str):
        forced = tool_choice == "required"
    else:
        forced = get_field(tool_choice, "type") != "allowed_tools" or get_field(tool_choice, "mode") == "required"
    if not forced:
        return None
    return json.dumps(dump(tool_choice), sort_keys=True, separators=(",", ":"), default=str)


def _find_latest_user_string_index(items: Sequence[Any]) -> int:
    for index in range(len(items) - 1, -1, -1):
        if _is_user_string_message(items[index]):
            return index
    return -1


class ModelResult:
    def __init__(self, options: Mapping[str, Any]) -> None:
        assert_unique_client_tool_names(options.get("tools"))
        self.options = dict(options)
        self._run_task: Optional[asyncio.Task[Any]] = None
        self._final_response: Any = None
        self._events: List[Any] = []
        self._tool_calls: List[ParsedToolCall] = []
        self._tool_outputs: List[Any] = []
        self._steps: List[StepResult] = []
        self._state: Any = None
        self._fresh_items_for_state: List[Any] = []
        self._context_store: Optional[ToolContextStore] = None
        self._condition = asyncio.Condition()
        self._run_done = False
        self._hooks = self.options.get("hooks")
        self._session_id = ""
        self._resuming_from_client_tools = False
        self._initial_request: Optional[Dict[str, Any]] = None
        self._session_start_emitted = False
        self._session_end_emitted = False
        self._all_tool_rounds = 0
        self._session_usage: Dict[str, Any] = {
            "model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
            "cost": 0.0,
            "has_cost": False,
        }
        if (self.options.get("approve_tool_calls") or self.options.get("reject_tool_calls")) and not self.options.get(
            "state"
        ):
            raise ValueError("approve_tool_calls and reject_tool_calls require a state accessor")
        # De-duplicate decisions so a call ID listed twice cannot execute an
        # approval-gated tool twice, and refuse contradictory input outright:
        # an ID that is both approved and rejected has no safe interpretation.
        self._approved_ids: List[str] = list(dict.fromkeys(self.options.get("approve_tool_calls") or []))
        self._rejected_ids: List[str] = list(dict.fromkeys(self.options.get("reject_tool_calls") or []))
        rejected_set = set(self._rejected_ids)
        conflicting = [call_id for call_id in self._approved_ids if call_id in rejected_set]
        if conflicting:
            raise ValueError(
                f"Tool call IDs cannot be both approved and rejected: {', '.join(conflicting)}. "
                "Pass each pending call ID in exactly one of approve_tool_calls or reject_tool_calls."
            )
        #: Set when a paused conversation was resumed without the decision it
        #: owes: the run sends nothing and every getter reports the pause.
        self._paused_without_decisions = False
        #: Forced tool_choice consumed in this logical run (see
        #: `_relax_forced_tool_choice`); persisted so resumes stay relaxed.
        self._consumed_forced_key: Optional[str] = None

    def _has_decisions(self) -> bool:
        return bool(self._approved_ids or self._rejected_ids)

    # -- transport -----------------------------------------------------------

    async def _send(self, request: Mapping[str, Any]) -> Any:
        client = self.options["client"]
        kwargs = sdk_request_kwargs(request)
        # Normalize input items at the transport boundary only — internal
        # state and stream events keep the upstream TS shapes:
        #  - Response items echoed back from a live turn are SDK pydantic
        #    models (e.g. OutputMessageItem); the request validator wants
        #    plain dicts, so dump() them.
        #  - Internal items use upstream's camelCase callId; the generated
        #    Python SDK validates snake_case call_id.
        if isinstance(kwargs.get("input"), list):
            normalized = []
            for item in kwargs["input"]:
                if not isinstance(item, Mapping):
                    item = dump(item)
                if isinstance(item, Mapping) and "callId" in item:
                    item = {("call_id" if k == "callId" else k): v for k, v in item.items()}
                normalized.append(item)
            kwargs["input"] = normalized
        request_options = dict(self.options.get("options") or {})
        headers = request_options.pop("headers", None) or request_options.pop("http_headers", None)
        if headers:
            kwargs["http_headers"] = dict(headers)
        kwargs.update(sdk_request_kwargs(request_options))
        responses = getattr(getattr(client, "beta"), "responses")
        if hasattr(responses, "send_async"):
            return await maybe_await(responses.send_async(**kwargs))
        return await asyncio.to_thread(responses.send, **kwargs)

    async def _coerce_response(self, value: Any) -> Any:
        if hasattr(value, "ok") and getattr(value, "ok") is False:
            raise getattr(value, "error", RuntimeError("Responses API request failed"))
        if is_async_iterable(value):
            completed = None
            received: List[Any] = []
            async for event in value:
                await self._append_event(event)
                received.append(event)
                typ = get_field(event, "type")
                if typ in {"response.completed", "response.incomplete"}:
                    completed = get_field(event, "response")
                if typ == "response.failed":
                    raise RuntimeError(str(get_field(event, "message", "Response failed")))
            if completed is None:
                raise RuntimeError("Stream ended without response.completed")
            assert_streamed_tool_calls_match_response(received, completed)
            return completed
        for attr in ("value", "response", "result"):
            if hasattr(value, attr):
                inner = getattr(value, attr)
                if inner is not None:
                    value = inner
                    break
        if isinstance(value, Mapping) and "value" in value and len(value) <= 2:
            value = value["value"]
        if isinstance(value, list) and value and all(isinstance(item, Mapping) and "type" in item for item in value):
            completed = None
            for event in value:
                await self._append_event(event)
                if event.get("type") in {"response.completed", "response.incomplete"}:
                    completed = event.get("response")
            if completed is not None:
                assert_streamed_tool_calls_match_response(value, completed)
            return completed or {"id": "response_from_events", "output": []}
        return value

    async def _send_and_track(
        self,
        request: Mapping[str, Any],
        turn_type: str,
        turn_number: int,
        *,
        caller_tool_choice: bool = True,
    ) -> Any:
        """Send a request, coerce its response, and emit PostModelCall for it.

        With `caller_tool_choice` (every dispatch except engine-owned
        overrides such as the forced-final `tool_choice: "none"` turn), the
        caller-configured `tool_choice` goes through the forced-choice policy:
        once a forced choice (`"required"`, a specific tool, or
        `allowed_tools` with `mode: "required"`) has produced a tool call in
        this logical run, it is relaxed to `"auto"` on later turns so the model
        can answer in text instead of calling tools until the step budget runs
        out. A choice is consumed only when the response that carried it
        called a tool; an unforced choice re-arms it.
        """
        started = time.monotonic()
        configured_key: Optional[str] = None
        if caller_tool_choice and "tool_choice" in request:
            configured = request.get("tool_choice")
            configured_key = _forced_tool_choice_key(configured)
            if configured_key is not None and configured_key == self._consumed_forced_key:
                request = {**dict(request), "tool_choice": _relax_forced_tool_choice(configured)}
        response = await self._coerce_response(await self._send(request))
        if caller_tool_choice:
            if configured_key is None:
                self._consumed_forced_key = None
            elif configured_key != self._consumed_forced_key and extract_tool_calls_from_response(response):
                self._consumed_forced_key = configured_key
        await self._emit_post_model_call(response, started, turn_type, turn_number)
        return response

    async def _append_event(self, event: Any) -> None:
        async with self._condition:
            self._events.append(event)
            self._condition.notify_all()

    def _ensure_task(self) -> asyncio.Task[Any]:
        if self._run_task is None:
            self._run_task = asyncio.create_task(self._run())
        return self._run_task

    # -- state -----------------------------------------------------------

    async def _load_state(self) -> None:
        accessor = self.options.get("state")
        if accessor is None:
            return
        loaded = await maybe_await(accessor.load())
        if loaded is not None:
            self._consumed_forced_key = loaded.consumed_forced_tool_choice_key
        self._resuming_from_client_tools = bool(loaded is not None and loaded.status == "awaiting_client_tools")
        self._state = loaded or create_initial_state()
        if (
            loaded is not None
            and loaded.status in ("awaiting_approval", "awaiting_hitl")
            and not self._has_decisions()
            and await self._stays_paused_without_decisions(loaded)
        ):
            # Paused for a human decision but none was supplied: stay paused.
            # Falling through would flip the status to "in_progress" and send
            # a fresh model request, advancing the conversation past calls
            # that still owe a decision. Every response/stream getter surfaces
            # this as an explicit "awaiting ..." error (see `_paused_error`).
            self._paused_without_decisions = True
            return
        if not self._resuming_from_client_tools:
            self._state = update_state(self._state, {"status": "in_progress"})
            await maybe_await(accessor.save(self._state))

    async def _resolve_initial_request(self) -> Dict[str, Any]:
        """Resolve the turn-0 request exactly once. Caller-supplied async
        parameter functions (including a function-valued `input`) must not run
        twice when the paused-resume check needs the resolved input before the
        normal initialization path does."""
        if self._initial_request is None:
            self._initial_request = await resolve_async_functions(self.options["request"], {"number_of_turns": 0})
        return self._initial_request

    async def _stays_paused_without_decisions(self, loaded: Any) -> bool:
        """Whether a state loaded as `awaiting_approval` / `awaiting_hitl` stays
        paused when the caller supplied no approve/reject decisions.

        - No pending calls: nothing is owed a decision, so continue with a
          normal request instead of stranding the conversation.
        - `awaiting_approval`: always paused. Only a recorded approve/reject
          decision releases an approval gate; a caller-supplied
          `function_call_output` for the gated call does not.
        - `awaiting_hitl`: paused unless `input` carries a
          `function_call_output` for a pending call -- the documented way to
          resume a HITL tool (it goes through `on_response_received`).
        """
        pending_ids = {call.id for call in (loaded.pending_tool_calls or [])}
        if not pending_ids:
            return False
        if loaded.status != "awaiting_hitl":
            return True
        input_value = (await self._resolve_initial_request()).get("input")
        if not isinstance(input_value, list):
            return True
        return not any(
            get_field(item, "type") == "function_call_output"
            and (get_field(item, "callId") or get_field(item, "call_id")) in pending_ids
            for item in input_value
        )

    def _paused_error(self) -> Optional[Exception]:
        """Describe the pause when the conversation is waiting on the caller
        (`awaiting_approval` / `awaiting_hitl`), or `None` when it is not."""
        state = self._state
        status = getattr(state, "status", None)
        if status not in ("awaiting_approval", "awaiting_hitl"):
            return None
        pending_ids = [call.id for call in (state.pending_tool_calls or [])]
        shown = ", ".join(pending_ids[:_MAX_PENDING_IDS_IN_ERROR])
        hidden = len(pending_ids) - _MAX_PENDING_IDS_IN_ERROR
        id_list = f"{shown}, ... ({hidden} more)" if hidden > 0 else shown
        if status == "awaiting_approval":
            what = "awaiting approval"
            how_to_resume = "Resume with approve_tool_calls and/or reject_tool_calls for the pending calls."
        else:
            what = "awaiting HITL input"
            how_to_resume = (
                "Resume by passing a function_call_output for a pending call ID in input, "
                "or with approve_tool_calls / reject_tool_calls (approving re-runs on_tool_called)."
            )
        input_value = self.options["request"].get("input")
        input_supplied = callable(input_value) or bool(input_value)
        message = (
            f"Conversation is {what} (status '{status}'); no response is available. "
            f"Pending tool call IDs: {id_list}. {how_to_resume} "
            "Use requires_approval() or get_pending_tool_calls() to inspect a paused conversation."
        )
        if input_supplied:
            message += " The input supplied with this call was not sent to the model and was not saved to state."
        return RuntimeError(message)

    async def _save_state(self, **updates: Any) -> None:
        accessor = self.options.get("state")
        if accessor is None or self._state is None:
            return
        if updates.get("status") == "complete":
            # A later conversational turn is a new logical run and may force
            # its first tool call again.
            self._consumed_forced_key = None
        updates["consumed_forced_tool_choice_key"] = self._consumed_forced_key
        self._state = update_state(self._state, updates)
        await maybe_await(accessor.save(self._state))

    def _response_output_items(self, response: Any) -> List[Any]:
        output = get_field(response, "output", []) or []
        return output if isinstance(output, list) else [output]

    async def _save_response_to_state(self, response: Any) -> None:
        if self._state is None:
            return
        messages = self._state.messages
        if self._fresh_items_for_state:
            messages = append_to_messages(messages, self._fresh_items_for_state)
            self._fresh_items_for_state = []
        messages = append_to_messages(messages, self._response_output_items(response))
        updates: Dict[str, Any] = {"messages": messages, "previous_response_id": get_field(response, "id")}
        if self._resuming_from_client_tools:
            # Manual calls stay durable until a resume request actually
            # produces a response -- clearing before the request would lose
            # the only copy if that request failed.
            updates["pending_tool_calls"] = None
            updates["status"] = "in_progress"
        self._resuming_from_client_tools = False
        await self._save_state(**updates)

    async def _save_tool_outputs_to_state(self, outputs: Sequence[Any]) -> None:
        if self._state is None or not outputs:
            return
        await self._save_state(messages=append_to_messages(self._state.messages, list(outputs)))

    async def _persist_client_tools_pause(self, response: Any, unresolved_calls: Sequence[ParsedToolCall]) -> None:
        """Persist state when the loop stops because of unresolved manual
        (client-executed) tool calls -- tools with neither `execute` nor
        `on_tool_called`. Mirrors the HITL pause path but uses the distinct
        `awaiting_client_tools` status so callers can tell the two apart.

        Without a StateAccessor nothing is persisted: `get_pending_tool_calls()`
        returns `[]` and the caller must read the unresolved calls off the
        response output directly.
        """
        self._final_response = response
        if self.options.get("state") is None or not unresolved_calls:
            return
        await self._save_state(pending_tool_calls=list(unresolved_calls), status="awaiting_client_tools")

    def _validate_final_response(self, response: Any, allow_empty_output: bool = False) -> None:
        """Validate the final response has required fields.

        Message prefixes (`Invalid final response: missing required fields`,
        `Invalid final response: empty or invalid output`) are load-bearing --
        callers and tests match on them. Detail is appended after an em dash;
        never reword the prefixes. This is a pure array-length check, so a
        `function_call`-only output passes.
        """
        response_id = get_field(response, "id")
        output = get_field(response, "output")
        if not response_id or output is None:
            missing = [name for name, absent in (("id", not response_id), ("output", output is None)) if absent]
            raise ValueError(f"Invalid final response: missing required fields: {', '.join(missing)}")
        if not isinstance(output, list) or len(output) == 0:
            if allow_empty_output:
                return
            if isinstance(output, list):
                detail = (
                    f'output array is empty (length 0) for response "{response_id}". The model returned no '
                    "output items. This can happen when the provider returns an empty final turn; see the "
                    "strict_final_response/allow_final_response options"
                )
            else:
                detail = f'output is not an array (got {_describe_non_record(output)}) for response "{response_id}"'
            raise ValueError(f"Invalid final response: empty or invalid output — {detail}")

    # -- hooks -----------------------------------------------------------

    def _has_approval_configured(self, tools: Sequence[Tool]) -> bool:
        if self.options.get("require_approval"):
            return True
        return any(tool_has_approval_configured(t) for t in tools)

    async def _emit_session_start(self, tools: Sequence[Tool]) -> None:
        if not self._hooks:
            return
        self._hooks.set_session_id(self._session_id)
        await self._hooks.emit(
            "SessionStart",
            {
                "config": {
                    "has_tools": bool(tools),
                    "has_approval": self._has_approval_configured(tools),
                    "has_state": self.options.get("state") is not None,
                }
            },
            session_id=self._session_id,
        )
        self._session_start_emitted = True

    async def _emit_session_end(self, reason: str) -> None:
        if not self._hooks or not self._session_start_emitted or self._session_end_emitted:
            return
        self._session_end_emitted = True
        payload: Dict[str, Any] = {"reason": reason}
        if self._session_usage["model_calls"] > 0:
            payload["total_usage"] = self._snapshot_session_usage()
        try:
            await self._hooks.emit("SessionEnd", payload, session_id=self._session_id)
        except Exception as error:  # noqa: BLE001 - teardown must never mask the real error
            warnings.warn(f"[SessionEnd] error during session teardown: {error}", stacklevel=2)

    def _snapshot_session_usage(self) -> Dict[str, Any]:
        """Materialize the running session aggregate. `cost` is present only
        when at least one response reported it (a summed `0` from cost-less
        responses would read as "free" rather than "unknown"). Single source
        of truth for both `SessionEnd.total_usage` and `get_usage()`, so the
        two cannot drift."""
        totals: Dict[str, Any] = {
            "model_calls": self._session_usage["model_calls"],
            "input_tokens": self._session_usage["input_tokens"],
            "output_tokens": self._session_usage["output_tokens"],
            "total_tokens": self._session_usage["total_tokens"],
            "cached_tokens": self._session_usage["cached_tokens"],
            "reasoning_tokens": self._session_usage["reasoning_tokens"],
        }
        if self._session_usage["has_cost"]:
            totals["cost"] = self._session_usage["cost"]
        return totals

    async def _emit_post_model_call(self, response: Any, started_at: float, turn_type: str, turn_number: int) -> None:
        # Accumulation runs unconditionally -- before the hooks short-circuit --
        # because `get_usage()` surfaces these totals to callers who configured
        # no hooks at all. Only the hook emit is gated on a HooksManager.
        usage = _extract_model_call_usage(get_field(response, "usage"))
        self._session_usage["model_calls"] += 1
        if usage:
            self._session_usage["input_tokens"] += usage["input_tokens"]
            self._session_usage["output_tokens"] += usage["output_tokens"]
            self._session_usage["total_tokens"] += usage["total_tokens"]
            self._session_usage["cached_tokens"] += usage["cached_tokens"]
            self._session_usage["reasoning_tokens"] += usage["reasoning_tokens"]
            if usage.get("cost") is not None:
                self._session_usage["cost"] += usage["cost"]
                self._session_usage["has_cost"] = True
        if not self._hooks:
            return
        payload = {
            "session_id": self._session_id,
            "response_id": get_field(response, "id", ""),
            "model": get_field(response, "model", "") or "",
            "duration_ms": (time.monotonic() - started_at) * 1000,
            "turn_type": turn_type,
            "turn_number": turn_number,
        }
        if usage:
            payload["usage"] = usage
        await self._hooks.emit("PostModelCall", payload, session_id=self._session_id)

    async def _emit_permission_request(self, call: ParsedToolCall, tools: Sequence[Tool]) -> Tuple[str, Optional[str]]:
        if not self._hooks:
            return "ask_user", None
        if isinstance(call.arguments, str):
            # Raw-string arguments mean the model produced invalid JSON. Fail
            # closed (fall through to the human approval flow).
            return "ask_user", None

        tool = next(
            (t for t in tools if is_client_tool(t) and get_tool_function(t).get("name") == call.name),
            None,
        )
        require_approval = get_tool_function(tool).get("require_approval") if tool else None
        if callable(require_approval) or self.options.get("require_approval"):
            risk_level = "high"
        elif require_approval is True:
            risk_level = "medium"
        else:
            risk_level = "low"

        emitted = await self._hooks.emit(
            "PermissionRequest",
            {
                "tool_name": call.name,
                "tool_input": call.arguments if isinstance(call.arguments, dict) else {},
                "risk_level": risk_level,
            },
            tool_name=call.name,
            session_id=self._session_id,
        )
        if not emitted.results:
            return "ask_user", None
        last = emitted.results[-1]
        return last.get("decision", "ask_user"), last.get("reason")

    async def _maybe_run_user_prompt_submit(self, input_value: Any) -> Any:
        if not self._hooks or input_value is None:
            return input_value

        if isinstance(input_value, str):
            prompt = input_value
            emitted = await self._hooks.emit("UserPromptSubmit", {"prompt": prompt}, session_id=self._session_id)
            if emitted.blocked:
                reject = next((r.get("reject") for r in emitted.results if r.get("reject")), None)
                raise ValueError(reject if isinstance(reject, str) else "Prompt rejected by hook")
            if emitted.mutated:
                return emitted.final_payload.get("prompt", prompt)
            return input_value

        if isinstance(input_value, list):
            target_index = _find_latest_user_string_index(input_value)
            if target_index == -1:
                return input_value
            prompt = input_value[target_index]["content"]
            emitted = await self._hooks.emit("UserPromptSubmit", {"prompt": prompt}, session_id=self._session_id)
            if emitted.blocked:
                reject = next((r.get("reject") for r in emitted.results if r.get("reject")), None)
                raise ValueError(reject if isinstance(reject, str) else "Prompt rejected by hook")
            if not emitted.mutated:
                return input_value
            mutated_prompt = emitted.final_payload.get("prompt", prompt)
            new_items = list(input_value)
            new_items[target_index] = {**new_items[target_index], "content": mutated_prompt}
            return new_items

        return input_value

    async def _inject_append_prompt_message(self, prompt: str, current_request: Dict[str, Any]) -> None:
        injected = {"role": "user", "content": prompt}
        if self._state is not None:
            next_messages = append_to_messages(self._state.messages, [injected])
            self._state = update_state(self._state, {"messages": next_messages})
            if self.options.get("state") is not None:
                await self._save_state()
        current_input = current_request.get("input")
        if isinstance(current_input, list):
            current_input.append(injected)
        elif current_input:
            current_request["input"] = [{"role": "user", "content": current_input}, injected]
        else:
            current_request["input"] = [injected]

    async def _run_stop_hook(self, force_resume_count: int, current_request: Dict[str, Any]) -> str:
        """Emit the Stop hook when a stop_when condition halts the loop.
        Returns "resume" when the loop should continue, "stop" otherwise."""
        if not self._hooks:
            return "stop"
        emitted = await self._hooks.emit("Stop", {"reason": "max_turns"}, session_id=self._session_id)
        should_force_resume = any(r.get("force_resume") is True for r in emitted.results)
        append_prompt = "\n".join(
            r.get("append_prompt")
            for r in emitted.results
            if isinstance(r.get("append_prompt"), str) and r.get("append_prompt")
        )
        if append_prompt:
            await self._inject_append_prompt_message(append_prompt, current_request)
        if not should_force_resume:
            return "stop"
        if force_resume_count >= _MAX_FORCE_RESUME_OVERRIDES:
            warnings.warn(
                f"[Stop hook] force_resume honored {_MAX_FORCE_RESUME_OVERRIDES} times without new "
                "progress; stopping to prevent an infinite loop.",
                stacklevel=2,
            )
            return "stop"
        return "resume"

    # -- tool execution ----------------------------------------------------

    def _find_tool(self, name: str, tools: Sequence[Tool]) -> Optional[Tool]:
        return next((t for t in tools if is_client_tool(t) and get_tool_function(t).get("name") == name), None)

    async def _run_tool_with_hooks(
        self, tool: Tool, call: ParsedToolCall, context: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single tool and emit the PreToolUse/PostToolUse/
        PostToolUseFailure lifecycle hooks around it.

        Returns a tagged outcome:
        - `{"type": "parse_error", "call", "error_message"}` -- the model
          produced invalid JSON for this call's arguments. No hooks fire.
        - `{"type": "hook_blocked", "call", "reason"}` -- PreToolUse blocked
          the call.
        - `{"type": "execution", "call", "result"}` -- the tool ran (or
          paused, if `result` is `None` for a HITL tool). `call` reflects any
          `mutated_input` piped by PreToolUse.
        """
        if isinstance(call.arguments, str):
            error_message = (
                f'Failed to parse tool call arguments for "{call.name}": The model provided invalid JSON. '
                f'Raw arguments received: "{call.arguments}". '
                "Please provide valid JSON arguments for this tool call."
            )
            return {"type": "parse_error", "call": call, "error_message": error_message}

        effective_call = call
        if self._hooks and call.pre_tool_use_applied is not True:
            original_input = call.arguments if isinstance(call.arguments, dict) else {}
            pre = await self._hooks.emit(
                "PreToolUse",
                {"tool_name": call.name, "tool_input": original_input},
                tool_name=call.name,
                session_id=self._session_id,
            )
            if pre.blocked:
                block = next((r.get("block") for r in pre.results if r.get("block")), None)
                reason = block if isinstance(block, str) else "Blocked by PreToolUse hook"
                return {"type": "hook_blocked", "call": call, "reason": reason}
            if pre.mutated:
                effective_call = dataclasses.replace(call, arguments=pre.final_payload.get("tool_input"))

        started = time.monotonic()
        result = await execute_tool(
            tool,
            effective_call,
            context,
            self._record_preliminary,
            self._context_store,
            self.options.get("shared_context_schema"),
        )
        duration_ms = (time.monotonic() - started) * 1000

        if self._hooks and result is not None:
            tool_input = effective_call.arguments if isinstance(effective_call.arguments, dict) else {}
            if result.get("error") is not None:
                await self._hooks.emit(
                    "PostToolUseFailure",
                    {"tool_name": effective_call.name, "tool_input": tool_input, "error": str(result["error"])},
                    tool_name=effective_call.name,
                    session_id=self._session_id,
                )
            else:
                await self._hooks.emit(
                    "PostToolUse",
                    {
                        "tool_name": effective_call.name,
                        "tool_input": tool_input,
                        "tool_output": result.get("result"),
                        "duration_ms": duration_ms,
                    },
                    tool_name=effective_call.name,
                    session_id=self._session_id,
                )

        return {"type": "execution", "call": effective_call, "result": result}

    async def _prepare_tool_calls_for_approval(
        self,
        calls: Sequence[ParsedToolCall],
        tools: Sequence[Tool],
    ) -> Tuple[List[ParsedToolCall], List[UnsentToolResult]]:
        """Apply PreToolUse once before approval classification."""
        prepared: List[ParsedToolCall] = []
        blocked: List[UnsentToolResult] = []
        for call in calls:
            tool = self._find_tool(call.name, tools)
            if (
                not self._hooks
                or call.pre_tool_use_applied is True
                or isinstance(call.arguments, str)
                or not tool
                or not is_auto_resolvable_tool(tool)
            ):
                prepared.append(call)
                continue

            original_input = call.arguments if isinstance(call.arguments, dict) else {}
            pre = await self._hooks.emit(
                "PreToolUse",
                {"tool_name": call.name, "tool_input": original_input},
                tool_name=call.name,
                session_id=self._session_id,
            )
            if pre.blocked:
                block = next((result.get("block") for result in pre.results if result.get("block")), None)
                reason = block if isinstance(block, str) else "Blocked by PreToolUse hook"
                blocked.append(create_rejected_result(call.id, call.name, reason))
                continue

            arguments = pre.final_payload.get("tool_input") if pre.mutated else call.arguments
            prepared.append(dataclasses.replace(call, arguments=arguments, pre_tool_use_applied=True))

        return prepared, blocked

    async def _resolve_approval_gate(
        self,
        calls: Sequence[ParsedToolCall],
        tools: Sequence[Tool],
        turn_context: Mapping[str, Any],
    ) -> Tuple[List[ParsedToolCall], List[ParsedToolCall], List[UnsentToolResult]]:
        """Return pending calls, executable calls, and resolved rejection results."""
        prepared, resolved_results = await self._prepare_tool_calls_for_approval(calls, tools)
        partition = await partition_tool_calls(
            prepared,
            tools,
            turn_context,
            self.options.get("require_approval"),
        )
        needs_approval = list(partition["requires_approval"])
        pending: List[ParsedToolCall] = []
        allowed_occurrences: set[int] = set()
        denied_occurrences: set[int] = set()

        if needs_approval and self._hooks:
            for call in needs_approval:
                decision, reason = await self._emit_permission_request(call, tools)
                if decision == "allow":
                    allowed_occurrences.add(id(call))
                elif decision == "deny":
                    denied_occurrences.add(id(call))
                    resolved_results.append(
                        create_rejected_result(call.id, call.name, reason or "Denied by PermissionRequest hook")
                    )
                else:
                    pending.append(call)
        else:
            pending.extend(needs_approval)

        pending_occurrences = {id(call) for call in pending}
        auto_execute_occurrences = {id(call) for call in partition["auto_execute"]}
        executable = [
            call
            for call in prepared
            if id(call) not in pending_occurrences
            and id(call) not in denied_occurrences
            and (id(call) in allowed_occurrences or id(call) in auto_execute_occurrences)
        ]
        return pending, executable, resolved_results

    async def _execute_calls_to_unsent_results(
        self,
        calls: Sequence[ParsedToolCall],
        tools: Sequence[Tool],
        turn_context: Mapping[str, Any],
    ) -> List[UnsentToolResult]:
        """Execute and persist auto-resolvable calls before an approval pause."""
        results: List[UnsentToolResult] = []
        for call in calls:
            tool = self._find_tool(call.name, tools)
            if not tool or not is_auto_resolvable_tool(tool):
                continue
            outcome = await self._run_tool_with_hooks(tool, call, turn_context)
            if outcome["type"] == "parse_error":
                results.append(create_rejected_result(call.id, call.name, outcome["error_message"]))
            elif outcome["type"] == "hook_blocked":
                results.append(create_rejected_result(call.id, call.name, outcome["reason"]))
            elif outcome["result"] is None:
                # HITL tools cannot produce an unsent result until resumed.
                continue
            elif outcome["result"].get("error") is not None:
                results.append(create_rejected_result(call.id, call.name, str(outcome["result"]["error"])))
            else:
                effective_call = outcome["call"]
                results.append(
                    create_unsent_result(
                        effective_call.id,
                        effective_call.name,
                        await self._unsent_model_output(effective_call, tool, outcome["result"].get("result")),
                    )
                )
        return results

    async def _record_preliminary(self, call_id: str, value: Any) -> None:
        await self._append_event(
            {
                "type": "tool.preliminary_result",
                "toolCallId": call_id,
                "tool_call_id": call_id,
                "result": value,
                "timestamp": _now_ms(),
            }
        )

    def _parsed_tool_input(self, tool: Tool, call: ParsedToolCall) -> Any:
        """`call.arguments` is the JSON-parsed wire payload. Parse it with the
        tool's `input_schema` so `to_model_output` sees the same input
        `execute` got; fall back to the raw arguments when parsing fails."""
        try:
            return validate_schema(get_tool_function(tool).get("input_schema"), call.arguments)
        except Exception:
            return call.arguments

    async def _apply_to_model_output(self, call: ParsedToolCall, tool: Tool, value: Any) -> Tuple[bool, str, Any]:
        """Run the tool's `to_model_output` mapper, if any. Returns
        `(applied, type, value)` where `type` is `"content"` or `"json"`."""
        converter = get_tool_function(tool).get("to_model_output")
        if not converter:
            return False, "", None
        converted = await maybe_await(converter({"output": value, "input": self._parsed_tool_input(tool, call)}))
        if isinstance(converted, Mapping) and converted.get("type") == "content":
            return True, "content", converted.get("value", [])
        if isinstance(converted, Mapping) and converted.get("type") == "json":
            return True, "json", converted.get("value")
        return False, "", None

    async def _unsent_model_output(self, call: ParsedToolCall, tool: Tool, output: Any) -> Any:
        """Unsent results are serialized by `unsent_results_to_api_format`,
        which does not know about `to_model_output`. Apply it here so resumed
        and approved calls send the model the same output as the main loop."""
        applied, _kind, value = await self._apply_to_model_output(call, tool, output)
        return value if applied else output

    async def _tool_result_to_output(
        self, call: ParsedToolCall, tool: Tool, result: Mapping[str, Any]
    ) -> Dict[str, Any]:
        if result.get("error") is not None:
            output: Any = json_dumps({"error": str(result["error"])})
        else:
            applied, kind, value = await self._apply_to_model_output(call, tool, result.get("result"))
            if applied and kind == "content":
                output = value
            elif applied:
                output = json_dumps(value)
            else:
                output = json_dumps(result.get("result"))
        return {"type": "function_call_output", "id": f"output_{call.id}", "callId": call.id, "output": output}

    def _rejected_output(self, call_id: str, reason: str) -> Dict[str, Any]:
        return {
            "type": "function_call_output",
            "id": f"output_{call_id}",
            "callId": call_id,
            "output": json_dumps({"error": reason}),
        }

    # -- the main loop -----------------------------------------------------

    async def _run(self) -> Any:
        session_end_reason = "complete"
        hooks = self._hooks
        try:
            await self._load_state()
            if self._paused_without_decisions:
                # No request, no state write, no SessionStart: the getters
                # report the pause (see `_paused_error`).
                return None
            self._session_id = self._state.id if self._state is not None else generate_conversation_id()

            tools: Sequence[Tool] = self.options.get("tools") or []
            await self._emit_session_start(tools)

            request = dict(await self._resolve_initial_request())
            request["stream"] = True
            base_input = request.get("input")

            if hooks and base_input is not None:
                base_input = await self._maybe_run_user_prompt_submit(base_input)
                request["input"] = base_input

            historical_input = normalize_input_to_array(self._state.messages) if self._state is not None else []
            fresh_input = normalize_input_to_array(base_input)
            if self.options.get("context") is not None:
                resolved_context = await resolve_context(self.options.get("context"), {"number_of_turns": 0})
                self._context_store = ToolContextStore(resolved_context)
            if tools and fresh_input:
                historical_function_calls = [
                    item for item in historical_input if get_field(item, "type") == "function_call"
                ]
                synthetic_input = [*historical_function_calls, *fresh_input]
                hooked_input = await apply_on_response_received_hooks(
                    synthetic_input,
                    tools,
                    {"number_of_turns": 0},
                    self._context_store,
                    self.options.get("shared_context_schema"),
                )
                hooked_array = normalize_input_to_array(hooked_input)
                fresh_input = hooked_array[len(historical_function_calls) :]
            if self._state is not None:
                request["input"] = append_to_messages(historical_input, fresh_input)
                if not (self._state.pending_tool_calls and self._has_decisions()):
                    self._fresh_items_for_state = list(fresh_input)
            elif fresh_input:
                request["input"] = fresh_input

            current_request = request
            is_resume_turn = self._resuming_from_client_tools
            if self._state is not None and self._state.pending_tool_calls and self._has_decisions():
                resume_request = await self._build_resume_request(request)
                if resume_request is None:
                    # Decisions covered only some of the pending calls (or an
                    # approved call needs re-approval after a PreToolUse
                    # mutation, or an approved HITL call paused again): the
                    # conversation stays paused and no request is sent.
                    return None
                current_request = resume_request
                is_resume_turn = True

            final_response = None
            force_resume_count = 0
            pending_final_directive = False

            for turn_number in range(_MAX_TURNS):
                if self.options.get("on_turn_start"):
                    await maybe_await(self.options["on_turn_start"]({"number_of_turns": turn_number}))
                await self._append_event(
                    {
                        "type": "turn.start",
                        "turnNumber": turn_number,
                        "turn_number": turn_number,
                        "timestamp": _now_ms(),
                    }
                )
                if turn_number == 0:
                    turn_type = "resume" if is_resume_turn else "initial"
                elif pending_final_directive:
                    turn_type = "final"
                else:
                    turn_type = "tool_round"
                pending_final_directive = False

                response = await self._send_and_track(
                    current_request,
                    turn_type,
                    turn_number,
                    # The forced-final turn's tool_choice "none" is an
                    # engine-owned override, not the caller's policy.
                    caller_tool_choice=turn_type != "final",
                )
                await self._append_event(
                    {
                        "type": "turn.end",
                        "turnNumber": turn_number,
                        "turn_number": turn_number,
                        "timestamp": _now_ms(),
                    }
                )
                if self.options.get("on_turn_end"):
                    await maybe_await(self.options["on_turn_end"]({"number_of_turns": turn_number}, response))
                calls = extract_tool_calls_from_response(response)
                self._tool_calls.extend(calls)
                final_response = response
                await self._save_response_to_state(response)
                text = extract_text_from_response(response)
                step = StepResult(
                    step_type="initial" if turn_number == 0 else "continue",
                    text=text,
                    tool_calls=calls,
                    tool_results=[],
                    response=response,
                    usage=get_field(response, "usage"),
                )
                self._steps.append(step)
                if not calls or not tools:
                    await self._save_state(status="complete")
                    break

                stop_when = self.options.get("stop_when")
                stop_conditions = list(stop_when) if isinstance(stop_when, list) else ([stop_when] if stop_when else [])
                stopped_by_stop_when = False
                while stop_conditions and await is_stop_condition_met(stop_conditions, self._steps):
                    stop_decision = await self._run_stop_hook(force_resume_count, current_request)
                    if stop_decision == "resume":
                        # Zero-cost retry: re-check the stop condition against
                        # the SAME already-fetched response/steps -- no new
                        # model request. Bare force_resume alone typically
                        # doesn't change anything the condition inspects, so
                        # this burns through the consecutive-override cap
                        # quickly unless append_prompt (injected above) or
                        # external state changes what the condition sees.
                        force_resume_count += 1
                        continue
                    session_end_reason = "max_turns"
                    stopped_by_stop_when = True
                    break

                if stopped_by_stop_when:
                    allow_final_response = self.options.get("allow_final_response")
                    final_response_enabled = allow_final_response is not False
                    resolvable_pending = [c for c in calls if self._call_is_auto_resolvable(c, tools)]
                    if not final_response_enabled and turn_number == 0 and resolvable_pending:
                        # With allow_final_response=False a met stop_when means
                        # no tool from this response runs. The first response
                        # still goes through the approval check (upstream
                        # evaluates it before the loop), but pausing for a gated
                        # call must not execute its auto-approved siblings: they
                        # are saved as skipped results so the paused turn stays
                        # well-formed for the resume request.
                        self._assert_unique_call_ids(calls, tools)
                        turn_context = {"number_of_turns": turn_number + 1, "turn_request": current_request}
                        requires_approval, executable_calls, gate_results = await self._resolve_approval_gate(
                            calls, tools, turn_context
                        )
                        if requires_approval:
                            if self.options.get("state") is None:
                                names = ", ".join(call.name for call in requires_approval)
                                raise ValueError(
                                    f"Tool(s) require approval but no state accessor is configured: {names}"
                                )
                            resolved_gate_ids = {result.call_id for result in gate_results}
                            gate_results.extend(
                                create_rejected_result(call.id, call.name, "Tool execution skipped: stop condition met")
                                for call in executable_calls
                                if call.id not in resolved_gate_ids
                            )
                            await self._save_state(
                                pending_tool_calls=requires_approval,
                                unsent_tool_results=gate_results or None,
                                status="awaiting_approval",
                            )
                            self._final_response = final_response
                            return final_response
                    if final_response_enabled and resolvable_pending:
                        self._assert_unique_call_ids(calls, tools)
                        turn_context = {"number_of_turns": turn_number + 1, "turn_request": current_request}
                        requires_approval, executable_calls, gate_results = await self._resolve_approval_gate(
                            calls, tools, turn_context
                        )
                        if requires_approval:
                            if self.options.get("state") is None:
                                names = ", ".join(call.name for call in requires_approval)
                                raise ValueError(
                                    f"Tool(s) require approval but no state accessor is configured: {names}"
                                )
                            gate_results.extend(
                                await self._execute_calls_to_unsent_results(executable_calls, tools, turn_context)
                            )
                            await self._save_state(
                                pending_tool_calls=requires_approval,
                                unsent_tool_results=gate_results or None,
                                status="awaiting_approval",
                            )
                            self._final_response = final_response
                            return final_response

                        final_outputs: List[Any] = unsent_results_to_api_format(gate_results)
                        resolved_gate_ids = {result.call_id for result in gate_results}
                        final_executed_calls: List[ParsedToolCall] = []
                        for call in executable_calls:
                            if call.id in resolved_gate_ids:
                                continue
                            matching_tool = self._find_tool(call.name, tools)
                            if matching_tool and is_auto_resolvable_tool(matching_tool):
                                outcome = await self._run_tool_with_hooks(matching_tool, call, turn_context)
                                if outcome["type"] == "parse_error":
                                    final_outputs.append(self._rejected_output(call.id, outcome["error_message"]))
                                elif outcome["type"] == "hook_blocked":
                                    final_outputs.append(self._rejected_output(call.id, outcome["reason"]))
                                else:
                                    result = outcome["result"]
                                    if result is not None:
                                        if result.get("error") is None:
                                            final_executed_calls.append(outcome["call"])
                                        output = await self._tool_result_to_output(
                                            outcome["call"], matching_tool, result
                                        )
                                        final_outputs.append(output)
                                        self._tool_outputs.append(output)
                                        await self._append_event(
                                            {
                                                "type": "tool.call_output",
                                                "output": output,
                                                "timestamp": _now_ms(),
                                            }
                                        )
                            else:
                                final_outputs.append(
                                    {
                                        "type": "function_call_output",
                                        "id": f"output_{call.id}",
                                        "callId": call.id,
                                        "output": "Tool execution skipped: step limit reached.",
                                    }
                                )
                        await self._save_tool_outputs_to_state(final_outputs)
                        # Apply nextTurnParams from the calls that actually ran
                        # so they affect the final text-coercion request
                        # (mirrors the in-loop behavior).
                        final_params = await execute_next_turn_params_functions(
                            final_executed_calls, tools, current_request
                        )
                        if final_params:
                            current_request = apply_next_turn_params_to_request(current_request, final_params)
                        current_request = self._build_final_request(
                            current_request, response, final_outputs, allow_final_response
                        )
                        pending_final_directive = True
                        continue
                    break

                self._assert_unique_call_ids(calls, tools)
                turn_context = {"number_of_turns": turn_number + 1, "turn_request": current_request}
                requires_approval, executable_calls, hook_resolved_unsent = await self._resolve_approval_gate(
                    calls,
                    tools,
                    turn_context,
                )

                if requires_approval:
                    if self.options.get("state") is None:
                        names = ", ".join(call.name for call in requires_approval)
                        raise ValueError(f"Tool(s) require approval but no state accessor is configured: {names}")
                    hook_resolved_unsent.extend(
                        await self._execute_calls_to_unsent_results(executable_calls, tools, turn_context)
                    )
                    save_kwargs: Dict[str, Any] = {
                        "pending_tool_calls": requires_approval,
                        "status": "awaiting_approval",
                    }
                    if hook_resolved_unsent:
                        save_kwargs["unsent_tool_results"] = hook_resolved_unsent
                    await self._save_state(**save_kwargs)
                    self._final_response = final_response
                    return final_response

                outputs: List[Any] = unsent_results_to_api_format(hook_resolved_unsent) if hook_resolved_unsent else []
                paused: List[ParsedToolCall] = []
                executed_calls: List[ParsedToolCall] = []
                for call in executable_calls:
                    tool = self._find_tool(call.name, tools)
                    if not tool or not is_auto_resolvable_tool(tool):
                        continue
                    # `tool_call` is threaded into the execute context by the
                    # executor from the call it actually runs (wire shape).
                    turn_context = {"number_of_turns": turn_number + 1, "turn_request": current_request}
                    outcome = await self._run_tool_with_hooks(tool, call, turn_context)
                    if outcome["type"] == "parse_error":
                        await self._append_event(
                            {
                                "type": "tool.result",
                                "toolCallId": call.id,
                                "tool_call_id": call.id,
                                "source": "mcp" if is_mcp_tool(tool) else "client",
                                "result": {"error": outcome["error_message"]},
                                "timestamp": _now_ms(),
                            }
                        )
                        output = self._rejected_output(call.id, outcome["error_message"])
                        outputs.append(output)
                        self._tool_outputs.append(output)
                        await self._append_event({"type": "tool.call_output", "output": output, "timestamp": _now_ms()})
                        continue
                    if outcome["type"] == "hook_blocked":
                        output = self._rejected_output(call.id, outcome["reason"])
                        outputs.append(output)
                        self._tool_outputs.append(output)
                        await self._append_event({"type": "tool.call_output", "output": output, "timestamp": _now_ms()})
                        continue

                    effective_call = outcome["call"]
                    result = outcome["result"]
                    if result is None:
                        paused.append(call)
                        continue
                    await self._append_event(
                        {
                            "type": "tool.result",
                            "toolCallId": call.id,
                            "tool_call_id": call.id,
                            "source": result.get("source", "mcp" if is_mcp_tool(tool) else "client"),
                            "result": {"error": str(result["error"])}
                            if result.get("error") is not None
                            else result.get("result"),
                            "timestamp": _now_ms(),
                        }
                    )
                    output = await self._tool_result_to_output(effective_call, tool, result)
                    outputs.append(output)
                    if result.get("error") is None:
                        # Only calls whose tool body ran and returned without
                        # error may drive nextTurnParams, with the input the
                        # tool actually ran on (including a PreToolUse
                        # mutation). Blocked, invalid, or failed calls must
                        # not change the next request.
                        executed_calls.append(effective_call)
                    self._tool_outputs.append(output)
                    await self._append_event({"type": "tool.call_output", "output": output, "timestamp": _now_ms()})
                    step.tool_results.append(result)

                if paused:
                    await self._save_tool_outputs_to_state(outputs)
                    await self._save_state(pending_tool_calls=paused, status="awaiting_hitl")
                    self._final_response = final_response
                    return final_response

                await self._save_tool_outputs_to_state(outputs)

                resolved_ids = {get_field(o, "callId") for o in outputs}
                unresolved_calls = [c for c in calls if c.id not in resolved_ids]
                if unresolved_calls:
                    await self._persist_client_tools_pause(response, unresolved_calls)
                    return final_response

                if not outputs:
                    break
                self._all_tool_rounds += 1
                force_resume_count = 0
                next_params = await execute_next_turn_params_functions(executed_calls, tools, current_request)
                if next_params:
                    current_request = apply_next_turn_params_to_request(current_request, next_params)
                current_request = self._build_followup_request(current_request, response, outputs)
            else:
                raise RuntimeError("call_model exceeded the 20-turn safety limit")

            # Tolerate an empty final response after at least one completed
            # tool round (mini-class models intermittently return an empty
            # final turn after the tool call was the answer): retry once,
            # then accept the empty output rather than reporting failure --
            # unless strict_final_response opts back into the legacy throw.
            can_tolerate_empty = self._all_tool_rounds > 0 and self.options.get("strict_final_response") is not True
            output = get_field(final_response, "output")
            is_empty_output = isinstance(output, list) and len(output) == 0
            if can_tolerate_empty and is_empty_output:
                retry_turn_number = self._all_tool_rounds + 1
                final_response = await self._retry_current_request(current_request, retry_turn_number)
                await self._save_response_to_state(final_response)
                output = get_field(final_response, "output")
                is_empty_output = isinstance(output, list) and len(output) == 0

            allow_empty_output = can_tolerate_empty and is_empty_output
            self._validate_final_response(final_response, allow_empty_output)
            self._final_response = final_response
            await self._save_state(status="complete")
            return final_response
        except Exception:
            session_end_reason = "error"
            raise
        finally:
            try:
                await self._emit_session_end(session_end_reason)
                if hooks:
                    await hooks.drain()
            except Exception as error:  # noqa: BLE001 - teardown must never mask the real error
                warnings.warn(f"[SessionEnd] error during session teardown: {error}", stacklevel=2)
            async with self._condition:
                self._run_done = True
                self._condition.notify_all()

    def _call_is_auto_resolvable(self, call: ParsedToolCall, tools: Sequence[Tool]) -> bool:
        tool = self._find_tool(call.name, tools)
        return bool(tool and is_auto_resolvable_tool(tool))

    def _assert_unique_call_ids(self, calls: Sequence[ParsedToolCall], tools: Sequence[Tool]) -> None:
        """Refuse a batch the loop would execute when two calls share a `call_id`.

        Approvals, rejections and tool outputs are all matched to calls by that
        id, so a repeated id would let one decision apply to a different call.
        The batch runs nothing. Batches with no executable calls are left to the
        caller, who owns correlation for manual tools.
        """
        if not any(self._call_is_auto_resolvable(call, tools) for call in calls):
            return
        seen: set[str] = set()
        for call in calls:
            if call.id in seen:
                raise ValueError(
                    f'Response contains more than one function_call with call_id "{call.id}"; '
                    "refusing to execute the batch because tool outputs are matched to calls by call_id."
                )
            seen.add(call.id)

    async def _build_resume_request(self, request: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply approve/reject decisions to the persisted pending calls.

        Returns the resume request, or `None` when the conversation stays
        paused: a pending call received no decision, an approved call whose
        PreToolUse-mutated arguments need their own approval is pending
        again, or an approved HITL call paused again.
        """
        self._fresh_items_for_state = []
        tools: Sequence[Tool] = self.options.get("tools") or []
        unsent: List[UnsentToolResult] = list(self._state.unsent_tool_results or []) if self._state else []
        pending: List[ParsedToolCall] = list(self._state.pending_tool_calls or []) if self._state is not None else []
        seen_ids: set[str] = {result.call_id for result in unsent}
        for call in pending:
            if call.id in seen_ids:
                raise ValueError(
                    f'Saved state has more than one tool call with call_id "{call.id}"; '
                    "refusing to resume because approvals are matched to calls by call_id."
                )
            seen_ids.add(call.id)

        # numberOfTurns is 1-indexed after the initial request (upstream
        # `allToolExecutionRounds.length + 1` on a fresh run).
        turn_context: Dict[str, Any] = {"number_of_turns": 1}
        hitl_paused_ids: set[str] = set()
        second_gate: Dict[str, ParsedToolCall] = {}
        # Each decision list is de-duplicated in __init__, so an approved
        # call executes at most once per resume.
        for call_id in self._approved_ids:
            found = next((candidate for candidate in pending if candidate.id == call_id), None)
            if found is None:
                continue
            call = found
            tool = self._find_tool(call.name, tools)
            if not tool or not is_auto_resolvable_tool(tool):
                unsent.append(create_rejected_result(call.id, call.name, "Tool not found or not executable"))
                continue

            if self._hooks and call.pre_tool_use_applied is not True and not isinstance(call.arguments, str):
                prepared = await self._prepare_after_initial_approval(call, tool, tools, turn_context)
                if prepared[0] == "blocked":
                    unsent.append(create_rejected_result(call.id, call.name, prepared[1]))
                    continue
                if prepared[0] == "pending":
                    second_gate[call.id] = prepared[2]
                    continue
                call = prepared[2]

            outcome = await self._run_tool_with_hooks(tool, call, turn_context)
            if outcome["type"] == "parse_error":
                unsent.append(create_rejected_result(call.id, call.name, outcome["error_message"]))
            elif outcome["type"] == "hook_blocked":
                unsent.append(create_rejected_result(call.id, call.name, outcome["reason"]))
            elif outcome["result"] is None:
                # HITL tool paused on approval -- keep the call visible via
                # pending_tool_calls (status becomes awaiting_hitl below).
                hitl_paused_ids.add(call.id)
            elif outcome["result"].get("error") is not None:
                unsent.append(create_rejected_result(call.id, call.name, str(outcome["result"]["error"])))
            else:
                effective_call = outcome["call"]
                unsent.append(
                    create_unsent_result(
                        call.id,
                        call.name,
                        await self._unsent_model_output(effective_call, tool, outcome["result"].get("result")),
                    )
                )

        for call_id in self._rejected_ids:
            rejected = next((candidate for candidate in pending if candidate.id == call_id), None)
            if rejected is None:
                continue
            # Upstream also emits PostToolUseFailure here for calls whose
            # PreToolUse ran after an initial approval. This port still runs
            # PreToolUse *before* the approval gate (the upstream two-phase
            # gate is not ported yet), so `pre_tool_use_applied` is set on
            # every held call and emitting here would fire the hook where
            # upstream does not. Left out until the two-phase gate lands.
            unsent.append(create_rejected_result(rejected.id, rejected.name, "Rejected by user"))

        processed = {
            call_id
            for call_id in [*self._approved_ids, *self._rejected_ids]
            if call_id not in hitl_paused_ids and call_id not in second_gate
        }
        remaining = [second_gate.get(call.id, call) for call in pending if call.id not in processed]
        unresolved_approvals = [call for call in remaining if call.id not in hitl_paused_ids]
        if unresolved_approvals:
            next_status = "awaiting_approval"
        elif hitl_paused_ids:
            next_status = "awaiting_hitl"
        else:
            next_status = "in_progress"

        if next_status != "in_progress":
            await self._save_state(
                pending_tool_calls=remaining or None,
                unsent_tool_results=unsent or None,
                status=next_status,
            )
            return None

        outputs = unsent_results_to_api_format(unsent)
        updated_messages = append_to_messages(self._state.messages, outputs) if self._state is not None else outputs
        await self._save_state(
            messages=updated_messages,
            pending_tool_calls=None,
            unsent_tool_results=None,
            status="in_progress",
        )
        return {**dict(request), "input": updated_messages, "stream": True}

    async def _prepare_after_initial_approval(
        self,
        call: ParsedToolCall,
        tool: Tool,
        tools: Sequence[Tool],
        turn_context: Mapping[str, Any],
    ) -> Tuple[str, str, ParsedToolCall]:
        """Run PreToolUse for an approved call that has not been through it.

        The human approved the arguments they saw. If PreToolUse rewrites
        them, the rewritten call needs its own approval check: returns
        `("ready", "", call)` to execute, `("blocked", reason, call)` to reject,
        or `("pending", "", mutated_call)` to pause again on the mutated
        arguments (marked `pre_tool_use_applied` so the hook does not run a
        second time on the next resume).
        """
        assert self._hooks is not None
        original_input = call.arguments if isinstance(call.arguments, dict) else {}
        pre = await self._hooks.emit(
            "PreToolUse",
            {"tool_name": call.name, "tool_input": original_input},
            tool_name=call.name,
            session_id=self._session_id,
        )
        if pre.blocked:
            block = next((result.get("block") for result in pre.results if result.get("block")), None)
            return "blocked", block if isinstance(block, str) else "Blocked by PreToolUse hook", call
        if not pre.mutated:
            return "ready", "", dataclasses.replace(call, pre_tool_use_applied=True)
        mutated = dataclasses.replace(call, arguments=pre.final_payload.get("tool_input"), pre_tool_use_applied=True)
        # A mutation that no longer satisfies the schema is refused rather
        # than executed or re-queued (upstream `validatePreparedMutation`).
        ok, parsed = _parse_tool_arguments(tool, mutated.arguments)
        if not ok or not isinstance(parsed, Mapping):
            invalid = f'PreToolUse produced invalid input for "{call.name}"'
            await self._emit_prepared_failure(mutated, invalid)
            return "blocked", invalid, mutated
        if not await self._mutated_input_requires_approval(mutated, tools, turn_context):
            return "ready", "", mutated
        decision, reason = await self._emit_permission_request(mutated, tools)
        if decision == "allow":
            return "ready", "", mutated
        if decision == "deny":
            denial = reason or "Denied by PermissionRequest hook"
            await self._emit_prepared_failure(mutated, denial)
            return "blocked", denial, mutated
        return "pending", "", mutated

    async def _mutated_input_requires_approval(
        self, call: ParsedToolCall, tools: Sequence[Tool], turn_context: Mapping[str, Any]
    ) -> bool:
        """Re-check approval for PreToolUse-mutated arguments. Only
        input-dependent checks can change their answer: a call-level
        `require_approval` function, or a function-valued tool-level
        `require_approval`. A blanket `require_approval=True` was already
        satisfied by the human's approval, so the mutated call runs."""
        call_level = self.options.get("require_approval")
        if callable(call_level):
            return await tool_requires_approval(call, tools, turn_context, call_level)
        tool = self._find_tool(call.name, tools)
        if tool is None or not callable(get_tool_function(tool).get("require_approval")):
            return False
        return await tool_requires_approval(call, tools, turn_context)

    async def _emit_prepared_failure(self, call: ParsedToolCall, reason: str) -> None:
        """PostToolUseFailure for a call that went through PreToolUse but will
        not execute (invalid mutation, PermissionRequest deny, user reject)."""
        if not self._hooks:
            return
        await self._hooks.emit(
            "PostToolUseFailure",
            {
                "tool_name": call.name,
                "tool_input": call.arguments if isinstance(call.arguments, dict) else {},
                "error": reason,
            },
            tool_name=call.name,
            session_id=self._session_id,
        )

    async def _iter_events_live(self) -> AsyncIterator[Any]:
        self._ensure_task()
        index = 0
        while True:
            async with self._condition:
                while index >= len(self._events) and not self._run_done:
                    await self._condition.wait()
                if index < len(self._events):
                    event = self._events[index]
                    index += 1
                elif self._run_done:
                    break
                else:
                    continue
            yield event
        await self.get_response()

    def _build_followup_request(
        self, request: Mapping[str, Any], response: Any, outputs: Sequence[Any]
    ) -> Dict[str, Any]:
        output_items = get_field(response, "output", []) or []
        if not isinstance(output_items, list):
            output_items = [output_items]
        return {
            **dict(request),
            "input": [*normalize_input_to_array(request.get("input")), *output_items, *outputs],
            "stream": True,
        }

    def _build_final_request(
        self, request: Mapping[str, Any], response: Any, outputs: Sequence[Any], allow_final_response: Any
    ) -> Dict[str, Any]:
        new_request = self._build_followup_request(request, response, outputs)
        # Forbid tool calls without dropping the `tools` block: removing it
        # would invalidate the prompt-cache prefix.
        if new_request.get("tools") is not None:
            new_request["tool_choice"] = "none"
        directive = (
            DEFAULT_FINAL_RESPONSE_DIRECTIVE
            if allow_final_response is True or allow_final_response is None
            else allow_final_response
        )
        if isinstance(directive, str) and directive:
            new_request["input"] = [
                *normalize_input_to_array(new_request.get("input")),
                {"role": "user", "content": directive},
            ]
        return new_request

    async def _retry_current_request(self, current_request: Mapping[str, Any], turn_number: int) -> Any:
        """Re-send the current resolved request once, forcing `tool_choice:
        "none"` when tools are present so the retry coerces a text turn
        instead of a fresh (silently dropped) function call."""
        new_request = dict(current_request)
        if new_request.get("tools") is not None:
            new_request["tool_choice"] = "none"
        new_request["stream"] = True
        return await self._send_and_track(new_request, "retry", turn_number, caller_tool_choice=False)

    # -- public streaming/result surface ------------------------------------

    async def get_response(self) -> Any:
        response = await self._ensure_task()
        if response is None:
            raise self._paused_error() or RuntimeError("Response not available")
        return response

    async def get_text(self) -> str:
        response = await self.get_response()
        return extract_text_from_response(response)

    async def get_text_stream(self) -> AsyncIterator[str]:
        self._ensure_task()
        yielded = False
        index = 0
        while True:
            async with self._condition:
                while index >= len(self._events) and not self._run_done:
                    await self._condition.wait()
                if index < len(self._events):
                    event = self._events[index]
                    index += 1
                elif self._run_done:
                    break
                else:
                    continue
            if get_field(event, "type") == "response.output_text.delta":
                yielded = True
                yield str(get_field(event, "delta", ""))
        await self.get_response()
        if not yielded:
            text = extract_text_from_response(self._final_response)
            if text:
                yield text

    async def get_reasoning_stream(self) -> AsyncIterator[str]:
        async for event in self._iter_events_live():
            if get_field(event, "type") == "response.reasoning_text.delta":
                yield str(get_field(event, "delta", ""))

    async def get_tool_stream(self) -> AsyncIterator[Dict[str, Any]]:
        async for event in self._iter_events_live():
            typ = get_field(event, "type")
            if typ == "response.function_call_arguments.delta":
                yield {"type": "delta", "content": get_field(event, "delta", "")}
            elif typ == "tool.preliminary_result":
                yield {
                    "type": "preliminary_result",
                    "toolCallId": get_field(event, "toolCallId"),
                    "tool_call_id": get_field(event, "tool_call_id", get_field(event, "toolCallId")),
                    "result": get_field(event, "result"),
                }
            elif typ == "tool.result":
                yield {
                    "type": "tool_result",
                    "toolCallId": get_field(event, "toolCallId"),
                    "tool_call_id": get_field(event, "tool_call_id", get_field(event, "toolCallId")),
                    "source": get_field(event, "source", "client"),
                    "result": get_field(event, "result"),
                }
            elif typ == "tool.call_output":
                yield {"type": "tool_call_output", "output": get_field(event, "output")}
            elif typ in {"turn.start", "turn.end"}:
                yield {
                    "type": typ,
                    "turnNumber": get_field(event, "turnNumber"),
                    "turn_number": get_field(event, "turn_number", get_field(event, "turnNumber")),
                    "timestamp": get_field(event, "timestamp"),
                }

    async def get_tool_calls_stream(self) -> AsyncIterator[ParsedToolCall]:
        seen_ids = set()
        names: Dict[str, str] = {}
        call_ids: Dict[str, str] = {}
        buffers: Dict[str, List[str]] = {}
        yielded_from_stream = False

        async for event in self._iter_events_live():
            typ = get_field(event, "type")
            if typ == "response.output_item.added":
                item = get_field(event, "item", {})
                if get_field(item, "type") == "function_call":
                    item_id = str(get_field(item, "id", get_field(item, "callId", "")))
                    names[item_id] = str(get_field(item, "name", ""))
                    call_ids[item_id] = str(get_field(item, "callId", get_field(item, "call_id", item_id)))
                    buffers.setdefault(item_id, [])
            elif typ == "response.function_call_arguments.delta":
                item_id = str(get_field(event, "itemId", get_field(event, "item_id", get_field(event, "callId", ""))))
                buffers.setdefault(item_id, []).append(str(get_field(event, "delta", "")))
            elif typ == "response.function_call_arguments.done":
                item_id = str(get_field(event, "itemId", get_field(event, "item_id", get_field(event, "callId", ""))))
                raw = get_field(event, "arguments", "".join(buffers.get(item_id, [])))
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    args = raw
                call = ParsedToolCall(
                    id=call_ids.get(item_id, item_id),
                    name=names.get(item_id, str(get_field(event, "name", ""))),
                    arguments=args,
                )
                seen_ids.add(call.id)
                yielded_from_stream = True
                yield call

        for call in self._tool_calls:
            if call.id not in seen_ids and (not yielded_from_stream or call.id):
                yield call

    async def get_tool_calls(self) -> List[ParsedToolCall]:
        await self.get_response()
        return list(self._tool_calls)

    async def get_full_responses_stream(self) -> AsyncIterator[Any]:
        async for event in self._iter_events_live():
            yield event

    async def get_new_messages_stream(self) -> AsyncIterator[Any]:
        yielded = []
        yielded_call_ids = set()

        def should_yield(item: Any) -> bool:
            if get_field(item, "type") != "function_call":
                return True
            name = get_field(item, "name")
            call_id = get_field(item, "callId", get_field(item, "call_id", get_field(item, "id", None)))
            if call_id in yielded_call_ids:
                return False
            matching = next(
                (
                    candidate
                    for candidate in self.options.get("tools") or []
                    if is_client_tool(candidate) and get_tool_function(candidate).get("name") == name
                ),
                None,
            )
            if not matching or not is_manual_tool(matching):
                return False
            if call_id is not None:
                yielded_call_ids.add(call_id)
            return True

        async for event in self._iter_events_live():
            typ = get_field(event, "type")
            if typ == "tool.call_output":
                item = get_field(event, "output")
                yielded.append(item)
                yield item
            if typ == "response.output_item.done":
                item = get_field(event, "item")
                if should_yield(item):
                    yielded.append(item)
                    yield item
        for item in normalize_input_to_array(get_field(self._final_response, "output", [])):
            if item not in yielded and should_yield(item):
                yield item
        for output in self._tool_outputs:
            if output not in yielded:
                yield output

    async def get_items_stream(self) -> AsyncIterator[Any]:
        async for item in self.get_new_messages_stream():
            yield item

    async def get_context_updates(self) -> AsyncIterator[Dict[str, Any]]:
        await self.get_response()
        if self._context_store:
            yield self._context_store.get_snapshot()

    async def get_full_chat_stream(self) -> AsyncIterator[Dict[str, Any]]:
        async for delta in self.get_text_stream():
            yield {"type": "content.delta", "delta": delta}
        yield {"type": "message.complete", "response": await self.get_response()}

    async def requires_approval(self) -> bool:
        # Inspection APIs keep working on a paused conversation, so they wait
        # for the run without going through get_response()'s pause error.
        await self._ensure_task()
        state = self._state
        return bool(
            state
            and (
                state.status in {"awaiting_approval", "awaiting_hitl", "awaiting_client_tools"}
                or state.pending_tool_calls
            )
        )

    async def get_pending_tool_calls(self) -> List[ParsedToolCall]:
        await self._ensure_task()
        return list(self._state.pending_tool_calls or []) if self._state else []

    async def get_state(self) -> Any:
        await self._ensure_task()
        return self._state

    async def get_usage(self) -> Dict[str, Any]:
        """Aggregate token/cost usage across every model call this run made.

        `get_response()` resolves to the *final* round's response, so in a
        multi-round tool loop its `usage` block covers only that generation.
        This sums the initial request, each tool-round follow-up, the
        empty-final retry, the `allow_final_response` final turn, and
        approval-resume requests. Same shape as the `SessionEnd` hook's
        `total_usage`: `model_calls`, `input_tokens`, `output_tokens`,
        `total_tokens`, `cached_tokens`, `reasoning_tokens`, and `cost` only
        when the server reported it.

        Waits for the run to finish, but never raises: a failed or paused run
        still consumed tokens, so it returns the totals accrued so far (and
        warns with the swallowed cause). Await `get_response()` to observe the
        failure itself.
        """
        try:
            await self._ensure_task()
        except Exception as error:  # noqa: BLE001 - documented never-raises accessor
            warnings.warn(f"[get_usage] run failed; reporting totals accrued so far: {error}", stacklevel=2)
        return self._snapshot_session_usage()

    def cancel(self) -> None:
        if self._run_task is not None:
            self._run_task.cancel()
