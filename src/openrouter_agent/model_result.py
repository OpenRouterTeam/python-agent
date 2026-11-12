from __future__ import annotations

import asyncio
import dataclasses
import json
import time
import warnings
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from pydantic import BaseModel
from typing_extensions import Literal

from ._utils import (
    dump,
    get_field,
    is_async_iterable,
    json_dumps,
    json_loads_maybe,
    maybe_await,
    sdk_request_kwargs,
    validate_schema,
)
from .async_params import has_async_functions, resolve_async_functions
from .conversation_state import (
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
from .doom_loop import (
    DoomLoopAction,
    DoomLoopMonitor,
    DoomLoopVerdict,
    LoopKeyResolution,
    ResolvedEscalationConfig,
    resolve_doom_loop_option,
    resolve_loop_key_material,
)
from .next_turn_params import apply_next_turn_params_to_request, execute_next_turn_params_functions
from .stop_conditions import is_stop_condition_met
from .stream_transformers import extract_text_from_response, extract_tool_calls_from_response, response_has_tool_calls
from .tool_context import ToolContextStore, resolve_context
from .tool_executor import apply_on_response_received_hooks, execute_tool
from .tool_types import (
    ParsedToolCall,
    StepResult,
    Tool,
    UnsentToolResult,
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


def _find_latest_user_string_index(items: Sequence[Any]) -> int:
    for index in range(len(items) - 1, -1, -1):
        if _is_user_string_message(items[index]):
            return index
    return -1


def _relax_forced_tool_choice(tool_choice: Any) -> Any:
    """Relax a spent forced tool choice: `required` / a specific tool ->
    `auto`; `allowed_tools` with `mode: required` -> same set with
    `mode: auto`. Everything else passes through unchanged."""
    if tool_choice is None or isinstance(tool_choice, str):
        return "auto" if tool_choice == "required" else tool_choice
    if get_field(tool_choice, "type") == "allowed_tools":
        if get_field(tool_choice, "mode") == "required":
            return {**dump(tool_choice), "mode": "auto"}
        return tool_choice
    return "auto"


def _is_forced_tool_choice(tool_choice: Any) -> bool:
    """Whether a caller-configured tool choice requires this turn to call a tool."""
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice == "required"
    return get_field(tool_choice, "type") != "allowed_tools" or get_field(tool_choice, "mode") == "required"


def _tool_choice_identity(tool_choice: Any) -> str:
    from .doom_loop import canonicalize_key_material

    return canonicalize_key_material(dump(tool_choice) if tool_choice is not None else None)


def _forced_tool_choice_key(tool_choice: Any) -> Optional[str]:
    """Stable identity for a concrete forced choice; None means tools are not forced."""
    return _tool_choice_identity(tool_choice) if _is_forced_tool_choice(tool_choice) else None


#: Sentinel marking a tool-call id that appeared MORE THAN ONCE in one batch
#: (upstream `DUPLICATE_CALL_ID`). Ids are model-emitted and nothing enforces
#: uniqueness; a colliding id poisons the loop-key cache entry rather than let
#: the last write win, or one call's doom-loop identity aliases onto another's.
_DUPLICATE_CALL_ID = object()

#: "No fallback identity" marker for doom-loop evaluations (upstream passes
#: `null`). A sentinel, because `None` is legitimate key material in Python.
_NO_FALLBACK = object()

#: Output-item fields that are volatile per invocation (ids, status, timing)
#: or are the result payload itself; excluded from server-tool identity.
_SERVER_TOOL_VOLATILE_KEYS = frozenset({"id", "type", "status", "output", "result", "results", "error"})


def _extract_server_tool_identity(item: Any) -> Optional[Dict[str, Any]]:
    """Identity material for a server-tool output item (upstream
    `extractServerToolIdentity`): every field except volatile ones and the
    result payload, so identical repeated requests fingerprint identically.
    Returns None when nothing identifying remains."""
    record = item if isinstance(item, Mapping) else dump(item)
    if not isinstance(record, Mapping):
        return None
    identity: Dict[str, Any] = {}
    for key, value in record.items():
        if key in _SERVER_TOOL_VOLATILE_KEYS or callable(value):
            continue
        identity[key] = value
    return identity or None


#: Cap on pending call IDs listed in a paused-conversation error message.
_MAX_PENDING_IDS_IN_ERROR = 10


def _describe_non_record(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    return "object"


class ModelResult:
    def __init__(self, options: Mapping[str, Any]) -> None:
        self.options = dict(options)
        self._run_task: Optional[asyncio.Task[Any]] = None
        self._final_response: Any = None
        self._events: List[Any] = []
        self._tool_calls: List[ParsedToolCall] = []
        self._tool_outputs: List[Any] = []
        self._call_names: Dict[str, str] = {}
        self._steps: List[StepResult] = []
        #: Executed tool rounds -- the step history stop conditions see
        #: (upstream `allToolExecutionRounds`): one entry per round whose
        #: calls actually ran, not one per model response.
        self._executed_rounds: List[StepResult] = []
        self._state: Any = None
        self._fresh_items_for_state: List[Any] = []
        self._context_store: Optional[ToolContextStore] = None
        self._condition = asyncio.Condition()
        self._run_done = False
        self._hooks = self.options.get("hooks")
        self._session_id = ""
        self._resuming_from_client_tools = False
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
        if (self._approved_ids or self._rejected_ids) and not self.options.get("state"):
            raise ValueError("approve_tool_calls and reject_tool_calls require a state accessor")
        # PermissionRequest 'deny' decisions parked between the approval gate
        # and the tool round, which synthesizes rejected outputs for them.
        self._hook_denied_calls: Dict[str, str] = {}
        # Approval is idempotent per response object and tool-call occurrence:
        # each call in a response is gated at most once per run.
        self._completed_approval_gates: set[Tuple[int, int]] = set()
        # Forced tool-choice relaxation policy (DEV-785). Only the consumed
        # semantic key is persisted; dynamic choices re-resolve every turn.
        self._consumed_forced_tool_choice_key: Optional[str] = None
        self._configured_tool_choice: Any = None
        self._configured_forced_tool_choice_key: Optional[str] = None
        self._prepared_effective_tool_choice: Any = None
        self._prepared_tool_choice_commit: Optional[Tuple[str, Optional[str]]] = None
        self._dispatched_tool_choice_commit: Optional[Tuple[str, Optional[str]]] = None
        self._dispatched_caller_tool_choice = False
        self._pending_next_request: Optional[Dict[str, Any]] = None
        # Doom-loop detection (None when the `doom_loop` option is off). The
        # monitor is a pure, serializable state machine; this class owns the
        # engine-side consequences (hook emission, block synthesis, steer
        # injection, escalation, stopping the loop).
        doom_loop_config = resolve_doom_loop_option(self.options.get("doom_loop"))
        self._doom_loop_monitor: Optional[DoomLoopMonitor] = (
            DoomLoopMonitor(doom_loop_config) if doom_loop_config is not None else None
        )
        self._doom_loop_escalation: Optional[ResolvedEscalationConfig] = (
            doom_loop_config["escalation"] if doom_loop_config is not None else None
        )
        # Armed by a 'stop' verdict; checked at the loop boundaries. Persisted
        # as `stop_verdict` so a condemned run stays condemned across
        # decision-only resumes; cleared by a fresh conversational turn.
        self._doom_loop_stop: Optional[DoomLoopVerdict] = None
        # Steer guidance queued by 'steer' (and 'escalate') verdicts, injected
        # as a user message before the next follow-up request. Persisted as
        # `pending_steer` so guidance queued right before a pause survives.
        self._pending_doom_loop_steer: List[str] = []
        # Monotonic round counter, bumped at every execution-batch boundary.
        self._doom_loop_round = 0
        # Final decision per (tool, fingerprint) in the current round, so
        # in-round duplicates reuse it (no second increment / hook emission).
        self._doom_loop_round_decisions: Dict[str, Dict[str, Any]] = {}
        # Loop-key resolutions computed while declaring the current round,
        # keyed by call id, so a `loop_key` callable runs at most once per call.
        self._doom_loop_round_key_material: Dict[str, Any] = {}
        # Armed by an 'escalate' verdict; consumed by the next follow-up
        # request as ONE-TURN overrides. Deliberately not persisted.
        self._pending_doom_loop_escalation: Optional[DoomLoopVerdict] = None
        # PreToolUse outcomes parked between the approval gates and execution
        # (upstream `preparedToolCalls`), keyed by call id:
        # - `{"type": "ready", "call": effective_call, "mutated": bool}` --
        #   PreToolUse already ran; execution uses `call` and must not emit
        #   PreToolUse again.
        # - `{"type": "blocked", "reason": str}` -- PreToolUse blocked, the
        #   mutated input failed validation / was denied, or the doom-loop
        #   checkpoint refused the call. `_run_tool_with_hooks` synthesizes
        #   the error output without running the tool.
        self._prepared_tool_calls: Dict[str, Dict[str, Any]] = {}
        # Second-phase (mutated-input) approval gates already run, so a call
        # is re-gated at most once per run: `(response_key, occurrence)`.
        self._completed_mutated_approval_gates: set[Tuple[Any, Any]] = set()

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
            iterator = value.__aiter__()
            try:
                async for event in iterator:
                    await self._append_event(event)
                    typ = get_field(event, "type")
                    if typ == "response.failed":
                        raise RuntimeError(str(get_field(event, "message", "Response failed")))
                    if typ in {"response.completed", "response.incomplete"}:
                        # The terminal event ends the response: stop reading
                        # rather than waiting for the source to close, which
                        # some transports deliver late (or never).
                        completed = get_field(event, "response")
                        break
            finally:
                closer = getattr(iterator, "aclose", None)
                if completed is not None and closer is not None:
                    try:
                        await closer()
                    except Exception:  # noqa: BLE001 - a failing close must not fail the turn
                        pass
            if completed is None:
                raise RuntimeError("Stream ended without response.completed")
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
            return completed or {"id": "response_from_events", "output": []}
        return value

    async def _send_and_track(self, request: Mapping[str, Any], turn_type: str, turn_number: int) -> Any:
        """Send a request, coerce its response, and emit PostModelCall for it."""
        self._begin_tool_choice_dispatch(request.get("tool_choice"))
        started = time.monotonic()
        response = await self._coerce_response(await self._send(request))
        # toolChoice constrains model output, so a forced requirement is
        # satisfied as soon as this turn emits a tool call. Record it before
        # any approval/HITL/client pause can persist the run.
        self._commit_tool_choice_dispatch(response_has_tool_calls(response))
        await self._emit_post_model_call(response, started, turn_type, turn_number)
        return response

    # -- forced tool-choice policy (DEV-785) ---------------------------------

    def _apply_resolved_forced_tool_choice_policy(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Record the freshly-resolved caller choice before deriving its wire
        value. An unforced turn clears the consumed key, re-arming the same
        forced value if it appears again later. A different forced key
        re-arms immediately."""
        self._configured_tool_choice = request.get("tool_choice")
        self._configured_forced_tool_choice_key = _forced_tool_choice_key(self._configured_tool_choice)
        if self._configured_forced_tool_choice_key is None:
            self._prepared_tool_choice_commit = (
                None if self._consumed_forced_tool_choice_key is None else ("clear", None)
            )
        elif self._configured_forced_tool_choice_key == self._consumed_forced_tool_choice_key:
            self._prepared_tool_choice_commit = None
        else:
            self._prepared_tool_choice_commit = ("consume", self._configured_forced_tool_choice_key)
        return self._apply_forced_tool_choice_policy(request)

    def _apply_forced_tool_choice_policy(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Derive the caller-configured tool choice for the active logical run.
        A forced choice that already produced a tool call is relaxed so the
        model can answer in text. Engine-owned one-turn overrides (`none` for
        finalization) are applied after this helper and therefore still win."""
        if (
            self._configured_forced_tool_choice_key is not None
            and self._configured_forced_tool_choice_key == self._consumed_forced_tool_choice_key
        ):
            tool_choice = _relax_forced_tool_choice(self._configured_tool_choice)
        else:
            tool_choice = self._configured_tool_choice
        self._prepared_effective_tool_choice = tool_choice
        if tool_choice is request.get("tool_choice"):
            return request
        updated = dict(request)
        if tool_choice is None:
            updated.pop("tool_choice", None)
        else:
            updated["tool_choice"] = tool_choice
        return updated

    def _begin_tool_choice_dispatch(self, actual_tool_choice: Any) -> None:
        """Bind the prepared caller-policy transition to the actual wire
        request. Engine overrides produce a different key and cannot commit."""
        self._dispatched_caller_tool_choice = _tool_choice_identity(actual_tool_choice) == _tool_choice_identity(
            self._prepared_effective_tool_choice
        )
        self._dispatched_tool_choice_commit = (
            self._prepared_tool_choice_commit if self._dispatched_caller_tool_choice else None
        )

    def _commit_tool_choice_dispatch(self, has_tool_calls: bool) -> None:
        """Commit only after the prepared caller choice reached the wire and a
        response materialized; a forced choice is consumed only when that
        response called a tool."""
        dispatched = self._dispatched_caller_tool_choice
        commit = self._dispatched_tool_choice_commit
        self._dispatched_caller_tool_choice = False
        self._dispatched_tool_choice_commit = None
        if not dispatched:
            return
        self._prepared_tool_choice_commit = None
        if commit is not None and commit[0] == "clear":
            self._consumed_forced_tool_choice_key = None
        elif commit is not None and commit[0] == "consume" and has_tool_calls:
            self._consumed_forced_tool_choice_key = commit[1]

    def _reset_forced_tool_choice_policy(self) -> None:
        """Clear all runtime policy state for a terminal run exit: a later
        conversational turn is a new logical run and may force again."""
        self._consumed_forced_tool_choice_key = None
        self._configured_forced_tool_choice_key = None
        self._configured_tool_choice = None
        self._prepared_effective_tool_choice = None
        self._prepared_tool_choice_commit = None
        self._dispatched_tool_choice_commit = None
        self._dispatched_caller_tool_choice = False

    async def _mark_state_complete(self) -> None:
        self._reset_forced_tool_choice_policy()
        await self._save_state(status="complete")

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
        self._resuming_from_client_tools = bool(loaded is not None and loaded.status == "awaiting_client_tools")
        self._state = loaded or create_initial_state()
        if loaded is not None:
            self._consumed_forced_tool_choice_key = getattr(loaded, "consumed_forced_tool_choice_key", None)
            self._restore_doom_loop_state(getattr(loaded, "doom_loop", None))

    def _restore_doom_loop_state(self, persisted: Any) -> None:
        """Rehydrate doom-loop state so a resumed run continues where it left
        off: streak counters always; the queued steer guidance; and the stop
        verdict under the condemnation rule -- a doom-stopped conversation
        STAYS stopped across decision-only resumes (approve/reject), while a
        fresh conversational turn clears it (operator input is new
        information). Streaks are kept either way."""
        monitor = self._doom_loop_monitor
        if monitor is None or persisted is None:
            return
        monitor.restore(persisted)
        if not isinstance(persisted, Mapping):
            return
        pending_steer = persisted.get("pending_steer", persisted.get("pendingSteer"))
        if isinstance(pending_steer, list):
            for message in pending_steer:
                if isinstance(message, str):
                    self._queue_doom_loop_steer(message)
        stop_verdict = persisted.get("stop_verdict", persisted.get("stopVerdict"))
        if isinstance(stop_verdict, Mapping) and stop_verdict and (self._approved_ids or self._rejected_ids):
            self._doom_loop_stop = cast(DoomLoopVerdict, dict(stop_verdict))

    async def _save_state(self, **updates: Any) -> None:
        accessor = self.options.get("state")
        if accessor is None or self._state is None:
            return
        # Persist only the consumed semantic tool-choice key, never the
        # callback or a copied effective tool_choice.
        updates["consumed_forced_tool_choice_key"] = self._consumed_forced_tool_choice_key
        # Piggyback the doom-loop detector snapshot on every persist so loop
        # memory survives serialize -> resume. Engine-owned fields ride along:
        # `stop_verdict` keeps a condemned run condemned across decision-only
        # resumes; `pending_steer` carries guidance that had no flush point.
        if self._doom_loop_monitor is not None:
            doom_loop_state: Dict[str, Any] = dict(self._doom_loop_monitor.get_state())
            if self._doom_loop_stop is not None:
                doom_loop_state["stop_verdict"] = dict(self._doom_loop_stop)
            if self._pending_doom_loop_steer:
                doom_loop_state["pending_steer"] = list(self._pending_doom_loop_steer)
            updates["doom_loop"] = doom_loop_state
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
        response_id = get_field(response, "id")
        output = get_field(response, "output")
        # Message prefixes ("Invalid final response: missing required fields",
        # "Invalid final response: empty or invalid output") are load-bearing --
        # callers and tests match on them. Detail is appended after an em dash.
        if not response_id or output is None:
            missing = [
                name for name, present in (("id", bool(response_id)), ("output", output is not None)) if not present
            ]
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
            raise ValueError(f"Invalid final response: empty or invalid output \u2014 {detail}")

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
        """Single source of truth for `SessionEnd.total_usage` and the public
        `get_usage()` accessor, so the two can never drift. `cost` is present
        only when at least one response reported it."""
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

    # -- two-phase approval preparation -----------------------------------------

    async def _prepare_tool_call_for_approval(self, call: ParsedToolCall) -> ParsedToolCall:
        """Run PreToolUse ahead of execution and park its outcome in
        `_prepared_tool_calls` (upstream `prepareToolCallForApproval`).
        Returns the effective call (the mutated input when a handler piped
        one). Raw-string (unparseable) arguments never reach the hook."""
        if not self._hooks or isinstance(call.arguments, str):
            return call
        pre = await self._hooks.emit(
            "PreToolUse",
            {"tool_name": call.name, "tool_input": call.arguments if isinstance(call.arguments, dict) else {}},
            tool_name=call.name,
            session_id=self._session_id,
        )
        if pre.blocked:
            block = next((r.get("block") for r in pre.results if r.get("block")), None)
            reason = block if isinstance(block, str) else "Blocked by PreToolUse hook"
            self._prepared_tool_calls[call.id] = {"type": "blocked", "reason": reason}
            return call
        effective = dataclasses.replace(call, arguments=pre.final_payload.get("tool_input")) if pre.mutated else call
        self._prepared_tool_calls[call.id] = {"type": "ready", "call": effective, "mutated": bool(pre.mutated)}
        return effective

    async def _mutated_input_requires_approval(
        self, call: ParsedToolCall, context: Mapping[str, Any], tools: Sequence[Tool]
    ) -> bool:
        """Re-check only approval sources whose answer can depend on input:
        the call-level `require_approval` option, or a callable tool-level
        `require_approval` (upstream `mutatedInputRequiresApproval`)."""
        if not tools:
            return False
        call_level = self.options.get("require_approval")
        if call_level is not None:
            return await tool_requires_approval(call, tools, context, call_level)
        tool = self._find_tool(call.name, tools)
        if tool is None or not callable(get_tool_function(tool).get("require_approval")):
            return False
        return await tool_requires_approval(call, tools, context)

    async def _emit_prepared_failure(self, call: ParsedToolCall, reason: str) -> None:
        """PostToolUseFailure for a call whose lifecycle entered PreToolUse but
        never executed (mutated-input denial/rejection, invalid mutation)."""
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

    async def _block_prepared_tool_call(self, call: ParsedToolCall, reason: str) -> None:
        self._prepared_tool_calls[call.id] = {"type": "blocked", "reason": reason}
        await self._emit_prepared_failure(call, reason)

    async def _validate_prepared_mutation(self, effective: ParsedToolCall, tools: Sequence[Tool]) -> str:
        """Validate hook-mutated input against the tool's input schema before
        it can be re-gated or executed. Returns "ready" or "blocked"."""
        tool = self._find_tool(effective.name, tools)
        if tool is None:
            return "ready"
        try:
            parsed = validate_schema(get_tool_function(tool).get("input_schema"), effective.arguments)
        except Exception:  # noqa: BLE001 - any validation failure blocks the call
            parsed = None
            valid = False
        else:
            valid = True
        if not valid or not isinstance(parsed, (Mapping, BaseModel)):
            await self._block_prepared_tool_call(effective, f'PreToolUse produced invalid input for "{effective.name}"')
            return "blocked"
        return "ready"

    async def _prepare_after_initial_approval(
        self,
        call: ParsedToolCall,
        context: Mapping[str, Any],
        tools: Sequence[Tool],
        gate_key: Tuple[Any, Any],
    ) -> str:
        """Second approval phase for a call that passed (or skipped) the
        initial gate (upstream `prepareAfterInitialApproval`): run PreToolUse,
        validate a mutated input, and re-gate it once when an input-dependent
        approval source now requires approval. Returns "ready", "blocked", or
        "pending" (the mutated call needs a human decision)."""
        effective = await self._prepare_tool_call_for_approval(call)
        prepared = self._prepared_tool_calls.get(call.id)
        if prepared is None or prepared["type"] != "ready" or not prepared["mutated"]:
            return "blocked" if prepared is not None and prepared["type"] == "blocked" else "ready"

        if await self._validate_prepared_mutation(effective, tools) == "blocked":
            return "blocked"
        if gate_key in self._completed_mutated_approval_gates:
            return "ready"
        self._completed_mutated_approval_gates.add(gate_key)
        if not await self._mutated_input_requires_approval(effective, context, tools):
            return "ready"

        decision, reason = await self._emit_permission_request(effective, tools)
        if decision == "allow":
            return "ready"
        if decision == "deny":
            await self._block_prepared_tool_call(effective, reason or "Denied by PermissionRequest hook")
            return "blocked"
        return "pending"

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
        self,
        tool: Tool,
        call: ParsedToolCall,
        context: Mapping[str, Any],
        run_pre_tool_use: bool = True,
    ) -> Dict[str, Any]:
        """Execute a single tool and emit the PreToolUse/PostToolUse/
        PostToolUseFailure lifecycle hooks around it.

        PreToolUse normally already ran during approval preparation
        (`_prepare_tool_call_for_approval`); its parked outcome is consumed
        here so the hook never runs twice for one call. It is emitted here
        only when no prepared outcome exists and `run_pre_tool_use` is True.

        Returns a tagged outcome:
        - `{"type": "parse_error", "call", "error_message"}` -- the model
          produced invalid JSON for this call's arguments. No hooks fire.
        - `{"type": "hook_blocked", "call", "reason"}` -- PreToolUse blocked
          the call (or its prepared outcome was blocked).
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

        prepared = self._prepared_tool_calls.pop(call.id, None)
        if prepared is not None and prepared["type"] == "blocked":
            return {"type": "hook_blocked", "call": call, "reason": prepared["reason"]}

        effective_call: ParsedToolCall = prepared["call"] if prepared is not None else call
        # Emit PreToolUse here only when the approval gate has not already
        # done so (e.g. an approved call resumed from legacy persisted state).
        if self._hooks and prepared is None and run_pre_tool_use:
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

    async def _record_preliminary(self, call_id: str, value: Any) -> None:
        name = self._call_names.get(call_id, "")
        await self._append_event(
            {
                "type": "tool.preliminary_result",
                "toolCallId": call_id,
                "tool_call_id": call_id,
                "toolName": name,
                "tool_name": name,
                "result": value,
                "timestamp": _now_ms(),
            }
        )

    async def _tool_result_to_output(
        self, call: ParsedToolCall, tool: Tool, result: Mapping[str, Any]
    ) -> Dict[str, Any]:
        if result.get("error") is not None:
            output: Any = json_dumps({"error": str(result["error"])})
        else:
            converter = get_tool_function(tool).get("to_model_output")
            if converter:
                converted = await maybe_await(converter({"output": result.get("result"), "input": call.arguments}))
                if isinstance(converted, Mapping) and converted.get("type") == "content":
                    output = converted.get("value", [])
                else:
                    output = json_dumps(result.get("result"))
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

    # -- doom-loop detection -------------------------------------------------

    async def _begin_doom_loop_round(self, batch: Sequence[ParsedToolCall], tools: Sequence[Tool]) -> None:
        """Advance the doom-loop round counter and declare the round's calls
        (upstream `beginDoomLoopRound`). A round's identity for one tool is
        the *set* of fingerprints it was called with, so the set must be
        complete before any of its calls is scored.

        INVARIANT -- every declared member must eventually be recorded. The
        filters below therefore mirror every path that skips the per-call
        checkpoint: unknown/manual tools (`is_auto_resolvable_tool`), calls a
        PermissionRequest hook denied, and `loop_key`-exempt calls. A
        declared-but-never-recorded phantom member silently resets the
        tool's round streak.
        """
        monitor = self._doom_loop_monitor
        if monitor is None:
            return
        self._doom_loop_round += 1
        self._doom_loop_round_decisions.clear()
        self._doom_loop_round_key_material.clear()

        declared: List[Any] = []
        for call in batch:
            tool = self._find_tool(call.name, tools)
            # Never checked => never recorded => must not be declared (and
            # its `loop_key` must not run). Precedes the malformed branch: a
            # raw-string call to an unknown or manual tool is never recorded.
            if tool is None or not is_auto_resolvable_tool(tool):
                continue
            if call.id in self._hook_denied_calls:
                continue
            if isinstance(call.arguments, str):
                declared.append({"tool_name": str(call.name), "key_material": call.arguments})
                continue
            # `resolve_loop_key_material` catches a raising `loop_key`, but the
            # field-list form reads the arguments, which can raise on its own.
            # Skip just that call; the per-call checkpoint retries it.
            try:
                resolution = resolve_loop_key_material(
                    get_tool_function(tool).get("loop_key"),
                    call.arguments if call.arguments is not None else {},
                )
            except Exception as error:  # noqa: BLE001 - detection must never fail a run
                warnings.warn(
                    f'[DoomLoop] could not resolve loop identity for "{call.name}" while declaring the round; '
                    f"excluding it from the round set: {error}",
                    stacklevel=2,
                )
                continue
            # Cache so the checkpoint does not invoke `loop_key` again. A
            # duplicate (model-emitted) call id poisons the entry so neither
            # call aliases the other's identity.
            if call.id in self._doom_loop_round_key_material:
                self._doom_loop_round_key_material[call.id] = _DUPLICATE_CALL_ID
            else:
                self._doom_loop_round_key_material[call.id] = resolution
            if resolution["kind"] == "exempt":
                continue
            declared.append({"tool_name": str(call.name), "key_material": resolution["key_material"]})
        await monitor.declare_round(self._doom_loop_round, declared)

    async def _evaluate_doom_loop_call(
        self,
        *,
        tool_name: str,
        key_material: Any,
        allow_block: bool,
        detector: Literal["tool-fingerprint", "server-tool-fingerprint"],
        fallback_key_material: Any = _NO_FALLBACK,
        call: Optional[ParsedToolCall] = None,
    ) -> Dict[str, Any]:
        """Evaluate one call against the monitor (upstream
        `enqueueDoomLoopEvaluation`). Upstream serializes evaluations on a
        promise chain because its round executes in parallel; this port runs
        a round's calls sequentially in emission order, so plain awaiting
        gives the same deterministic recording order.

        Unhashable key material degrades to `fallback_key_material`; if there
        is none (or it is also unhashable) detection is skipped for the call.
        Returns `{"action": ..., "message"?: ...}` (`"proceed"` when no rung
        fired). Shared side effects (steer queue, escalation, stop arm) are
        applied here.
        """
        monitor = self._doom_loop_monitor
        if monitor is None:
            return {"action": "proceed"}
        try:
            record = await monitor.record_tool_call(
                tool_name, key_material, self._doom_loop_round, allow_block=allow_block, detector=detector
            )
        except Exception as error:  # noqa: BLE001 - detection must never fail a call
            has_fallback = fallback_key_material is not _NO_FALLBACK
            warnings.warn(
                f'[DoomLoop] could not fingerprint call to "{tool_name}"; '
                f"{'falling back to full arguments' if has_fallback else 'skipping detection for this call'}: "
                f"{error}",
                stacklevel=2,
            )
            if not has_fallback:
                return {"action": "proceed"}
            try:
                record = await monitor.record_tool_call(
                    tool_name, fallback_key_material, self._doom_loop_round, allow_block=allow_block, detector=detector
                )
            except Exception as fallback_error:  # noqa: BLE001
                warnings.warn(
                    f'[DoomLoop] fallback identity for "{tool_name}" also unhashable; '
                    f"skipping detection for this call: {fallback_error}",
                    stacklevel=2,
                )
                return {"action": "proceed"}

        # Same (tool, fingerprint) already decided this round: reuse the
        # decision -- no second streak increment, no second hook emission.
        decision_key = f"{tool_name}\n{record['fingerprint']}"
        if record["duplicate_in_round"]:
            return self._doom_loop_round_decisions.get(decision_key, {"action": "proceed"})
        verdict = record.get("verdict")
        if verdict is None:
            decision: Dict[str, Any] = {"action": "proceed"}
            self._doom_loop_round_decisions[decision_key] = decision
            return decision
        action = await self._apply_doom_loop_verdict(verdict, call)
        decision = {"action": action, "message": verdict["message"]}
        self._doom_loop_round_decisions[decision_key] = decision
        self._apply_doom_loop_side_effects(action, verdict)
        return decision

    async def _check_doom_loop_before_execution(self, tool: Tool, call: ParsedToolCall) -> Tuple[bool, str]:
        """Doom-loop checkpoint for one parsed call, before PreToolUse and
        execution (upstream `checkDoomLoopBeforeExecution`). Returns
        `(blocked, reason)`; `stop` also blocks the condemned call."""
        if self._doom_loop_monitor is None:
            return False, ""
        call_arguments = call.arguments if call.arguments is not None else {}
        cached = self._doom_loop_round_key_material.get(call.id)
        resolution: LoopKeyResolution
        if cached is not None and cached is not _DUPLICATE_CALL_ID:
            resolution = cast(LoopKeyResolution, cached)
        else:
            try:
                resolution = resolve_loop_key_material(get_tool_function(tool).get("loop_key"), call_arguments)
            except Exception as error:  # noqa: BLE001 - detection must never fail a run
                warnings.warn(
                    f'[DoomLoop] could not resolve loop identity for "{call.name}"; '
                    f"skipping detection for this call: {error}",
                    stacklevel=2,
                )
                return False, ""
        if resolution["kind"] == "exempt":
            return False, ""
        key_material = resolution["key_material"]
        if resolution["kind"] == "fallback":
            warnings.warn(f'[DoomLoop] tool "{call.name}": {resolution["warning"]}', stacklevel=2)
        decision = await self._evaluate_doom_loop_call(
            tool_name=str(call.name),
            key_material=key_material,
            fallback_key_material=_NO_FALLBACK if key_material is call_arguments else call_arguments,
            allow_block=True,
            detector="tool-fingerprint",
            call=call,
        )
        if decision["action"] in ("block", "stop"):
            return True, decision.get("message") or (
                f'Doom loop suspected: tool "{call.name}" is repeating identical calls. '
                "Repeating the call will not change the result. Take a different approach."
            )
        return False, ""

    async def _apply_doom_loop_verdict(
        self, verdict: DoomLoopVerdict, call: Optional[ParsedToolCall] = None
    ) -> DoomLoopAction:
        """Emit the DoomLoopDetected hook and resolve the final action. The
        LAST `override_action` in the chain wins. Non-blockable verdicts
        (text, server tools) downgrade 'block' to 'observe'; 'escalate'
        without a mechanism or budget downgrades to 'observe'."""
        action: DoomLoopAction = verdict["action"]
        if self._hooks:
            payload: Dict[str, Any] = {
                "detector": verdict["detector"],
                "action": verdict["action"],
                "streak": verdict["streak"],
                "fingerprint": verdict["fingerprint"],
                "message": verdict["message"],
            }
            if "tool_name" in verdict:
                payload["tool_name"] = verdict["tool_name"]
            if call is not None and isinstance(call.arguments, dict):
                payload["tool_input"] = call.arguments
            emitted = await self._hooks.emit(
                "DoomLoopDetected",
                payload,
                tool_name=verdict.get("tool_name"),
                session_id=self._session_id,
            )
            for result in emitted.results:
                override = result.get("override_action") if isinstance(result, Mapping) else None
                if override:
                    action = override
        if verdict["detector"] != "tool-fingerprint" and action == "block":
            action = "observe"
        if action == "escalate" and not (
            self._doom_loop_monitor is not None and self._doom_loop_monitor.can_escalate()
        ):
            action = "observe"
        return action

    def _apply_doom_loop_side_effects(self, action: str, verdict: DoomLoopVerdict) -> None:
        """Side effects shared by every checkpoint: queue steer guidance, arm
        an escalation for the next request, or arm the stop verdict. 'block'
        has none -- the calling checkpoint synthesizes the blocked output."""
        if action == "steer":
            self._queue_doom_loop_steer(verdict["message"])
        elif action == "escalate":
            # Latch the FIRST escalation verdict for the next request: a second
            # verdict in the same window must not double-spend the budget.
            if self._pending_doom_loop_escalation is None:
                self._pending_doom_loop_escalation = verdict
                self._queue_doom_loop_steer(
                    f"{verdict['message']} An escalated turn follows: use the additional guidance to change course."
                )
        elif action == "stop":
            self._doom_loop_stop = verdict

    def _apply_doom_loop_escalation(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Consume a pending escalation into ONE-TURN request overrides
        (upstream `takeDoomLoopEscalationOverrides`) and return the request to
        dispatch. The caller's request is never mutated, so the following turn
        reverts automatically. Burns one unit of the budget at APPLICATION
        time.

        - `model`: replace the model for this dispatch (dropping any `models`
          fallback list so it cannot shadow the override).
        - `advisor`: append an `openrouter:advisor` server tool and pin
          `tool_choice` to it so the model must consult it this turn.
        """
        verdict = self._pending_doom_loop_escalation
        config = self._doom_loop_escalation
        monitor = self._doom_loop_monitor
        self._pending_doom_loop_escalation = None
        if verdict is None or config is None or monitor is None or not monitor.can_escalate():
            return request
        monitor.consume_escalation()
        escalated: Dict[str, Any] = dict(request)
        if config.get("model") is not None:
            escalated["model"] = config["model"]
            escalated.pop("models", None)
        advisor = config.get("advisor")
        if advisor is not None and advisor is not False:
            # Upstream spells the parameter `forwardTranscript`; this port's
            # wire is the generated Python SDK's snake_case
            # (`AdvisorServerToolConfig.forward_transcript`).
            advisor_parameters: Dict[str, Any] = {
                "forward_transcript": True,
                "instructions": (
                    "You are an escalation advisor. The executing model appears stuck in a loop: "
                    f"{verdict['message']} Diagnose why its approach is failing and give concrete, "
                    "specific instructions for a DIFFERENT approach. Do not restate the problem."
                ),
                **(advisor if isinstance(advisor, Mapping) else {}),
            }
            existing_tools = request.get("tools")
            escalated["tools"] = [
                *(existing_tools if isinstance(existing_tools, list) else []),
                {"type": "openrouter:advisor", "parameters": advisor_parameters},
            ]
            escalated["tool_choice"] = {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "openrouter:advisor"}],
            }
        return escalated

    async def _check_doom_loop_for_response(self, response: Any) -> None:
        """Step-level checkpoint on a fresh model response (upstream
        `checkDoomLoopForResponse`): the text detectors, then server-tool
        fingerprints. Server tools never pass through `_run_tool_with_hooks`,
        so their repetition is detected here, post-execution (no block)."""
        monitor = self._doom_loop_monitor
        if monitor is None:
            return
        verdict: Optional[DoomLoopVerdict]
        try:
            verdict = await monitor.record_assistant_text(extract_text_from_response(response))
        except Exception as error:  # noqa: BLE001 - detection must never fail a run
            warnings.warn(f"[DoomLoop] text checkpoint failed; skipping for this response: {error}", stacklevel=2)
            verdict = None
        if verdict is not None:
            action = await self._apply_doom_loop_verdict(verdict)
            self._apply_doom_loop_side_effects(action, verdict)

        for item in self._response_output_items(response):
            item_type = get_field(item, "type")
            if item_type is None or item_type in ("message", "reasoning", "function_call"):
                continue
            identity = _extract_server_tool_identity(item)
            if identity is None:
                continue
            await self._evaluate_doom_loop_call(
                tool_name=f"server:{item_type}",
                key_material=identity,
                allow_block=False,
                detector="server-tool-fingerprint",
            )

    def _queue_doom_loop_steer(self, message: str) -> None:
        """Queue steer guidance, deduplicating identical messages."""
        if message not in self._pending_doom_loop_steer:
            self._pending_doom_loop_steer.append(message)

    async def _flush_doom_loop_steer(self, current_request: Dict[str, Any]) -> None:
        """Inject queued steer guidance as one user message before the next
        follow-up request. Only called at a safe point (a fully-paired round):
        a user message between a dangling function_call and its output would
        be invalid history, so pauses persist the queue instead."""
        if not self._pending_doom_loop_steer:
            return
        prompt = "\n".join(self._pending_doom_loop_steer)
        self._pending_doom_loop_steer = []
        await self._inject_append_prompt_message(prompt, current_request)

    async def _seal_doom_loop_stop(self, response: Any, resolved_call_ids: Optional[set[str]] = None) -> None:
        """Halt for an armed stop verdict keeping persisted history
        well-formed: every function_call in `response` without an output yet
        gets a synthesized halt-error output, so a stateful resume never sends
        a dangling function_call (providers reject those with a 400)."""
        verdict = self._doom_loop_stop
        unresolved = [
            call
            for call in extract_tool_calls_from_response(response)
            if resolved_call_ids is None or call.id not in resolved_call_ids
        ]
        if not unresolved:
            return
        halt_message = (
            f"Run halted by doom-loop detection: {verdict['message']}"
            if verdict is not None
            else "Run halted by doom-loop detection."
        )
        await self._save_tool_outputs_to_state([self._rejected_output(call.id, halt_message) for call in unresolved])

    # -- approval gate -------------------------------------------------------

    def _is_decision_resume(self) -> bool:
        """Resuming an approval/HITL pause with explicit approve/reject
        decisions (`process_approval_decisions` upstream)."""
        state = self._state
        return bool(
            state is not None
            and state.status in ("awaiting_approval", "awaiting_hitl")
            and state.pending_tool_calls
            and (self._approved_ids or self._rejected_ids)
        )

    def _stays_paused_without_decisions(self, resolved_input: Any) -> bool:
        """Paused for a human decision but none was supplied: stay paused.

        Falling through would flip the status to in_progress and send a fresh
        model request, advancing the conversation past calls that still owe a
        decision. An `awaiting_hitl` conversation still resumes the documented
        way -- a `function_call_output` for a paused call in `input`.
        """
        state = self._state
        if state is None or state.status not in ("awaiting_approval", "awaiting_hitl"):
            return False
        if self._approved_ids or self._rejected_ids:
            return False
        pending_ids = {call.id for call in state.pending_tool_calls or []}
        if not pending_ids:
            return False
        if state.status != "awaiting_hitl":
            return True
        if not isinstance(resolved_input, list):
            return True
        return not any(
            get_field(item, "type") == "function_call_output"
            and get_field(item, "callId", get_field(item, "call_id")) in pending_ids
            for item in resolved_input
        )

    def _paused_error(self) -> Optional[Exception]:
        """Describe the pause when the conversation is waiting on the caller
        (`awaiting_approval` / `awaiting_hitl`), or None when it is not."""
        state = self._state
        status = state.status if state is not None else None
        if status not in ("awaiting_approval", "awaiting_hitl"):
            return None
        pending_ids = [call.id for call in (state.pending_tool_calls or [])]
        shown = ", ".join(pending_ids[:_MAX_PENDING_IDS_IN_ERROR])
        hidden = len(pending_ids) - _MAX_PENDING_IDS_IN_ERROR
        id_list = f"{shown}, ... ({hidden} more)" if hidden > 0 else shown
        if status == "awaiting_approval":
            what = "awaiting approval"
            how = "Resume with approve_tool_calls and/or reject_tool_calls for the pending calls."
        else:
            what = "awaiting HITL input"
            how = (
                "Resume by passing a function_call_output for a pending call ID in input, "
                "or with approve_tool_calls / reject_tool_calls (approving re-runs on_tool_called)."
            )
        raw_input = self.options["request"].get("input")
        input_supplied = callable(raw_input) or bool(raw_input)
        return RuntimeError(
            f"Conversation is {what} (status '{status}'); no response is available. "
            f"Pending tool call IDs: {id_list}. {how} "
            "Use requires_approval() or get_pending_tool_calls() to inspect a paused conversation."
            + (
                " The input supplied with this call was not sent to the model and was not saved to state."
                if input_supplied
                else ""
            )
        )

    def _response_unavailable_error(self, fallback: str = "Response not available") -> Exception:
        return self._paused_error() or RuntimeError(fallback)

    async def _execute_auto_approve_tools(
        self, calls: Sequence[ParsedToolCall], tools: Sequence[Tool], turn_context: Mapping[str, Any]
    ) -> List[UnsentToolResult]:
        """Execute auto-approved calls while pausing for approval and keep
        their results as unsent, so the resume path replays them without
        re-executing."""
        results: List[UnsentToolResult] = []
        for call in calls:
            tool = self._find_tool(call.name, tools)
            if not tool or not is_auto_resolvable_tool(tool):
                continue
            try:
                outcome = await self._run_tool_with_hooks(tool, call, turn_context)
            except Exception as error:  # noqa: BLE001 - mirrors upstream Promise.allSettled
                results.append(create_rejected_result(call.id, call.name, str(error)))
                continue
            if outcome["type"] == "parse_error":
                await self._append_tool_result_event(call, tool, {"error": outcome["error_message"]})
                results.append(create_rejected_result(call.id, call.name, outcome["error_message"]))
            elif outcome["type"] == "hook_blocked":
                results.append(create_rejected_result(call.id, call.name, outcome["reason"]))
            elif outcome["result"] is None:
                # HITL tool paused -- no unsent result for this call.
                continue
            elif outcome["result"].get("error") is not None:
                results.append(create_rejected_result(call.id, call.name, str(outcome["result"]["error"])))
            else:
                results.append(create_unsent_result(call.id, call.name, outcome["result"].get("result")))
        return results

    async def _handle_approval_check(
        self,
        calls: Sequence[ParsedToolCall],
        current_round: int,
        response: Any,
        tools: Sequence[Tool],
        skip_auto_execution: bool = False,
    ) -> bool:
        """Gate a response's tool calls on approval. Returns True when the run
        must pause (`awaiting_approval`).

        Each call occurrence in a response is gated at most once per run, so
        the pre-loop gate, the in-loop gate, and the `allow_final_response`
        gate never re-emit `PermissionRequest` or re-run predicates for an
        already-resolved call. Nothing is executed when no call needs a
        gate -- the normal tool round runs every call exactly once. When
        pausing, auto-approved siblings run now (or, with
        `skip_auto_execution`, are recorded as skipped) and are persisted as
        unsent results for the resume.
        """
        if not tools:
            return False
        unseen, blocked = await self._classify_initial_approval_calls(calls, response, tools)
        if not unseen and not blocked:
            return False

        turn_context = {"number_of_turns": current_round}
        partition = await partition_tool_calls(unseen, tools, turn_context, self.options.get("require_approval"))
        needs_approval = list(partition["requires_approval"])
        auto_execute = list(partition["auto_execute"])

        denied, still_pending, initial_survivors = await self._resolve_approval_phases(
            needs_approval, auto_execute, calls, response, tools, turn_context
        )

        if not still_pending:
            for call, reason in denied:
                self._hook_denied_calls[call.id] = reason
            return False

        if self.options.get("state") is None:
            names = ", ".join(call.name for call in still_pending)
            raise ValueError(
                f"Tool(s) require approval but no state accessor is configured: {names}. "
                'Provide a StateAccessor via the "state" parameter to enable approval workflows.'
            )

        pending_ids = {call.id for call in still_pending}
        # Doom-blocked calls ride along: `_execute_auto_approve_tools` turns
        # them into rejected unsent results without executing them. Survivors
        # consume their prepared PreToolUse outcome there.
        executable_now = [*[c for c in initial_survivors if c.id not in pending_ids], *blocked]
        if skip_auto_execution:
            unsent = [
                create_rejected_result(call.id, call.name, "Tool execution skipped: stop condition met")
                for call in executable_now
            ]
        else:
            unsent = await self._execute_auto_approve_tools(executable_now, tools, turn_context)
        combined = [*unsent, *[create_rejected_result(call.id, call.name, reason) for call, reason in denied]]

        # A pending call whose PreToolUse already ran (it paused on the
        # mutated-input gate) persists its effective arguments plus the
        # marker, so the resume does not run the hook again.
        pending_to_save = [
            dataclasses.replace(call, pre_tool_use_applied=True) if call.id in self._prepared_tool_calls else call
            for call in still_pending
        ]
        save_kwargs: Dict[str, Any] = {"pending_tool_calls": pending_to_save, "status": "awaiting_approval"}
        if combined:
            save_kwargs["unsent_tool_results"] = combined
        await self._save_state(**save_kwargs)
        self._prepared_tool_calls.clear()
        self._hook_denied_calls.clear()
        self._final_response = response
        return True

    async def _resolve_approval_phases(
        self,
        needs_approval: Sequence[ParsedToolCall],
        auto_execute: Sequence[ParsedToolCall],
        calls: Sequence[ParsedToolCall],
        response: Any,
        tools: Sequence[Tool],
        turn_context: Mapping[str, Any],
    ) -> Tuple[List[Tuple[ParsedToolCall, str]], List[ParsedToolCall], List[ParsedToolCall]]:
        """Run both approval phases (upstream `resolveApprovalPhases`).

        Phase 1 gates the model-issued arguments: PermissionRequest for every
        call needing approval. Phase 2 runs PreToolUse for every
        auto-resolvable survivor (auto-approved or hook-allowed) so its
        effective arguments are known before execution; a mutated input is
        validated and re-gated by input-dependent approval sources. Returns
        `(denied, still_pending, initial_survivors)`.
        """
        denied: List[Tuple[ParsedToolCall, str]] = []
        still_pending: List[ParsedToolCall] = []
        if self._hooks:
            for call in needs_approval:
                decision, reason = await self._emit_permission_request(call, tools)
                if decision == "deny":
                    denied.append((call, reason or "Denied by PermissionRequest hook"))
                elif decision != "allow":
                    still_pending.append(call)
        else:
            still_pending = list(needs_approval)

        pending_ids = {call.id for call in still_pending}
        denied_ids = {call.id for call, _ in denied}
        initial_survivors = [
            *auto_execute,
            *[c for c in needs_approval if c.id not in pending_ids and c.id not in denied_ids],
        ]
        # Occurrence identity of each call within its response, matching the
        # initial gate's `(id(response), index)` keying.
        occurrences = {id(call): index for index, call in enumerate(calls)}
        for call in initial_survivors:
            tool = self._find_tool(call.name, tools)
            if tool is None or not is_auto_resolvable_tool(tool):
                continue
            gate_key = (id(response), occurrences.get(id(call), call.id))
            if await self._prepare_after_initial_approval(call, turn_context, tools, gate_key) == "pending":
                prepared = self._prepared_tool_calls.get(call.id)
                if prepared is not None and prepared["type"] == "ready":
                    still_pending.append(prepared["call"])
        return denied, still_pending, initial_survivors

    async def _classify_initial_approval_calls(
        self, calls: Sequence[ParsedToolCall], response: Any, tools: Sequence[Tool]
    ) -> Tuple[List[ParsedToolCall], List[ParsedToolCall]]:
        """Mark each not-yet-gated call occurrence as gated and run the
        doom-loop checkpoint on it (upstream `classifyInitialApprovalCalls`).

        The checkpoint runs on the model-issued arguments BEFORE approval
        partitioning, PermissionRequest, and PreToolUse. Returns `(unseen,
        blocked)`: `blocked` calls were refused by the checkpoint and are
        excluded from partitioning; `_run_tool_with_hooks` later synthesizes
        their error output without executing them.
        """
        response_key = id(response)
        unseen_candidates = [
            call for index, call in enumerate(calls) if (response_key, index) not in self._completed_approval_gates
        ]
        await self._begin_doom_loop_round(unseen_candidates, tools)

        unseen: List[ParsedToolCall] = []
        blocked: List[ParsedToolCall] = []
        for index, call in enumerate(calls):
            gate = (response_key, index)
            if gate in self._completed_approval_gates:
                continue
            self._completed_approval_gates.add(gate)
            tool = self._find_tool(call.name, tools)
            if self._doom_loop_monitor is not None and tool is not None and is_auto_resolvable_tool(tool):
                if isinstance(call.arguments, str):
                    # Malformed call: its identity is the raw string.
                    decision = await self._evaluate_doom_loop_call(
                        tool_name=str(call.name),
                        key_material=call.arguments,
                        allow_block=True,
                        detector="tool-fingerprint",
                        call=call,
                    )
                    is_blocked = decision["action"] in ("block", "stop")
                    reason = decision.get("message") or "Blocked by doom loop"
                else:
                    is_blocked, reason = await self._check_doom_loop_before_execution(tool, call)
                if is_blocked:
                    self._prepared_tool_calls[call.id] = {"type": "blocked", "reason": reason}
                    blocked.append(call)
                    continue
            unseen.append(call)
        return unseen, blocked

    async def _process_approval_decisions(self, request: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply approve/reject decisions to a paused conversation.

        Returns the resume request, or None when the conversation stays paused
        (some pending calls still owe a decision, or an approved HITL call
        paused again).
        """
        self._fresh_items_for_state = []
        state = self._state
        assert state is not None
        tools: Sequence[Tool] = self.options.get("tools") or []
        pending: List[ParsedToolCall] = list(state.pending_tool_calls or [])
        unsent: List[UnsentToolResult] = list(state.unsent_tool_results or [])
        self._call_names.update({call.id: call.name for call in pending})
        hitl_paused: set[str] = set()
        # Calls that pause again after a PreToolUse mutation stay pending,
        # marked so the next resume does not re-run the hook.
        second_gate_paused: set[str] = set()
        turn_context = {"number_of_turns": self._all_tool_rounds + 1}
        persisted_key = f"persisted:{state.previous_response_id or 'unknown'}"

        for call_id in self._approved_ids:
            index = next((i for i, c in enumerate(pending) if c.id == call_id), None)
            if index is None:
                continue
            call = pending[index]
            tool = self._find_tool(call.name, tools)
            if not tool or not is_auto_resolvable_tool(tool):
                unsent.append(create_rejected_result(call_id, call.name, "Tool not found or not executable"))
                continue
            if not call.pre_tool_use_applied:
                # Approved on the model-issued arguments (or legacy state
                # without the marker): PreToolUse runs now, and a mutated
                # input needs its own approval.
                prepared = await self._prepare_after_initial_approval(
                    call, turn_context, tools, (persisted_key, call_id)
                )
                if prepared == "pending":
                    ready = self._prepared_tool_calls.get(call_id)
                    if ready is not None and ready["type"] == "ready":
                        pending[index] = dataclasses.replace(ready["call"], pre_tool_use_applied=True)
                    second_gate_paused.add(call_id)
                    continue
            outcome = await self._run_tool_with_hooks(tool, call, turn_context, run_pre_tool_use=False)
            if outcome["type"] == "parse_error":
                await self._append_tool_result_event(call, tool, {"error": outcome["error_message"]})
                unsent.append(create_rejected_result(call_id, call.name, outcome["error_message"]))
            elif outcome["type"] == "hook_blocked":
                unsent.append(create_rejected_result(call_id, call.name, outcome["reason"]))
            elif outcome["result"] is None:
                # HITL tool paused on approval -- keep it visible on
                # pending_tool_calls (status becomes awaiting_hitl below).
                hitl_paused.add(call_id)
            elif outcome["result"].get("error") is not None:
                unsent.append(create_rejected_result(call_id, call.name, str(outcome["result"]["error"])))
            else:
                unsent.append(create_unsent_result(call_id, call.name, outcome["result"].get("result")))

        for call_id in self._rejected_ids:
            rejected = next((c for c in pending if c.id == call_id), None)
            if rejected is None:
                continue
            call = rejected
            reason = "Rejected by user"
            if call.pre_tool_use_applied:
                # The call's lifecycle already entered PreToolUse (it paused
                # on the mutated-input gate), so the rejection is a failure.
                await self._emit_prepared_failure(call, reason)
            self._prepared_tool_calls.pop(call_id, None)
            self._hook_denied_calls.pop(call_id, None)
            unsent.append(create_rejected_result(call_id, call.name, reason))

        processed = {
            cid
            for cid in [*self._approved_ids, *self._rejected_ids]
            if cid not in hitl_paused and cid not in second_gate_paused
        }
        remaining = [c for c in pending if c.id not in processed]
        remaining_unresolved = [c for c in remaining if c.id not in hitl_paused]
        if remaining_unresolved:
            next_status = "awaiting_approval"
        elif hitl_paused:
            next_status = "awaiting_hitl"
        else:
            next_status = "in_progress"
        if next_status == "in_progress" and self._doom_loop_stop is not None:
            # This approval continuation is terminal: no model request
            # follows, so a later fresh run must not inherit its consumed
            # caller choice.
            self._reset_forced_tool_choice_policy()
        await self._save_state(
            status=next_status,
            pending_tool_calls=remaining or None,
            unsent_tool_results=unsent or None,
        )
        if next_status != "in_progress":
            # Queued steer guidance rides doom_loop.pending_steer in the
            # state persisted above; the resumed run delivers it later.
            self._prepared_tool_calls.clear()
            self._hook_denied_calls.clear()
            return None

        # A doom stop re-armed from persisted state (decision-only resume of
        # a condemned run): halt WITHOUT the unsent-results model request.
        # The executed outputs stay persisted as unsent results.
        if self._doom_loop_stop is not None:
            return None

        # Continue with the unsent results. SDK-generated outputs are
        # appended as-is: `on_response_received` is reserved for
        # caller-supplied outputs.
        outputs = unsent_results_to_api_format(unsent)
        updated_messages = append_to_messages(state.messages, outputs)
        await self._save_state(messages=updated_messages, unsent_tool_results=None)
        return {**dict(request), "input": updated_messages, "stream": True}

    async def _append_tool_result_event(self, call: ParsedToolCall, tool: Tool, result: Any) -> None:
        await self._append_event(
            {
                "type": "tool.result",
                "toolCallId": call.id,
                "tool_call_id": call.id,
                "toolName": call.name,
                "tool_name": call.name,
                "source": "mcp" if is_mcp_tool(tool) else "client",
                "result": result,
                "timestamp": _now_ms(),
            }
        )

    # -- tool rounds ------------------------------------------------------------

    async def _execute_round(
        self,
        calls: Sequence[ParsedToolCall],
        tools: Sequence[Tool],
        turn_context: Mapping[str, Any],
        step: StepResult,
    ) -> Tuple[List[Any], List[ParsedToolCall], List[ParsedToolCall]]:
        """Execute one round of tool calls. Returns `(outputs, paused,
        executed_calls)`. Calls denied by a PermissionRequest hook get a
        synthesized rejected output instead of executing."""
        outputs: List[Any] = []
        paused: List[ParsedToolCall] = []
        executed_calls: List[ParsedToolCall] = []
        for call in calls:
            denied_reason = self._hook_denied_calls.pop(call.id, None)
            if denied_reason is not None:
                output = self._rejected_output(call.id, denied_reason)
                outputs.append(output)
                self._tool_outputs.append(output)
                await self._append_event({"type": "tool.call_output", "output": output, "timestamp": _now_ms()})
                continue
            tool = self._find_tool(call.name, tools)
            if not tool or not is_auto_resolvable_tool(tool):
                continue
            outcome = await self._run_tool_with_hooks(tool, call, turn_context)
            if outcome["type"] == "parse_error":
                await self._append_tool_result_event(call, tool, {"error": outcome["error_message"]})
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
                    "toolName": call.name,
                    "tool_name": call.name,
                    "source": result.get("source", "mcp" if is_mcp_tool(tool) else "client"),
                    "result": {"error": str(result["error"])}
                    if result.get("error") is not None
                    else result.get("result"),
                    "timestamp": _now_ms(),
                }
            )
            output = await self._tool_result_to_output(effective_call, tool, result)
            outputs.append(output)
            executed_calls.append(effective_call)
            self._tool_outputs.append(output)
            await self._append_event({"type": "tool.call_output", "output": output, "timestamp": _now_ms()})
            step.tool_results.append(result)
        return outputs, paused, executed_calls

    async def _resolve_request_for_turn(self, current_request: Dict[str, Any], number_of_turns: int) -> Dict[str, Any]:
        """Re-resolve async/callable request params for a turn, preserving the
        accumulated input (upstream `resolveAsyncFunctionsForTurn`)."""
        if not has_async_functions(self.options["request"]):
            return current_request
        resolved = await resolve_async_functions(self.options["request"], {"number_of_turns": number_of_turns})
        resolved["stream"] = True
        if "input" in current_request:
            resolved["input"] = current_request["input"]
        return self._apply_resolved_forced_tool_choice_policy(resolved)

    # -- the main loop -----------------------------------------------------

    async def _run(self) -> Any:
        session_end_reason = "complete"
        hooks = self._hooks
        try:
            await self._load_state()
            self._session_id = self._state.id if self._state is not None else generate_conversation_id()

            tools: Sequence[Tool] = self.options.get("tools") or []
            request = await resolve_async_functions(self.options["request"], {"number_of_turns": 0})

            # Paused for a human decision but none was supplied: stay paused.
            # No request is sent and the supplied input is neither sent nor
            # saved; every response/stream getter raises an explicit pause
            # error (see `_paused_error`).
            if self._stays_paused_without_decisions(request.get("input")):
                return None

            decision_resume = self._is_decision_resume()
            if self._state is not None and not self._resuming_from_client_tools and not decision_resume:
                await self._save_state(status="in_progress")

            await self._emit_session_start(tools)

            request = self._apply_resolved_forced_tool_choice_policy(request)
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
                if not decision_resume:
                    self._fresh_items_for_state = list(fresh_input)
            elif fresh_input:
                request["input"] = fresh_input

            current_request = request
            is_resume_turn = self._resuming_from_client_tools
            if decision_resume:
                resumed = await self._process_approval_decisions(request)
                if resumed is None:
                    # A doom stop on an approval continuation: like a pause, no
                    # response exists, but the session ends as doom-stopped.
                    if (
                        self._doom_loop_stop is not None
                        and self._state is not None
                        and self._state.status == "in_progress"
                    ):
                        session_end_reason = "doom_loop"
                        self._reset_forced_tool_choice_policy()
                        await self._save_state()
                    return None
                current_request = resumed
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

                # Escalation recovery: one-turn overrides (model swap / forced
                # advisor consult) applied to a tool-round follow-up dispatch
                # only; `current_request` keeps the configured model/tools, so
                # the next turn reverts automatically.
                dispatch_request = (
                    self._apply_doom_loop_escalation(current_request) if turn_type == "tool_round" else current_request
                )
                response = await self._send_and_track(dispatch_request, turn_type, turn_number)
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
                self._call_names.update({call.id: call.name for call in calls})
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
                # The forced final turn (tool_choice "none") ends the run: any
                # tool calls it still carries are not executed.
                if turn_type == "final":
                    break
                # Doom-loop text + server-tool checkpoint for the fresh response.
                await self._check_doom_loop_for_response(response)
                if not calls or not tools:
                    # A text verdict may have armed a stop: report the run as
                    # doom-stopped, not as a normal completion.
                    if self._doom_loop_stop is not None:
                        session_end_reason = "doom_loop"
                    await self._mark_state_complete()
                    break

                stop_when = self.options.get("stop_when")
                stop_conditions = list(stop_when) if isinstance(stop_when, list) else ([stop_when] if stop_when else [])

                if turn_number == 0:
                    # With allow_final_response=False, a met stop_when means no
                    # tool from this response runs. Evaluate it before the
                    # approval check so pausing for an approval-gated call does
                    # not execute its auto-approved siblings.
                    skip_auto_execution = (
                        self.options.get("allow_final_response") is False
                        and bool(stop_conditions)
                        and await is_stop_condition_met(stop_conditions, self._executed_rounds)
                    )
                    if await self._handle_approval_check(calls, 0, response, tools, skip_auto_execution):
                        return final_response
                    # All calls manual / not auto-resolvable: surface them as
                    # pending client tools instead of completing.
                    if not any(self._call_is_auto_resolvable(c, tools) for c in calls):
                        await self._persist_client_tools_pause(response, calls)
                        return final_response

                # A doom-loop 'stop' verdict armed by this response's text (or
                # an earlier checkpoint) halts the run before any further
                # spend. Seal first: this response's calls have not executed,
                # so they need synthesized halt outputs.
                if self._doom_loop_stop is not None:
                    await self._seal_doom_loop_stop(response)
                    session_end_reason = "doom_loop"
                    break

                stopped_by_stop_when = False
                while stop_conditions and await is_stop_condition_met(stop_conditions, self._executed_rounds):
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

                round_context = {"number_of_turns": turn_number + 1, "turn_request": current_request}

                if stopped_by_stop_when:
                    # stop_when broke the loop while the model was still
                    # emitting tool calls: execute them so they have matching
                    # outputs, then make one more tool_choice "none" request to
                    # coerce a final text response. Default-on.
                    allow_final_response = self.options.get("allow_final_response")
                    final_response_enabled = allow_final_response is not False
                    if final_response_enabled and any(self._call_is_auto_resolvable(c, tools) for c in calls):
                        # Gate these calls exactly like a normal round would:
                        # this path executes real tools.
                        if await self._handle_approval_check(calls, turn_number + 1, response, tools):
                            return final_response
                        current_request = await self._resolve_request_for_turn(current_request, turn_number + 1)
                        outputs, paused, executed_calls = await self._execute_round(calls, tools, round_context, step)
                        self._record_executed_round(calls, response, outputs, tools)
                        # Persist real outputs BEFORE the HITL pause check so a
                        # partial batch keeps the regular tool's output.
                        await self._save_tool_outputs_to_state(outputs)
                        if paused:
                            await self._save_state(pending_tool_calls=paused, status="awaiting_hitl")
                            self._final_response = final_response
                            return final_response
                        self._all_tool_rounds += 1
                        # A doom stop armed while executing the halted turn's
                        # calls: seal and skip the final text-coercion request
                        # ("no further model requests after stop").
                        if self._doom_loop_stop is not None:
                            await self._seal_doom_loop_stop(response, {get_field(o, "callId") for o in outputs})
                            session_end_reason = "doom_loop"
                            break
                        await self._apply_next_turn_params(executed_calls, tools, current_request)
                        current_request = self._pending_next_request or current_request
                        self._pending_next_request = None
                        # Pair any unexecuted call with a stub output so every
                        # function_call in the *request* has a matching output.
                        # Stubs are NOT persisted to state.
                        executed_ids = {get_field(o, "callId") for o in outputs}
                        stubs = [
                            {
                                "type": "function_call_output",
                                "id": f"output_{call.id}",
                                "callId": call.id,
                                "output": "Tool execution skipped: step limit reached.",
                            }
                            for call in calls
                            if call.id not in executed_ids
                        ]
                        current_request = self._build_final_request(
                            current_request, response, [*outputs, *stubs], allow_final_response
                        )
                        pending_final_directive = True
                        continue
                    break

                if await self._handle_approval_check(calls, turn_number + 1, response, tools):
                    return final_response

                current_request = await self._resolve_request_for_turn(current_request, turn_number + 1)
                outputs, paused, executed_calls = await self._execute_round(calls, tools, round_context, step)
                self._record_executed_round(calls, response, outputs, tools)

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

                # A doom stop armed during this round halts before the next
                # model request. Seal defensively: the condemned call already
                # has a blocked output, but other calls may be unresolved.
                if self._doom_loop_stop is not None:
                    if outputs:
                        self._all_tool_rounds += 1
                    await self._seal_doom_loop_stop(response, resolved_ids)
                    session_end_reason = "doom_loop"
                    break

                if not outputs:
                    break
                self._all_tool_rounds += 1
                force_resume_count = 0
                # Inject queued doom-loop steer guidance so the very next
                # request carries it. It lands before this round's call/output
                # items (the injection point predates the follow-up
                # accumulation) -- matching upstream.
                await self._flush_doom_loop_steer(current_request)
                await self._apply_next_turn_params(executed_calls, tools, current_request)
                current_request = self._pending_next_request or current_request
                self._pending_next_request = None
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
            await self._mark_state_complete()
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

    def _record_executed_round(
        self, calls: Sequence[ParsedToolCall], response: Any, outputs: Sequence[Any], tools: Sequence[Tool]
    ) -> None:
        """Record an executed tool round as a stop-condition step. `tool_results`
        is client-tool-centric; server-tool output items are surfaced on
        `server_tool_results` so conditions can react to either class."""
        tool_results: List[Dict[str, Any]] = []
        for output in outputs:
            call_id = get_field(output, "callId")
            name = next((c.name for c in calls if c.id == call_id), "")
            matched = self._find_tool(name, tools)
            raw = get_field(output, "output")
            tool_results.append(
                {
                    "tool_call_id": call_id,
                    "tool_name": name,
                    "source": "mcp" if matched is not None and is_mcp_tool(matched) else "client",
                    "result": json_loads_maybe(raw) if isinstance(raw, str) else raw,
                }
            )
        server_items = [
            item
            for item in self._response_output_items(response)
            if get_field(item, "type") not in (None, "message", "reasoning", "function_call")
        ]
        self._executed_rounds.append(
            StepResult(
                step_type="continue",
                text=extract_text_from_response(response),
                tool_calls=list(calls),
                tool_results=tool_results,
                response=response,
                server_tool_results=server_items,
                usage=get_field(response, "usage"),
            )
        )

    async def _apply_next_turn_params(
        self, executed_calls: Sequence[ParsedToolCall], tools: Sequence[Tool], current_request: Dict[str, Any]
    ) -> None:
        """Apply `next_turn_params` from executed tools. A tool-computed
        `tool_choice` becomes the new caller-level policy, not a one-turn
        override, so it re-runs the resolved-policy bookkeeping."""
        self._pending_next_request = None
        next_params = await execute_next_turn_params_functions(executed_calls, tools, current_request)
        if not next_params:
            return
        next_request = apply_next_turn_params_to_request(current_request, next_params)
        if "tool_choice" in next_params:
            next_request = self._apply_resolved_forced_tool_choice_policy(next_request)
        self._pending_next_request = next_request

    def _call_is_auto_resolvable(self, call: ParsedToolCall, tools: Sequence[Tool]) -> bool:
        tool = self._find_tool(call.name, tools)
        return bool(tool and is_auto_resolvable_tool(tool))

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
        followup = {
            **dict(request),
            "input": [*normalize_input_to_array(request.get("input")), *output_items, *outputs],
            "stream": True,
        }
        # A forced tool choice has served its purpose once a tool round has
        # executed; keeping it would forbid the model from ever answering in
        # text (DEV-785). Relax it so the model can call again or answer.
        return self._apply_forced_tool_choice_policy(followup)

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
        return await self._send_and_track(new_request, "retry", turn_number)

    # -- public streaming/result surface ------------------------------------

    async def get_response(self) -> Any:
        """The final round's response. Raises an explicit pause error when a
        resume left the conversation paused without producing a response."""
        await self._ensure_task()
        if self._final_response is None:
            raise self._response_unavailable_error()
        return self._final_response

    async def get_doom_loop_verdict(self) -> Optional[DoomLoopVerdict]:
        """The doom-loop verdict that stopped this run, or None when the run
        was not stopped by doom-loop detection (including when detection is
        off). Resolves after the tool loop settles. For per-event observation
        use the `DoomLoopDetected` hook."""
        await self._ensure_task()
        return self._doom_loop_stop

    async def get_usage(self) -> Dict[str, Any]:
        """Aggregate token/cost usage across every model call this run made
        (initial request, each tool-round follow-up, the empty-final retry,
        the `allow_final_response` final turn, and approval-resume requests).

        Same shape as the `SessionEnd` hook's `total_usage`. Never raises: a
        failed run still consumed tokens, so the totals accrued so far are
        returned. `cost` is present only when the server reported it.
        """
        try:
            await self._ensure_task()
        except Exception as error:  # noqa: BLE001 - documented never-raises accessor
            warnings.warn(f"[get_usage] run failed; reporting totals accrued so far: {error}", stacklevel=2)
        return self._snapshot_session_usage()

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
                    "toolName": get_field(event, "toolName"),
                    "tool_name": get_field(event, "tool_name", get_field(event, "toolName")),
                    "result": get_field(event, "result"),
                }
            elif typ == "tool.result":
                yield {
                    "type": "tool_result",
                    "toolCallId": get_field(event, "toolCallId"),
                    "tool_call_id": get_field(event, "tool_call_id", get_field(event, "toolCallId")),
                    "toolName": get_field(event, "toolName"),
                    "tool_name": get_field(event, "tool_name", get_field(event, "toolName")),
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
                    call_ids[item_id] = str(get_field(item, "callId", item_id))
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

    def cancel(self) -> None:
        if self._run_task is not None:
            self._run_task.cancel()
