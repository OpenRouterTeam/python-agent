from __future__ import annotations

import asyncio
import dataclasses
import json
import time
import warnings
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Sequence, Tuple

from ._utils import dump, get_field, is_async_iterable, json_dumps, maybe_await, sdk_request_kwargs
from .async_params import resolve_async_functions
from .conversation_state import (
    append_to_messages,
    create_initial_state,
    create_rejected_result,
    create_unsent_result,
    generate_conversation_id,
    partition_tool_calls,
    unsent_results_to_api_format,
    update_state,
)
from .next_turn_params import apply_next_turn_params_to_request, execute_next_turn_params_functions
from .stop_conditions import is_stop_condition_met
from .stream_transformers import extract_text_from_response, extract_tool_calls_from_response
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


class ModelResult:
    def __init__(self, options: Mapping[str, Any]) -> None:
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
            async for event in value:
                await self._append_event(event)
                typ = get_field(event, "type")
                if typ in {"response.completed", "response.incomplete"}:
                    completed = get_field(event, "response")
                if typ == "response.failed":
                    raise RuntimeError(str(get_field(event, "message", "Response failed")))
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
        started = time.monotonic()
        response = await self._coerce_response(await self._send(request))
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
        self._resuming_from_client_tools = bool(loaded is not None and loaded.status == "awaiting_client_tools")
        self._state = loaded or create_initial_state()
        if not self._resuming_from_client_tools:
            self._state = update_state(self._state, {"status": "in_progress"})
            await maybe_await(accessor.save(self._state))

    async def _save_state(self, **updates: Any) -> None:
        accessor = self.options.get("state")
        if accessor is None or self._state is None:
            return
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
        if not response_id or output is None:
            raise ValueError("Invalid final response: missing required fields")
        if not isinstance(output, list) or len(output) == 0:
            if allow_empty_output:
                return
            raise ValueError("Invalid final response: empty or invalid output")

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
            total_usage = {
                "model_calls": self._session_usage["model_calls"],
                "input_tokens": self._session_usage["input_tokens"],
                "output_tokens": self._session_usage["output_tokens"],
                "total_tokens": self._session_usage["total_tokens"],
                "cached_tokens": self._session_usage["cached_tokens"],
                "reasoning_tokens": self._session_usage["reasoning_tokens"],
            }
            if self._session_usage["has_cost"]:
                total_usage["cost"] = self._session_usage["cost"]
            payload["total_usage"] = total_usage
        try:
            await self._hooks.emit("SessionEnd", payload, session_id=self._session_id)
        except Exception as error:  # noqa: BLE001 - teardown must never mask the real error
            warnings.warn(f"[SessionEnd] error during session teardown: {error}", stacklevel=2)

    async def _emit_post_model_call(self, response: Any, started_at: float, turn_type: str, turn_number: int) -> None:
        if not self._hooks:
            return
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
        if self._hooks:
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
        await self._append_event(
            {
                "type": "tool.preliminary_result",
                "toolCallId": call_id,
                "tool_call_id": call_id,
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

    # -- the main loop -----------------------------------------------------

    async def _run(self) -> Any:
        session_end_reason = "complete"
        hooks = self._hooks
        try:
            await self._load_state()
            self._session_id = self._state.id if self._state is not None else generate_conversation_id()

            tools: Sequence[Tool] = self.options.get("tools") or []
            await self._emit_session_start(tools)

            request = await resolve_async_functions(self.options["request"], {"number_of_turns": 0})
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
                if not (
                    self._state.pending_tool_calls
                    and (self.options.get("approve_tool_calls") or self.options.get("reject_tool_calls"))
                ):
                    self._fresh_items_for_state = list(fresh_input)
            elif fresh_input:
                request["input"] = fresh_input

            current_request = request
            is_resume_turn = self._resuming_from_client_tools
            if (
                self._state is not None
                and self._state.pending_tool_calls
                and (self.options.get("approve_tool_calls") or self.options.get("reject_tool_calls"))
            ):
                current_request = await self._build_resume_request(request)
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

                response = await self._send_and_track(current_request, turn_type, turn_number)
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
                    if final_response_enabled and resolvable_pending:
                        final_outputs: List[Any] = []
                        turn_context = {"number_of_turns": turn_number + 1, "turn_request": current_request}
                        for call in calls:
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
                        current_request = self._build_final_request(
                            current_request, response, final_outputs, allow_final_response
                        )
                        pending_final_directive = True
                        continue
                    break

                partition = await partition_tool_calls(
                    calls, tools, {"number_of_turns": turn_number + 1}, self.options.get("require_approval")
                )
                requires_approval = list(partition["requires_approval"])
                hook_resolved_unsent: List[UnsentToolResult] = []
                if requires_approval and hooks:
                    still_pending: List[ParsedToolCall] = []
                    for call in requires_approval:
                        decision, reason = await self._emit_permission_request(call, tools)
                        if decision == "allow":
                            promo_tool = self._find_tool(call.name, tools)
                            if promo_tool and is_auto_resolvable_tool(promo_tool):
                                outcome = await self._run_tool_with_hooks(
                                    promo_tool,
                                    call,
                                    {
                                        "number_of_turns": turn_number + 1,
                                        "tool_call": call,
                                        "turn_request": current_request,
                                    },
                                )
                                if outcome["type"] == "parse_error":
                                    hook_resolved_unsent.append(
                                        create_rejected_result(call.id, call.name, outcome["error_message"])
                                    )
                                elif outcome["type"] == "hook_blocked":
                                    hook_resolved_unsent.append(
                                        create_rejected_result(call.id, call.name, outcome["reason"])
                                    )
                                elif outcome["result"] is None:
                                    still_pending.append(call)
                                elif outcome["result"].get("error") is not None:
                                    hook_resolved_unsent.append(
                                        create_rejected_result(call.id, call.name, str(outcome["result"]["error"]))
                                    )
                                else:
                                    hook_resolved_unsent.append(
                                        create_unsent_result(call.id, call.name, outcome["result"].get("result"))
                                    )
                            else:
                                still_pending.append(call)
                        elif decision == "deny":
                            hook_resolved_unsent.append(
                                create_rejected_result(call.id, call.name, reason or "Denied by PermissionRequest hook")
                            )
                        else:
                            still_pending.append(call)
                    requires_approval = still_pending

                if requires_approval:
                    if self.options.get("state") is None:
                        names = ", ".join(call.name for call in requires_approval)
                        raise ValueError(f"Tool(s) require approval but no state accessor is configured: {names}")
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
                for call in partition["auto_execute"]:
                    tool = self._find_tool(call.name, tools)
                    if not tool or not is_auto_resolvable_tool(tool):
                        continue
                    turn_context = {
                        "number_of_turns": turn_number + 1,
                        "tool_call": call,
                        "turn_request": current_request,
                    }
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
                            "preliminaryResults": result.get("preliminary_results"),
                            "preliminary_results": result.get("preliminary_results"),
                            "timestamp": _now_ms(),
                        }
                    )
                    output = await self._tool_result_to_output(effective_call, tool, result)
                    outputs.append(output)
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

    async def _build_resume_request(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        self._fresh_items_for_state = []
        approved = set(self.options.get("approve_tool_calls") or [])
        rejected = set(self.options.get("reject_tool_calls") or [])
        outputs: List[Any] = []
        if self._state is not None and self._state.unsent_tool_results:
            outputs.extend(unsent_results_to_api_format(self._state.unsent_tool_results))
        pending: List[ParsedToolCall] = self._state.pending_tool_calls if self._state is not None else []
        for call in pending or []:
            tool = self._find_tool(call.name, self.options.get("tools") or [])
            if call.id in rejected:
                outputs.append(self._rejected_output(call.id, "Tool call rejected by user"))
            elif call.id in approved and tool and is_auto_resolvable_tool(tool):
                turn_context = {"number_of_turns": 0}
                outcome = await self._run_tool_with_hooks(tool, call, turn_context)
                if outcome["type"] == "parse_error":
                    outputs.append(self._rejected_output(call.id, outcome["error_message"]))
                elif outcome["type"] == "hook_blocked":
                    outputs.append(self._rejected_output(call.id, outcome["reason"]))
                elif outcome["result"] is not None:
                    outputs.append(await self._tool_result_to_output(outcome["call"], tool, outcome["result"]))
        updated_messages = append_to_messages(self._state.messages, outputs) if self._state is not None else outputs
        await self._save_state(
            messages=updated_messages,
            pending_tool_calls=None,
            unsent_tool_results=None,
            status="in_progress",
        )
        return {**dict(request), "input": updated_messages, "stream": True}

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
        return await self._send_and_track(new_request, "retry", turn_number)

    # -- public streaming/result surface ------------------------------------

    async def get_response(self) -> Any:
        return await self._ensure_task()

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
                    "preliminaryResults": get_field(event, "preliminaryResults"),
                    "preliminary_results": get_field(event, "preliminary_results"),
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
        await self.get_response()
        state = self._state
        return bool(
            state
            and (
                state.status in {"awaiting_approval", "awaiting_hitl", "awaiting_client_tools"}
                or state.pending_tool_calls
            )
        )

    async def get_pending_tool_calls(self) -> List[ParsedToolCall]:
        await self.get_response()
        return list(self._state.pending_tool_calls or []) if self._state else []

    async def get_state(self) -> Any:
        await self.get_response()
        return self._state

    def cancel(self) -> None:
        if self._run_task is not None:
            self._run_task.cancel()
