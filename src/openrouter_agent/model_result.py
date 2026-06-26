from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Sequence

from ._utils import get_field, is_async_iterable, json_dumps, maybe_await, sdk_request_kwargs
from .async_params import resolve_async_functions
from .conversation_state import (
    append_to_messages,
    create_initial_state,
    partition_tool_calls,
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
    get_tool_function,
    is_auto_resolvable_tool,
    is_client_tool,
    is_manual_tool,
)
from .turn_context import normalize_input_to_array

GetResponseOptions = Dict[str, Any]


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
        if (self.options.get("approve_tool_calls") or self.options.get("reject_tool_calls")) and not self.options.get(
            "state"
        ):
            raise ValueError("approve_tool_calls and reject_tool_calls require a state accessor")

    async def _send(self, request: Mapping[str, Any]) -> Any:
        client = self.options["client"]
        kwargs = sdk_request_kwargs(request)
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

    async def _append_event(self, event: Any) -> None:
        async with self._condition:
            self._events.append(event)
            self._condition.notify_all()

    def _ensure_task(self) -> asyncio.Task[Any]:
        if self._run_task is None:
            self._run_task = asyncio.create_task(self._run())
        return self._run_task

    async def _load_state(self) -> None:
        accessor = self.options.get("state")
        if accessor is None:
            return
        loaded = await maybe_await(accessor.load())
        self._state = loaded or create_initial_state()
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
        await self._save_state(messages=messages, previous_response_id=get_field(response, "id"))

    async def _save_tool_outputs_to_state(self, outputs: Sequence[Any]) -> None:
        if self._state is None or not outputs:
            return
        await self._save_state(messages=append_to_messages(self._state.messages, list(outputs)))

    async def _run(self) -> Any:
        await self._load_state()
        tools: Sequence[Tool] = self.options.get("tools") or []
        request = await resolve_async_functions(self.options["request"], {"number_of_turns": 0})
        request["stream"] = True
        base_input = request.get("input")
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
        if (
            self._state is not None
            and self._state.pending_tool_calls
            and (self.options.get("approve_tool_calls") or self.options.get("reject_tool_calls"))
        ):
            current_request = await self._build_resume_request(request)
        final_response = None
        try:
            for turn_number in range(20):
                if self.options.get("on_turn_start"):
                    await maybe_await(self.options["on_turn_start"]({"number_of_turns": turn_number}))
                await self._append_event(
                    {
                        "type": "turn.start",
                        "turnNumber": turn_number,
                        "turn_number": turn_number,
                        "timestamp": int(time.time() * 1000),
                    }
                )
                response = await self._coerce_response(await self._send(current_request))
                await self._append_event(
                    {
                        "type": "turn.end",
                        "turnNumber": turn_number,
                        "turn_number": turn_number,
                        "timestamp": int(time.time() * 1000),
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
                if not calls:
                    await self._save_state(status="complete")
                    break
                stop_when = self.options.get("stop_when")
                stop_conditions = list(stop_when) if isinstance(stop_when, list) else ([stop_when] if stop_when else [])
                if stop_conditions and await is_stop_condition_met(stop_conditions, self._steps):
                    if self.options.get("allow_final_response"):
                        final_outputs = []
                        for call in calls:
                            matching_tool = next(
                                (
                                    candidate
                                    for candidate in tools
                                    if is_client_tool(candidate)
                                    and get_tool_function(candidate).get("name") == call.name
                                ),
                                None,
                            )
                            if matching_tool and is_auto_resolvable_tool(matching_tool):
                                result = await execute_tool(
                                    matching_tool,
                                    call,
                                    {
                                        "number_of_turns": turn_number + 1,
                                        "tool_call": call,
                                        "turn_request": current_request,
                                    },
                                    self._record_preliminary,
                                    self._context_store,
                                    self.options.get("shared_context_schema"),
                                )
                                if result is not None:
                                    output = await self._tool_result_to_output(call, matching_tool, result)
                                    final_outputs.append(output)
                                    self._tool_outputs.append(output)
                                    await self._append_event(
                                        {
                                            "type": "tool.call_output",
                                            "output": output,
                                            "timestamp": int(time.time() * 1000),
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
                            current_request, response, final_outputs, self.options.get("allow_final_response")
                        )
                        continue
                    break
                partition = await partition_tool_calls(
                    calls, tools, {"number_of_turns": turn_number + 1}, self.options.get("require_approval")
                )
                if partition["requires_approval"]:
                    if self.options.get("state") is None:
                        names = ", ".join(call.name for call in partition["requires_approval"])
                        raise ValueError(f"Tool(s) require approval but no state accessor is configured: {names}")
                    await self._save_state(
                        pending_tool_calls=partition["requires_approval"], status="awaiting_approval"
                    )
                    break
                outputs = []
                paused: List[ParsedToolCall] = []
                executed_calls: List[ParsedToolCall] = []
                for call in partition["auto_execute"]:
                    tool = next(
                        (
                            candidate
                            for candidate in tools
                            if is_client_tool(candidate) and get_tool_function(candidate).get("name") == call.name
                        ),
                        None,
                    )
                    if not tool or not is_auto_resolvable_tool(tool):
                        continue
                    result = await execute_tool(
                        tool,
                        call,
                        {"number_of_turns": turn_number + 1, "tool_call": call, "turn_request": current_request},
                        self._record_preliminary,
                        self._context_store,
                        self.options.get("shared_context_schema"),
                    )
                    if result is None:
                        paused.append(call)
                        continue
                    await self._append_event(
                        {
                            "type": "tool.result",
                            "toolCallId": call.id,
                            "tool_call_id": call.id,
                            "result": {"error": str(result["error"])}
                            if result.get("error") is not None
                            else result.get("result"),
                            "preliminaryResults": result.get("preliminary_results"),
                            "preliminary_results": result.get("preliminary_results"),
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    output = await self._tool_result_to_output(call, tool, result)
                    outputs.append(output)
                    executed_calls.append(call)
                    self._tool_outputs.append(output)
                    await self._append_event(
                        {"type": "tool.call_output", "output": output, "timestamp": int(time.time() * 1000)}
                    )
                    step.tool_results.append(result)
                if paused:
                    await self._save_tool_outputs_to_state(outputs)
                    await self._save_state(pending_tool_calls=paused, status="awaiting_hitl")
                    break
                if not outputs:
                    break
                await self._save_tool_outputs_to_state(outputs)
                next_params = await execute_next_turn_params_functions(executed_calls, tools, current_request)
                if next_params:
                    current_request = apply_next_turn_params_to_request(current_request, next_params)
                current_request = self._build_followup_request(current_request, response, outputs)
            else:
                raise RuntimeError("call_model exceeded the 20-turn safety limit")
            self._final_response = final_response
            return final_response
        finally:
            async with self._condition:
                self._run_done = True
                self._condition.notify_all()

    async def _record_preliminary(self, call_id: str, value: Any) -> None:
        await self._append_event(
            {
                "type": "tool.preliminary_result",
                "toolCallId": call_id,
                "tool_call_id": call_id,
                "result": value,
                "timestamp": int(time.time() * 1000),
            }
        )

    async def _build_resume_request(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        self._fresh_items_for_state = []
        approved = set(self.options.get("approve_tool_calls") or [])
        rejected = set(self.options.get("reject_tool_calls") or [])
        outputs: List[Any] = []
        if self._state is not None and self._state.unsent_tool_results:
            for item in self._state.unsent_tool_results:
                if item.error:
                    output = json_dumps({"error": item.error})
                else:
                    output = json_dumps(item.output)
                outputs.append(
                    {
                        "type": "function_call_output",
                        "id": f"output_{item.call_id}",
                        "callId": item.call_id,
                        "output": output,
                    }
                )
        pending: List[ParsedToolCall] = self._state.pending_tool_calls if self._state is not None else []
        for call in pending or []:
            tool = next(
                (
                    candidate
                    for candidate in self.options.get("tools") or []
                    if is_client_tool(candidate) and get_tool_function(candidate).get("name") == call.name
                ),
                None,
            )
            if call.id in rejected:
                outputs.append(
                    {
                        "type": "function_call_output",
                        "id": f"output_{call.id}",
                        "callId": call.id,
                        "output": json_dumps({"error": "Tool call rejected by user"}),
                    }
                )
            elif call.id in approved and tool and is_auto_resolvable_tool(tool):
                result = await execute_tool(
                    tool, call, {"number_of_turns": 0}, self._record_preliminary, self._context_store
                )
                if result is not None:
                    outputs.append(await self._tool_result_to_output(call, tool, result))
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
        self, request: Mapping[str, Any], response: Any, outputs: Sequence[Any], final: Any
    ) -> Dict[str, Any]:
        new_request = self._build_followup_request(request, response, outputs)
        new_request.pop("tools", None)
        new_request.pop("tool_choice", None)
        new_request.pop("parallel_tool_calls", None)
        if isinstance(final, str) and final:
            new_request["input"] = [
                *normalize_input_to_array(new_request.get("input")),
                {"role": "user", "content": final},
            ]
        return new_request

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
        return bool(state and (state.status in {"awaiting_approval", "awaiting_hitl"} or state.pending_tool_calls))

    async def get_pending_tool_calls(self) -> List[ParsedToolCall]:
        await self.get_response()
        return list(self._state.pending_tool_calls or []) if self._state else []

    async def get_state(self) -> Any:
        await self.get_response()
        return self._state

    def cancel(self) -> None:
        if self._run_task is not None:
            self._run_task.cancel()
