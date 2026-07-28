# openrouter-agent Changelog

## 0.8.0 Sync

- **Lifecycle hooks system** (the headline feature of this release): `HooksManager`, `HookName`, `HookEntry`, and nine built-in lifecycle hooks — `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `UserPromptSubmit`, `Stop`, `PermissionRequest`, `SessionStart`, `SessionEnd`, and `PostModelCall`. Pass a `HooksManager` instance, or an inline `{hook_name: [HookEntry(...)]}` dict of built-in hooks, via `hooks=` on `call_model`.
  - `PreToolUse` can block a tool call or mutate its input before execution; `PermissionRequest` can pre-empt the approval gate with `allow`/`deny`/`ask_user`; `UserPromptSubmit` can reject or rewrite the user's prompt before the first request goes out; `Stop` can force the loop past a `stop_when` hit (as a zero-cost in-memory retry, no extra model request) and inject a follow-up prompt.
  - `SessionStart`/`SessionEnd` fire exactly once per run, paired even when a no-tools stream raises; `SessionEnd` carries aggregated token/cost usage totals across every model call in the run. `PostModelCall` fires once per model request with a `turn_type` label (`initial`, `resume`, `tool_round`, `final`, `retry`) and per-call usage.
  - A single `HooksManager` instance is safe to share across concurrent `call_model` runs — session identity is threaded per emit rather than stored as shared mutable state.
- **Versioned conversation-state serialization**: `serialize_conversation_state` / `deserialize_conversation_state` / `CONVERSATION_STATE_VERSION` for durable, cross-process state storage. Deserializing a version-less legacy blob normalizes to version 1; a future/unknown version raises `UnsupportedStateVersionError`, and malformed or missing-field JSON raises `InvalidStateError`, instead of a store silently misinterpreting an incompatible shape. The existing `StateAccessor` `load()`/`save()` contract is unchanged — these helpers are opt-in.
- **Unresolved manual tool calls now pause cleanly**: when the model calls a manual tool (`execute=False`, no `on_tool_called`) and there is no way to auto-resolve it, the run now stops with conversation status `"awaiting_client_tools"` and the unresolved calls available via `get_pending_tool_calls()` / `get_state()`, instead of silently dropping the call. This also applies to a mixed round of auto-executable and manual calls: the auto-executable outputs are still persisted before the pause.
- **`allow_final_response` is now on by default**: when a `stop_when` condition halts the loop mid-tool-call, `call_model` now makes one more turn with `tool_choice="none"` by default (tools stay in the request so the prompt-cache prefix survives) instead of requiring the caller to opt in. Bare `True` or omitting the option appends a new default directive (`DEFAULT_FINAL_RESPONSE_DIRECTIVE`) as a user message so models that emit tool-call syntax as text don't leak an unparsed call into the final answer; a non-empty string still overrides the wording, `""` appends nothing, and `False` disables the extra turn entirely.
- **Empty final responses after a completed tool round are now tolerated**: some models intermittently return an empty final turn right after a tool call was effectively the answer. That case is now retried once (forcing `tool_choice="none"`) and, if still empty, accepted rather than raised as an error. Pass `strict_final_response=True` to restore the old strict behavior. A run with no completed tool rounds still raises on an empty/invalid final response as before.
- **MCP-style tool result discrimination**: `mark_mcp()` / `is_mcp_tool()` let a tool be branded as originating from a remote MCP-style server without changing its execution or wire shape; tool results and `tool.result` stream events now carry a `source: "client" | "mcp"` field so consumers can tell precisely-typed local results apart from untyped remote ones.

## Previous Sync (0.7.2)

- Completed the Python port against the current `@openrouter/agent` 0.7.2 surface, keeping the package focused on the Responses API, tool orchestration, streaming, state, approval/HITL, tool context, stop conditions, and Claude/Chat compatibility.
- Strengthened stateful pause/resume behavior so approval and HITL pauses persist the model tool-call turn and resume with `function_call` before `function_call_output`, matching upstream multi-turn semantics.
- Expanded stream parity with turn boundary events, tool result and tool-call-output events, reusable consumers, and streamed function-call argument reconstruction in `ModelResult.get_tool_calls_stream()`.
- Improved generator-tool parity for live preliminary events and Python's final-yield convention, including broad `dict` event/output schema handling.
- Added broader behavioral tests for Responses API use, request options, `allow_final_response`, next-turn params, user-input persistence, hook firing, Claude unsupported-content carriers, image conversion, manual-tool filtering, and approval/HITL resume ordering.
- Compatibility notes: Python typing remains looser than TypeScript tuple/conditional inference by design; `extract_unsupported_content` and Claude reasoning summary conversion are usable but should be tightened in a future parity pass.

## 0.7.2 Python Port

- Refreshed the port against the current `@openrouter/agent` 0.7.2 package snapshot, including the load-bearing tool loop, streaming, HITL, state, context, stop-condition, and Claude/Chat compatibility behavior documented upstream.
- Added the first standalone Python package for the OpenRouter Agent toolkit, aligned with the upstream `@openrouter/agent` 0.7.2 package surface.
- Ported the Responses API orchestration entry point as `call_model(...)` and the `OpenRouter.call_model(...)` convenience method, using `client.beta.responses.send_async` rather than Chat Completions.
- Added Python-native tool definitions for regular, generator, manual, HITL, and OpenRouter server tools, with Pydantic v2 schema validation and JSON Schema generation in place of Zod.
- Added `ModelResult` consumption methods for final text/response, text and reasoning streams, tool streams, tool-call streams, full response events with `turn.start`/`turn.end`, new message/items streams, context updates, pending approvals, and state inspection.
- Added conversation-state helpers, stop conditions, next-turn parameter helpers, SDK hook normalization, request-option passthrough, tool context, tool event broadcasting, reusable stream support, and Claude/OpenAI Chat compatibility helpers.
- Ported deterministic pytest coverage for tool factories, schema sanitization, tool execution, context, state, streaming broadcasters, message-format conversion, hook normalization, generator event streaming, stop-token accounting, and a mocked Responses API tool loop. Live e2e tests skip cleanly unless `OPENROUTER_API_KEY` is set.

### TypeScript to Python Name Mappings

- `callModel` -> `call_model`; `serverTool` -> `server_tool`.
- `getText`, `getResponse`, `getTextStream`, `getReasoningStream`, `getToolStream`, `getToolCallsStream`, `getToolCalls`, `getFullResponsesStream`, `getNewMessagesStream` -> snake_case `ModelResult` methods.
- `stepCountIs`, `hasToolCall`, `maxTokensUsed`, `maxCost`, `finishReasonIs` -> `step_count_is`, `has_tool_call`, `max_tokens_used`, `max_cost`, `finish_reason_is`.
- `fromClaudeMessages`, `toClaudeMessage`, `fromChatMessages`, `toChatMessage` -> `from_claude_messages`, `to_claude_message`, `from_chat_messages`, `to_chat_message`.
- `ToolContextStore`, `ToolEventBroadcaster`, `ToolType`, and `ModelResult` keep their class/enum names.

### Compatibility Notes

- Pydantic v2 replaces Zod. Tool inputs and outputs are validated at execution boundaries and schemas are sanitized before being sent to the Responses API.
- Python async generators cannot return a value the way JavaScript async generators can; generator tools validate yielded values against event/output schemas, emit preliminary events live, and treat the output-shaped final yield as the final result.
- Python static typing cannot exactly reproduce upstream TypeScript tuple/conditional inference for per-tool input/output/context narrowing. Runtime parity is prioritized; exported aliases and `py.typed` provide best-effort type support.
- SDK type re-exports now bind to the real generated Python SDK symbols that appear in the public API, so users can import request/response/item/event/hook types from `openrouter_agent` without object fallbacks.
- The current Python SDK has no `beforeCreateRequest` hook type matching the TypeScript SDK; `BeforeCreateRequestContext` and `BeforeCreateRequestHook` are exposed as structural Protocols until the generated SDK adds native symbols.
