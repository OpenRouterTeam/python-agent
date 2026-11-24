# openrouter-agent-sdk Changelog

## Unreleased — partial backport of later `@openrouter/agent` fixes

Not released. The package version stays `0.8.2`. This adds selected fixes and small features from upstream `@openrouter/agent` 0.9.0–1.3.0 on top of the 0.8.0 port. Upstream's larger 0.9.0+ features are **not** included yet: async/background/deferred tools, the `task` tool, steering and subagent tools, doom-loop detection, run-level cancellation and tool timeouts, tool sets, stream replay compaction, and the MCP integration.

### Approval and resume

- **Behavior change:** `approve_tool_calls` / `reject_tool_calls` drop duplicate IDs, so an approved call runs only once. An ID that appears in both lists raises `ValueError` before anything runs.
- **Behavior change:** resuming an `awaiting_approval` conversation without decisions now leaves it paused. Previously it sent a new model request. No request is sent and the supplied `input` is not saved. The same applies when decisions cover only some of the pending calls: the undecided calls stay pending. `awaiting_hitl` still resumes when `input` carries a `function_call_output` for a pending call.
- While a conversation stays paused, `get_response()`, `get_text()`, `get_tool_calls()` and the stream getters raise an error that names the pause and lists up to 10 pending call IDs. `requires_approval()`, `get_pending_tool_calls()` and `get_state()` still work.
- **Behavior change:** an approved HITL call whose `on_tool_called` returns `None` again stays pending, with status `awaiting_hitl`. Previously it was dropped.
- **Behavior change:** approving a call that no longer has an executable tool now records a "Tool not found or not executable" error output. Previously the call was dropped.
- When a `PreToolUse` hook rewrites an approved call's arguments on resume:
  - Input-dependent approval checks run again on the new arguments. These are a callable `require_approval`, either on the tool or on the request. A plain `require_approval=True` does not ask again.
  - Rewritten arguments that fail `input_schema` are blocked and reported through `PostToolUseFailure`.
- **Behavior change:** the rejection message is now `"Rejected by user"`, matching upstream.
- With `allow_final_response=False` and a `stop_when` already met on the first response, pausing for an approval-gated call no longer runs that call's auto-approved siblings. They are saved as skipped results.

### Tool loop

- A response cut off at `max_output_tokens` no longer runs any of its tool calls, including the ones that completed before the cut-off. The run ends with that `incomplete` response.
- **Behavior change:** a forced `tool_choice` is relaxed to `"auto"` once it has produced a tool call. This covers `"required"`, a specific tool, or `allowed_tools` with `mode: "required"`. The relaxed state survives approval and client-tool resumes: it is stored in the new `ConversationState.consumed_forced_tool_choice_key` field, which is additive within state version 1. It resets when the run completes.
- `next_turn_params` can now set `tool_choice`, and its context always includes the current `tool_choice`.
- **Behavior change:** `next_turn_params` now runs only for calls whose tool ran and returned without error, and receives the input the tool actually ran on.
- `to_model_output` now receives the parsed input, the same value `execute` gets. It also runs for results produced during approval resumes and for results saved before an approval pause.
- Tool execute contexts now always carry `tool_call` as a wire-shaped `function_call` item. Before this, the main loop passed a `ParsedToolCall` there, and the resume and final-turn paths passed nothing.
- **Behavior change:** generator tools no longer collect `preliminary_results` on the result or the `tool.result` event. Yields are still streamed live as `tool.preliminary_result` events.
- Final-response validation errors now say what is wrong (empty output array with the response ID, wrong output type, or which required fields are missing). The original message prefixes are unchanged.

### New API

- `ModelResult.get_usage()` returns token and cost totals across every model call in the run. It never raises; a failed or paused run returns what it accrued so far.
- `call_model(..., active_tools=[...])` sends and runs only the named client tools; server tools are always kept. An empty tool list leaves `tools` out of the request.
- `tool(..., strict=True | False)` is sent on the wire tool definition. When `strict` is not set, the tool definition now carries `"strict": null`.
- `tool(..., wire_input_schema={...})` lets a manual tool send its own JSON Schema to the model.
- `mark_mcp(tool, annotations=..., wire_input_schema=...)` records an MCP server's tool annotations and sends the server's own object schema to the model.
- `from_chat_messages` now turns assistant `tool_calls` into `function_call` items instead of dropping them.

### Context and hooks

- **Behavior change:** `set_context` / `set_shared_context` now store the schema's parsed value for each field they update. The merged context is validated first, so model-level validators apply. An invalid update raises and leaves the stored context unchanged.
- Blocking hooks fail closed: a `PreToolUse` or `UserPromptSubmit` chain that is aborted before every handler has run now counts as blocked.

### Compatibility notes

- `call_model` raises `NotImplementedError` for upstream options this port does not support yet (`doom_loop`, `signal`, `tool_timeout_ms`, `tool_concurrency`, `async_tools`, `stream_replay`). Previously they were forwarded to the API.
- `tool()` raises `ValueError` if `wire_input_schema` is passed to a tool that is not manual. Upstream enforces this only in its type definitions.
- `get_usage()` reports a failed run with a Python warning (upstream logs to the console). On an approval-resumed run it waits for the whole resumed loop. Upstream counts only the resume request there.
- Function-valued request parameters (for example a callable `tool_choice`) are still resolved once per run, not per turn. A dynamic forced `tool_choice` therefore does not re-arm between turns as it does upstream.
- `PreToolUse` still runs before the approval check on a fresh run. Upstream now asks for approval on the model's original arguments and runs `PreToolUse` afterwards.

## 0.8.2 — stop-condition approval fix

Downstream-only fix on top of the 0.8.0 port; no new upstream work.

- **Security:** tool calls that run when a `stop_when` condition fires (final-response handling) now go through the same approval check as a normal turn. Calls that require approval, including tools with `require_approval=True`, pause as `awaiting_approval` instead of executing. Before this, they ran with no approval decision.
- `call_model` raises when two client tools share a name (shipped in 0.8.1 but missing from its notes).

## 0.8.1 — approval fix

Downstream-only fix on top of the 0.8.0 port; no new upstream work.

- **Security:** callable `require_approval` predicates and call-level approval checks now see the arguments after `input_schema` parsing (coercions and defaults applied), the same values `execute` receives. Arguments that fail parsing require approval. Before this, a value such as the string `"true"` could pass a strict predicate and then run as boolean `True`.

## 0.8.0 — first PyPI release

First published release. The version is `0.8.0`, not `0.0.1`, because this
package's version tracks the ported `@openrouter/agent` version — so a release
states plainly which upstream version it is a port of. The library surface is the
0.8.0 port described under *Packaged surface* below.

### Packaging

- **The PyPI distribution is `openrouter-agent-sdk`.** The import is unchanged (`from openrouter_agent import ...`). `openrouter-agent` on PyPI belongs to an unrelated third-party project, so it was never available to this package; the rename is what makes publishing possible. A distribution name differing from the import name is normal (`scikit-learn`/`sklearn`).
- **Requires Python 3.10+**, and depends on **`openrouter>=1.1,<2`**. These move together: every `openrouter` 1.x release requires Python 3.10 or newer — the SDK dropped 3.9 at 1.0.0. Python 3.9 reached EOL in October 2025, and as the first published release there is no earlier version to fall back to on 3.9.
- Added repository/issues/changelog URLs and trove classifiers to the package metadata, and stopped shipping the port tooling (`.upstreamer/`, `.github/`, `scripts/upstream`) inside the sdist.

### Packaged surface (the 0.8.0 sync)

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
