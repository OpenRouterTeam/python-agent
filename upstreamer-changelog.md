# openrouter-agent-sdk Changelog

## Unreleased (partial sync toward `@openrouter/agent` 1.2.0, not yet publishable)

This is partial work toward the upstream 0.9.0–1.2.0 releases. The package version stays at `0.8.0` because the upstream async-tool system (background/deferred tools, the `task` tool, steering, subagent tools, per-tool cancellation/timeouts, tool concurrency) has **not** been ported yet. Do not publish this state.

### New

- **Doom-loop detection** (opt-in `doom_loop=True` or a config dict). It flags repeated identical tool calls (including same-tool fan-outs and invalid-JSON repeats), repeated server-tool calls, and repeated text. It responds through a graduated ladder: `observe`, `steer`, `escalate`, `block`, `stop`.
  - The new `DoomLoopDetected` hook fires on each detection, and `SessionEnd` gets a `"doom_loop"` reason.
  - Tools declare call identity with `loop_key=`.
  - Detector state persists in `ConversationState.doom_loop`.
  - `ModelResult.get_doom_loop_verdict()` reports a stop.
  - The primitives (`DoomLoopMonitor`, `resolve_doom_loop_option`, `fingerprint_tool_call`, …) are exported. Fingerprints match the shared cross-port conformance vectors.
- **Tool sets:** `create_tool_set(...)` with `activate` / `deactivate` / `activate_when` / `deactivate_when`. There is also a new `active_tools=[...]` option on `call_model`, which filters the client tools sent to the model; server tools always pass. A filtered-to-empty tool list now omits `tools` from the request entirely.
- **`await result.get_usage()`:** aggregate token/cost totals across every model call in a run, with the same shape as `SessionEnd.total_usage`.
  - It never raises.
  - Usage now accrues even when no hooks are configured.
- **`next_turn_params` can set `tool_choice`**, e.g. to widen an `allowed_tools` set without changing `tools`.
  - Context functions now see upstream's defaults for absent fields.
  - Invalid keys are warned about and skipped.
  - Non-callable values are ignored.
  - Non-object call arguments raise a descriptive error.
- **New tool options:**
  - `tool(..., strict=True)` is serialized onto the wire definition.
  - `tool(..., wire_input_schema={...})` sends a caller-owned JSON Schema while `input_schema` keeps validating.
  - `server_tool(config, {"id": ...})` sets a stable id (default `server:<type>`).
  - `mark_mcp(tool, {"loop_key", "wire_input_schema"})`.
- **The user agent now includes the package version:** `openrouter-agent-sdk/<version>`.
- **`ReusableReadableStream` and `ToolEventBroadcaster`** support replay compaction (`stream_replay="active-consumers"`) and release their backlog when consumers leave or the stream is cancelled.
- **`from_chat_messages`** now emits `function_call` items for assistant tool calls, which were previously dropped.

### Behavior changes

- **Stop conditions now count executed tool rounds, as upstream does,** not every model response. `step_count_is(n)` therefore allows one more tool round than before. Use `step_count_is(0)` to stop on the first response's tool calls.
- **A forced `tool_choice` relaxes to `auto` once it has produced a tool call.** This covers `"required"`, a specific tool, and `allowed_tools` with `mode: "required"`. The relaxation persists across approval/HITL/client-tool resumes, so the model can answer in text instead of being forced to call tools until the step budget runs out. Callable `tool_choice` (and other callable request params) are now re-resolved every turn.
- **Approval hardening:**
  - **Duplicate IDs:** a call ID listed twice in `approve_tool_calls` / `reject_tool_calls` no longer executes twice. Passing the same ID in both lists raises before anything runs.
  - **Resuming with no decision:** resuming an `awaiting_approval` conversation with no decision stays paused instead of sending a new request. The input passed with that call is not sent or saved. An `awaiting_hitl` conversation still resumes with a `function_call_output` for a paused call.
  - **Partial decisions:** the remaining calls stay pending.
  - **Getter errors while paused:** `get_response()`, `get_text()` and the stream getters raise an explicit "Conversation is awaiting approval ..." error that lists pending call IDs. `requires_approval()`, `get_pending_tool_calls()` and `get_state()` keep working.
  - **Auto-approved sibling calls:** when a turn pauses for approval, its auto-approved sibling calls now run before the pause and are replayed on resume. Previously they were left without outputs. If `stop_when` is already met and `allow_final_response=False`, they are recorded as skipped instead of executed.
  - **`allow_final_response` path:** this path now enforces the approval gate. Each call is gated at most once per run.
  - **`require_approval` predicates:** they receive schema-parsed arguments (defaults applied), and schema-invalid arguments fail closed.
  - **`PreToolUse` timing:** `PreToolUse` now runs before execution, as part of the approval gate. A mutated input is validated and re-gated, and a pending call records `pre_tool_use_applied=True` so the hook is not applied twice on resume.
  - **Rejection output:** a rejection now produces `{"error": "Rejected by user"}`.
- **Truncated responses:** a response truncated at `max_output_tokens` never executes its tool calls, including calls completed before the cut-off. The run finalizes with the incomplete response.
- **Generator tools** no longer accumulate a `preliminary_results` list on the result or on `tool.result` events. Yields are still delivered live.
- **Tool events:** `tool.result` and `tool.preliminary_result` events, and the `get_tool_stream()` projection, now carry `toolName` / `tool_name`.
- **Execute context:** the execute context's `tool_call` is now always populated with a wire-shaped `function_call` item, including on the streaming loop and on direct executor calls.
- **Streams:** a stream that never closes after its terminal `response.completed` event no longer hangs the run.
- **`is_claude_style_messages`** now requires a Claude-specific content block, as upstream does.
- **`is_stop_condition_met`** evaluates every condition and stops only on an exact `True`.
- **Final-response error messages:** `Invalid final response` errors now name the defect (empty array vs. wrong type, and which fields are missing). The existing message prefixes are unchanged.

### Compatibility notes

- **Predicate arguments:** for Pydantic `input_schema`s, approval predicates receive `model_dump()`ed dicts, because the predicate contract is a mapping of arguments.
- **`strict` and `description`:** when unset they are omitted from the wire definition, where upstream sends `null`.
- **MCP:** the MCP integration upstream moved into `@openrouter/agent/mcp` is not part of this package.

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
