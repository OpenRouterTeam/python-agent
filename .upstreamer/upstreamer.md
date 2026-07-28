---
upstream: OpenRouterTeam/typescript-agent
downstream: OpenRouterTeam/python-agent
model: openrouter/~anthropic/claude-opus-latest
---

# Python Port Contract — `@openrouter/agent` → `openrouter-agent`

This repository is an async-first Python port of the OpenRouter TypeScript Agent
SDK (`@openrouter/agent`, in `packages/agent/` upstream). TypeScript is the
**reference spec**. Behavioral divergence is a bug unless it appears in the
Idiomatic Divergences section below.

## Scope

Port **only** `packages/agent/` from upstream.

Explicitly out of scope:

- `packages/mcp/` (`@openrouter/mcp`). Not ported. Do not begin porting it as a
  side effect of a sync. Adding it is a deliberate contract change.
- Upstream JS/TS infrastructure: `package.json`, `pnpm-lock.yaml`, `tsconfig*`,
  `turbo.json`, `biome.json`, vitest config, `.changeset/`, `.github/`, `esm/`
  build output, `node_modules/`.
- Upstream README/docs prose. This repo's `README.md` is its own product surface;
  update it when the public API changes, do not translate upstream's.

## Substrate Pin

The port sits on the generated `openrouter` Python SDK and must not reimplement
HTTP, auth, retries, or model schemas.

- Target: `openrouter>=0.10.2` (current `pyproject.toml` pin).
- `call_model` sends through `client.beta.responses.send_async` — the Responses
  API, matching upstream's Responses path. Do not switch to Chat Completions.

Do **not** bump the `openrouter` dependency on your own initiative. Upstream
tracks `@openrouter/sdk`; the Python generated SDK moves independently and is
currently at a much newer major. Crossing that boundary is a breaking change
that needs its own PR. If an upstream change *requires* a newer `openrouter`,
stop and report it as a blocker rather than bumping.

## Package Version

`pyproject.toml` `version` tracks the ported `@openrouter/agent` version. Read it
from the upstream `packages/agent/package.json` at the target commit and set it
to match. If the target commit is between releases, keep the last released
version and note the drift in the final report.

## Required Public API

Every symbol below must be importable from `openrouter_agent` and behaviorally
faithful. This list is the parity floor — the verifier enforces presence, the
eval enforces behavior.

Entry point and client:
- `call_model`, `OpenRouter` (with `call_model` convenience method)

Tools:
- `tool`, `server_tool` — regular, generator, manual, HITL, and server tool
  shapes; Pydantic v2 schema validation and JSON Schema generation; schema
  sanitization before the request

Result consumption (`ModelResult`):
- `get_text`, `get_response`, `get_text_stream`, `get_reasoning_stream`,
  `get_tool_stream`, `get_tool_calls_stream`, `get_tool_calls`,
  `get_full_responses_stream`, `get_new_messages_stream`
- Turn boundary events (`turn.start` / `turn.end`), context updates, pending
  approvals, state inspection

State:
- `create_initial_state`, `append_to_messages`, `update_state`,
  `partition_tool_calls`
- **Versioned serialization contract** (upstream #66): `serialize_conversation_state`,
  `deserialize_conversation_state`, `CONVERSATION_STATE_VERSION`, and the
  invalid/mismatched-state error types. Round-trip must be stable and version
  mismatch must raise rather than silently accept.

Stop conditions:
- `step_count_is`, `has_tool_call`, `max_tokens_used`, `max_cost`,
  `finish_reason_is`

Lifecycle hooks (upstream #7, #67 — the 0.8.0 headline):
- `HooksManager` and its options, hook name enum / `BUILT_IN_HOOKS`, hook
  definition/entry/handler/registry types, tool matchers, `PostModelCall`
  telemetry, `SessionEnd` usage totals
- Session-id threading must be per-emit so a shared manager is concurrency-safe
- `SessionEnd` / drain must be guaranteed on no-tools stream error paths

Final-turn control:
- `allow_final_response`, `DEFAULT_FINAL_RESPONSE_DIRECTIVE` (upstream #68 —
  bare `True` uses the default directive; a string appends that string)

Manual/HITL state:
- `awaiting_client_tools` persistence for unresolved manual tool calls
  (upstream #64)

Compatibility:
- `from_claude_messages`, `to_claude_message`, `from_chat_messages`,
  `to_chat_message`
- `extract_unsupported_content`, `has_unsupported_content`,
  `get_unsupported_content_summary`
- Claude content block / role enums, `is_claude_style_messages`

Support:
- Tool context (`ToolContextStore`, context builder), `ToolEventBroadcaster`,
  reusable streams, next-turn params, async param resolution, request options
  passthrough, turn context, item/stream type guards, `SDKHooks`

## Naming Map

TypeScript `camelCase` → Python `snake_case` for functions and methods. Classes
and enums keep their names (`ModelResult`, `ToolContextStore`,
`ToolEventBroadcaster`, `ToolType`, `HooksManager`, `HookName`).

Established mappings — do not re-derive:

| TypeScript | Python |
| --- | --- |
| `callModel` | `call_model` |
| `serverTool` | `server_tool` |
| `getText` / `getResponse` | `get_text` / `get_response` |
| `getTextStream` / `getReasoningStream` | `get_text_stream` / `get_reasoning_stream` |
| `getToolStream` / `getToolCallsStream` / `getToolCalls` | `get_tool_stream` / `get_tool_calls_stream` / `get_tool_calls` |
| `getFullResponsesStream` / `getNewMessagesStream` | `get_full_responses_stream` / `get_new_messages_stream` |
| `stepCountIs`, `hasToolCall`, `maxTokensUsed`, `maxCost`, `finishReasonIs` | `step_count_is`, `has_tool_call`, `max_tokens_used`, `max_cost`, `finish_reason_is` |
| `fromClaudeMessages` / `toClaudeMessage` | `from_claude_messages` / `to_claude_message` |
| `fromChatMessages` / `toChatMessage` | `from_chat_messages` / `to_chat_message` |
| `createInitialState` / `appendToMessages` / `updateState` | `create_initial_state` / `append_to_messages` / `update_state` |
| `partitionToolCalls` | `partition_tool_calls` |
| `serializeConversationState` / `deserializeConversationState` | `serialize_conversation_state` / `deserialize_conversation_state` |
| `CONVERSATION_STATE_VERSION` | `CONVERSATION_STATE_VERSION` |
| `DEFAULT_FINAL_RESPONSE_DIRECTIVE` | `DEFAULT_FINAL_RESPONSE_DIRECTIVE` |
| `allowFinalResponse` | `allow_final_response` |
| `onToolCalled` / `onResponseReceived` | `on_tool_called` / `on_response_received` |
| `approveToolCalls` / `rejectToolCalls` | `approve_tool_calls` / `reject_tool_calls` |
| `extractUnsupportedContent` | `extract_unsupported_content` |

If upstream adds a symbol with no mapping here, follow the convention and
**report the new mapping** so it can be added to this table.

## Idiomatic Divergences

Deliberate and permanent. Do not converge these toward TypeScript.

1. **Pydantic v2 replaces Zod.** Tool inputs/outputs validated at execution
   boundaries; schemas sanitized before hitting the Responses API.
2. **Async-first.** `async`/`await` throughout; `send_async` not `send`.
3. **Generator tools.** Python async generators cannot return a value the way JS
   async generators can. Generator tools validate yielded values against
   event/output schemas, emit preliminary events live, and treat the
   output-shaped final yield as the final result.
4. **Typing is looser.** Python cannot reproduce TypeScript's tuple/conditional
   inference for per-tool input/output/context narrowing. Runtime parity is
   prioritized; exported aliases plus `py.typed` give best-effort static support.
   Do not contort the runtime to chase a type-level feature.
5. **Cancellation** uses Python-native mechanisms, not `AbortSignal`.
6. **Structural Protocols** stand in where the generated Python SDK has no
   matching symbol (e.g. `BeforeCreateRequestContext` / `BeforeCreateRequestHook`
   until the generated SDK exposes them natively). Bind to real generated symbols
   as soon as they exist.

## Output Shape

This repo IS the downstream. Write in place:

```text
src/openrouter_agent/     # port lives here, one module per upstream lib module
tests/unit/               # deterministic tests
tests/e2e/                # live tests, must skip without OPENROUTER_API_KEY
pyproject.toml            # version tracks ported @openrouter/agent version
README.md                 # this package's own docs
upstreamer-changelog.md   # user-facing port notes
```

Module naming follows the existing layout: upstream `lib/tool-executor.ts` →
`src/openrouter_agent/tool_executor.py`. Keep that correspondence for new
modules so the port stays navigable against the reference.

## Verification

`.upstreamer/scripts/verify.sh` must pass: `ruff`, `mypy`, `pytest`, plus the
required-public-API presence check and no-TS-artifact check.

Then `.upstreamer/eval.md` must return PASS or PASS WITH WARNINGS from a fresh
review context before state advances.

## Final Report

Include: upstream delta, modules and public API touched, tests added, any new
naming mappings derived, any parity gap deliberately left open with reasoning,
verifier result, eval result and report path, and whether the `openrouter`
substrate pin blocked anything.
