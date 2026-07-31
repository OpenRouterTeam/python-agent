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

- Target: `openrouter>=1.1,<2` (current `pyproject.toml` pin).
- `call_model` sends through `client.beta.responses.send_async` — the Responses
  API, matching upstream's Responses path. Do not switch to Chat Completions.

The pin was moved from `>=0.10.2` to `>=1.1,<2` deliberately, in the PR that set
this package up for PyPI publishing — not by a sync run. Two consequences worth
knowing before touching it again:

- **It forced the Python floor to 3.10.** Every `openrouter` 1.x release requires
  Python `>=3.10` (the SDK dropped 3.9 at 1.0.0), so `requires-python` is now
  `>=3.10`. Python 3.9 reached EOL in October 2025.
- **The upper bound is load-bearing.** A 2.x could move the Responses API surface
  this port binds to. `<2` means a new major cannot silently break installs.

Do **not** bump the `openrouter` dependency on your own initiative. Upstream
tracks `@openrouter/sdk`; the Python generated SDK moves independently. Crossing
a major boundary is a breaking change that needs its own PR, with the test suite
and `mypy` verified against the new major first. If an upstream change *requires*
a newer `openrouter`, stop and report it as a blocker rather than bumping.

## Package Identity

Fixed. A sync must not change any of these:

| | Value | Why |
|---|---|---|
| PyPI distribution name | `openrouter-agent-sdk` | `openrouter-agent` is taken on PyPI by an unrelated third-party project. Do not "correct" the name to match upstream's `@openrouter/agent`. |
| Import name | `openrouter_agent` | The import path is unaffected by the distribution name and stays aligned with upstream. A PyPI name differing from the import name is normal (`scikit-learn`/`sklearn`). |

`[project.urls]` and `classifiers` are repo-owned publishing metadata, not port
output. Leave them alone.

## Package Version

Two independent numbers. Do not collapse them.

| Where | What | Who changes it |
|---|---|---|
| `.upstreamer/state.yaml` `upstream_agent_version` | The `@openrouter/agent` version actually ported. Read from upstream `packages/agent/package.json` at the target commit. | **A sync must update this alongside `upstream_commit`.** The verifier checks it against upstream. |
| `pyproject.toml` `version` | The PyPI distribution version of `openrouter-agent-sdk`. | Humans, at release time. A sync must **not** touch it. |

They were the same field until the package was prepared for PyPI. They had to
split: `openrouter-agent-sdk` is a brand-new project on PyPI and starts at
`0.0.1`, so tying the distribution version to upstream's `0.8.0` would advertise a
release history that does not exist — and would burn every version number between.

Version honesty still has teeth. The verifier compares
`upstream_agent_version` against upstream's `package.json` and **fails** if it
drifts or is missing, so a port cannot record a version it did not achieve. If the
target commit is between releases, keep the last released version there and note
the drift in the final report.

Publishing is gated on the distribution version: a released version can never be
reused on PyPI, even after a yank.

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

## Test Parity

Upstream's test suite is the most precise statement of the behavior contract that
exists. Porting the source without porting the tests produces a package that
compiles, passes its own assertions, and is a version behind on real behavior —
the exact failure this pipeline exists to prevent. So tests are in scope, not a
nice-to-have.

**Upstream tests are part of the port.**

- Maintain a 1:1 file mapping: upstream `tests/unit/foo-bar.test.ts` →
  `tests/unit/test_foo_bar.py`. When a sync touches an upstream test file, port
  the corresponding cases in the same run.
- When upstream **adds** a test file, port it. When upstream **changes**
  assertions in one, update the Python counterpart to match. An upstream test
  file with no Python counterpart is a parity gap — report it explicitly in the
  final report, with the invariant it protects and a severity assessment.
- A behavioral change ported without a test asserting it is incomplete work.

**Assert upstream behavior, not the port's own shape.** These are the recurring
ways a test looks like coverage without being coverage. All are rejectable:

- Membership-only assertions on event streams. Assert **order and count**:
  `types.count("turn.end") == 1` and `types.index(...) < types.index(...)`, not
  `"turn.end" in types`. A membership check passes when an event fires twice,
  fires out of order, or carries the wrong payload.
- Vacuous conditional asserts. `assert x is None if "x" in d else True` is
  `assert True` on the missing branch.
- `assert x is not None` as a test's *only* assertion. As a mypy-narrowing line
  before a real assertion it is fine; alone it asserts almost nothing.
- `assert len(xs) > 0` where the invariant is *which* items are present.
- Re-asserting a stub's own canned data, or that a type guard returns True for an
  object the test built with that guard's marker.
- Asserting an execution *happened* when the invariant is that it happened
  **exactly once**. Double-execution of a side-effecting tool is a real upstream
  regression class; only a count catches it.

**Use the shared fixtures.** `tests/_fixtures.py` provides `make_response`,
`function_call_item`, `text_response`, `tool_call_response`, `QueuedClient`, and
`MemoryStateAccessor`. Do not hand-roll new fake clients or partial response
dicts: `make_response` populates every field the real Responses API returns, so a
stub cannot be more forgiving than production. If a test needs bespoke transport
behavior (error injection, streaming), build its payloads with these builders.

**Coverage may not decay.** `--cov-fail-under` in `.github/workflows/ci.yaml` is a
ratchet. A port run may raise it, never lower it. If new ported source drops
coverage below the floor, the missing tests are part of the port — write them.

**Comment deliberate divergences at the assertion.** Where the port must assert
something different from upstream, say why with a source reference. Known case:
outgoing `function_call_output` items use snake_case `call_id`, not upstream's
`callId`, because `ModelResult._send` normalizes at the transport boundary
(`model_result.py:148-156`). Without the comment, a later reader "fixes" it back
and breaks the test.

**Keep tests deterministic.** Never port a timing race as `asyncio.sleep`. Gate on
`asyncio.Event` so ordering is explicit — see
`tests/unit/test_turn_end_race_condition.py`. Construct `asyncio` primitives
inside the async test body: `asyncio.Condition()` binds the running loop eagerly
on 3.9 and lazily on 3.13, so module-scope construction breaks on 3.9 only.

**Prefer the public API over private internals.** Upstream tests sometimes cast to
an internal type and call a private method. Where this port's internals differ
(e.g. it has no `_execute_tools_if_needed` — the loop is inlined in
`ModelResult._run`), drive the same invariant through `call_model` instead. The
test then survives the next sync's refactors.

## Verification

`.upstreamer/scripts/verify.sh` must pass: `ruff`, `mypy` (over `src` **and**
`tests`), `pytest`, the coverage floor, plus the required-public-API presence
check and no-TS-artifact check.

Then `.upstreamer/eval.md` must return PASS or PASS WITH WARNINGS from a fresh
review context before state advances.

## Final Report

Include: upstream delta, modules and public API touched, tests added, any new
naming mappings derived, any parity gap deliberately left open with reasoning,
verifier result, eval result and report path, and whether the `openrouter`
substrate pin blocked anything.
