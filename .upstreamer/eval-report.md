FAIL

# Port Parity Eval — Python (`@openrouter/agent` 0.8.0 → 1.2.0) — BANKRUPTCY

Target upstream commit `50a1ec2a16f5c163136f81b208511b1c01e87874` (level with the
`@openrouter/agent@1.2.0` tag); last ported `680bceb4598f228d3e2ec58e2416e4335cdff059`
(0.8.0). Incremental run, force 0.

**Outcome: bankruptcy declared. `.upstreamer/state.yaml` was NOT advanced.**
The mechanical verifier fails (version drift: `pyproject.toml=0.8.0` vs upstream
`1.2.0`) and two independent fresh-context evals returned `FAIL`. The next run
must retry this same delta.

## Why the port cannot pass within this run

The delta spans five upstream releases (0.9.0, 0.10.0, 0.11.0, 1.0.0/1.0.1, 1.1.0,
1.2.0; ~49k lines across `packages/agent`). Its 0.9.0 headline, upstream #90
(`e8d7d6d`): unified `run()` tools with `sync`/`background`/`deferred` lifecycles,
`awaiting_async_tools`, `resume_tool_results`, `AsyncToolRegistry`, the universal
`task` tool and check-ins, steering (`send_to_task`, `queue_user_message`),
`tool.agent()` subagents, per-tool cancellation/timeouts (`signal`,
`tool_timeout_ms`), and tool concurrency (`tool_concurrency`, `Semaphore`). None of it
is ported. It is several thousand lines of engine changes plus about 10 upstream
test files. That is not a focused fix, and the contract does not list it as a
divergence. As a result, no version is both honest and verifier-passing:
- `0.8.0` fails the verifier.
- `1.2.0` overstates what was ported, which is a FAIL under eval.md's version-honesty rule.

## Attempts

1. **Initial port** of the delta, excluding #90 and MCP. The first fresh eval
   (`FAIL`) found the following:
   - #90 is missing.
   - c610b6e's two-phase PreToolUse-before-approval gate is missing. This one is
     security-relevant: a hook could mutate approved arguments without being
     re-gated.
   - `stream_replay` is not plumbed into `ModelResult`.
   - There are smaller HITL/context/usage gaps.
2. **Focused fix:** ported the two-phase approval preparation. This adds the
   `pre_tool_use_applied` marker, schema validation of mutated input, re-gating by
   input-dependent approval sources, and no second PreToolUse on resume. The 5
   skipped upstream cases were un-skipped and pass. A second fresh eval re-ran
   (`FAIL`) and confirmed the approval gate now matches upstream exactly. The
   remaining blockers are #90 and version honesty.
3. A third attempt would not change the verdict. The remaining gap is a feature
   port, not a fix, so the run stops here as the skill directs.

## Second (final) fresh eval — verbatim


# Parity Eval (independent re-run): python-agent working tree vs `@openrouter/agent` @ 50a1ec2

- Last ported: `680bceb` (`@openrouter/agent` 0.8.0). Target: `50a1ec2` (`chore: version packages (#159)`). `git describe` says the target is exactly the `@openrouter/agent@1.2.0` tag.
- Delta: 50 upstream commits touching `packages/agent`, covering releases **0.9.0, 0.10.0, 0.11.0, 1.0.0, 1.0.1, 1.1.0, 1.2.0**. In `src` and tests: 142 files, about +49k/-1.3k lines.
- Port under review: the uncommitted working tree. 22 modules are modified, 3 are new (`doom_loop.py`, `tool_set.py`, `user_agent.py`), and 33 test files are new, plus `tests/vectors/doom-loop-fingerprints.json`.
- What I ran:
  - `uv run pytest tests/unit -q`: 738 passed, 9 skipped, 23 xfailed.
  - `.upstreamer/scripts/verify.sh`: **FAIL (1 failure)**. ruff, format, mypy, pytest, the 83% coverage floor and the 31-symbol API check all pass. The one failure is version drift.

## Verdict rationale (short)

The work that was ported is mostly faithful, and much of it is good:
- doom-loop detection
- the c610b6e approval split into two phases
- approval-decision validation (1601bff/1008d7e)
- stopWhen-before-approval (844845f)
- truncation guard, `tool_choice` relaxation, `next_turn_params.tool_choice`
- `get_usage`, ToolSet, user agent
- replay compaction in `ReusableStream` and the broadcaster

But the tree is **not** a port of 1.2.0. It leaves out the headline of 0.9.0 (`e8d7d6d`: unified `run()` tools, background/deferred lifecycles, `resumeToolResults`, the universal `task` tool, steering, `tool.agent()` subagents, per-tool cancellation/timeouts, tool concurrency). That feature is not a documented divergence in the contract.

The declared version also fails mechanically: it stays at 0.8.0 while upstream is 1.2.0. I can't find a version that is both honest and verifier-clean:
- 0.8.0 understates the content and fails the verifier.
- 1.2.0 would overstate it, and the eval says that is a FAIL.

Separately, the port has several strict-xfail "PORT GAP"s in HITL, state and usage behaviour. Advancing `state.yaml` to `50a1ec2` would mark e8d7d6d as ported, and the next sync would skip it for good. That is exactly the failure this eval exists to stop.

---

## 1. Version honesty: FAIL

- `pyproject.toml:11` is `version = "0.8.0"`. Upstream `packages/agent/package.json` at the target is `"version": "1.2.0"`. verify.sh: `FAIL: version drift: pyproject.toml=0.8.0, upstream @openrouter/agent=1.2.0`.
- **0.8.0 is not an accurate label for the tree either.** The tree ships behaviour from 0.9.0 through 1.2.0:
  - doom loop (0.9.0, `78c562e`)
  - `get_usage` (`a629cf1`)
  - `tool_choice` relaxation (`0efdbb0`)
  - ToolSet (`8d2ed61`)
  - `from_chat_messages` emitting `function_call` items (`dacf4d1`, 1.0.0)
  - manual-tool wire schema (`4dce84e`, 1.0.0)
  - user-agent identification (`ddab365`)
  - truncation guard (`69aae63`)
  - stopWhen-before-approval (1.0.1)
  - MCP own JSON Schema (1.1.0)
  - approval-decision hardening (1.2.0)
- **1.2.0 would overstate it.** The 0.9.0 minor (`e8d7d6d`, upstream CHANGELOG.md lines 324+) is almost entirely missing (see §2). The contract forbids setting the version "past what was actually ported", and the eval says overstating is a FAIL.
- No version both passes the verifier and tells the truth. So the tree can't be accepted at this target. This is a scope problem, not a one-line fix.
- Release-tag note (contract, Package Version): verify.sh says "ported tree is level with the 1.2.0 release tag (publishable)". That is true of the *target commit* and not of the *ported content*. Publishing this tree as 1.2.0 would ship a 1.2.0 that is missing 0.9.0's API.

## 2. Public API completeness

### Contract list: PASS
All 31 required symbols import from `openrouter_agent` (verify.sh check, and I confirmed by import). I spot-checked behaviour on the approval, state, hooks and stop paths below.

### New upstream root exports in `packages/agent/src/index.ts` (680bceb..50a1ec2)

**Present:**
- Doom loop: `DoomLoopMonitor`, `canonicalize_key_material`, `fingerprint_key_material`, `fingerprint_tool_call`, `detect_text_repetition`, `resolve_doom_loop_option`, `resolve_loop_key_material`, `DEFAULT_DOOM_LOOP_LADDER`, `DEFAULT_MAX_ESCALATIONS`, `MAX_CANONICALIZE_DEPTH`
- Async params: `TOOL_SET_SNAPSHOT`
- Tools and context: `mark_mcp`, `build_tool_execute_context`, `ToolContextStore`, `ToolEventBroadcaster`
- Tool sets: `create_tool_set`, `ToolSet`
- `ModelResult`: `get_usage`, `get_items_stream`, `get_doom_loop_verdict`

**Missing at runtime (none appear in Idiomatic Divergences):**

| Upstream export | Module | Python name per naming convention |
|---|---|---|
| `resumeToolResults`, `ToolTaskAlreadySettledError` | `inner-loop/resume-tool-results.ts` | `resume_tool_results`, `ToolTaskAlreadySettledError` |
| `AgentTranscriptSource` | `lib/agent-tool.ts` | `AgentTranscriptSource` |
| `AsyncToolRegistry` | `lib/async-tool-registry.ts` | `AsyncToolRegistry` |
| `TASK_TOOL_NAME`, `TaskToolInputSchema` | `lib/tool-check.ts` | same |
| `acquireAll`, `Semaphore` | `lib/tool-concurrency.ts` | `acquire_all`, `Semaphore` |
| `DEFAULT_TASK_LOG_LIMITS`, `ToolTask` | `lib/tool-task.ts` | same |
| `isAgentTool`, `isDeferredHandle`, `isLongRunningTool`, `isToolAsyncStartedEvent`, `isToolAsyncSettledEvent`, `isUnifiedTool` | `lib/tool-types.ts` | `is_agent_tool`, … |
| `tool.background`, `tool.deferred`, `tool.agent`; `tool({ run, lifecycle })` | `lib/tool.ts` | `tool()` has no `run`/`lifecycle` (`src/openrouter_agent/tool.py`) |
| `ModelResult.getAsyncTasks/cancelTask/sendToTask/queueUserMessage` | `lib/model-result.ts:7527-7573` | absent from `model_result.py` |
| `ConversationStatus 'awaiting_async_tools'`, `ConversationState.pendingAsyncTools` | `lib/tool-types.ts:1823,1922` | absent from `tool_types.ConversationState`. A TS blob carrying `pendingAsyncTools` is silently dropped by `deserialize_conversation_state`. |
| callModel options `signal`, `toolTimeoutMs`, `toolConcurrency`, `asyncTools`; per-tool `timeoutMs`, `maxConcurrency` | | absent |

Upstream subpath exports added to `package.json` with no Python module:
- `./tool-concurrency`, `./async-tool-registry`, `./resume-tool-results`, `./tool-task`, `./tool-check`, `./agent-tool`
- `./mcp`, `./mcp/*` (see §9)

The contract requires "one Python module per upstream lib module". `lib/agent-tool.ts`, `async-tool-registry.ts`, `async-tools.ts`, `tool-check.ts`, `tool-concurrency.ts`, `tool-task.ts` and `inner-loop/resume-tool-results.ts` have no counterpart.

`ModelResult.queueUserMessage` is general steering, not specific to async tools, and it is also missing.

**New naming mappings to add to the contract table** (derived by the port, all consistent with convention):
- `getUsage` → `get_usage`
- `getItemsStream` → `get_items_stream`
- `getDoomLoopVerdict` → `get_doom_loop_verdict`
- `createToolSet` → `create_tool_set`
- `markMcp` → `mark_mcp`
- `doomLoop` → `doom_loop`
- `loopKey` → `loop_key`
- `strictFinalResponse` → `strict_final_response`
- `preToolUseApplied` → `pre_tool_use_applied`
- `TOOL_SET_SNAPSHOT` unchanged

## 3. Load-bearing loop: parity on what exists, with one pre-existing undocumented divergence

- I read `ModelResult._run` (`model_result.py:1878-2206`):
  - Request sequence, accumulated input via `_build_followup_request`, stop-condition evaluation, the `allow_final_response` final turn with stubs for unexecuted calls, and the empty-final retry all match upstream.
  - `max_output_tokens` truncation skips execution (`stream_transformers.py`, `test_max_output_tokens_truncation.py`).
  - `next_turn_params.tool_choice` becomes the new caller-level policy (`_apply_next_turn_params`, :2245).
  - Forced `tool_choice` is relaxed on follow-ups.
- **Divergence (pre-existing, undocumented, now asserted upstream):**
  - Upstream executes one round's calls concurrently (`Promise.allSettled`, `model-result.ts:3093` and :3973). It was concurrent before this delta too.
  - The port runs them one by one: `_execute_round` (`model_result.py:1812`, a `for call in calls: await …`) and `_execute_auto_approve_tools` (:1470).
  - Upstream `tool-cancellation.test.ts` "unbounded by default (all calls run at once)" pins concurrency. It is unported.
  - Output order matches; wall-clock and overlapping side-effect behaviour does not.
  - This should be fixed, or added as a documented divergence.
- No cancellation of individual tools or of the run by signal, and no tool timeouts (`tool_timeout` error output). `cancel()` (`model_result.py:2550`) cancels the asyncio task. That is acceptable under divergence 5 for whole-run cancel. Per-tool `timeout_ms` / `tool_timeout_ms` and the `tool_timeout` error code have no equivalent.

## 4. Streaming: PASS on the ported surface

- `ReusableStream` and `ToolEventBroadcaster` replay compaction and backlog release (`17418f7`, `250e64e`) are ported, with `test_reusable_stream.py` (12 tests) and `test_replay_buffer_compaction.py` (8 tests).
- `turn.start` / `turn.end` stay paired per turn (:1961-1992).
- The late-close terminal-event case for PostModelCall (`test_hooks_post_model_call.py`) asserts exact counts and payload.
- **Gap:** upstream `model-result-replay-compaction.test.ts` (6 cases: terminal response after the active consumer trims history, `streamReplay` stripped from follow-up and resolved requests, cached terminal response, cleanup-failure handling) has no Python counterpart. These streaming invariants are FAIL-class under the eval. Port them, or show they are covered elsewhere.
- No `tool.async_started` / `tool.async_settled` events (they come with the missing async tools).

## 5. State: parity on the ported surface, one gap

- Versioned serialization works:
  - unchanged `CONVERSATION_STATE_VERSION`
  - an unsupported version raises before shape checks
  - `pre_tool_use_applied` is written only when `True`, and legacy blobs load as `False` (`conversation_state.py:104-109`, `:169-180`)
  - tests: `test_conversation_state_serialization.py:148-187`
- `tool_requires_approval` (`conversation_state.py:227-268`) matches c610b6e:
  - the call-level check gets the normalized copy when parsing succeeds and the raw call otherwise
  - the tool-level predicate gets parsed args
  - a schema-invalid or non-mapping parse fails closed
- **Gap:** `ConversationState` has no `pending_async_tools` and no `awaiting_async_tools` status (see §2).
- Minor: `deserialize_conversation_state` builds `ParsedToolCall(**item)`, so an unknown key in a pending call raises `TypeError` instead of `InvalidStateError`.
- **Gap (strict xfail, state/approval):** `test_get_usage.py:316` `test_does_not_advance_an_approval_resumed_run_when_read_before_the_loop`.
  - Upstream `getUsage()` (`model-result.ts:7704-7712`) only observes an approval-resumed run (`isResumingFromApproval` guard).
  - The port's `get_usage()` (`model_result.py:2354`) awaits `_ensure_task()`, which drives the entire run: it executes approved tools, makes further model requests and marks the conversation complete.
  - So reading usage has side effects, including spend, that a TS user doesn't get.
  - This is a behavioural gap in new delta API on the approval/state path. It is pinned by an xfail but is not a contract divergence.

## 6. Approval / HITL ordering: c610b6e is faithfully ported; HITL output gaps remain

- **Two-phase gate.**
  - `_handle_approval_check` / `_resolve_approval_phases` / `_prepare_after_initial_approval` (`model_result.py:1493-1618`, :808-839) match upstream `handleApprovalCheck` / `resolveApprovalPhases` / `prepareAfterInitialApproval` (`model-result.ts:3272-3384`, :3188-3258, :2871-2904) step for step.
  - Phase 1 runs partition plus PermissionRequest on the model-issued args.
  - Phase 2 runs PreToolUse for auto-resolvable survivors. A mutated input is schema-validated, then re-gated once (mutated-gate key) by input-dependent sources only (`_mutated_input_requires_approval`).
  - A call that pauses on the second gate is persisted with `pre_tool_use_applied=True` (:1557-1560, matching upstream :3367-3372).
  - On resume:
    - an approved call without the marker runs PreToolUse and may re-pause (:1697-1709)
    - one with the marker skips PreToolUse (`run_pre_tool_use=False`)
    - a rejected call with the marker emits `PostToolUseFailure` (:1731-1734), matching upstream :6079-6163
  - **This means PreToolUse runs before the final approval decision for survivors, and runs only after the human decision for calls pending from phase 1.** That is upstream's ordering exactly. The marker is honoured.
  - Doom-loop classification runs before partitioning (`_classify_initial_approval_calls`, matching `classifyInitialApprovalCalls`).
- **Mixed turns.**
  - Auto-executable siblings run before the pause and are stored as `unsent_tool_results` along with hook denials.
  - With `allow_final_response is False` and a met `stop_when`, they are recorded as "skipped" on turn 0 only (844845f; same call sites as upstream :6417-6421).
  - Resume appends unsent outputs after the persisted messages (:1778-1781).
  - The approval gate is now also enforced on the `allow_final_response` path (:2083).
- **Approval decisions** (1601bff/1008d7e):
  - the same ID in both lists raises (:240-248)
  - a resume with no decision stays paused
  - HITL output resumes still work
  - tests: `test_approval_gate_regressions.py` (51 cases vs 42 upstream; 2 skipped)
- **Gate identity:**
  - The port keys gates on `(id(response), index)`. Upstream uses a per-response occurrence counter (`approvalResponseKey` / `assignApprovalCallOccurrences`, :2752-2789).
  - This is safe today because every response stays alive in `self._steps`, so `id()` can't be reused within a run.
  - The subset-call contract is skipped with a reason (`test_approval_gate_regressions.py:975`).
  - It is fragile and undocumented. Warning only.
- **HITL gaps (strict xfails, pre-existing but now exposed; FAIL-class area):** in `test_hitl_tool.py`:
  - `:382` — `apply_on_response_received_hooks` (`tool_executor.py`) validates and rewrites caller outputs for *any* client tool. Upstream only touches HITL tools (`buildHitlToolMap`, `tool-executor.ts:1018`). A regular tool's caller-supplied output that doesn't match its schema gets turned into an error wrapper.
  - `:564` — when a hook throws on a content-array output, the port emits a JSON string. Upstream emits a content array (`formatHookError`).
  - `:861` — the port serializes the schema-validated hook return and drops extra keys. Upstream serializes the raw return.
- `test_tool_context.py:152,178` (strict xfails): `set_context` does not validate against the context schema, and is not a no-op when the schema is absent. Upstream `tool-context.ts:113-133, 237-240`.

## 7. Hooks: PASS

- Session id is threaded per emit (`session_id=self._session_id`).
- `SessionEnd` and drain sit in `finally` (:2197-2206), with teardown errors downgraded to warnings.
- A `DoomLoopDetected` hook and `"doom_loop"` SessionEnd reason are present.
- The 231fb65 tool call is threaded into the hook context, and `hooks` / `active_tools` are stripped from the outgoing request on both resolution paths (`test_hooks_session_lifecycle.py`; asserts exact event pairs).
- `test_hooks_session_lifecycle.py` and `test_hooks_post_model_call.py` port only the delta cases. That is acceptable because the base cases live in `test_model_result_hooks.py`, but it breaks the 1:1 file mapping.

## 8. Doom-loop integration: PASS (well covered)

- `doom_loop.py` (1289 lines) is wired into `_classify_initial_approval_calls`, `_begin_doom_loop_round`, `_check_doom_loop_for_response`, the steer flush (:2166), escalation dispatch overrides (:1981), the stop seal (:2049, :2099, :2151), and state persistence/restore (:503-544). Steer is deferred across HITL pauses, as upstream does.
- Upstream fingerprint vectors are shared (`tests/vectors/doom-loop-fingerprints.json`).
- Test counts match or exceed upstream:

| Test file | Port | Upstream |
|---|---|---|
| doom-loop | 96 | 66 |
| integration | 21 | 21 |
| fanout | 27 | 27 |
| remediation | 23 | 21 |
| escalation | 16 | 16 |
| public-api | 6 | 6 |

- The integration fixtures build on `tests/_fixtures.py` builders (`Script` wraps `make_response`/`function_call_item`) and assert execution counts.
- Upstream's "late-result delivery as forward progress" can't apply here, because there are no async tools.

## 9. Contract scope vs upstream moving MCP into `packages/agent/src/mcp`

- Since `787cbf8` (#102), the MCP client lives inside `packages/agent`:
  - `src/mcp/*`, 22 files
  - subpath exports `@openrouter/agent/mcp`, `/mcp/create-mcp-tools`, `/mcp/types`, `/mcp/schema`, `/mcp/cache`
  - `@openrouter/mcp` is now a compatibility facade
- Read literally, the contract's Scope ("Port only `packages/agent/`") now *includes* MCP. Its intent ("`packages/mcp/` … Not ported … Adding it is a deliberate contract change") excludes it.
- The port took the intent route: no `mcp` subpackage. It kept only root-level MCP touchpoints (`mark_mcp`, `is_mcp_tool`, `source: "mcp"`, and the MCP tool's own JSON Schema on the wire from 907beec, in `test_mcp_tool_branding.py`).
- That is a reasonable reading, but it is **undocumented**. The contract needs an explicit amendment that names `packages/agent/src/mcp/**` and `tests/unit/mcp/**` as out of scope. Without one, every future sync is ambiguous.
- verify.sh's parity diff only globs `tests/unit/*.test.ts`, so the 16 MCP test files are invisible to it:
  - `build-tools`, `cache`, `call-tool-shape`, `create-mcp-tools`, `elicitation`, `json-schema-to-zod`, `list-tools-pagination`, `loop-key`, `mcp-connection`, `mcp-sdk`, `protocol-era`, `rehydrate`, `resource-tools`, `result-mapper`, `version`, `wire-input-schema`
- Severity: **warning, if the contract is amended**. Otherwise it is an unported in-scope module.
- `mcp/loop-key.test.ts` and `mcp/wire-input-schema.test.ts` also test root-level behaviour: `loopKey` on wrapped tools, and the wire schema. Check that their root-level assertions are covered by `test_doom_loop*.py` and `test_manual_tool_wire_schema.py`.

## 10. Test parity

### Unported upstream test files (`comm -23`, 33 files plus 16 under `tests/unit/mcp/`)

**New in this delta and FAIL-class** (they cover the tool loop, streaming, approval or state):

| Upstream file | Cases | Invariant | Severity |
|---|---|---|---|
| `async-tool-background.test.ts` | 8 | background lifecycle, pending placeholder, `tool_task_result` injection, `onRunEnd` | FAIL (tool loop) |
| `async-tool-deferred.test.ts` | 19 | deferred pause `awaiting_async_tools`, `.resolve/.fail/.cancel`, double-settle error, tool_choice on resume | FAIL (state / HITL-like pause) |
| `async-tool-registry.test.ts` | 5 | registry tracking/settling | FAIL (state) |
| `task-tool-actions.test.ts` | 11 | universal `task` tool check/steer/result/cancel | FAIL (tool loop) |
| `task-tool-integration.test.ts` | 7 | task tool inside the loop | FAIL (tool loop) |
| `tool-check.test.ts` | 13 | custom `check` config dispatch | FAIL (tool loop) |
| `agent-tool.test.ts` | 9 | `tool.agent()` subagents | FAIL (tool loop) |
| `steering.test.ts` | 4 | `ToolTask` inbox, `sendToTask` | FAIL (tool loop) |
| `tool-cancellation.test.ts` | 10 | `ctx.signal`, `tool_timeout`, run abort, `cancel()` aborts tools, **concurrency incl. unbounded-by-default**, semaphore release | FAIL (tool loop) |
| `run-cancellation.test.ts` | 7 | run `signal` into every dispatch, pre-aborted fails before network, `timeoutMs` composition | FAIL (loop / streaming). Partly divergence 5, but the invariants (no dispatch after abort, per-dispatch timeout) still need Python equivalents. |
| `model-result-replay-compaction.test.ts` | 6 | terminal response after replay trim, `streamReplay` stripping | FAIL (streaming) |
| `startup-import-closure.test.ts` | 1 | optional deps not in the root import closure | warning |
| `package-boundaries.test.ts` | 0 | packaging only | cosmetic |

**Pre-existing (unported at 680bceb as well, unchanged in this delta).** These were accepted by the previous eval, presumably as covered by `test_model_result_hooks.py` and others. I note them, and did not re-grade them as delta gaps:
- `assistant-message-images`, `build-tool-call-stream`, `conversation-state` (42 cases), `create-tool`, `get-items-stream-manual-tools`
- `hooks-contract-fixes`, `hooks-emit(-adversarial)`, `hooks-manager-adversarial`, `hooks-matchers-adversarial`, `hooks-resolve(-adversarial)`, `hooks-types-adversarial`, `hooks-void-schema`
- `manual-tool-calls-adversarial`, `model-result-stream-detection`, `schema-sanitization`, `shared-context`, `tool-type-compat`, `user-input-persistence`

The 1:1 mapping is still unmet for them. `conversation-state.test.ts` (42 cases) and `user-input-persistence.test.ts` cover state and are worth closing.

Type-level `.test-d.ts` files (`async-tool-types`, `filter-tools-by-ids`, `full-field-examples`, `manual-tool-wire-schema`, `message-compat-types`, `resolved-tools`, `server-tool-id`, `tool-name-correlation`, `tool-shared-name`) are a warning only (divergence 4).

### Quality of new tests

- Generally strong:
  - order and count assertions on event sequences
  - exact execution counts (`ExecutionCounter`, `Counter`)
  - requests asserted by index
  - fixtures built from `tests/_fixtures.py`; bespoke transports (`FailingClient(QueuedClient)`, async-generator streams) still use the shared builders
- Flags:
  - `test_tool_name_events.py:212`: `assert len(tool_events) >= 1`. The invariant is which events fire and how many.
  - `test_tool_name_events.py:1-12` docstring says `toolName` is "pinned by a strict xfail", but there is no xfail in that file now. The docstring is stale.
  - `test_tool_orchestrator.py`: 17 of 27 cases are strict xfails against a non-existent `execute_tool_loop` / result-helper API. `lib/tool-orchestrator.ts` is not a root export and is unreferenced upstream, so this is a warning. But the file is 63% xfail, and it is xfail scaffolding, not coverage.
  - Skips with reasons naming unported async/agent/cancellation features: `test_server_tool.py:114,119`, `test_tool_context.py:118,123`, `test_call_model_active_tools.py:74`, `test_approval_gate_regressions.py:717`, `test_tool_choice_relaxation.py:320` (the `interrupted` status path). These confirm the gaps in §2 and §3.

### Coverage floor
`--cov-fail-under=83` in `.github/workflows/ci.yaml:80` is unchanged and matches verify.sh. It was not lowered.

## 11. Divergences and repo-owned files

- **Undocumented divergences (findings):**
  - async/unified tools, task/steering/agent tools, tool cancellation/timeout/concurrency (§2)
  - sequential round execution (§3)
  - `get_usage` drives the run (§5)
  - HITL `on_response_received` differences (§6)
  - `set_context` schema handling (§6)
  - MCP subpath (§9)
  - Skip/xfail reasons call several of these "Out of scope", but the contract has no such scope entry.
- **Repo-owned files intact:** `.github/`, `LICENSE`, and `pyproject.toml` (`[project.urls]`, classifiers, distribution name `openrouter-agent-sdk`) are unchanged. The user-agent token uses the PyPI name (divergence 7, `user_agent.py`).
- **Docs:**
  - README was updated for doom loop and the PreToolUse ordering, and its claims match the code.
  - `upstreamer-changelog.md` was **not** updated for this delta, so there is no user-facing note that 0.9.0's async-tool feature set is absent.

---

## What's needed to pass

1. **Close the scope gap, or narrow the target.** Choose one:
   - (a) Port `e8d7d6d` and its follow-ups:
     - `tool(run=…, lifecycle=…)`, `tool.background`, `tool.deferred`, `tool.agent`
     - `AsyncToolRegistry`, `resume_tool_results`, `ToolTaskAlreadySettledError`
     - the universal `task` tool (`TASK_TOOL_NAME`, `TaskToolInputSchema`, `check` config), `ToolTask` with steering inbox and `DEFAULT_TASK_LOG_LIMITS`
     - `Semaphore`/`acquire_all`, `tool_concurrency`, `max_concurrency`
     - per-tool `timeout_ms` / `tool_timeout_ms`; `ModelResult.get_async_tasks/cancel_task/send_to_task/queue_user_message`
     - `pending_async_tools` and the `awaiting_async_tools` / `interrupted` statuses
     - the `is_*` guards and `tool.async_*` events
     - Then set `version = "1.2.0"`.
   - (b) Retarget the run at the last commit before `e8d7d6d`, so the version and content can be made consistent. Then port forward in a later sync.
   - (c) Amend the contract with an explicit, reasoned Idiomatic Divergence or Scope exclusion for the async-tool system. Only then can the version be set to 1.2.0 without overstating.
2. **Make verify.sh pass.** The version must equal upstream `package.json` (1.2.0) *and* be honest, which follows from item 1.
3. **Fix `get_usage()`** so it only observes on an approval-resumed run (upstream `model-result.ts:7704-7712`). Turn `test_get_usage.py:316` from xfail into a passing test.
4. **Fix the HITL `apply_on_response_received_hooks` gaps** (`test_hitl_tool.py:382, :564, :861`) and `set_context` schema handling (`test_tool_context.py:152, :178`). Otherwise, record each one as a documented divergence with its reasoning.
5. **Port or justify sequential round execution.** Either run round calls concurrently (matching `Promise.allSettled`) or document it as a divergence. Port the concurrency cases of `tool-cancellation.test.ts`.
6. **Port `model-result-replay-compaction.test.ts`** (streaming), and add Python equivalents of the `run-cancellation.test.ts` invariants under divergence 5.
7. **Amend the contract's Scope** to say explicitly whether `packages/agent/src/mcp/**` and `tests/unit/mcp/**` are in or out. Extend verify.sh's parity diff to recurse into `tests/unit/**`.
8. **Clean up the tests.** Replace `len(tool_events) >= 1` in `test_tool_name_events.py:212` with an exact count, and fix the stale docstring.
9. **Update `upstreamer-changelog.md`** with the delta and every parity gap left open. Add the new naming mappings listed in §2 to the contract table.

## Recommended human action

1. **Decide on #90 (`e8d7d6d`) explicitly. Pick one:**
   - (a) Schedule a dedicated port of the async-tool system and its follow-ups.
     Candidate modules: `tool_task`, `tool_check`, `async_tool_registry`,
     `async_tools`, `agent_tool`, `tool_concurrency`, `resume_tool_results`, plus
     `ModelResult` additions. Tests: async-tool-background/deferred/registry,
     task-tool-actions/integration, tool-check, agent-tool, steering,
     tool-cancellation, run-cancellation. Then re-run this sync.
   - (b) Amend the contract to make the async-tool system a documented, versioned
     divergence, and define a version strategy that does not claim ≥0.9.0 without it.
2. **Amend the contract scope for MCP.** Upstream moved it into
   `packages/agent/src/mcp` (`@openrouter/agent/mcp`), so "port only
   `packages/agent/`" now literally includes it. This run treated it as out of
   scope, following the contract's intent. Also make `verify.sh`'s parity check
   recurse into `tests/unit/**`.
3. **Add the new naming mappings to the contract table** (listed in the final
   report).
4. **Keep the partial work in this run's working tree.** It is verified green
   apart from the version check. Choose one:
   - land it on a branch the next sync builds on;
   - discard it, in which case the next run re-ports it.

   Do not publish to PyPI from it.
5. **Known remaining non-#90 gaps** (strict-xfail pinned in tests):
   - `get_usage()` drives an approval-resumed run instead of only observing it.
   - Three HITL `on_response_received` output-handling differences.
   - `set_context` schema validation and the no-schema no-op.
   - `tool_orchestrator.execute_tool_loop` and its four result helpers.
   - `stream_replay` is accepted but not wired into `ModelResult`'s event buffer.
   - Round tool calls run sequentially; upstream runs them concurrently.
