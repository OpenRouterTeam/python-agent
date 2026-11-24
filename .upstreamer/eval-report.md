FAIL

# Port Parity Eval: `@openrouter/agent` 0.8.0 → 1.3.0 (680bceb → 0fe4cb1)

Reviewer: an independent fresh-context re-grade, following `.upstreamer/eval.md` and the contract in `.upstreamer/upstreamer.md`. It re-checks the tree after the converter's first fix attempt for the defects the previous eval listed.

Under review: the uncommitted working tree.
- 13 `src/` files are modified, 14 test files are new, and 6 test-side files are modified (4 unit test files plus `tests/_fixtures.py`).
- `pyproject.toml` still has `version = "0.8.2"`.
- `state.yaml` still points at `680bceb`.

Line references are to the current working tree.

## 1. Verdict rationale

**FAIL** (unchanged from the previous eval). Each of the following is a FAIL condition under `eval.md` on its own.

1. **The version claim cannot be made honest.**
   - The target `0fe4cb1` is tagged `@openrouter/agent@1.3.0`, so the verifier wants `version = 1.3.0`.
   - The port only reflects a subset of the 0.9.0–1.3.0 patch fixes. These are missing:
     - the headline 0.9.0 features: #90 async/unified `run()` tools and #73/#89 doom-loop;
     - the 0.10.0 tool-set (#31);
     - the 0.11.0 replay compaction (#109);
     - run-level `signal`;
     - the MCP subpath.
   - Claiming 1.3.0 would overstate the port by three minor versions, and claiming even 0.9.0 would also overstate it.
   - The earliest delta commit (`78c562e`, doom-loop #73) is not ported. So no upstream commit after 680bceb is fully covered, and `state.yaml` cannot truthfully move forward.
   - The verifier still fails on version drift: `pyproject.toml=0.8.2, upstream @openrouter/agent=1.3.0`.
2. **The approval ordering still differs from upstream's two-phase gate (#94, shipped in 0.11.0, inside this delta). D1 is not fixed.**
   - On a fresh run the port runs `PreToolUse` before approval classification for every call, including calls it then holds for a human (`model_result.py:780-816`, called from `_resolve_approval_gate` :825).
   - The predicate and `PermissionRequest` see the mutated arguments.
   - The pending call is persisted with the mutated arguments and `pre_tool_use_applied=True`.
   - The D2/D3 fixes only take effect on pending calls without that flag. On a fresh run with hooks, that means legacy states only. Because D1 is unfixed, the D3 rejection fix now also causes a new hook divergence on the common path (R1, §8).
3. **Large load-bearing upstream features are missing.** These change loop, state (`awaiting_async_tools`, persisted `doomLoop` state) and hooks behaviour:
   - async tools, the unified lifecycle, the `task` tool, steering, `tool.agent()` subagents, and per-tool timeouts and concurrency;
   - doom-loop detection;
   - run-level cancellation;
   - tool-sets;
   - stream-replay compaction.
4. **Test parity.** 51 upstream unit test files have no Python counterpart. Many cover the tool loop, state, approval/HITL, hooks, or streaming (§4).

The patch-level fixes that were ported are mostly faithful and reasonably tested (§2). Mechanical checks are green except for the version:
- ruff and mypy pass;
- 254 unit tests pass;
- the coverage floor of 83% is met;
- all 31 required symbols are exported.

## 2. Per-change parity table

| Upstream change | Status | Evidence (port) | Notes |
|---|---|---|---|
| #131 decision dedupe / conflict throw | Ported | `ModelResult.__init__` `_approved_ids`/`_rejected_ids` via `dict.fromkeys`, plus a conflict `ValueError` | Message matches upstream wording, with snake_case names. |
| #131 decision-less resume stays paused; partial decisions stay paused; getters raise the pause error | Ported | `_load_state` :337, `_stays_paused_without_decisions` :372, `_paused_error` :399, `_build_resume_request` returns `None` :1478-1484, `get_response` :1637, inspection getters use `_ensure_task` | Logic matches upstream. The resume gate is `pending_tool_calls and has_decisions` (:1009), not upstream's `status in (awaiting_approval, awaiting_hitl) and has_decisions` (low-severity edge for `awaiting_client_tools` states). |
| #94 two-phase approval gate | **Divergent (partially improved)** | Initial gate: `_prepare_tool_calls_for_approval` :780-816, `_resolve_approval_gate` :818-861. Post-approval: `_prepare_after_initial_approval` :1496-1542, `_mutated_input_requires_approval` :1544-1558, `_emit_prepared_failure` :1560-1574 | The post-approval phase now mirrors upstream (D2 and D3 fixed). The initial phase is still single-phase (D1 not fixed). |
| #143 stopWhen before approval on the first response (`allow_final_response=False`) | Ported | `model_result.py:~1104-1135` | Siblings are saved as `"Tool execution skipped: stop condition met"` (:1125). |
| #155 toModelOutput with parsed input, including the resume path | Ported | `_parsed_tool_input` :907, `_apply_to_model_output` :916, `_unsent_model_output` :929, used in `_build_resume_request` :1450 | |
| #153 nextTurnParams only for successful calls, using the effective (mutated) input | Ported | tool-round path in `model_result.py` | |
| #114 nextTurnParams may set `tool_choice`; the context always carries the key | Ported | `next_turn_params.py:8-28` | |
| #161 context setters store parsed values and validate the merged context | Ported | `tool_context.py:56-74`, :136-144 | |
| #160 an aborted blocking hook chain fails closed | Ported | `hooks_emit.py:71-79` | Matches the upstream diff. |
| #125 max_output_tokens truncation: no tool calls executed | Ported | `stream_transformers.py:41-77` | |
| #100 forced toolChoice relaxation, including across resumes and the persisted key | Partially ported | `_relax_forced_tool_choice` :123, `_forced_tool_choice_key` :144, `_send_and_track` :289, `ConversationState.consumed_forced_tool_choice_key` | Static choices match upstream. The dynamic (async-param) `tool_choice` re-arm cannot be reached because async params are resolved once (`_resolve_initial_request` :363). Upstream re-resolves them on every turn. |
| #97 `getUsage` | Partially ported | `get_usage` :1831-1852, `_snapshot_session_usage` :547 | D4 is not fixed: `get_usage()` drives the loop on approval-resumed runs. |
| #99 strict passthrough | Ported | `tool.py`, `tool_executor.py` | |
| #118 / #146 `wireInputSchema` | Ported (agent side) | `tool.py` (`wire_input_schema`, `mark_mcp`), `tool_executor.py` | Runtime raise for non-manual tools; undocumented divergence (D7). |
| #148 MCP annotations | Partially ported (brand only) | `mark_mcp(annotations=…)` | No `McpToolAnnotations` and no `createMCPTools`. Depends on the MCP scope (§5). |
| #93 fromChatMessages tool calls | Ported | `chat_compat.py:30-53` | |
| #110 preliminaryResults dropped | Ported | `tool_executor.py`, tool.result events | |
| #91 toolCall threaded into the execute context | Ported | `tool_executor.py` `to_function_call_item` | |
| #95 validateFinalResponse messages | Ported | `_validate_final_response` :484 | 7 of 17 upstream cases ported. |
| #31 `activeTools` | Ported | `call_model.py:35-38` | 6 of 9 upstream cases. The rest need tool-sets. |
| #31 `createToolSet` / tool-set | **Missing** | none | |
| #162 duplicate client tool names | Ported (earlier run) | `tool_types.py`, `ModelResult.__init__` | |
| #120 user agent | Ported (earlier run) | `openrouter.py` | Contract divergence 7. |
| #90 async tools / unified `run()` / lifecycles / `task` tool / steering / `tool.agent()` / per-tool cancellation and timeouts / concurrency | **Missing** | No registry, task tool, lifecycle, or `awaiting_async_tools` status | `call_model` now refuses `async_tools`, `tool_timeout_ms`, `tool_concurrency` (D6). |
| #73 / #89 doom-loop detection, escalation, fan-out | **Missing** | none | `call_model` now refuses `doom_loop`. |
| Run-level `signal` cancellation | **Missing** | Only `ModelResult.cancel()` (:1854) | `call_model` now refuses `signal`. |
| #109 / #116 stream replay compaction (`streamReplay`) | **Missing** | Unbounded `ModelResult._events` | `call_model` now refuses `stream_replay`. |
| #102 / #140 / #146 / #148 MCP under `packages/agent/src/mcp` | **Missing** (scope ambiguous, §5) | none | 25 upstream source files, 16 test files. |
| #158 SDK upgrade to 1.5.85 | N/A | The pinned SDK is a Python SDK | |

## 3. Concrete findings (status after fix attempt 1)

**D1. NOT FIXED. The approval gate runs PreToolUse before the human decision.**

`model_result.py:780-816` (`_prepare_tool_calls_for_approval`) is still called first from `_resolve_approval_gate` :825.

Upstream (`handleApprovalCheck` / `resolveApprovalPhases`, model-result.ts 3199-3410):
- `partitionToolCalls` and `PermissionRequest` run on the raw arguments.
- `PreToolUse` runs only for initial survivors, via `prepareAfterInitialApproval`.
- A held call is persisted with its original arguments and without `preToolUseApplied`.

Port:
- `PreToolUse` fires for every call first.
- The predicate and `PermissionRequest` see the mutated arguments.
- Every held call is persisted with `pre_tool_use_applied=True`.

Re-verified: a fresh run with a non-mutating `PreToolUse` hook on a `require_approval=True` tool fires `PreToolUse` once before the pause and persists `pre_tool_use_applied=[True]`. Upstream fires 0 times and stores no flag.

`test_pre_tool_use_mutation_is_approved_once_and_persists_across_resume` still pins the port's single-phase shape. The upstream block "approval uses the post-PreToolUse arguments that execute would receive" (approval-gate-regressions.test.ts 444-1088) is still not ported.

**D2. FIXED. Re-gating after a resume-time mutation is limited to input-dependent checks.**

`_mutated_input_requires_approval` (`model_result.py:1544-1558`) mirrors upstream `mutatedInputRequiresApproval` (2795-2818):
- If a callable call-level `require_approval` exists, it is evaluated on the mutated call.
- Otherwise only a function-valued tool-level `require_approval` is evaluated.
- A blanket `True` returns `False`.

Verified by reproduction: on a legacy pending call (no flag), a resume with `approve_tool_calls=["c1"]` and a mutating `PreToolUse` behaves as upstream:
- the tool executes once, with `{"v": "mutated"}`;
- `PreToolUse` fires once;
- `PostToolUse` sees the mutated input;
- one follow-up request is sent and the status ends `complete`.

The function-valued path still pauses again with the mutated arguments and `pre_tool_use_applied=True`, which matches upstream `secondGatePausedIds`.

Small difference: upstream returns `false` when `options.tools` is absent even if a call-level function exists. That branch cannot be reached in the port, because the approved call's tool has already been found.

New test `test_blanket_require_approval_runs_the_approved_call_after_a_pre_tool_use_mutation` (`tests/unit/test_approval_gate_regressions.py:683`) asserts three things: the call executed exactly once with the mutated value, exactly one request was sent, and the status is `complete`. That is real parity coverage. It does not assert the `PreToolUse` count. I checked that manually: it is 1.

**D3. FIXED (post-approval path). See R1 for the interaction with D1.**

| Case | Port | Upstream | Result |
|---|---|---|---|
| Invalid mutation | `_prepare_after_initial_approval` :1526-1532 parses the mutation with `_parse_tool_arguments` and requires a mapping. On failure it blocks with `PreToolUse produced invalid input for "<name>"` and emits `PostToolUseFailure`. | `validatePreparedMutation` / `blockPreparedToolCall` (2835-2872) | Matches |
| PermissionRequest deny on the mutated call | :1538-1541 emits `PostToolUseFailure` with the deny reason (default `Denied by PermissionRequest hook`). | `prepareAfterInitialApproval` 2902-2904 | Matches |
| Reject a call with `pre_tool_use_applied` | :1454-1462 emits `PostToolUseFailure` with "Rejected by user", then records the rejected result. | 6212-6222 | Matches |

Verified by reproduction:
- Function-valued gate, then a mutation, then a `PermissionRequest` deny with reason "nope": the call does not execute, the model output is `{"error":"nope"}`, and exactly one `PostToolUseFailure` fires, with error `nope`.
- An invalid mutation `{"v": 123}` produces the upstream error output and exactly one failure hook.

Tests:
- `test_a_mutation_that_fails_the_input_schema_is_blocked_on_resume` (:707) asserts the exact output list and the exact failure list.
- `test_rejecting_a_prepared_call_closes_its_lifecycle_with_post_tool_use_failure` (:732) asserts the exact `(tool_name, error)` list and the exact output.
- **Gap:** no test covers the `PermissionRequest` deny → `PostToolUseFailure` path. It was verified only by my reproduction.

**D4. NOT FIXED. `get_usage()` drives the tool loop on an approval-resumed run.**

`model_result.py:1848-1852` still awaits `_ensure_task()`. Upstream `getUsage` (7761-7787) counts only the resume dispatch and leaves the loop unadvanced. The upstream cases "does not advance an approval-resumed run when read before the loop" and "counts a STREAMING resume generation without advancing the loop" are not ported.

**D5. PARTIALLY FIXED.**

Fixed:
- The resume turn context is now `{"number_of_turns": 1}` (:1408). Verified: an approved tool's execute context receives `number_of_turns == 1`. No test pins this.
- The decision-resume rejection reason is now `"Rejected by user"` (:1458), matching upstream 6217. `create_rejected_result`'s default `"Tool call rejected by user"` (`conversation_state.py:265`) matches upstream's `createRejectedResult` default (conversation-state.ts:424), so it is correct as is.

Not fixed:
- The decision-resume path still emits `SessionStart` (:974) and runs `UserPromptSubmit` on the input (:979-981). Upstream's resume path skips both.
- The resume gate still keys on `pending_tool_calls` rather than status (:1009).

**D6. FIXED.**

`call_model.py:14-29` raises `NotImplementedError` before any request for non-`None` `doom_loop`, `signal`, `tool_timeout_ms`, `tool_concurrency`, `async_tools`, or `stream_replay`. `test_unsupported_upstream_options_are_refused_instead_of_sent` (`tests/unit/test_call_model.py:114-119`) checks every key and asserts `client.requests == []`.

Minor warning: `doom_loop=False` is also refused. Upstream `resolveDoomLoopOption(false)` means "disabled" (doom-loop.ts:392), which is exactly the port's behaviour, so accepting `False` would be the more faithful choice.

**D7. NOT FIXED. Undocumented divergences** (should go in the compatibility notes):
- `tool()` raises at runtime on `wire_input_schema` for non-manual tools. Upstream enforces this only at the type level.
- The `ConversationState` key `consumed_forced_tool_choice_key` is snake_case.
- `get_usage` warns instead of raising.
- `NotImplementedError` is raised for the unported engine options (D6).

**Test quality of the new tests.**
- Shared fixtures are used throughout (`QueuedClient`, `make_response`, `tool_call_response`, `MemoryStateAccessor`, the `recording` helper).
- Request-count and exactly-once assertions are exact. There is no `asyncio.sleep` and no hand-rolled client.
- `assert x is not None` lines are only used for narrowing.
- Case coverage is still thin compared with upstream:

  | File | Port | Upstream |
  |---|---|---|
  | approval-gate-regressions | 23 | 43 |
  | final-response-validation | 7 | 17 |
  | get-usage | 8 | 13 |
  | hooks-contract-fixes | 6 | 16 |
  | next-turn-params | 3 | 12 |
  | server-tool | 5 | 16 |
  | shared-context | 4 | 16 |
  | call-model-active-tools | 6 | 9 |

- The three new approval tests use a hand-built legacy state. Two get it by pausing without hooks; the reject test sets `pre_tool_use_applied=True` by hand. This exercises the upstream post-approval code paths, but not the fresh-run shape upstream produces.

**CI / repo-owned files.**
- `--cov-fail-under=83` in `.github/workflows/ci.yaml` is unchanged.
- `.github/`, `LICENSE`, `pyproject.toml` identity and version, and `state.yaml` are untouched.

## 4. Lists

### Fixable defects in the ported subset (still open)

1. **D1** `model_result.py:780-861`: implement the two-phase gate.
   - Partition and emit `PermissionRequest` on the raw arguments.
   - Run `PreToolUse` only for initial survivors, via `_prepare_after_initial_approval`.
   - Persist human-held calls with their original arguments and no flag.
   - Port upstream tests 444-1088, and fix `test_pre_tool_use_mutation_is_approved_once_and_persists_across_resume`.
   - This also resolves R1.
2. **D4** `model_result.py:1831-1852`: on a decision-resume run, `get_usage()` must count only the resume dispatch and must not advance the loop. Port the two upstream get-usage cases.
3. **D5 (remainder)**:
   - Skip `SessionStart` and `UserPromptSubmit` on the decision-resume path (:972-981).
   - Gate the resume on `status in ("awaiting_approval", "awaiting_hitl")` (:1009).
   - Add a test pinning `number_of_turns == 1` on resume.
4. **D3 test gap**: add a test for a `PermissionRequest` deny on a mutated call → exactly one `PostToolUseFailure`.
5. **D6 nit**: accept `doom_loop=False`.
6. **D7**: record the divergences listed above in `upstreamer-changelog.md` / Parity Notes.

### Unported upstream features (gaps)

- **#90 async tools:**
  - the unified `run()` interface and the `background`/`deferred` lifecycles;
  - `AsyncToolRegistry`;
  - the `task` tool (check/steer/result/cancel), steering, and `tool.agent()` subagents;
  - per-tool `timeout_ms`/`max_concurrency`, plus `Semaphore`/`acquireAll`;
  - the `awaiting_async_tools` status, `resumeToolResults`, and `getAsyncTasks`.
- **#73 / #89 doom-loop:** `DoomLoopMonitor`, the escalation ladder, fan-out detection, remediation/steer, persisted `doomLoop` state, `getDoomLoopVerdict`, and the `DoomLoopDetected` hook.
- Run-level `signal`, `tool_timeout_ms`, and `tool_concurrency`.
- **#31 tool-set:** `createToolSet`, `inferTools`/`.resolve()`, and snapshot metadata stripping.
- **#109 / #116 stream replay compaction** (`streamReplay`).
- Per-turn async-param re-resolution. Upstream's dynamic `tool_choice` re-arm depends on it.
- The `packages/agent/src/mcp` subpath (`createMCPTools`, `rehydrateMCPTools`, cache, elicitation, resource tools, `McpToolAnnotations`). Scope is ambiguous, see §5.

### Unported upstream test files (from `verify.sh` / `comm -23`, 51 files)

| Upstream test | Invariant | Severity |
|---|---|---|
| doom-loop, doom-loop-escalation, doom-loop-fanout, doom-loop-integration, doom-loop-public-api, doom-loop-remediation | repeated-call detection, escalation, stop verdict across resumes | FAIL (loop/state/hooks) |
| async-tool-background, async-tool-deferred, async-tool-registry, task-tool-actions, task-tool-integration, steering, agent-tool | async lifecycle, task check-ins, deferred resume, subagents | FAIL (loop/state) |
| run-cancellation, tool-cancellation, tool-orchestrator | cancellation/timeouts, concurrency, execution order | FAIL (loop) |
| tool-check | task-tool check semantics | FAIL (loop) |
| model-result-replay-compaction, replay-buffer-compaction, reusable-stream | multi-consumer replay correctness and memory bounds | FAIL (streaming) |
| tool-set | activeTools/tool-set resolution | warning→FAIL (request shape) |
| hitl-tool | HITL pause/resume, context threading | FAIL (approval/HITL) |
| hooks-emit, hooks-emit-adversarial, hooks-manager-adversarial, hooks-matchers-adversarial, hooks-resolve(+adversarial), hooks-types-adversarial, hooks-void-schema, hooks-post-model-call, hooks-session-lifecycle | hook chain semantics, PostModelCall, SessionEnd | FAIL (hooks); partly covered by `test_hooks_manager.py` / `test_model_result_hooks.py` |
| conversation-state, stop-conditions, turn-context, user-input-persistence | state helpers, stop predicates, turn context, input persistence | FAIL (state/loop) |
| manual-tool-calls-adversarial, get-items-stream-manual-tools, build-tool-call-stream, model-result-stream-detection, tool-name-events, tool-event-broadcaster | manual-tool edge cases, item/stream surfaces | FAIL (streaming) |
| tool-context, schema-sanitization, create-tool, tool-type-compat | context store, schema sanitization, tool factory | warning |
| openrouter | client wrapper (user agent, hooks) | warning (partly in `test_user_agent.py`) |
| claude-type-guards, assistant-message-images | compat guards/images | warning |
| package-boundaries, startup-import-closure | TS packaging/import graph | warning (largely N/A for Python) |
| mcp/* (16 files) | MCP subpath | depends on the scope decision |
| `*.test-d.ts` (type-level) | type inference | warning (contract divergence 4) |

## 5. Contract ambiguity: MCP scope

The contract says "Port only `packages/agent/`" and that `packages/mcp/` is not ported. Upstream #102 moved MCP into `packages/agent/src/mcp` and exports it as `@openrouter/agent/mcp`, and #140, #146 and #148 landed there.

- **Read literally**, MCP is now in scope.
- **Read by intent** ("do not begin porting MCP as a side effect of a sync"), it is out of scope.

The converter treated it as out of scope, which is defensible, but the decision is not recorded anywhere. The contract needs an explicit amendment either way. This does not change the verdict.

## 6. Version honesty

- The target `0fe4cb1` is tagged `@openrouter/agent@1.3.0`, and the verifier reports the tree as level with the tag.
- **Claiming 1.3.0 would overstate the port.** It is missing the 0.9.0 async tools and doom-loop, the 0.10.0 tool-set, and the 0.11.0 replay compaction. PyPI versions cannot be reused.
- The highest truthful claim is **0.8.x**: upstream 0.8.0 plus Python backports of selected later patch fixes. Those are #91, #93, #95, #97 (partial), #99, #100 (partial), #110, #114, #118, #125, #131, #143, #146 (agent side), #153, #155, #160, #161, #162, and the post-approval part of #94.
- `state.yaml` must stay at `680bceb`. The earliest delta commit (`78c562e`, #73) is not ported.

## 7. Recommended human action

1. **Do not advance `state.yaml`, and do not set `version = 1.3.0`.** Do not publish from this tree.
2. Have the converter fix D1 next. D1 is the root cause of the remaining approval divergence and of R1. Then fix D4 and the rest of D5, and port the matching upstream approval-gate-regressions and get-usage cases.
3. Decide the sync strategy for the large features (#90, #73/#89, #31, #109, run-level `signal`):
   - port each in its own focused run; or
   - amend the contract to make them documented deferrals, with a version policy (for example, stay on 0.8.x and record the backports in `upstreamer-changelog.md`).
4. Amend the contract's Scope section on `packages/agent/src/mcp`.
5. Add the naming mappings once confirmed:

   | TypeScript | Python |
   |---|---|
   | `getUsage` | `get_usage` |
   | `activeTools` | `active_tools` |
   | `wireInputSchema` | `wire_input_schema` |
   | `consumedForcedToolChoiceKey` | `consumed_forced_tool_choice_key` |
   | `toFunctionCallItem` | `to_function_call_item` |
   | `isTruncatedAtMaxOutputTokens` | `is_truncated_at_max_output_tokens` |
   | `responseHasToolCalls` | `response_has_tool_calls` |
   | `mutatedInputRequiresApproval` | `_mutated_input_requires_approval` (private) |
   | `emitPreparedFailure` | `_emit_prepared_failure` (private) |

## 8. Re-run after fix attempt 1

**Commands run**
- `uv run pytest tests/unit -q`: 254 passed. There is one asyncio `ResourceWarning` for an unclosed event loop, which is not a failure.
- `.upstreamer/scripts/verify.sh`: every check passes except one, `version drift: pyproject.toml=0.8.2, upstream @openrouter/agent=1.3.0`. Passing checks are uv sync, lockfile, ruff check, ruff format, mypy, pytest with the 83% coverage floor, 31/31 required symbols, the release-tag level check, no TS artifacts, and repo-owned files present. It also lists the 51 unported upstream test files.

**Fix verification.** I read the code against upstream `prepareAfterInitialApproval` (2874-2908), `mutatedInputRequiresApproval` (2795-2818), `validatePreparedMutation` (2854-2872), `blockPreparedToolCall` (2835-2852), `emitPreparedFailure` (2820-2833) and `processApprovalDecisions` (6086-6260). I also ran reproductions.

| Defect | Claimed | Verified status |
|---|---|---|
| D2 | blanket `True` no longer re-gates | **Fixed.** Executes once with the mutated input and completes. The function-valued path still re-pauses, as upstream does. |
| D3 | invalid-mutation block, deny/reject PostToolUseFailure, "Rejected by user" | **Fixed** on the post-approval path. All three emit exactly one `PostToolUseFailure` with upstream's reason strings. No test covers the deny path. |
| D5 | number_of_turns 1 on resume | **Partially fixed.** `number_of_turns` is 1 (verified, but no test pins it) and the reason string is fixed. `SessionStart`/`UserPromptSubmit` still run on resume, and the gate still keys on `pending_tool_calls` rather than status. |
| D6 | refuse unported engine options | **Fixed.** Tested for all six keys with no request sent. Nit: `doom_loop=False` is refused. |
| D1, D4, D7 | not attempted | **Not fixed.** Confirmed still present. |

**Regression found**

**R1.** `PostToolUseFailure("Rejected by user")` now fires on ordinary fresh-run rejections when hooks are configured (`model_result.py:1459-1461`).
- **Cause:** D1 persists every held call with `pre_tool_use_applied=True` (:814). The new D3 rejection branch therefore fires for the common case.
- **Reproduction:** a hooks-enabled fresh run on a `require_approval=True` tool pauses with the flag set. Resuming with `reject_tool_calls=["c1"]` emits `PostToolUseFailure` once, with error `Rejected by user`. Upstream emits it zero times, because held calls never ran `PreToolUse` and so carry no flag.
- **Assessment:** this is internally consistent with the port's own (divergent) lifecycle, since `PreToolUse` did run. It is still a new observable hook difference from upstream in the common path; before this fix attempt the port matched upstream here.
- **Fix:** fixing D1 removes it.
- **Test gap:** `test_rejecting_a_prepared_call_closes_its_lifecycle_with_post_tool_use_failure` sets the flag by hand, so it does not exercise this path.

No other regressions found. All previously passing tests still pass, and the updated resume rejection text did not break any existing assertion.

**Verdict after re-run: FAIL.** D1 (approval/hooks ordering) and D4 are still open, the large features and their test files are still missing, and the version claim still cannot be made honest.

## 9. Converter fix attempt 2, and the bankruptcy decision

The converter wrote this section after the re-run in §8. It is not part of the
independent review.

**Fix attempt 2 (R1).** The `PostToolUseFailure("Rejected by user")` emit on
the reject path has been removed (`_build_resume_request`, comment at the
rejection loop). Ordinary rejections again fire no hook, matching upstream on
that path. A test pins this:
`test_rejecting_a_held_call_emits_no_post_tool_use_failure`. Upstream emits the
hook for calls whose PreToolUse ran after an initial approval. That emit goes
back in once D1, the two-phase gate, is ported. Attempt 2 also added
`test_a_permission_request_deny_on_mutated_arguments_blocks_and_reports_failure`,
covering the D3 deny case §8 said had no test. After attempt 2 the verifier
fails only on version drift, and all 255 unit tests pass at 86% coverage
against the 83% floor.

**Bankruptcy.** The eval still returns **FAIL**, and no fix inside this run's
scope can change that.

- **Unported features.** The remaining FAIL conditions are entire upstream
  features:
  - async/unified `run()` tools, the `task` tool, steering and `tool.agent()`
    subagents (#90)
  - doom-loop detection with cross-port fingerprint vectors (#73, #89)
  - run-level `signal` cancellation and per-tool timeouts and concurrency
  - tool sets (#31)
  - stream replay compaction (#109, #116)
  - per-turn re-resolution of async request params
  - the upstream two-phase approval gate (D1)

  Together these are roughly 10k lines of upstream source, most of it
  interleaved in the 7.8k-line `model-result.ts`, plus about 25k lines of
  upstream tests. This is not a focused fix.
- **Version.** The verifier requires `version = 1.3.0`. Declaring it would
  overstate the port, which the contract forbids ("Never set it past what was
  actually ported"). So the version stays `0.8.2` and the verifier's version
  check fails on purpose.
- **Nothing to advance to.** The earliest commit in the delta (`78c562e`,
  doom-loop) is unported. No intermediate upstream commit is fully covered, so
  `state.yaml` cannot advance at all.

`.upstreamer/state.yaml` therefore stays at
`680bceb4598f228d3e2ec58e2416e4335cdff059`, and the next run retries the same
delta.

**What this run does leave in the tree.** These are verified, tested
backports of the upstream patch-level fixes listed as Ported in §2:
- #131 approval-decision validation and paused resumes
- #143, #155, #153, #114, #161, #160, #125
- #100 for static choices
- #97, with D4 open
- #99, #118 and #146 on the agent side
- #93, #110, #91, #95, #31 `active_tools`

The verifier is green apart from the version check. The next run should treat
these as already present and reconcile them, not redo them.

**Recommended human action.**
1. Do not publish from this tree. `0.8.2` plus backports is not any upstream
   release.
2. Split the sync into staged runs. Two ways to do it:
   - Add a contract option to target an intermediate upstream commit or tag,
     for example port `@openrouter/agent@0.9.0` (async tools and doom loop)
     first.
   - Run it as several human-reviewed PRs, one per feature family: (a) the D1
     two-phase approval gate and the per-turn async-param re-resolution;
     (b) run-level cancellation and timeouts; (c) doom-loop, against
     `tests/vectors/doom-loop-fingerprints.json`; (d) async tools, task tool,
     steering and `tool.agent()`; (e) tool sets; (f) replay compaction.
3. Decide the MCP scope (§5). Upstream moved MCP into `packages/agent/src/mcp`,
   while the contract excludes `packages/mcp`. Porting it would also add an
   MCP Python SDK dependency, which is a substrate change.
4. Once features are in, set `version` to the highest upstream version whose
   features are all present, and only then advance `state.yaml`.
