PASS WITH WARNINGS

# Port Parity Eval — Python (`@openrouter/agent` 0.7.2 → 0.8.0), second pass

Fresh re-review against upstream commit `680bceb4598f228d3e2ec58e2416e4335cdff059`,
delta baseline `adc7939f4b7ed85b1a060d13433b8be6063cff73`, scoped to `packages/agent/`.
This is a second-pass review after a prior FAIL with three findings. All three
are verified fixed below by reading the code directly (not by trusting the
converter's summary), and a broader sweep of the rest of the delta and the
eval's Required Qualities turned up no new material gap — only pre-existing,
cosmetic items.

## Verdict rationale

All three prior findings are genuinely fixed, confirmed by tracing execution
and by breaking the fix and watching the test fail. No required public API
symbol is missing, no load-bearing loop / state / approval-ordering / hooks
behavior regressed, and the declared version (`0.8.0`) matches what's
actually ported. Remaining items are documentation/coverage gaps that don't
change behavior — hence WARNINGS, not FAIL.

## Finding 1 (prior FAIL #1) — Stop hook `force_resume` zero-cost retry: FIXED, verified

`src/openrouter_agent/model_result.py:677-694`, inside `_run()`:

```python
while stop_conditions and await is_stop_condition_met(stop_conditions, self._steps):
    stop_decision = await self._run_stop_hook(force_resume_count, current_request)
    if stop_decision == "resume":
        force_resume_count += 1
        continue
    session_end_reason = "max_turns"
    stopped_by_stop_when = True
    break
```

- `is_stop_condition_met` (`stop_conditions.py`) is a pure function over
  `self._steps` (already-collected `StepResult`s) — it does not call `_send`
  or `_send_and_track`. The `continue` re-enters the `while` and re-evaluates
  this pure check against the *same* steps; no request is sent on the
  "resume" path. This is a faithful match of upstream's
  `model-result.ts:3050-3068` (`shouldStopExecution()` / `runStopHook()` /
  `continue`), confirmed by reading both side by side.
- `tests/unit/test_model_result_hooks.py::test_stop_hook_force_resume_is_a_zero_cost_retry_no_extra_model_request`
  genuinely pins this: the mock `QueuedResponses` has exactly one queued
  response, and asserts `len(client.beta.responses.requests) == 1` after a
  run in which the Stop hook is invoked twice (once forcing resume, once
  not). If the "resume" branch sent a real request, the mock's second
  `pop(0)` would raise `IndexError` on the empty list.
- I verified this is a real regression guard, not an accidental pass: I
  temporarily replaced the `continue` branch with a version that calls
  `await self._send_and_track(...)` (simulating the old bug), reran the test,
  and both `test_stop_hook_force_resume_is_a_zero_cost_retry_no_extra_model_request`
  and `test_stop_hook_force_resume_then_falls_through_to_normal_tool_round`
  failed immediately with `IndexError: pop from empty list`. Reverted
  afterward; `git diff --stat` confirms the file is back to its pre-sabotage
  state and the tests pass again.
- Minor cosmetic nit: the second test's docstring says the halted round's
  tool calls "execute via the normal round path (not the final-directive
  coercion)" — tracing the code, they actually go through the *same*
  final-directive-coercion branch (`stopped_by_stop_when` → `resolvable_pending`
  execute → `tool_choice: "none"` follow-up), which is also what upstream does
  (`model-result.ts:3205+`). The assertions themselves (text output, request
  count) are correct; only the prose comment is a little misleading. Not a
  behavioral gap.

## Finding 2 (prior FAIL #2) — MCP tool-branding surface: FIXED, verified

- `tool.py:57-64` — `mark_mcp()` returns `{**tool_to_mark, "_mcp": True}`, a
  new dict; the underlying `"function"` value is not copied (shallow), same
  as upstream's `{...toolToMark, _mcp: true}`. Confirmed non-mutating via
  `tests/unit/test_mcp_tool_branding.py::test_mark_mcp_is_non_mutating_and_is_mcp_tool_detects_the_brand`
  (`base is not branded`, `branded["function"] is base["function"]`).
- `tool_types.py:264-270` — `McpBranded` alias and `is_mcp_tool()` structural
  check (`tool.get("_mcp") is True`), matching upstream's `isMcpTool`.
- `tool_executor.py` — `source = "mcp" if is_mcp_tool(tool) else "client"` is
  computed and threaded through all three execute paths
  (`execute_regular_tool`, `execute_generator_tool`, `execute_hitl_tool`,
  lines 62/86/151) and returned in every result dict (success and error
  branches).
- `model_result.py:825, 852` — the `tool.result` / `tool_result` stream event
  carries `"source"`, preferring the executor-computed `result.get("source", ...)`
  and falling back to `is_mcp_tool(tool)` for the parse-error path where no
  executor result exists.
- `test_mcp_tool_branding.py::test_mcp_branded_tool_result_carries_source_mcp_in_tool_result_event`
  and `test_regular_client_tool_result_carries_source_client` both drive
  `call_model(...)` end-to-end through a mocked Responses API and assert
  `source` on the actual `tool_result` event from `get_tool_stream()` — this
  is genuine parity coverage, not a unit-level stub check.

## Finding 3 (prior FAIL #3) — Over-exported hook internals: FIXED, verified

- `src/openrouter_agent/__init__.py` — grepped the import block and `__all__`:
  `BUILT_IN_HOOKS`, `BUILT_IN_HOOK_NAMES`, `matches_tool`, `resolve_hooks`,
  `execute_handler_chain` appear nowhere in either.
- No wildcard (`import *`) re-export exists anywhere in `__init__.py` that
  could smuggle these back in. `resolve_hooks` is used internally by
  `call_model.py` via `from .hooks_resolve import resolve_hooks` but is never
  re-exported from the package `__init__`.
- `.upstreamer/scripts/verify.sh`'s `REQUIRED_SYMBOLS` list was inspected —
  it does not reference any of the five removed names, so the mechanical
  verifier's required-symbol check is unaffected by the removal.
- This matches upstream's own `index.ts`, which explicitly documents (in a
  code comment added in this exact delta) that `matchesTool`, `resolveHooks`,
  `BUILT_IN_HOOKS`, and raw schema objects are deliberately not exported.

## Finding 4 (prior FAIL #4) — Thin hook test coverage: FIXED, verified

Read all four new tests in `tests/unit/test_model_result_hooks.py` in full:

- `test_user_prompt_submit_can_reject_a_string_prompt` — asserts the raised
  `ValueError` message and, critically, `len(client.beta.responses.requests) == 0`,
  proving the model was never called after rejection (not just that an
  exception was raised somewhere).
- `test_user_prompt_submit_can_mutate_a_string_prompt` — asserts the
  *actually-sent* request's `input[0]["content"]` contains the mutated text,
  reading `client.beta.responses.requests[0]`, i.e., proving the mutation
  reached the wire request, not just the hook's return value.
- `test_user_prompt_submit_mutates_last_user_message_in_array_input` — same,
  for array-shaped `input`, asserting `sent_input[-1]["content"]`.
- `test_pre_tool_use_mutated_input_actually_reaches_tool_execute` — the tool's
  `execute` callback records its received `params` into a list; the test
  asserts the *tool implementation actually observed* the hook's
  `mutated_input`. I traced the plumbing: `HookEntry` mutation config
  (`hooks_types.py:117`, `mutations={"mutated_input": "tool_input"}`) maps the
  hook's `mutated_input` return key onto `final_payload["tool_input"]`, which
  `model_result.py:492-493` reads via `pre.final_payload.get("tool_input")`
  and substitutes into `effective_call.arguments` before `execute_tool()` is
  called. This is a genuine end-to-end proof, not an API-surface check.

All four exercise actual behavior with concrete, falsifiable assertions
(request contents, request counts, executed-tool arguments) — not "call and
check nothing crashed."

## Full battery re-run

- `bash .upstreamer/scripts/verify.sh` → `PASS: 0 failures` (uv sync, ruff
  check, ruff format, mypy, pytest tests/unit, all 31 required public-API
  symbols present, version `0.8.0` matches upstream `package.json`, no
  leaked TS artifacts, all repo-owned files present).
- `uv run pytest tests/unit tests/e2e -q` → `102 passed` (e2e tests skip
  cleanly without `OPENROUTER_API_KEY`, per contract).

## Broader sweep beyond the four claimed fixes

Re-read the full upstream delta file list
(`git -C tmp/upstreamer/upstream diff --name-status adc7939..680bceb -- packages/agent/`)
and spot-checked every source file not already covered above:

- **`conversation-state.ts` (versioned serialization, upstream #66)** — fully
  present and correct in `conversation_state.py`: `CONVERSATION_STATE_VERSION`,
  `InvalidStateError`, `UnsupportedStateVersionError`,
  `serialize_conversation_state` / `deserialize_conversation_state` with the
  same version-check-before-structural-validation ordering, same required
  fields (`id`, `messages`, `status`), same "absence of `version` means v1"
  policy. Covered by `tests/unit/test_conversation_state_serialization.py`.
- **`call-model.ts` / `async-params.ts`** — `strict_final_response` and
  `hooks` (via `resolve_hooks`) are both threaded through `call_model.py` and
  `async_params.py`'s reserved-key list, matching upstream's added fields.
- **`tool.ts` / `tool-types.ts` (the other ~85% of their diffs)** — almost
  entirely TypeScript generic-variance engineering (`TContext` → `TCtx` /
  `ContextFromSchema`, `bivarianceHack` method-syntax tricks so concretely
  typed tools stay assignable to the wide `Tool` union). Zero runtime
  behavior attached; correctly not chased in Python per Idiomatic Divergence
  #4 ("Typing is looser... do not contort the runtime to chase a type-level
  feature").
- **`reusable-stream.ts`** — adds a `get isComplete()` getter. Not part of
  the Required Public API list and not load-bearing (it's a cache-hit
  optimization signal); not found ported 1:1 in `reusable_stream.py`, but
  this is pre-existing from before this delta's scope and cosmetic (no
  behavioral test depends on it upstream either).
- **`tool-orchestrator.ts` (`executeToolLoop`)** — upstream added a `source`
  field here too, but this function is dead code in upstream itself: it is
  not exported from `index.ts` and not imported by any other upstream
  source file (only mentioned in a code *comment* in one test). The port's
  `tool_orchestrator.py` has always been (since the original 0.7.2 port, not
  this delta) a thin re-export shim (`ModelResult`, `execute_tool`,
  `partition_tool_calls`) rather than a translation of `executeToolLoop`, so
  there's no `source`-field gap to fix — the shim never had the field to
  begin with, and the real tool-execution path (`model_result.py` +
  `tool_executor.py`) does have `source` correctly threaded (see Finding 2).
  Not a new gap; pre-existing and inconsequential since the dead code isn't
  reachable either upstream or downstream.
- **Hooks system files** (`hooks-emit.ts`, `hooks-manager.ts`,
  `hooks-matchers.ts`, `hooks-resolve.ts`, `hooks-schemas.ts`,
  `hooks-types.ts`) — all six have Python counterparts
  (`hooks_emit.py`, `hooks_manager.py`, `hooks_matchers.py`,
  `hooks_resolve.py`, `hooks_schemas.py`, `hooks_types.py`). Spot-checked
  `hooks_matchers.py` against `hooks-matchers.ts` line by line: `None`/wildcard,
  exact string, compiled regex, and predicate-callable branches all match
  (Python's `re.Pattern` has no `lastIndex`-style statefulness, so the
  upstream comment about resetting `RegExp.lastIndex` has no Python
  equivalent bug to guard against — correctly omitted, not a gap).
  Session-id-per-emit threading (`hooks_manager.py:51-60`) is present and
  tested (`test_hooks_manager.py::test_session_id_threads_per_emit_for_shared_manager`).
- **SessionEnd/drain on no-tools error paths** — behaviorally verified via a
  throwaway script: a transport that raises on `send_async` for a *no-tools*
  `call_model` still fires `SessionStart` then `SessionEnd` with
  `reason: "error"` before the exception propagates (the `try/except/finally`
  wrapping the entire `_run()` body guarantees this uniformly for tool and
  no-tool paths alike). This is correct but has no dedicated committed test
  pinning it — a pre-existing coverage gap, not a regression from this delta.
- **"Tool executes exactly once per round" regression** (upstream's
  `tool-execution-once.test.ts`, guarding against a historical bug where
  `handleApprovalCheck` pre-executed auto-approve tools and the main loop
  re-executed them) — verified behaviorally via a throwaway script with a
  mixed auto-tool + approval-gated-tool round and a `PermissionRequest`
  `"allow"` decision: each tool's `execute` fired exactly once. Correct, but
  there's no dedicated Python test file pinning this specific historical
  regression by name — again a coverage gap, not a behavioral gap.
- **Approval/HITL resume ordering** (`function_call` before
  `function_call_output`) — already covered by pre-existing
  `test_parity_requirements.py::test_approval_pause_persists_tool_call_turn_and_resume_orders_output_after_call`
  and `test_hitl_pause_persists_tool_call_turn_and_resume_orders_output_after_call`,
  and by the new `test_manual_tool_pending_state.py::test_mixed_auto_and_manual_round_persists_auto_output_and_pauses_manual`
  for the specific "auto output recorded before the pause" mixed-round case
  this delta's contract calls out. No regression found.

## Documentation gap (new finding, cosmetic)

`upstreamer-changelog.md` was not updated for this 0.7.2 → 0.8.0 sync. It
still reads "Latest Sync ... against the current `@openrouter/agent` 0.7.2
surface" and has no entry for the hooks system (upstream #7/#67, explicitly
called "the 0.8.0 headline" in the contract), MCP tool branding, versioned
conversation-state serialization (upstream #66), the `allow_final_response`
default-on `tool_choice: "none"` behavior change (upstream #68), or
`awaiting_client_tools` (upstream #64) — all of which *are* correctly
reflected in `README.md` (diff confirms new "Lifecycle Hooks" section,
updated Stop Conditions section, and a new manual-tools paragraph). The
contract's Output Shape section lists `upstreamer-changelog.md` alongside
`README.md` as this repo's own product surface / user-facing port notes;
leaving it stale after a headline feature release is a real (if purely
documentation-level) gap. Does not affect behavior, the public API, or any
Required Quality in eval.md — hence a warning, not a blocker.

## Idiomatic Divergences section check

The Stop-hook zero-cost-retry semantics are not a divergence — the port now
matches upstream's exact `runStopHook`/`shouldStopExecution` loop structure
(see Finding 1), so nothing needs to be added to
`.upstreamer/upstreamer.md`'s Idiomatic Divergences section for it. No other
undocumented divergence was found in this sweep; the existing six divergences
(Pydantic v2, async-first, generator tools, looser typing, Python-native
cancellation, structural protocols for `BeforeCreateRequestContext`) still
accurately describe the only permanent deltas from upstream.

## Summary

- Public API completeness: PASS (31/31 required symbols, plus the full
  hooks/MCP-branding/versioned-state surface added in this delta).
- Version honesty: PASS (`0.8.0` == upstream `package.json` at target commit).
- Load-bearing loop: PASS, including the fixed Stop-hook zero-cost retry.
- Streaming / State / Approval-HITL ordering / Hooks / Compatibility helpers:
  PASS, no regressions found beyond the four already-fixed items.
- Repo-owned files: PASS (LICENSE, README.md, pyproject.toml, scripts/upstream
  all present and not clobbered).
- Warnings: `upstreamer-changelog.md` stale (documentation only); a few
  historical-regression behaviors (SessionEnd-on-error, exactly-once tool
  execution) are behaviorally correct but lack dedicated pinning tests; one
  test docstring (`test_stop_hook_force_resume_then_falls_through_to_normal_tool_round`)
  slightly mischaracterizes which code path runs, though its assertions are
  correct; `reusable_stream.py` lacks upstream's new `is_complete` property
  (non-load-bearing).

**Verdict: PASS WITH WARNINGS.** All three prior FAIL findings are
substantively and verifiably fixed — not just superficially patched. The
remaining items are documentation or coverage nits with no behavioral impact.
