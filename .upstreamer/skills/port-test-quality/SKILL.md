---
name: port-test-quality
description: Port upstream tests and keep the Python suite an honest check on upstream behavior. Use when a sync adds or changes tests, when coverage drops below the floor, when writing a test for newly ported behavior, or when reviewing whether a port's tests actually assert parity.
---

# Port Test Quality

Execution discipline for the test half of the port. The binding rules live in the
**Test Parity** section of `.upstreamer/upstreamer.md`; this skill is how to
satisfy them. If the two conflict, the contract wins — report the conflict.

## Why this exists

A port can compile, export every required symbol, pass its own test suite, and
still be a version behind on real behavior. The suite is what makes that failure
mode visible or invisible, so tests are part of the port, not a follow-up.

The concrete history: this repo once had 15 test files against upstream's 46 at the
same commit. All 104 tests passed. Three upstream invariants had no counterpart —
`turn.end` dropped by a broadcaster race, a tool executing twice per round, and a
mixed auto+manual round sending an orphaned `function_call` that providers reject
with a 400. Every one of those bugs could have been present with CI fully green.

## Step 1: Diff the suites before writing anything

Never infer coverage. Enumerate it:

```bash
ls tmp/upstreamer/upstream/packages/agent/tests/unit/*.test.ts \
  | sed 's|.*/||;s|\.test\.ts$||;s|-|_|g' | sort > /tmp/up.txt
ls tests/unit/test_*.py | sed 's|.*/test_||;s|\.py$||' | sort > /tmp/port.txt
comm -23 /tmp/up.txt /tmp/port.txt   # upstream tests with no Python counterpart
```

Also diff the delta's test changes directly, since a *changed* upstream test is as
load-bearing as a new one:

```bash
git -C tmp/upstreamer/upstream diff <last>..<target> -- packages/agent/tests
```

Maintain the 1:1 mapping: `foo-bar.test.ts` → `tests/unit/test_foo_bar.py`. That
mapping is what makes this diff meaningful; breaking it hides gaps.

## Step 2: Triage by what breaks in production

Port highest-severity first. Rank by blast radius, not by file size:

| Severity | Area | Why |
| --- | --- | --- |
| HIGH | Tool loop: exactly-once execution, mixed auto+manual rounds, turn boundaries | Double-executes a side-effecting tool, or emits a request the provider rejects |
| HIGH | Approval / HITL ordering | Auto-tool output lost before a pause; wrong resume order |
| HIGH | State serialization, version mismatch | Silent data corruption across a resume |
| MEDIUM | Hooks firing/ordering, session-id threading, `SessionEnd` on error paths | Telemetry and permission gates silently stop working |
| MEDIUM | Streaming fan-out, error propagation to every consumer | A consumer hangs or ends silently |
| LOW | Compat round-trips, schema sanitization breadth, type-level tests | Recoverable, usually a rejected request |

Report anything you leave unported, with its invariant and severity. An
unreported gap is how the next run skips past it.

## Step 3: Write the test against upstream behavior

Read the upstream test and port **the invariant**, not the syntax.

**Assert order and count, never mere membership.**

```python
# NO — passes if turn.end fires twice, or before turn.start
assert "turn.end" in [event["type"] for event in events]

# YES
assert types.count("turn.end") == 1
assert types.index("turn.start") < types.index("turn.end")
```

**Assert exactly-once, not that-it-happened.** Use a counter object, not
`lambda p, c: calls.append(...) or {...}` — `list.append` returns `None`, so the
lambda both trips mypy's `func-returns-value` and can silently return the wrong
tool output. See `tests/unit/test_tool_execution_once.py`.

**Assert request counts.** "No follow-up request was sent" is a real invariant;
`len(client.requests) == 1` is how you state it. `QueuedResponses` raises a
descriptive `AssertionError` on queue exhaustion precisely so an unexpected extra
turn names itself.

**Rejectable patterns** — each looks like coverage and is not:

- `assert x is not None` as a test's only assertion (fine as a mypy-narrowing line
  before a real one)
- `assert len(xs) > 0` where the invariant is *which* items are present
- `assert v is None if "k" in d else True` — `assert True` on the missing branch
- Re-asserting a stub's own canned data, or that a type guard accepts an object
  built with that guard's marker

## Step 4: Use the shared fixtures

Import from `tests/_fixtures.py`; do not hand-roll fake clients or partial
response dicts.

```python
from tests._fixtures import QueuedClient, function_call_item, make_response, text_response
```

`make_response` populates every field the real Responses API returns (`status`,
`model`, `object`, `created_at`, `tool_choice`, …). Partial stubs are how a port
passes its own suite while mishandling production payloads — a stub must never be
more forgiving than the API.

Builders emit upstream's **camelCase** wire shape (`callId`) because that is what
the port's internals consume. `assert_matches_sdk_response_shape` converts and
validates against the generated SDK's model, so a required-field change in the SDK
fails loudly instead of drifting.

If a test needs bespoke transport behavior (error injection, streaming, conditional
branching), keep the bespoke class but build its payloads with these builders.

## Step 5: Keep it deterministic

- **Never port a timing race as `asyncio.sleep`.** Upstream races a 20ms stream
  tick against a 5ms executor; as wall-clock timing it flips under CI load. Gate
  on `asyncio.Event` so ordering is explicit — see
  `tests/unit/test_turn_end_race_condition.py`.
- **Construct `asyncio` primitives inside the async test body.**
  `asyncio.Condition()` binds the running loop eagerly on 3.9 and lazily on 3.13,
  so module-scope construction fails on 3.9 only — and CI now tests 3.9.
- **Do not assert on pending-task counts.** `ToolEventBroadcaster._wake()` fires
  `create_task(self._notify())` and never awaits it, leaving orphan tasks by
  design.
- Prove it: `for i in $(seq 1 50); do uv run pytest <file> -q || break; done`.

## Step 6: Prefer the public API over private internals

Upstream tests sometimes cast to an internal type and call a private method. Where
this port's internals differ, drive the same invariant through `call_model`.

Known case: upstream calls `executeToolsIfNeeded()`; this port has no such method
— the tool loop is inlined in `ModelResult._run`. Driving `call_model` gives the
identical assertion and survives the next sync's refactors.

## Step 7: Document deliberate divergences at the assertion

Where the port must assert differently from upstream, say why with a file
reference, or a later reader "fixes" it back:

```python
# Upstream asserts `callId`. This port emits snake_case `call_id`: _send
# normalizes at the transport boundary (model_result.py:148-156). Deliberate —
# do not change back to callId, it will fail.
assert outputs[0]["call_id"] == "call_auto_1"
```

If you derive a new divergence, add it to the contract's Test Parity section as
well as the assertion.

## Step 8: Verify before handing off

```bash
uv run pytest tests/unit -q                                    # all green
uv run pytest tests/unit -q --cov --cov-fail-under=<floor>     # floor holds
uv run mypy src tests                                          # tests are checked too
uv run ruff format . && uv run ruff check .
.upstreamer/scripts/verify.sh
```

The coverage floor in `.github/workflows/ci.yaml` is a **ratchet**: raise it when
coverage rises, never lower it. If newly ported source drops coverage below the
floor, the missing tests are part of the port — write them rather than lowering the
number.

## Report

In the final port report, include: upstream test files ported; upstream test files
**not** ported with invariant and severity; new tests added and the invariant each
pins; any new divergence documented; coverage before and after; and whether the
floor moved.
