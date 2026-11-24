# openrouter-agent-sdk

`openrouter-agent-sdk` is a Python agent toolkit for OpenRouter. It ports the public behavior of `@openrouter/agent` into an async-first Python package: Responses API calls, client and server tools, streaming consumption, multi-turn state, approval and human-in-the-loop gates, tool context, stop conditions, and Claude/OpenAI Chat format compatibility.

This package builds on the official `openrouter` Python SDK. It does not reimplement HTTP, auth, retries, or model schemas; `call_model` sends requests through `client.beta.responses.send_async`, the same Responses API surface used by the TypeScript package.

> **This package is a port.** `@openrouter/agent` (TypeScript) is the reference
> spec; this repo is kept in sync automatically. See [PORTING.md](PORTING.md).

## Install

```bash
pip install openrouter-agent-sdk
# or
uv add openrouter-agent-sdk
```

The distribution is `openrouter-agent-sdk`; the import is `openrouter_agent`:

```python
from openrouter_agent import call_model, tool
```

Requires Python 3.10+ (the `openrouter` SDK dropped 3.9 at 1.0.0).

## Quick Start

```python
import asyncio
from pydantic import BaseModel
from openrouter_agent import OpenRouter, call_model, tool

class WeatherInput(BaseModel):
    location: str

class WeatherOutput(BaseModel):
    temperature: int
    condition: str
    location: str

async def main() -> None:
    client = OpenRouter(api_key="YOUR_API_KEY")

    weather_tool = tool(
        name="get_weather",
        description="Get the current weather for a location",
        input_schema=WeatherInput,
        output_schema=WeatherOutput,
        execute=lambda params, ctx: WeatherOutput(
            temperature=72,
            condition="sunny",
            location=params.location,
        ),
    )

    result = call_model(
        client,
        {
            "model": "openai/gpt-4o",
            "input": "What is the weather in San Francisco?",
            "tools": [weather_tool],
        },
    )

    print(await result.get_text())

asyncio.run(main())
```

## Streaming

`call_model` returns a `ModelResult`. The result can be consumed multiple ways; each method works from the same completed run so concurrent consumers do not steal events from each other.

```python
result = call_model(client, {"model": model, "input": prompt, "tools": tools})

text = await result.get_text()
response = await result.get_response()

async for delta in result.get_text_stream():
    print(delta, end="")

async for delta in result.get_reasoning_stream():
    print(delta, end="")

async for event in result.get_tool_stream():
    print(event)

async for call in result.get_tool_calls_stream():
    print(call.name, call.arguments)
```

## Tool Variants

Regular tools execute automatically when the model emits a matching `function_call`.

```python
search = tool(
    name="search",
    input_schema=SearchInput,
    output_schema=SearchOutput,
    execute=run_search,
)
```

Generator tools yield progress events and a final output. Python async generators cannot return a final value, so the final yield is treated as the output and earlier yields are preliminary events.

```python
analysis = tool(
    name="analyze",
    input_schema=AnalysisInput,
    event_schema=ProgressEvent,
    output_schema=AnalysisOutput,
    execute=analyze_stream,
)
```

Manual tools are advertised to the model but not auto-executed. A manual tool may pass `wire_input_schema=` to send its own JSON Schema to the model instead of one generated from `input_schema`.

```python
confirm = tool(name="confirm_action", input_schema=ConfirmInput, execute=False)
```

Any client tool accepts `strict=True` / `strict=False` to request provider-enforced schema adherence on tool-call arguments; it is sent on the wire tool definition (omitted means `null`, the provider default). Pass `active_tools=[...names]` to `call_model` to send and execute only a subset of `tools` (server tools are always kept).

HITL tools use `on_tool_called`; returning a value auto-resolves, returning `None` pauses with pending tool calls in state.

```python
approve_wire = tool(
    name="approve_wire",
    input_schema=WireInput,
    output_schema=WireDecision,
    on_tool_called=ask_human,
)
```

Server tools pass through to OpenRouter unchanged.

```python
from openrouter_agent import server_tool

web = server_tool({"type": "web_search_2025_08_26", "max_results": 5})
```

## Approval, Context, and State

Use `require_approval` on a tool or `require_approval` on the request to pause sensitive calls before execution. Approval resume requires a state accessor with async `load()` and `save()` methods.

Resume with `approve_tool_calls` / `reject_tool_calls`. Duplicate IDs are ignored (each approved call runs once), and an ID in both lists raises before anything runs. A paused conversation stays paused until every pending call has a decision: resuming without decisions (or with decisions for only some calls) sends no model request, and `get_response()`, `get_text()`, `get_tool_calls()` and the stream getters raise an error naming the pause and the pending call IDs, while `requires_approval()`, `get_pending_tool_calls()` and `get_state()` keep working. An `awaiting_hitl` conversation also resumes when `input` carries a `function_call_output` for a pending call.

Manual tools (`execute=False`, no `on_tool_called`) pause the loop with status `"awaiting_client_tools"` when the model calls them, instead of silently dropping the call. Read the unresolved calls via `get_pending_tool_calls()` / `get_state()`, execute them yourself, and continue by calling `call_model` again with `function_call_output` items in `input`.

For durable cross-process storage, serialize state with `serialize_conversation_state` / `deserialize_conversation_state` rather than storing raw dataclass fields. The wire format is versioned (`CONVERSATION_STATE_VERSION`); a version mismatch raises `UnsupportedStateVersionError` and malformed JSON raises `InvalidStateError`, so a store can never silently misinterpret a future shape.

Tool context is kept outside the model transcript. Provide a context mapping with per-tool keys and optional `shared` state. Tool execution receives `ctx["local"]`, `ctx["shared"]`, `ctx["set_context"]`, and `ctx["set_shared_context"]`.

```python
result = call_model(
    client,
    {
        "model": model,
        "input": "List all users",
        "tools": [query_db],
        "context": {"query_db": {"connection_string": "postgres://localhost/app"}},
        "stop_when": step_count_is(5),
    },
)
```

## Usage

`get_response()` returns the *final* round's response, so its `usage` covers one generation. `await result.get_usage()` returns totals across every model call the run made (`model_calls`, `input_tokens`, `output_tokens`, `total_tokens`, `cached_tokens`, `reasoning_tokens`, and `cost` when reported). It waits for the run and never raises: a failed or paused run returns what was accrued so far.

## Lifecycle Hooks

Pass a `HooksManager` (or an inline `{hook_name: [HookEntry(...)]}` dict of built-in hooks) via `hooks=` to observe or intervene in a run: `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `UserPromptSubmit`, `Stop`, `PermissionRequest`, `SessionStart`, `SessionEnd`, and `PostModelCall`. Handlers receive a validated payload dict and a `LifecycleHookContext` (`session_id`, `hook_name`, `cancel_event`).

```python
from openrouter_agent import HookEntry, HookName, HooksManager

hooks = HooksManager()
hooks.on(HookName.PreToolUse.value, HookEntry(handler=lambda payload, ctx: None, matcher="delete_file"))
hooks.on(HookName.SessionEnd.value, HookEntry(handler=lambda payload, ctx: print(payload["total_usage"])))

result = call_model(client, {"model": model, "input": prompt, "tools": tools, "hooks": hooks})
```

`SessionStart` fires once per run with a config summary; `SessionEnd` fires once with aggregated `total_usage` (summed across every `PostModelCall`) and is guaranteed to fire — and any pending async hook work drained — even when a no-tools stream raises. `PreToolUse` can block a call (`{"block": "reason"}`) or mutate its input (`{"mutated_input": {...}}`); `PermissionRequest` can pre-empt the approval gate with `{"decision": "allow" | "deny" | "ask_user"}`; `Stop` can force the loop to keep going past a `stop_when` hit with `{"force_resume": True, "append_prompt": "..."}`. A `HooksManager` instance is safe to share across concurrent `call_model` runs — session identity is threaded per emit, not stored as manager-level mutable state.

## Stop Conditions

The built-ins mirror the TypeScript package and OR together when provided as a list:

- `step_count_is(n)`
- `has_tool_call(name)`
- `max_tokens_used(n)`
- `max_cost(dollars)`
- `finish_reason_is(reason)`

A forced `tool_choice` (`"required"`, a specific tool, or `allowed_tools` with `mode: "required"`) is relaxed to `"auto"` once it has produced a tool call in a run (including across approval and client-tool resumes), so the model can answer in text instead of calling tools until the step budget runs out. A tool's `next_turn_params` may set `tool_choice` for the next turn (for example, to widen an `allowed_tools` set without touching `tools`); `next_turn_params` only runs for calls whose tool body ran and returned without error. A response truncated at `max_output_tokens` executes none of its tool calls; the run ends with that `incomplete` response.

When a stop condition fires while the model is still emitting tool calls, `call_model` makes one more turn with `tool_choice="none"` by default (tools stay in the request so the prompt-cache prefix survives) so the run ends with a natural-language answer. `allow_final_response` tunes this: `True` or omitted appends `DEFAULT_FINAL_RESPONSE_DIRECTIVE` as a user message, a non-empty string replaces the wording, `""` appends nothing, and `False` disables the extra turn entirely.

## Format Compatibility

Use `from_claude_messages` / `to_claude_message` for Anthropic-style messages and `from_chat_messages` / `to_chat_message` for OpenAI Chat-style messages. Content that cannot be represented directly is carried as `unsupported_content` instead of being silently discarded.

## Development

```bash
uv sync --frozen --all-extras

uv run pytest tests/unit -q              # deterministic suite
uv run pytest tests/unit --cov           # with coverage
uv run mypy src tests                    # types, tests included
uv run ruff check . && uv run ruff format --check .
```

`tests/e2e/` runs against the live OpenRouter API and skips cleanly without
`OPENROUTER_API_KEY`:

```bash
OPENROUTER_API_KEY=sk-or-... uv run pytest tests/e2e -q
```

Tests share fixtures from `tests/_fixtures.py` — `make_response`,
`function_call_item`, `text_response`, `tool_call_response`, `QueuedClient`,
`MemoryStateAccessor`. Use them instead of hand-rolling a fake client:
`make_response` populates every field the real Responses API returns, so a stub
cannot be more permissive than production.

CI runs the suite on Python 3.10, 3.11, and 3.13, type-checks `src` and `tests`,
enforces a coverage floor, and verifies the built wheel imports in isolation.
Because this package is a port, tests are held to upstream behavior — see the
Test Parity section of `.upstreamer/upstreamer.md` and [PORTING.md](PORTING.md).

## Parity Notes

This is a faithful Python port of the `@openrouter/agent` public surface, with Python-native names (`call_model`, `server_tool`, `get_text_stream`) and Pydantic v2 schemas in place of Zod. Runtime behavior is preserved where the Python SDK exposes matching Responses API types. Static type inference is necessarily looser than TypeScript conditional types; the package ships `py.typed`, `Protocol`/dataclass aliases, and clear runtime validation rather than pretending to reproduce TypeScript tuple inference exactly.
