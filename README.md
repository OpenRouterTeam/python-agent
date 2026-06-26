# openrouter-agent

`openrouter-agent` is a Python agent toolkit for OpenRouter. It ports the public behavior of `@openrouter/agent` into an async-first Python package: Responses API calls, client and server tools, streaming consumption, multi-turn state, approval and human-in-the-loop gates, tool context, stop conditions, and Claude/OpenAI Chat format compatibility.

This package builds on the official `openrouter` Python SDK. It does not reimplement HTTP, auth, retries, or model schemas; `call_model` sends requests through `client.beta.responses.send_async`, the same Responses API surface used by the TypeScript package.

## Install

```bash
pip install openrouter-agent
# or
uv add openrouter-agent
```

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

Manual tools are advertised to the model but not auto-executed.

```python
confirm = tool(name="confirm_action", input_schema=ConfirmInput, execute=False)
```

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

## Stop Conditions

The built-ins mirror the TypeScript package and OR together when provided as a list:

- `step_count_is(n)`
- `has_tool_call(name)`
- `max_tokens_used(n)`
- `max_cost(dollars)`
- `finish_reason_is(reason)`

Set `allow_final_response=True` or a string to ask for one final no-tools turn when a stop condition fires on a tool-call turn.

## Format Compatibility

Use `from_claude_messages` / `to_claude_message` for Anthropic-style messages and `from_chat_messages` / `to_chat_message` for OpenAI Chat-style messages. Content that cannot be represented directly is carried as `unsupported_content` instead of being silently discarded.

## Parity Notes

This is a faithful Python port of the `@openrouter/agent` public surface, with Python-native names (`call_model`, `server_tool`, `get_text_stream`) and Pydantic v2 schemas in place of Zod. Runtime behavior is preserved where the Python SDK exposes matching Responses API types. Static type inference is necessarily looser than TypeScript conditional types; the package ships `py.typed`, `Protocol`/dataclass aliases, and clear runtime validation rather than pretending to reproduce TypeScript tuple inference exactly.
