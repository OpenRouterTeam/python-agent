# openrouter-agent-sdk

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/openrouter-agent-sdk.svg)](https://pypi.org/project/openrouter-agent-sdk/)

Typed tool definitions, multi-consumer streaming, and conversation state
management for the OpenRouter Responses API.

## Installation

```bash
pip install openrouter-agent-sdk
```

## Quickstart

```python
import asyncio

from openrouter import OpenRouter
from openrouter_agent import call_model, tool
from pydantic import BaseModel


client = OpenRouter(api_key="...")


class AddInput(BaseModel):
    a: float
    b: float


class AddOutput(BaseModel):
    result: float


add = tool(
    name="add",
    description="Add two numbers",
    input_schema=AddInput,
    output_schema=AddOutput,
    execute=lambda params, ctx: AddOutput(result=params.a + params.b),
)


async def main():
    result = await call_model(client, {
        "model": "openai/gpt-4o",
        "input": "What is 2 + 3?",
        "tools": [add],
    })
    print(result.get_text())


asyncio.run(main())
```

## Streaming

`ModelResult` supports several concurrent consumption patterns from the same
underlying response.

### Text stream

```python
result = await call_model(client, {
    "model": "openai/gpt-4o",
    "input": "Explain quantum computing briefly.",
})

async for delta in result.get_text_stream():
    print(delta, end="", flush=True)
```

### Full response stream

Receive every event (API responses, tool events, turn lifecycle):

```python
async for event in result.get_full_responses_stream():
    print(event)
```

## Stop conditions

Control when the agentic loop terminates by passing one or more stop
conditions in the request.

```python
from openrouter_agent import (
    call_model,
    step_count_is,
    has_tool_call,
    max_tokens_used,
    max_cost,
    finish_reason_is,
)

result = await call_model(client, {
    "model": "openai/gpt-4o",
    "input": "Research this topic thoroughly.",
    "tools": [my_tool],
    "stop_when": [
        step_count_is(5),          # stop after 5 steps
        has_tool_call("done"),     # stop when the "done" tool is called
        max_tokens_used(10_000),   # stop after 10k tokens
    ],
})
```

Available stop conditions:

| Factory | Triggers when |
|---|---|
| `step_count_is(n)` | The loop has executed `n` steps |
| `has_tool_call(name)` | A tool with the given name has been called |
| `max_tokens_used(n)` | Cumulative token usage reaches `n` |
| `max_cost(dollars)` | Estimated cost reaches the given amount |
| `finish_reason_is(reason)` | The model's finish reason matches |

## Format compatibility

Convert existing message histories from Claude or OpenAI Chat format into
the OpenResponses input format used by the SDK.

```python
from openrouter_agent import from_claude_messages, from_chat_messages

# From Claude/Anthropic format
items = from_claude_messages(claude_messages)

# From OpenAI Chat format
items = from_chat_messages(chat_messages)

result = await call_model(client, {
    "model": "openai/gpt-4o",
    "input": items,
})
```

## Tool approval

Tools can require human approval before execution:

```python
guarded = tool(
    name="delete_file",
    description="Delete a file from disk",
    input_schema=DeleteInput,
    execute=delete_file_impl,
    require_approval=True,
)
```

Or use a custom approval function:

```python
async def check_approval(tool_call, context):
    return tool_call.name != "dangerous_operation"

result = await call_model(client, {
    "model": "openai/gpt-4o",
    "input": "Clean up old files.",
    "tools": [guarded],
    "require_approval": check_approval,
})
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Type check
mypy
```

## License

[MIT](LICENSE)
