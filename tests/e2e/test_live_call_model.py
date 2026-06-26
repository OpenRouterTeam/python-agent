from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY is required for OpenRouter e2e tests",
)


async def test_live_call_model_smoke() -> None:
    from openrouter_agent import OpenRouter, call_model

    client = OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])
    result = call_model(client, {"model": "openai/gpt-4o-mini", "input": "Reply with the word pong."})
    text = await result.get_text()
    assert "pong" in text.lower()
