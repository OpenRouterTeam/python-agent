"""Port of upstream `tests/unit/hooks-post-model-call.test.ts`.

Only the case added by the 680bceb..50a1ec2 sync delta is ported here
("emits when the source close arrives after the terminal event"). The rest of
the upstream file's PostModelCall/SessionEnd invariants are covered in
`tests/unit/test_model_result_hooks.py` (once-per-turn labels, SessionEnd
usage aggregation, SessionEnd on transport failure).
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Tuple, cast


from openrouter_agent import HookEntry, HookName, HooksManager, call_model
from tests._fixtures import QueuedClient, text_response, usage_block


def collect_hooks() -> Tuple[HooksManager, List[Dict[str, Any]], List[Dict[str, Any]]]:
    hooks = HooksManager()
    calls: List[Dict[str, Any]] = []
    ends: List[Dict[str, Any]] = []

    def on_call(payload: Dict[str, Any], ctx: Any) -> None:
        calls.append(payload)

    def on_end(payload: Dict[str, Any], ctx: Any) -> None:
        ends.append(payload)

    hooks.on(HookName.PostModelCall.value, HookEntry(handler=on_call))
    hooks.on(HookName.SessionEnd.value, HookEntry(handler=on_end))
    return hooks, calls, ends


async def test_emits_when_the_source_close_arrives_after_the_terminal_event() -> None:
    never_closes = asyncio.Event()
    streamed = text_response("r_late_close", "hello back", usage=usage_block())

    async def late_close_stream() -> AsyncIterator[Dict[str, Any]]:
        yield {"type": "response.completed", "response": streamed, "sequence_number": 0}
        # No close: the source keeps the consumer waiting forever.
        await never_closes.wait()

    # Bespoke transport: a streaming body whose payload is still built by the
    # shared builders.
    client = QueuedClient([cast(Dict[str, Any], late_close_stream())])
    hooks, calls, ends = collect_hooks()
    result = call_model(client, {"model": "test-model", "input": "hi", "hooks": hooks})

    async def drain() -> None:
        async for _item in result.get_items_stream():
            pass

    try:
        # Not a timing race: the outcome is binary (terminates vs. blocks
        # forever). The bound only converts a permanent hang into a failure.
        await asyncio.wait_for(drain(), timeout=2.0)
    finally:
        result.cancel()

    assert len(client.requests) == 1
    assert len(calls) == 1
    assert calls[0]["response_id"] == "r_late_close"
    assert calls[0]["turn_type"] == "initial"
    assert len(ends) == 1
    assert ends[0]["total_usage"]["model_calls"] == 1
    assert ends[0]["total_usage"]["total_tokens"] == 150
