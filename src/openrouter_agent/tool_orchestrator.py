"""Tool orchestration re-exports.

Ported for 1:1 module parity with upstream `lib/tool-orchestrator.ts`, which is
itself unreferenced upstream. This port drives tool execution from the inlined
loop in `ModelResult._run`.

Kept deliberately: the porting contract (`.upstreamer/upstreamer.md`) requires one
Python module per upstream lib module, so deleting this would be a parity
regression that the next sync re-creates. Excluded from the coverage floor in
`pyproject.toml` rather than deleted. Do not re-litigate.
"""

from __future__ import annotations

from .model_result import ModelResult
from .tool_executor import execute_tool
from .conversation_state import partition_tool_calls

__all__ = ["ModelResult", "execute_tool", "partition_tool_calls"]
