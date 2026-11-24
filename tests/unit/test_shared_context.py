"""Ported from upstream tests/unit/shared-context.test.ts, region "context
setters store schema-validated values" (#161).

`set_context` / `set_shared_context` store the schema's parsed output for each
updated field and validate the merged context against the full schema before
storing it, so model-level validators are enforced at write time. An update
that fails validation raises and leaves the stored context unchanged.
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from openrouter_agent import ToolContextStore, tool
from openrouter_agent.tool_context import build_tool_execute_context
from openrouter_agent.tool_types import SHARED_CONTEXT_KEY

_TURN = {"number_of_turns": 1}


def _my_tool(context_schema: object = None) -> dict:
    return tool(name="my_tool", input_schema=dict, execute=lambda _p, _c: None, context_schema=context_schema)


def test_set_shared_context_stores_the_parsed_output_of_field_transforms() -> None:
    class Shared(BaseModel):
        level: int

        @field_validator("level")
        @classmethod
        def _clamp(cls, value: int) -> int:
            return min(value, 0)

    store = ToolContextStore({SHARED_CONTEXT_KEY: {"level": 0}})
    ctx = build_tool_execute_context(_my_tool(), _TURN, store, Shared)

    ctx["set_shared_context"]({"level": 1})

    assert dict(ctx["shared"]) == {"level": 0}


def test_set_context_stores_the_parsed_output_of_field_transforms() -> None:
    class Local(BaseModel):
        mode: str

        @field_validator("mode")
        @classmethod
        def _normalize(cls, value: str) -> str:
            return value.strip().lower()

    store = ToolContextStore({"my_tool": {"mode": "read"}})
    ctx = build_tool_execute_context(_my_tool(Local), _TURN, store)

    ctx["set_context"]({"mode": "  WRITE "})

    assert dict(ctx["local"]) == {"mode": "write"}


def test_set_shared_context_rejects_an_update_that_breaks_a_model_level_validator() -> None:
    class Shared(BaseModel):
        role: Literal["guest", "staff"]
        access: Literal["read", "write"]

        @model_validator(mode="after")
        def _staff_only_write(self) -> "Shared":
            if self.access == "write" and self.role != "staff":
                raise ValueError("write access requires staff")
            return self

    initial = {"role": "guest", "access": "read"}
    store = ToolContextStore({SHARED_CONTEXT_KEY: initial})
    ctx = build_tool_execute_context(_my_tool(), _TURN, store, Shared)

    with pytest.raises(ValidationError):
        ctx["set_shared_context"]({"access": "write"})
    assert dict(ctx["shared"]) == initial

    ctx["set_shared_context"]({"role": "staff"})
    ctx["set_shared_context"]({"access": "write"})
    assert dict(ctx["shared"]) == {"role": "staff", "access": "write"}


def test_unknown_keys_are_dropped_and_only_updated_fields_are_stored() -> None:
    class Local(BaseModel):
        mode: str = "read"
        retries: int = 3

    store = ToolContextStore({"my_tool": {}})
    ctx = build_tool_execute_context(_my_tool(Local), _TURN, store)

    ctx["set_context"]({"mode": "write", "unknown": True})

    # Defaults for fields the update did not touch are not written back.
    assert dict(ctx["local"]) == {"mode": "write"}
