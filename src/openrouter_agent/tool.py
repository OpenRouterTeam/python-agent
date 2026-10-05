from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from typing_extensions import TypedDict

from .tool_types import SHARED_CONTEXT_KEY, ToolType, get_tool_function, is_client_tool


_TASK_TOOL_NAME = "task"  # mirrors tool_check.TASK_TOOL_NAME (import-cycle free)


def _check_reserved_name(name: str) -> None:
    if name == SHARED_CONTEXT_KEY:
        raise ValueError('Tool name "shared" is reserved for shared context. Choose a different name.')
    if name == _TASK_TOOL_NAME:
        raise ValueError(
            f'Tool name "{_TASK_TOOL_NAME}" is reserved for the built-in task-interaction tool. '
            "Choose a different name."
        )


def tool(
    *,
    name: str,
    input_schema: Any = None,
    description: Optional[str] = None,
    execute: Any = None,
    output_schema: Any = None,
    event_schema: Any = None,
    context_schema: Any = None,
    next_turn_params: Any = None,
    require_approval: Any = None,
    on_tool_called: Any = None,
    on_response_received: Any = None,
    to_model_output: Any = None,
    strict: Optional[bool] = None,
    wire_input_schema: Optional[Mapping[str, Any]] = None,
    loop_key: Any = None,
    timeout_ms: Optional[float] = None,
    max_concurrency: Optional[int] = None,
    run: Any = None,
    lifecycle: Optional[str] = None,
    ack: Any = None,
    grace_ms: Optional[float] = None,
    poll_after_ms: Optional[int] = None,
    check: Any = None,
    log_limits: Any = None,
) -> Dict[str, Any]:
    """Create a client tool.

    Shapes (dispatch order matches upstream ``tool()``):

    - **unified** -- ``run=`` (async function or async generator) plus
      ``lifecycle="sync" | "background" | "deferred"`` (default ``"sync"``).
      Long-running lifecycles require an ``output_schema``. Deferred tools get
      ``.resolve()`` / ``.fail()`` / ``.cancel()`` completion methods (see
      `DeferredTool`).
    - **HITL** -- ``on_tool_called=`` (requires ``output_schema``).
    - **manual** -- ``execute=False``; may supply a caller-owned JSON Schema via
      ``wire_input_schema`` for wire serialization.
    - **generator** -- ``event_schema=`` plus an async-generator ``execute``.
    - **regular** -- ``execute=`` function.

    Shared options: ``strict`` (forwarded on the wire definition),
    ``loop_key`` (doom-loop call identity: a function over the validated
    arguments, a field-name list, or ``False`` to exempt), ``timeout_ms``
    (per-call deadline), ``max_concurrency`` (per-tool concurrency cap).
    """
    _check_reserved_name(name)
    fn: Dict[str, Any] = {"name": name, "input_schema": input_schema}

    def common() -> None:
        for key, value in (
            ("description", description),
            ("strict", strict),
            ("context_schema", context_schema),
            ("next_turn_params", next_turn_params),
            ("require_approval", require_approval),
            ("loop_key", loop_key),
            ("timeout_ms", timeout_ms),
            ("max_concurrency", max_concurrency),
            ("to_model_output", to_model_output),
        ):
            if value is not None:
                fn[key] = value

    if callable(run):
        resolved_lifecycle = lifecycle or "sync"
        if resolved_lifecycle not in ("sync", "background", "deferred"):
            raise ValueError(f'Tool "{name}": unknown lifecycle {resolved_lifecycle!r}')
        if resolved_lifecycle != "sync" and output_schema is None:
            raise ValueError(
                f"Tool \"{name}\" (lifecycle: '{resolved_lifecycle}') must declare an output_schema. "
                "Long-running results are validated when they settle — possibly long after the round "
                "that started them."
            )
        fn["lifecycle"] = resolved_lifecycle
        fn["run"] = run
        if output_schema is not None:
            fn["output_schema"] = output_schema
        common()
        for key, value in (
            ("event_schema", event_schema),
            ("ack", ack),
            ("grace_ms", grace_ms),
            ("poll_after_ms", poll_after_ms),
            ("check", check),
            ("log_limits", log_limits),
        ):
            if value is not None:
                fn[key] = value
        built = {"type": ToolType.Function.value, "function": fn}
        if resolved_lifecycle == "deferred":
            return DeferredTool(built)
        return built

    if output_schema is not None:
        fn["output_schema"] = output_schema
    if event_schema is not None:
        fn["event_schema"] = event_schema
    common()
    if on_tool_called is not None:
        if output_schema is None:
            raise ValueError(f'HITL tool "{name}" must declare an output_schema.')
        fn["on_tool_called"] = on_tool_called
        if on_response_received is not None:
            fn["on_response_received"] = on_response_received
    elif execute is not False:
        if execute is None:
            raise ValueError(f'Tool "{name}" must provide execute, execute=False, run, or on_tool_called.')
        fn["execute"] = execute
    elif wire_input_schema is not None:
        fn["wire_input_schema"] = wire_input_schema
    return {"type": ToolType.Function.value, "function": fn}


class DeferredTool(dict):  # type: ignore[type-arg]
    """A built ``lifecycle="deferred"`` tool: still a plain tool dict (it flows
    into ``tools=[...]`` unchanged) plus typed completion methods that route
    through `resume_tool_results` with this tool bound -- so ``output`` is
    validated against the tool's ``output_schema`` and a task id handed to an
    external system cannot settle a DIFFERENT tool's task through them.

    SECURITY: these inject values the model treats as tool results.
    Authenticate the webhook/caller before invoking them.
    """

    async def _complete(self, client: Any, request: Mapping[str, Any], entry: Dict[str, Any], options: Any) -> Any:
        from .resume_tool_results import resume_tool_results

        resume: Dict[str, Any] = {
            "state": request["state"],
            "tools": [self],
            "expect_tool_name": get_tool_function(self).get("name"),
            "results": [{"task_id": request["task_id"], **entry}],
        }
        if request.get("if_settled") is not None:
            resume["if_settled"] = request["if_settled"]
        run = request.get("run")
        if run is not None:
            run_tools = [t for t in (run.get("tools") or []) if t is not self]
            resume["run"] = {**dict(run), "tools": [self, *run_tools]}
        return await resume_tool_results(client, resume, options)

    async def resolve(self, client: Any, request: Mapping[str, Any], options: Any = None) -> Any:
        """Supply the task's successful result. With ``run`` config the
        conversation continues immediately and the `ModelResult` is returned;
        without, the result is recorded on state and ``None`` is returned."""
        return await self._complete(client, request, {"output": request.get("output")}, options)

    async def fail(self, client: Any, request: Mapping[str, Any], options: Any = None) -> Any:
        """Report the task as failed (same continue-or-record semantics)."""
        error = request.get("error")
        message = str(error) if isinstance(error, BaseException) else str(error if error is not None else "Task failed")
        return await self._complete(client, request, {"error": message}, options)

    async def cancel(self, client: Any, request: Mapping[str, Any], options: Any = None) -> Any:
        """Cancel the task (same continue-or-record semantics)."""
        reason = request.get("reason")
        return await self._complete(
            client,
            request,
            {"error": reason if isinstance(reason, str) else "Task cancelled", "status": "cancelled"},
            options,
        )


class ServerToolOptions(TypedDict, total=False):
    """Options for `server_tool` (upstream `ServerToolOptions`).

    `id` overrides the default tool-set ID (`server:{config["type"]}`), useful
    when two server tools of the same type need distinct activation IDs.
    """

    id: str


def server_tool(config: Mapping[str, Any], options: Optional[ServerToolOptions] = None) -> Dict[str, Any]:
    """Create an OpenRouter server-executed tool.

    Each server tool carries a stable tool-set ``id`` (default
    ``server:{config["type"]}``) so `ToolSet` activation APIs can address it.
    The ``id`` never reaches the wire: `convert_tools_to_api_format` sends only
    ``config``.
    """
    explicit_id = options.get("id") if options is not None else None
    if explicit_id == "":
        raise ValueError("Server tool ID must not be empty")
    tool_id = explicit_id if explicit_id is not None else f"server:{config.get('type')}"
    return {"_brand": "server-tool", "config": dict(config), "id": tool_id}


def get_server_tool_id(server: Mapping[str, Any]) -> str:
    """Stable tool-set ID of a server tool.

    Mirrors upstream `defaultServerId` in `tool-set.ts`: the explicit ``id``
    when it is a non-empty string, otherwise the synthesized
    ``server:{config["type"]}`` (covers hand-written server tools that predate
    the ``id`` field).
    """
    tool_id = server.get("id")
    if isinstance(tool_id, str) and len(tool_id) > 0:
        return tool_id
    config = server.get("config")
    tool_type = config.get("type") if isinstance(config, Mapping) else None
    return f"server:{tool_type}"


def mark_mcp(tool_to_mark: Dict[str, Any], options: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Add the additive MCP brand to an already-built client tool (see
    `is_mcp_tool`). Non-mutating: returns a shallow copy carrying `_mcp`, so
    the tool's runtime behavior and wire shape are unchanged -- only the
    `is_mcp_tool` check now identifies it as MCP-originated. Used by
    `@openrouter/mcp`-equivalent integrations to mark wrapped remote tools.

    ``options["loop_key"]`` attaches a doom-loop identity to the wrapped tool
    -- the only injection point for MCP tools, whose remote definitions cannot
    carry client-side functions.
    """
    marked = {**tool_to_mark, "_mcp": True}
    loop_key = (options or {}).get("loop_key")
    if loop_key is not None and is_client_tool(marked):
        marked["function"] = {**get_tool_function(marked), "loop_key": loop_key}
    return marked


def _attach_agent_builder() -> None:
    from .agent_tool import agent_tool

    setattr(tool, "agent", agent_tool)


_attach_agent_builder()
