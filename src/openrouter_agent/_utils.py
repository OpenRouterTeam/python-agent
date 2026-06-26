from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterable, Iterable, Mapping
from typing import Any, Callable, Dict, List

from pydantic import BaseModel, TypeAdapter


def get_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def set_field(value: Any, name: str, new_value: Any) -> Any:
    if isinstance(value, dict):
        copied = dict(value)
        copied[name] = new_value
        return copied
    try:
        copied = value.model_copy(update={name: new_value})
        return copied
    except Exception:
        setattr(value, name, new_value)
        return value


def dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, exclude_none=True)
    if hasattr(value, "dict"):
        return value.dict()
    return value


def to_snake(name: str) -> str:
    out: List[str] = []
    for ch in name:
        if ch.isupper() and out:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


_CAMEL_MAP = {
    "maxOutputTokens": "max_output_tokens",
    "toolChoice": "tool_choice",
    "parallelToolCalls": "parallel_tool_calls",
    "previousResponseId": "previous_response_id",
    "topP": "top_p",
    "topK": "top_k",
    "frequencyPenalty": "frequency_penalty",
    "presencePenalty": "presence_penalty",
    "promptCacheKey": "prompt_cache_key",
    "serviceTier": "service_tier",
    "safetyIdentifier": "safety_identifier",
    "maxToolCalls": "max_tool_calls",
}


def sdk_request_kwargs(request: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in request.items():
        if value is None:
            result[_CAMEL_MAP.get(key, key)] = value
        elif key in _CAMEL_MAP:
            result[_CAMEL_MAP[key]] = value
        else:
            result[key] = value
    return result


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def is_async_iterable(value: Any) -> bool:
    return hasattr(value, "__aiter__")


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def json_loads_maybe(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def json_dumps(value: Any) -> str:
    return json.dumps(dump(value), separators=(",", ":"), ensure_ascii=False)


def sanitize_json_schema(schema: Any) -> Any:
    if isinstance(schema, list):
        return [sanitize_json_schema(item) for item in schema]
    if isinstance(schema, dict):
        return {k: sanitize_json_schema(v) for k, v in schema.items() if not str(k).startswith("~")}
    return schema


def schema_to_json_schema(schema: Any) -> Dict[str, Any]:
    if schema is None:
        return {"type": "object", "properties": {}}
    if isinstance(schema, dict):
        return sanitize_json_schema(schema)
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return sanitize_json_schema(schema.model_json_schema())
    if hasattr(schema, "json_schema"):
        return sanitize_json_schema(schema.json_schema())
    if hasattr(schema, "model_json_schema"):
        return sanitize_json_schema(schema.model_json_schema())
    if isinstance(schema, TypeAdapter):
        return sanitize_json_schema(schema.json_schema())
    return {"type": "object", "properties": {}}


def validate_schema(schema: Any, value: Any) -> Any:
    if schema is None:
        return value
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_validate(value)
    if hasattr(schema, "validate_python"):
        return schema.validate_python(value)
    if hasattr(schema, "model_validate"):
        return schema.model_validate(value)
    if callable(schema) and not isinstance(schema, dict):
        return schema(value)
    return value


def is_content_array(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(item, Mapping) and item.get("type") in {"input_text", "input_image", "input_file"}
            for item in value
        )
    )


async def collect_async_iterable(value: AsyncIterable[Any]) -> List[Any]:
    return [item async for item in value]


def iter_sync(value: Iterable[Any]) -> Iterable[Any]:
    return value


def callable_name(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__name__", fn.__class__.__name__)
