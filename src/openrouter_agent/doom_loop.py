"""Doom-loop detection for the tool-execution loop.

Port of upstream ``lib/doom-loop.ts`` (``@openrouter/agent``).

A "doom loop" is a run that stops making progress while continuing to spend:
the model re-issues the same tool call with identical arguments (including
repeated *empty* calls and repeated unparseable calls), or emits the same
text tokens over and over. This module provides the deterministic detection
primitives; ``ModelResult`` wires them into the loop behind the ``doom_loop``
option on ``call_model``.

Design principles (verbatim from upstream, see the TS module docstring):

- **Deterministic.** Every verdict is a pure function of the recorded
  sequence of tool calls / assistant texts. No wall clocks, no randomness.
- **Cross-port fingerprints.** Key material is canonicalized per RFC 8785
  (JCS) and hashed with SHA-256 over the UTF-8 bytes. This port does NOT use
  ``json.dumps`` for numbers (``json.dumps(-0.0)`` is ``-0.0``, which is not
  JCS); it implements the ECMAScript ``Number::toString`` / ``JSON.stringify``
  serialization so canonical strings are byte-identical to the TypeScript
  reference. ``tests/vectors/doom_loop_fingerprints.json`` (copied from
  upstream) is the conformance contract.
- **Tool-declared identity.** A tool opts into precise loop identity via
  ``loop_key``: a function computing key material, a field-name list, or
  ``False`` to exempt the tool.
- **Round-scoped streaks.** A round's identity for one tool is the *set* of
  fingerprints it was called with; declare the set before scoring any of its
  calls (see :meth:`DoomLoopMonitor.declare_round`).
- **Graduated response.** observe -> steer -> escalate -> block -> stop.
- **Never the cause of failure.** Internal errors degrade to a fallback
  identity or skip detection for that call.

Python-specific notes (deliberate divergences, documented here once):

- Every exported camelCase name maps to snake_case; classes keep their names.
  Dict keys (verdicts, records, config, serialized state) are snake_case
  (``tool_name``, ``duplicate_in_round``, ``round_fingerprints``,
  ``call_streaks``, ``escalations_used``, ``stop_verdict``, ``pending_steer``),
  matching the snake_case ``ConversationState``.
- Fingerprinting is synchronous under the hood (``hashlib``), but every
  function that is ``async`` upstream (``fingerprint_key_material``,
  ``fingerprint_tool_call``, ``DoomLoopMonitor.declare_round`` /
  ``record_tool_call`` / ``record_assistant_text``) stays ``async def`` so call
  sites match the reference. Canonicalization errors therefore surface when
  the coroutine is awaited.
- JavaScript has both ``undefined`` and ``null``; Python only has ``None``.
  ``None`` maps to JSON ``null`` everywhere. Consequently a ``loop_key``
  *declaration* of ``None`` means "absent" (full arguments), and a ``loop_key``
  *function* returning ``None`` exempts the call (upstream ``null``) — the
  upstream "returned ``undefined`` -> fallback with warning" branch has no
  Python equivalent. Callable values inside mappings are dropped (upstream
  drops ``undefined``/function/symbol entries) and serialize as ``null`` in
  sequences / at top level.
- Values with no JSON representation raise :class:`DoomLoopCanonicalizationError`
  (a ``ValueError``): non-finite floats, ints too large for an IEEE double
  (upstream would have parsed those to ``Infinity``), sets, bytes, arbitrary
  objects, non-string mapping keys (upstream: bigint and friends), plus
  circular references and nesting deeper than :data:`MAX_CANONICALIZE_DEPTH`.
  Python ``int`` is serialized as the IEEE double JavaScript would have parsed
  (exact up to 2**53, same rounding beyond), so the same wire JSON yields the
  same fingerprint in both ports.
- ``console.warn`` maps to ``warnings.warn``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast, overload

from pydantic import BaseModel
from typing_extensions import Literal, NotRequired, Required, TypedDict

__all__ = [
    "DEFAULT_DOOM_LOOP_LADDER",
    "DEFAULT_MAX_ESCALATIONS",
    "MAX_CANONICALIZE_DEPTH",
    "DoomLoopAction",
    "DoomLoopCallRecord",
    "DoomLoopCanonicalizationError",
    "DoomLoopConfig",
    "DoomLoopDetectedPayload",
    "DoomLoopDetectedResult",
    "DoomLoopDetectorKind",
    "DoomLoopEscalationConfig",
    "DoomLoopLadder",
    "DoomLoopMonitor",
    "DoomLoopOption",
    "DoomLoopSerializedState",
    "DoomLoopStreak",
    "DoomLoopTextOptions",
    "DoomLoopVerdict",
    "LoopKeyResolution",
    "ResolvedDoomLoopConfig",
    "ResolvedEscalationConfig",
    "ResolvedTextOptions",
    "TextRepetitionResult",
    "canonicalize_key_material",
    "detect_text_repetition",
    "fingerprint_key_material",
    "fingerprint_tool_call",
    "resolve_doom_loop_option",
    "resolve_ladder_action",
    "resolve_loop_key_material",
]

# region Public Types

#: Escalation actions, weakest to strongest:
#:
#: - ``observe``  — emit the ``DoomLoopDetected`` hook only.
#: - ``steer``    — inject a corrective user message before the next model turn
#:   (same mechanism as the Stop hook's ``append_prompt``); the call still runs.
#: - ``escalate`` — one-turn recovery on the NEXT request: stronger model and/or
#:   a forced ``openrouter:advisor`` consult, per the ``escalation`` config.
#:   Bounded by ``escalation.max_escalations``. The call still runs.
#: - ``block``    — refuse the tool call and synthesize an error
#:   ``function_call_output`` (same shape as a PreToolUse block).
#: - ``stop``     — halt the run before the next model request
#:   (``SessionEnd.reason == "doom_loop"``).
DoomLoopAction = Literal["observe", "steer", "escalate", "block", "stop"]

#: Which detector produced a verdict.
DoomLoopDetectorKind = Literal["tool-fingerprint", "server-tool-fingerprint", "text-repetition", "text-streak"]

#: A ladder threshold: a streak count, or ``False`` to disable the rung.
Threshold = Union[int, Literal[False]]


class DoomLoopVerdict(TypedDict):
    """A doom-loop detection event (upstream ``DoomLoopVerdict``).

    ``streak`` is the repetition count that crossed a ladder threshold.
    ``tool_name`` is present for fingerprint verdicts. ``message`` is used
    verbatim for block outputs and steer messages.
    """

    detector: DoomLoopDetectorKind
    action: DoomLoopAction
    streak: int
    fingerprint: str
    tool_name: NotRequired[str]
    message: str


class DoomLoopCallRecord(TypedDict):
    """Result of recording one tool call (upstream ``DoomLoopCallRecord``).

    ``duplicate_in_round`` is True when the same ``(tool_name, fingerprint)``
    was already recorded in this round — the streak did NOT increment and the
    caller should reuse the decision it applied for the first occurrence.
    ``verdict`` is absent when no ladder rung fired.
    """

    fingerprint: str
    streak: int
    duplicate_in_round: bool
    verdict: NotRequired[DoomLoopVerdict]


class DoomLoopLadder(TypedDict, total=False):
    """Streak thresholds per action; ``False`` disables a rung. Strongest
    crossed rung wins (stop > block > escalate > steer > observe); thresholds
    compare with ``>=``."""

    observe: Threshold
    steer: Threshold
    escalate: Threshold
    block: Threshold
    stop: Threshold


class DoomLoopEscalationConfig(TypedDict, total=False):
    """Recovery configuration for the ``escalate`` rung.

    - ``model``: model slug for the NEXT turn only.
    - ``advisor``: force an ``openrouter:advisor`` consult on the next turn;
      ``True`` for defaults or a dict passed through as the advisor tool's
      ``parameters``. ``False`` is an explicit opt-out (not a mechanism).
    - ``max_escalations``: cap on escalations per run (default 2).
    """

    model: str
    advisor: Union[bool, Dict[str, Any]]
    max_escalations: int


class DoomLoopTextOptions(TypedDict, total=False):
    """Options for text-loop detection (defaults: enabled, 16, 4, 12, 400)."""

    enabled: bool
    max_period_tokens: int
    min_repeats: int
    min_covered_tokens: int
    max_window_tokens: int


class DoomLoopConfig(TypedDict, total=False):
    """Configuration for doom-loop detection. ``doom_loop=True`` uses all defaults."""

    ladder: DoomLoopLadder
    #: ``False`` disables text detection; a dict tunes it. Default on.
    text: Union[bool, DoomLoopTextOptions]
    #: Recovery behavior for the ``escalate`` rung. Off unless configured.
    escalation: DoomLoopEscalationConfig


#: The ``doom_loop`` option on ``call_model``: ``True`` for defaults, or a config mapping.
DoomLoopOption = Union[bool, DoomLoopConfig, Mapping[str, Any]]


class DoomLoopStreak(TypedDict):
    """Consecutive-repetition counter for one identity (a tool, or step text).

    ``round_fingerprints`` (multi-call rounds only) is the full fingerprint set
    of the tool's last round; ``call_streaks`` maps fingerprint -> consecutive
    rounds that exact call has been issued in. Neither is present on text
    streaks.
    """

    fingerprint: str
    streak: int
    round_fingerprints: NotRequired[List[str]]
    call_streaks: NotRequired[Dict[str, int]]


class DoomLoopSerializedState(TypedDict):
    """Plain-JSON detector state, persisted inside ``ConversationState.doom_loop``.

    ``tools`` / ``text`` / ``escalations_used`` are owned by the monitor;
    ``stop_verdict`` / ``pending_steer`` are owned by the engine, which merges
    them into the persisted blob alongside :meth:`DoomLoopMonitor.get_state`.
    """

    tools: Dict[str, DoomLoopStreak]
    text: NotRequired[DoomLoopStreak]
    stop_verdict: NotRequired[DoomLoopVerdict]
    pending_steer: NotRequired[List[str]]
    escalations_used: NotRequired[int]


class LoopKeyResolution(TypedDict, total=False):
    """Result of resolving a tool's ``loop_key`` against one call's arguments.

    - ``{"kind": "exempt"}``
    - ``{"kind": "key", "key_material": ...}``
    - ``{"kind": "fallback", "key_material": ..., "warning": str}``
    """

    kind: Required[Literal["exempt", "key", "fallback"]]
    key_material: Any
    warning: str


class TextRepetitionResult(TypedDict):
    """A detected repeating token block at the tail of a response."""

    #: Consecutive repeats of the block (feeds the ladder as the streak).
    repeats: int
    #: Block length in whitespace-delimited tokens.
    period_tokens: int
    #: repeats x period_tokens.
    covered_tokens: int
    #: The repeating block itself.
    sample: str


class ResolvedEscalationConfig(TypedDict, total=False):
    """Escalation config with the budget resolved."""

    model: str
    advisor: Union[bool, Dict[str, Any]]
    max_escalations: Required[int]


class ResolvedTextOptions(TypedDict):
    enabled: bool
    max_period_tokens: int
    min_repeats: int
    min_covered_tokens: int
    max_window_tokens: int


class ResolvedDoomLoopConfig(TypedDict):
    """Fully-resolved config used by the monitor."""

    ladder: Dict[str, Threshold]
    text: ResolvedTextOptions
    escalation: Optional[ResolvedEscalationConfig]


class DoomLoopDetectedPayload(BaseModel):
    """Payload of the ``DoomLoopDetected`` lifecycle hook (upstream
    ``DoomLoopDetectedPayloadSchema`` in ``hooks-schemas.ts``).

    Defined as a Pydantic model (not a TypedDict) because the hook registry's
    ``HookDefinition`` validates payloads with ``BaseModel`` classes, like
    every other built-in hook in ``hooks_schemas.py``.
    """

    #: Which detector fired: identical client/server tool calls, or repeated text.
    detector: DoomLoopDetectorKind
    #: The ladder action the engine resolved for this streak.
    action: DoomLoopAction
    #: Consecutive repetition count that crossed a ladder rung.
    streak: int
    #: Deterministic fingerprint of the repeated unit (call identity or text).
    fingerprint: str
    #: Present for tool-fingerprint verdicts.
    tool_name: Optional[str] = None
    #: Present for tool-fingerprint verdicts: the repeated call's arguments.
    tool_input: Optional[Dict[str, Any]] = None
    #: Explanation used for block outputs / steer messages.
    message: str


class DoomLoopDetectedResult(BaseModel):
    """Result of a ``DoomLoopDetected`` handler (upstream
    ``DoomLoopDetectedResultSchema``).

    ``override_action`` overrides the engine's resolved action for THIS event;
    the last override in the handler chain wins. ``block`` on a non-tool
    verdict downgrades to ``observe``; ``escalate`` without config/budget
    downgrades to ``observe``.
    """

    override_action: Optional[DoomLoopAction] = None


class DoomLoopCanonicalizationError(ValueError):
    """Raised by :func:`canonicalize_key_material` for key material RFC 8785
    cannot represent (non-finite numbers, unrepresentable types, circular
    references, nesting deeper than :data:`MAX_CANONICALIZE_DEPTH`)."""


# endregion

# region Defaults & Config Resolution

#: Default ladder: observe at 2 consecutive identical rounds, block at 3,
#: stop at 6. Steer and escalate are off by default. Read-only.
DEFAULT_DOOM_LOOP_LADDER: Mapping[str, Threshold] = MappingProxyType(
    {
        "observe": 2,
        "steer": False,
        "escalate": False,
        "block": 3,
        "stop": 6,
    }
)

#: Default escalation budget when the rung is enabled without a cap.
DEFAULT_MAX_ESCALATIONS = 2

_DEFAULT_TEXT_OPTIONS: Mapping[str, Any] = MappingProxyType(
    {
        "enabled": True,
        "max_period_tokens": 16,
        "min_repeats": 4,
        "min_covered_tokens": 12,
        "max_window_tokens": 400,
    }
)

#: Ladder rungs ordered weakest -> strongest, for dead-rung analysis.
_LADDER_ORDER: Tuple[str, ...] = ("observe", "steer", "escalate", "block", "stop")


def _is_number(value: Any) -> bool:
    """JS ``typeof value === 'number'``: int/float but never bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    if not _is_number(value):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # pragma: no cover - ints beyond float range
        return False


def _sanitize_threshold(value: Any, fallback: Threshold) -> Threshold:
    if value is False:
        return False
    if _is_finite_number(value) and value >= 1:
        return int(math.floor(value))
    return fallback


def _warn(message: str) -> None:
    warnings.warn(message, stacklevel=3)


def _warn_on_ladder_hazards(ladder: Mapping[str, Threshold]) -> None:
    """Warn about accepted-but-probably-wrong configurations: ``block`` with
    ``stop`` disabled (unbounded block/re-issue ping-pong) and dead rungs."""
    if ladder["block"] is not False and ladder["stop"] is False:
        _warn(
            "[DoomLoop] ladder has block enabled with stop disabled: a model that keeps "
            "re-issuing a blocked call loops indefinitely (each blocked round still costs a "
            "model request). Bound the run with stop_when, or enable the stop rung."
        )
    for weak_index, weak_name in enumerate(_LADDER_ORDER):
        weak_threshold = ladder[weak_name]
        if weak_threshold is False:
            continue
        for strong_name in _LADDER_ORDER[weak_index + 1 :]:
            strong_threshold = ladder[strong_name]
            if strong_threshold is not False and weak_threshold >= strong_threshold:
                _warn(
                    f'[DoomLoop] ladder rung "{weak_name}" ({weak_threshold}) can never fire: '
                    f'stronger rung "{strong_name}" ({strong_threshold}) already wins at that streak '
                    "(strongest crossed rung is applied)."
                )
                break  # one warning per dead rung is enough


def _pick(mapping: Mapping[str, Any], key: str, default: Any) -> Any:
    """JS ``mapping.key ?? default``."""
    value = mapping.get(key)
    return default if value is None else value


@overload
def resolve_doom_loop_option(
    option: Union[Literal[True], DoomLoopConfig, Mapping[str, Any]],
) -> ResolvedDoomLoopConfig: ...


@overload
def resolve_doom_loop_option(option: Optional[DoomLoopOption]) -> Optional[ResolvedDoomLoopConfig]: ...


def resolve_doom_loop_option(option: Optional[DoomLoopOption]) -> Optional[ResolvedDoomLoopConfig]:
    """Normalize the ``doom_loop`` option. ``None`` / ``False`` -> ``None``
    (detection off). ``True`` or a config mapping always yields a resolved
    config, so ``DoomLoopMonitor(resolve_doom_loop_option(True))`` type-checks.
    """
    if option is None or option is False:
        return None
    config: Mapping[str, Any] = {} if option is True else option  # type: ignore[assignment]
    ladder_input: Mapping[str, Any] = config.get("ladder") or {}
    text_raw = config.get("text")
    text_input: Any = {} if text_raw is None or text_raw is True else text_raw
    ladder: Dict[str, Threshold] = {
        name: _sanitize_threshold(ladder_input.get(name), DEFAULT_DOOM_LOOP_LADDER[name]) for name in _LADDER_ORDER
    }
    _warn_on_ladder_hazards(ladder)

    # Escalation recovery: usable only when the config names at least one
    # mechanism. `advisor: False` is an explicit opt-out, not a mechanism —
    # `{"advisor": False}` alone must read the same as `{}`.
    escalation_input: Optional[Mapping[str, Any]] = config.get("escalation")
    advisor: Any = escalation_input.get("advisor") if escalation_input is not None else None
    has_advisor = advisor is not None and advisor is not False
    has_model = escalation_input is not None and escalation_input.get("model") is not None
    escalation: Optional[ResolvedEscalationConfig] = None
    if escalation_input is not None and (has_model or has_advisor):
        cap: Any = escalation_input.get("max_escalations")
        # Key order mirrors upstream ({model?, advisor?, maxEscalations}).
        built: Dict[str, Any] = {}
        if has_model:
            built["model"] = escalation_input["model"]
        if has_advisor:
            built["advisor"] = advisor
        built["max_escalations"] = (
            int(math.floor(cap)) if _is_finite_number(cap) and cap >= 1 else DEFAULT_MAX_ESCALATIONS
        )
        escalation = cast(ResolvedEscalationConfig, built)
        if ladder["escalate"] is False:
            _warn(
                "[DoomLoop] escalation config provided but the escalate ladder rung is disabled; "
                "set ladder.escalate to a streak threshold for recovery to trigger."
            )
    elif ladder["escalate"] is not False:
        _warn(
            "[DoomLoop] ladder.escalate is enabled but no escalation config (model/advisor) was "
            "provided; the rung is skipped and verdicts fall through to weaker rungs."
        )

    text: ResolvedTextOptions
    if text_input is False:
        text = ResolvedTextOptions(
            enabled=False,
            max_period_tokens=_DEFAULT_TEXT_OPTIONS["max_period_tokens"],
            min_repeats=_DEFAULT_TEXT_OPTIONS["min_repeats"],
            min_covered_tokens=_DEFAULT_TEXT_OPTIONS["min_covered_tokens"],
            max_window_tokens=_DEFAULT_TEXT_OPTIONS["max_window_tokens"],
        )
    else:
        text = ResolvedTextOptions(
            enabled=_pick(text_input, "enabled", _DEFAULT_TEXT_OPTIONS["enabled"]),
            max_period_tokens=_pick(text_input, "max_period_tokens", _DEFAULT_TEXT_OPTIONS["max_period_tokens"]),
            min_repeats=_pick(text_input, "min_repeats", _DEFAULT_TEXT_OPTIONS["min_repeats"]),
            min_covered_tokens=_pick(text_input, "min_covered_tokens", _DEFAULT_TEXT_OPTIONS["min_covered_tokens"]),
            max_window_tokens=_pick(text_input, "max_window_tokens", _DEFAULT_TEXT_OPTIONS["max_window_tokens"]),
        )

    return ResolvedDoomLoopConfig(ladder=ladder, escalation=escalation, text=text)


# endregion

# region loop_key Resolution


def resolve_loop_key_material(loop_key: Any, args: Mapping[str, Any]) -> LoopKeyResolution:
    """Resolve a tool's ``loop_key`` declaration against one call's validated
    arguments.

    - absent (``None``) -> the full arguments object.
    - ``False`` -> the tool is statically exempt.
    - list/tuple of field names -> the declarative subset (empty list, or every
      field absent, falls back to full arguments with a warning).
    - callable -> called with the arguments. ``None`` exempts THIS call (the
      upstream ``null``); a raise falls back to the full arguments with a
      warning. An awaitable result is rejected with a fallback (``loop_key``
      must be synchronous, like upstream).
    - anything else -> fallback with an "unsupported shape" warning.
    """
    if loop_key is None:
        return {"kind": "key", "key_material": args}
    if loop_key is False:
        return {"kind": "exempt"}
    if isinstance(loop_key, (list, tuple)):
        # An empty field list would make EVERY call to the tool fingerprint
        # identically; fall back to full arguments with a warning.
        if len(loop_key) == 0:
            return {
                "kind": "fallback",
                "key_material": args,
                "warning": (
                    "loopKey is an empty field list, which would give every call to this tool the same "
                    "identity; falling back to full arguments. Use `false` to exempt the tool."
                ),
            }
        subset: Dict[str, Any] = {}
        for field in loop_key:
            if isinstance(field, str) and field in args:
                subset[field] = args[field]
        # Every declared field missing is the same collapse by another route.
        if len(subset) == 0:
            names = ", ".join(f for f in loop_key if isinstance(f, str))
            return {
                "kind": "fallback",
                "key_material": args,
                "warning": (
                    f"loopKey fields [{names}] are all absent from the arguments, which would give every "
                    "call the same identity; falling back to full arguments."
                ),
            }
        return {"kind": "key", "key_material": subset}
    if callable(loop_key):
        try:
            result = loop_key(args)
        except Exception as error:
            return {
                "kind": "fallback",
                "key_material": args,
                "warning": f"loopKey threw ({error}); falling back to full arguments",
            }
        if inspect.isawaitable(result):
            close = getattr(result, "close", None)
            if callable(close):
                close()
            return {
                "kind": "fallback",
                "key_material": args,
                "warning": "loopKey returned an awaitable; loopKey must be synchronous. Falling back to full arguments",
            }
        if result is None:
            return {"kind": "exempt"}
        return {"kind": "key", "key_material": result}
    return {
        "kind": "fallback",
        "key_material": args,
        "warning": f"loopKey has unsupported shape ({type(loop_key).__name__}); falling back to full arguments",
    }


# endregion

# region Canonicalization & Fingerprinting

#: Maximum nesting depth accepted by :func:`canonicalize_key_material`.
MAX_CANONICALIZE_DEPTH = 64

_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def _combine_surrogate_pairs(value: str) -> str:
    """Python strings may carry a well-formed UTF-16 surrogate PAIR as two code
    points; JavaScript sees that as one astral character. Combine them so the
    port serializes exactly what ``JSON.stringify`` would. Lone surrogates
    survive as-is."""
    if _SURROGATE_RE.search(value) is None:
        return value
    return value.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")


def _escape_lone_surrogate(match: re.Match[str]) -> str:
    return f"\\u{ord(match.group(0)):04x}"


def _quote_string(value: str) -> str:
    """ECMAScript ``JSON.stringify`` on a string (the JCS string form):
    ``"``/``\\``/control characters escaped (short forms for \\b\\f\\n\\r\\t,
    lowercase ``\\u00XX`` otherwise), lone surrogates as lowercase ``\\udXXX``,
    everything else literal."""
    normalized = _combine_surrogate_pairs(value)
    quoted = json.dumps(normalized, ensure_ascii=False)
    if _SURROGATE_RE.search(quoted) is None:
        return quoted
    return _SURROGATE_RE.sub(_escape_lone_surrogate, quoted)


def _format_number(value: float) -> str:
    """ECMAScript ``Number::toString`` (the JCS number form) for a finite double.

    ``repr`` yields the shortest round-tripping digit string (the same digits
    ES requires); this re-lays them out per ES rules: ``-0`` -> ``0``, integers
    below 1e21 in plain digits, ``1e21`` -> ``1e+21``, ``1e-7`` -> ``1e-7``.
    """
    if value == 0:
        return "0"
    if value < 0:
        return "-" + _format_number(-value)
    text = repr(value)
    if "e" in text:
        mantissa, exponent_text = text.split("e")
        exponent = int(exponent_text)
    else:
        mantissa, exponent = text, 0
    if "." in mantissa:
        integer_part, fraction_part = mantissa.split(".")
    else:
        integer_part, fraction_part = mantissa, ""
    digits = integer_part + fraction_part
    n = len(integer_part) + exponent
    stripped = digits.lstrip("0")
    n -= len(digits) - len(stripped)
    digits = stripped.rstrip("0")
    k = len(digits)
    if k <= n <= 21:
        return digits + "0" * (n - k)
    if 0 < n <= 21:
        return f"{digits[:n]}.{digits[n:]}"
    if -6 < n <= 0:
        return "0." + "0" * (-n) + digits
    e = n - 1
    sign = "+" if e >= 0 else "-"
    if k == 1:
        return f"{digits}e{sign}{abs(e)}"
    return f"{digits[0]}.{digits[1:]}e{sign}{abs(e)}"


def _utf16_sort_key(key: str) -> bytes:
    """JCS sorts object keys by UTF-16 code units; big-endian UTF-16 bytes
    compare in exactly that order."""
    return key.encode("utf-16-be", "surrogatepass")


def _is_js_function(value: Any) -> bool:
    return callable(value) and not isinstance(value, (Mapping, list, tuple, str))


_NON_FINITE_MESSAGE = (
    "Cannot canonicalize non-finite number (NaN/Infinity) for doom-loop fingerprinting: RFC 8785 has no representation"
)


def canonicalize_key_material(value: Any) -> str:
    """RFC 8785 (JCS) canonical JSON of ``value``: mapping keys sorted by UTF-16
    code units (recursively), sequences in order, strings and finite numbers
    serialized per ECMAScript ``JSON.stringify``. Insensitive to key insertion
    order.

    Callable entries in mappings are dropped (JSON semantics for JS
    functions); callables in sequences / at the top level serialize as
    ``null``.

    Raises:
        DoomLoopCanonicalizationError: on values RFC 8785 cannot represent —
            non-finite floats, ints beyond the IEEE double range, sets, bytes,
            arbitrary objects, non-string mapping keys — plus circular
            references and nesting deeper than :data:`MAX_CANONICALIZE_DEPTH`.
            Real tool arguments (parsed JSON) never trigger these; only
            computed ``loop_key`` material can, and the engine falls back.
    """
    seen: Set[int] = set()

    def canon(v: Any, depth: int) -> str:
        if depth > MAX_CANONICALIZE_DEPTH:
            raise DoomLoopCanonicalizationError(
                f"Cannot canonicalize key material nested deeper than {MAX_CANONICALIZE_DEPTH} levels "
                "for doom-loop fingerprinting"
            )
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            # JS parses JSON integers into IEEE doubles; mirror that so the same
            # wire JSON fingerprints identically across ports.
            try:
                as_float = float(v)
            except OverflowError:
                raise DoomLoopCanonicalizationError(_NON_FINITE_MESSAGE) from None
            return _format_number(as_float)
        if isinstance(v, float):
            if not math.isfinite(v):
                raise DoomLoopCanonicalizationError(_NON_FINITE_MESSAGE)
            return _format_number(v)
        if isinstance(v, str):
            return _quote_string(v)
        if isinstance(v, (Mapping, list, tuple)):
            marker = id(v)
            if marker in seen:
                raise DoomLoopCanonicalizationError(
                    "Cannot canonicalize circular key material for doom-loop fingerprinting"
                )
            seen.add(marker)
            try:
                if isinstance(v, (list, tuple)):
                    return "[" + ",".join(canon(item, depth + 1) for item in v) + "]"
                keys: List[str] = []
                for key, entry in v.items():
                    if not isinstance(key, str):
                        raise DoomLoopCanonicalizationError(
                            f"Cannot canonicalize non-string mapping key of type {type(key).__name__} for "
                            "doom-loop fingerprinting: JSON object keys are strings"
                        )
                    if _is_js_function(entry):
                        continue
                    keys.append(key)
                keys.sort(key=_utf16_sort_key)
                return "{" + ",".join(f"{_quote_string(key)}:{canon(v[key], depth + 1)}" for key in keys) + "}"
            finally:
                seen.discard(marker)
        if _is_js_function(v):
            # function — not representable; JSON semantics say null here.
            return "null"
        raise DoomLoopCanonicalizationError(
            f"Cannot canonicalize {type(v).__name__} for doom-loop fingerprinting: RFC 8785 has no "
            "representation (convert to a JSON value in loop_key)"
        )

    return canon(value, 0)


def _sha256_hex(text: str) -> str:
    """SHA-256 over UTF-8, lowercase hex. Lone surrogates (only reachable via a
    tool name — canonical strings escape them) encode as U+FFFD, matching the
    WHATWG ``TextEncoder`` upstream uses."""
    normalized = _combine_surrogate_pairs(text)
    if _SURROGATE_RE.search(normalized) is not None:
        normalized = _SURROGATE_RE.sub("\ufffd", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


async def fingerprint_key_material(key_material: Any) -> str:
    """``sha256(utf8(jcs(key_material)))``, lowercase hex (64 chars)."""
    return _sha256_hex(canonicalize_key_material(key_material))


async def fingerprint_tool_call(tool_name: str, key_material: Any) -> str:
    """``sha256(utf8(tool_name + "\\n" + jcs(key_material)))``, lowercase hex.

    The tool name participates so ``search({q})`` and ``fetch({q})`` never
    share a fingerprint. This exact construction is the cross-port contract —
    see ``tests/vectors/doom_loop_fingerprints.json``.
    """
    return _sha256_hex(f"{tool_name}\n{canonicalize_key_material(key_material)}")


def _fingerprint_tool_call_sync(tool_name: str, key_material: Any) -> str:
    return _sha256_hex(f"{tool_name}\n{canonicalize_key_material(key_material)}")


# endregion

# region Text Repetition Detection

# ECMAScript `\s` (WhiteSpace + LineTerminator). Python's `\s` / str.split()
# differ (e.g. Python treats \x1c-\x1f as whitespace, JS treats \ufeff as
# whitespace), and tokenization is part of the deterministic contract.
_JS_WHITESPACE = "\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_JS_WS_RUN = re.compile(f"[{_JS_WHITESPACE}]+")
_JS_TRIM = re.compile(f"\\A[{_JS_WHITESPACE}]+|[{_JS_WHITESPACE}]+\\Z")


def _js_trim(text: str) -> str:
    return _JS_TRIM.sub("", text)


def _js_tail(text: str, budget: int) -> str:
    """``text.length > budget ? text.slice(-budget) : text`` in UTF-16 code
    units, as JavaScript measures strings."""
    if budget <= 0:
        # JS `slice(-0)` is `slice(0)`: the whole string.
        return text
    if len(text) * 2 <= budget:
        # Even all-astral text has at most 2 code units per code point.
        return text
    # The last `budget` code points always cover the last `budget` code units.
    candidate = text[-budget:]
    encoded = candidate.encode("utf-16-le", "surrogatepass")
    if len(candidate) == len(text) and len(encoded) // 2 <= budget:
        return text
    return encoded[-budget * 2 :].decode("utf-16-le", "surrogatepass")


def _option(options: Optional[Mapping[str, Any]], key: str) -> Any:
    if options is None:
        return _DEFAULT_TEXT_OPTIONS[key]
    return _pick(options, key, _DEFAULT_TEXT_OPTIONS[key])


def detect_text_repetition(text: str, options: Optional[Mapping[str, Any]] = None) -> Optional[TextRepetitionResult]:
    """Detect a period-p token block repeating at the *tail* of ``text``.

    Pure and deterministic: whitespace tokenization, suffix comparison, no
    heuristics beyond the thresholds (``max_period_tokens``, ``min_repeats``,
    ``min_covered_tokens``, ``max_window_tokens`` — snake_case keys of
    :class:`DoomLoopTextOptions`). Only a fixed character budget (64 chars per
    window token, in UTF-16 code units like upstream) is sliced off the tail
    before tokenization. Ties on covered tokens prefer the smallest period.

    Returns ``None`` when no block meets the thresholds.

    KNOWN LIMIT: token blocks must repeat *exactly*; paraphrased loops miss.
    """
    max_period_tokens = _option(options, "max_period_tokens")
    min_repeats = _option(options, "min_repeats")
    min_covered_tokens = _option(options, "min_covered_tokens")
    max_window_tokens = _option(options, "max_window_tokens")

    char_budget = max_window_tokens * 64
    tail = _js_tail(text, char_budget)
    trimmed = _js_trim(tail)
    all_tokens = [token for token in _JS_WS_RUN.split(trimmed) if token]
    tokens = all_tokens[-max_window_tokens:] if max_window_tokens > 0 else all_tokens
    count = len(tokens)
    if count < max(min_covered_tokens, 2):
        return None

    best: Optional[TextRepetitionResult] = None
    max_period = min(max_period_tokens, count // 2)
    for period in range(1, max_period + 1):
        repeats = 1
        last_block = tokens[count - period :]
        start = count - 2 * period
        while start >= 0:
            if tokens[start : start + period] != last_block:
                break
            repeats += 1
            start -= period
        covered_tokens = repeats * period
        if repeats >= min_repeats and covered_tokens >= min_covered_tokens:
            if best is None or covered_tokens > best["covered_tokens"]:
                best = TextRepetitionResult(
                    repeats=repeats,
                    period_tokens=period,
                    covered_tokens=covered_tokens,
                    sample=" ".join(last_block),
                )
    return best


# endregion

# region Ladder Resolution


def resolve_ladder_action(
    ladder: Mapping[str, Any],
    streak: int,
    *,
    allow_block: bool,
    allow_escalate: bool = True,
) -> Optional[DoomLoopAction]:
    """Map a streak onto the strongest crossed ladder rung.

    ``allow_block=False`` is used for text and server-tool verdicts (nothing
    to block): a block-level streak falls through to escalate/steer/observe,
    while stop still stops. ``allow_escalate=False`` disables the escalate
    rung (no mechanism configured, or budget exhausted). A missing or
    ``False`` threshold disables its rung.
    """

    def meets(name: str) -> bool:
        threshold = ladder.get(name)
        if threshold is None or threshold is False:
            return False
        return bool(streak >= threshold)

    if meets("stop"):
        return "stop"
    if allow_block and meets("block"):
        return "block"
    if allow_escalate and meets("escalate"):
        return "escalate"
    if meets("steer"):
        return "steer"
    if meets("observe"):
        return "observe"
    return None


# endregion

# region Monitor


@dataclass
class _StreakEntry:
    """In-memory streak entry: the serialized shape plus run-local round data.

    - ``round``: round of the most recent record (never serialized — a resumed
      run must always increment on its first record).
    - ``round_fingerprints``: this round's COMPLETE declared set (sorted), or
      the lone fingerprint.
    - ``seen_this_round``: fingerprints of ``round`` already recorded (in-round
      duplicate collapse).
    - ``prior_round_fingerprints`` / ``prior_streak`` / ``prior_call_streaks``:
      the previous round's baseline, fixed at the round transition so arrival
      order within a round can never affect a score.
    - ``call_streaks``: the CURRENT round's per-call counts, grown in place.
    """

    fingerprint: str
    streak: int
    round: Optional[int] = None
    round_fingerprints: Optional[List[str]] = None
    seen_this_round: Optional[Set[str]] = None
    prior_round_fingerprints: Optional[List[str]] = None
    prior_streak: Optional[int] = None
    prior_call_streaks: Optional[Dict[str, int]] = None
    call_streaks: Optional[Dict[str, int]] = None


@dataclass
class _DeclaredRound:
    round: int
    fingerprints: Dict[str, List[str]]


#: The shared empty seen-set. Never mutated (the write path allocates a fresh set).
_EMPTY_SEEN: FrozenSet[str] = frozenset()

_ACTION_STRENGTH: Mapping[str, int] = MappingProxyType({"observe": 0, "steer": 1, "escalate": 2, "block": 3, "stop": 4})


def _is_valid_streak(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("fingerprint"), str)
        and _is_finite_number(value.get("streak"))
    )


def _call_streaks_reconstructible(entry: _StreakEntry) -> bool:
    """True when per-call counts carry no information beyond the round set and
    round streak (every member counts exactly the round streak), so
    ``get_state`` may omit them and ``restore`` rebuilds them exactly."""
    counts = entry.call_streaks
    if counts is None:
        return True
    members = entry.round_fingerprints if entry.round_fingerprints is not None else [entry.fingerprint]
    return len(counts) == len(members) and all(key in members and counts[key] == entry.streak for key in counts)


def _restore_streak_entry(entry: Mapping[str, Any]) -> _StreakEntry:
    """Rebuild one tool's in-memory streak entry from its persisted shape.

    Malformed sets fall back to the lone fingerprint; per-call counts are
    validated entry-by-entry; the round streak is clamped to [1, 1_000_000]
    (fail-open: a corrupt streak degrades the entry rather than dropping it).
    ``round`` is intentionally unset: the first resumed record is a new round.
    """
    fingerprint: str = entry["fingerprint"]
    raw_set = entry.get("round_fingerprints")
    persisted_set: List[str]
    if isinstance(raw_set, (list, tuple)) and len(raw_set) > 1 and all(isinstance(v, str) for v in raw_set):
        persisted_set = sorted(raw_set)
    else:
        persisted_set = [fingerprint]
    persisted_call_streaks: Dict[str, int] = {}
    raw_counts = entry.get("call_streaks")
    if isinstance(raw_counts, Mapping):
        for call_fingerprint, count in raw_counts.items():
            if isinstance(call_fingerprint, str) and _is_finite_number(count) and count >= 1:
                persisted_call_streaks[call_fingerprint] = int(math.floor(count))
    streak = min(max(int(math.floor(entry["streak"])), 1), 1_000_000)
    return _StreakEntry(
        fingerprint=fingerprint,
        streak=streak,
        round_fingerprints=persisted_set,
        # Absent per-call counts mean get_state() proved them reconstructible:
        # rebuild them for the WHOLE set, not just the last-recorded member.
        call_streaks=(
            persisted_call_streaks if persisted_call_streaks else {member: streak for member in persisted_set}
        ),
    )


def _grow_call_streaks(previous: Optional[_StreakEntry], fingerprint: str, call_streak: int) -> Dict[str, int]:
    """This round's per-call accumulator, grown in place (O(1) per record).
    A fresh dict is created at each round transition, so the previous round's
    baseline (``prior_call_streaks``) is never disturbed."""
    if previous is not None and previous.call_streaks is not None:
        previous.call_streaks[fingerprint] = call_streak
        return previous.call_streaks
    return {fingerprint: call_streak}


def _summarize_round(fingerprints: Sequence[str]) -> str:
    """Short, stable identity for a round's fingerprint set."""
    shown = [value[:8] for value in fingerprints[:3]]
    if len(fingerprints) > 3:
        return f"{'+'.join(shown)}+{len(fingerprints) - 3} more…"
    return f"{'+'.join(shown)}…"


def _sets_match(left: Sequence[str], right: Sequence[str]) -> bool:
    """Set equality over two sorted fingerprint lists."""
    return len(left) == len(right) and all(a == b for a, b in zip(left, right, strict=True))


def _build_tool_verdict_message(
    *,
    tool_name: str,
    fingerprint: str,
    call_set: Sequence[str],
    round_streak: int,
    call_streak: int,
    round_declared: bool,
) -> str:
    """Verdict text naming what actually repeated. Per-call (or undeclared)
    verdicts are fingerprint-free AND count-free so the steer rung's exact-text
    dedupe collapses one round's evidence to at most two strings per tool."""
    if call_streak > round_streak or not round_declared:
        return (
            f'Doom loop suspected: this exact "{tool_name}" call has been repeated across '
            "consecutive rounds. Repeating it will not change the result. "
            "Take a different approach, or explain why repetition is required."
        )
    if len(call_set) > 1:
        return (
            f'Doom loop suspected: tool "{tool_name}" was invoked in {round_streak} consecutive rounds '
            f"with the same set of {len(call_set)} parallel calls "
            f"(round identity {_summarize_round(call_set)}). Reissuing the same fan-out "
            "will not change the results. Take a different approach, or explain why repetition is required."
        )
    return (
        f'Doom loop suspected: tool "{tool_name}" was invoked in {round_streak} consecutive rounds '
        f"with identical arguments (fingerprint {fingerprint[:16]}…). Repeating the call "
        "will not change the result. Take a different approach, or explain why repetition is required."
    )


def _stronger_verdict(a: Optional[DoomLoopVerdict], b: Optional[DoomLoopVerdict]) -> Optional[DoomLoopVerdict]:
    if a is None:
        return b
    if b is None:
        return a
    # Within-response wins ties: its message names the repeated block.
    return b if _ACTION_STRENGTH[b["action"]] > _ACTION_STRENGTH[a["action"]] else a


class DoomLoopMonitor:
    """Pure state machine over the recorded transcript: feed it tool calls and
    assistant texts, get verdicts back. Holds no engine concerns (hook
    emission, blocking, steering, stopping, the steer queue and the stop
    verdict live in ``ModelResult``).

    State is bounded, plain JSON (one streak entry per distinct tool name +
    one text streak + the consumed escalation budget) and round-trips through
    ``ConversationState.doom_loop`` via :meth:`get_state` / :meth:`restore`.
    """

    def __init__(self, config: ResolvedDoomLoopConfig, initial_state: Any = None) -> None:
        self._config = config
        # A plain dict is safe for hostile tool names ("__proto__" is inert data in Python).
        self._tools: Dict[str, _StreakEntry] = {}
        self._text: Optional[_StreakEntry] = None
        # Escalation recoveries consumed by this conversation. Persisted so a
        # resumed run cannot reset its budget.
        self._escalations_used = 0
        # The current round's declared per-tool fingerprint sets. Run-local.
        self._declared_round: Optional[_DeclaredRound] = None
        # Per-object fingerprint memo for the declared round (upstream uses a
        # WeakMap). Python dicts/lists are not weak-referenceable, so it is
        # keyed by id() while holding a strong reference (so ids cannot be
        # reused), populated by declare_round and reset at each declaration —
        # bounded to one round's key material.
        self._fingerprint_memo: Dict[int, Tuple[Any, Dict[str, str]]] = {}
        if initial_state is not None:
            self.restore(initial_state)

    async def declare_round(self, round: int, calls: Sequence[Mapping[str, Any]]) -> None:
        """Declare the complete set of calls a round will make, before any of
        them is recorded. Each entry is ``{"tool_name": str, "key_material": Any}``.

        Idempotent per round, and safe to skip: per-call detection needs no
        declaration. What it adds is round-set evidence (the fan-out scored as
        one unit). Unhashable key material is skipped with a warning; the
        caller's own fallback chain handles it at record time. Never serialized.
        """
        self._fingerprint_memo = {}
        collected: Dict[str, Set[str]] = {}
        for call in calls:
            tool_name: str = call["tool_name"]
            key_material = call.get("key_material")
            try:
                fingerprint = self._fingerprint_once(tool_name, key_material, memoize=True)
            except Exception as error:
                warnings.warn(
                    f'[DoomLoop] could not fingerprint a "{tool_name}" call while declaring round '
                    f"{round}; excluding it from the round set: {error}",
                    stacklevel=2,
                )
                continue
            collected.setdefault(tool_name, set()).add(fingerprint)
        self._declared_round = _DeclaredRound(
            round=round,
            fingerprints={tool_name: sorted(members) for tool_name, members in collected.items()},
        )

    def _fingerprint_once(self, tool_name: str, key_material: Any, *, memoize: bool = False) -> str:
        memoizable = isinstance(key_material, (dict, list))
        if memoizable:
            hit = self._fingerprint_memo.get(id(key_material))
            if hit is not None and hit[0] is key_material:
                cached = hit[1].get(tool_name)
                if cached is not None:
                    return cached
        fingerprint = _fingerprint_tool_call_sync(tool_name, key_material)
        if memoizable and memoize:
            slot = self._fingerprint_memo.get(id(key_material))
            if slot is None or slot[0] is not key_material:
                slot = (key_material, {})
                self._fingerprint_memo[id(key_material)] = slot
            slot[1][tool_name] = fingerprint
        return fingerprint

    def can_escalate(self) -> bool:
        """True when the escalate rung can still fire: a mechanism is
        configured and the budget is not exhausted."""
        escalation = self._config["escalation"]
        return escalation is not None and self._escalations_used < escalation["max_escalations"]

    def consume_escalation(self) -> None:
        """Consume one escalation from the budget. The ENGINE calls this when it
        actually applies the recovery — not at verdict time."""
        self._escalations_used += 1

    def get_state(self) -> DoomLoopSerializedState:
        """Snapshot the serializable detector state (deep copy). Round markers
        are dropped. ``stop_verdict`` / ``pending_steer`` are engine-owned and
        merged in by the engine."""
        tools: Dict[str, DoomLoopStreak] = {}
        for name, entry in self._tools.items():
            streak = DoomLoopStreak(fingerprint=entry.fingerprint, streak=entry.streak)
            if entry.round_fingerprints is not None and len(entry.round_fingerprints) > 1:
                # COPIED, not aliased: the caller may mutate the saved blob.
                streak["round_fingerprints"] = list(entry.round_fingerprints)
            if entry.call_streaks is not None and not _call_streaks_reconstructible(entry):
                streak["call_streaks"] = dict(entry.call_streaks)
            tools[name] = streak
        state = DoomLoopSerializedState(tools=tools)
        if self._text is not None:
            state["text"] = DoomLoopStreak(fingerprint=self._text.fingerprint, streak=self._text.streak)
        if self._escalations_used > 0:
            state["escalations_used"] = self._escalations_used
        return state

    def restore(self, state: Any) -> None:
        """Restore persisted state (from ``ConversationState.doom_loop``).
        Invalid blobs are ignored with a warning; engine-owned fields
        (``stop_verdict``, ``pending_steer``) are ignored here."""
        if not isinstance(state, Mapping):
            warnings.warn("[DoomLoop] Ignoring invalid persisted doom-loop state", stacklevel=2)
            return
        tools: Dict[str, _StreakEntry] = {}
        raw_tools = state.get("tools")
        if isinstance(raw_tools, Mapping):
            for name, entry in raw_tools.items():
                if isinstance(name, str) and _is_valid_streak(entry):
                    tools[name] = _restore_streak_entry(entry)
        self._tools = tools
        raw_text: Any = state.get("text")
        self._text = (
            _StreakEntry(fingerprint=raw_text["fingerprint"], streak=raw_text["streak"])
            if _is_valid_streak(raw_text)
            else None
        )
        used: Any = state.get("escalations_used")
        self._escalations_used = int(math.floor(used)) if _is_finite_number(used) and used >= 0 else 0

    async def record_tool_call(
        self,
        tool_name: str,
        key_material: Any,
        round: int,
        *,
        allow_block: bool = True,
        detector: Literal["tool-fingerprint", "server-tool-fingerprint"] = "tool-fingerprint",
    ) -> DoomLoopCallRecord:
        """Record one tool call and return the streak record plus any verdict.

        - Per tool: interleaved calls to *other* tools do not reset a streak.
        - Per round: the same ``(tool, fingerprint)`` recorded again in the
          SAME round is a duplicate (``duplicate_in_round``), no increment.
        - A round's identity is its declared fingerprint *set*; per-call
          streaks run alongside, and the stronger of the two decides.
        - ``allow_block=False`` for post-execution (server tool) records.

        Raises whatever :func:`canonicalize_key_material` raises for
        unhashable key material; callers must catch and fall back.
        """
        fingerprint = self._fingerprint_once(tool_name, key_material)
        previous = self._tools.get(tool_name)
        is_same_round = previous is not None and previous.round is not None and previous.round == round

        declared: Optional[List[str]] = (
            self._declared_round.fingerprints.get(tool_name)
            if self._declared_round is not None and self._declared_round.round == round
            else None
        )
        declared_member = declared is not None and fingerprint in declared

        seen: Union[Set[str], FrozenSet[str]] = _EMPTY_SEEN
        if is_same_round and previous is not None and previous.seen_this_round is not None:
            seen = previous.seen_this_round
        duplicate_in_round = fingerprint in seen

        # The baseline every call of this round is measured against.
        prior_set: Optional[List[str]]
        prior_streak: int
        prior_call_streaks: Dict[str, int]
        if is_same_round and previous is not None:
            prior_set = previous.prior_round_fingerprints
            prior_streak = previous.prior_streak if previous.prior_streak is not None else 0
            prior_call_streaks = previous.prior_call_streaks if previous.prior_call_streaks is not None else {}
        elif previous is not None:
            prior_set = (
                previous.round_fingerprints if previous.round_fingerprints is not None else [previous.fingerprint]
            )
            prior_streak = previous.streak
            prior_call_streaks = previous.call_streaks if previous.call_streaks is not None else {}
        else:
            prior_set = None
            prior_streak = 0
            prior_call_streaks = {}

        def score(members: Sequence[str]) -> int:
            return prior_streak + 1 if prior_set is not None and _sets_match(prior_set, members) else 1

        # PER-CALL evidence: consecutive rounds THIS exact fingerprint was issued in.
        call_streak = prior_call_streaks.get(fingerprint, 0) + 1

        call_set: List[str] = declared if declared_member and declared is not None else [fingerprint]
        streak = score(call_set)
        round_set: List[str] = declared if declared is not None else [fingerprint]

        if seen is _EMPTY_SEEN:
            next_seen: Set[str] = {fingerprint}
        else:
            next_seen = seen  # type: ignore[assignment]
            next_seen.add(fingerprint)

        self._tools[tool_name] = _StreakEntry(
            # A non-member must not become the identity paired with `streak`.
            fingerprint=(
                previous.fingerprint
                if declared is not None and not declared_member and previous is not None
                else fingerprint
            ),
            round=round,
            seen_this_round=next_seen,
            round_fingerprints=round_set,
            streak=score(round_set),
            prior_round_fingerprints=prior_set,
            prior_streak=prior_streak,
            prior_call_streaks=prior_call_streaks,
            call_streaks=_grow_call_streaks(previous if is_same_round else None, fingerprint, call_streak),
        )

        effective_streak = max(streak, call_streak)
        action = resolve_ladder_action(
            self._config["ladder"],
            effective_streak,
            allow_block=allow_block,
            allow_escalate=self.can_escalate(),
        )
        if action is None:
            return DoomLoopCallRecord(
                fingerprint=fingerprint, streak=effective_streak, duplicate_in_round=duplicate_in_round
            )
        return DoomLoopCallRecord(
            fingerprint=fingerprint,
            streak=effective_streak,
            duplicate_in_round=duplicate_in_round,
            verdict=DoomLoopVerdict(
                detector=detector,
                action=action,
                streak=effective_streak,
                fingerprint=fingerprint,
                tool_name=tool_name,
                message=_build_tool_verdict_message(
                    tool_name=tool_name,
                    fingerprint=fingerprint,
                    call_set=call_set,
                    round_streak=streak,
                    call_streak=call_streak,
                    round_declared=declared_member,
                ),
            ),
        )

    def reset_text_streak(self) -> None:
        """Clear the cross-step text streak (called by the engine when a late
        async-tool result is injected — observable forward progress). Tool
        streaks are deliberately kept."""
        self._text = None

    async def record_assistant_text(self, text: str) -> Optional[DoomLoopVerdict]:
        """Record one step's assistant text and return the strongest verdict of
        the two text detectors (``text-repetition`` within the response,
        ``text-streak`` across steps). Empty/whitespace-only text is a no-op:
        it neither counts nor resets the cross-step streak."""
        if not self._config["text"]["enabled"]:
            return None
        normalized = _js_trim(_JS_WS_RUN.sub(" ", text))
        if len(normalized) == 0:
            return None

        within_response: Optional[DoomLoopVerdict] = None
        repetition = detect_text_repetition(normalized, self._config["text"])
        if repetition is not None:
            action = resolve_ladder_action(
                self._config["ladder"],
                repetition["repeats"],
                allow_block=False,
                allow_escalate=self.can_escalate(),
            )
            if action is not None:
                within_response = DoomLoopVerdict(
                    detector="text-repetition",
                    action=action,
                    streak=repetition["repeats"],
                    fingerprint=await fingerprint_key_material(repetition["sample"]),
                    message=(
                        f'Doom loop suspected: the response repeats "{repetition["sample"]}" '
                        f"{repetition['repeats']} times in a row. Stop repeating and take a different approach."
                    ),
                )

        fingerprint = await fingerprint_key_material(normalized)
        previous = self._text
        streak = previous.streak + 1 if previous is not None and previous.fingerprint == fingerprint else 1
        self._text = _StreakEntry(fingerprint=fingerprint, streak=streak)
        cross_step: Optional[DoomLoopVerdict] = None
        cross_action = resolve_ladder_action(
            self._config["ladder"],
            streak,
            allow_block=False,
            allow_escalate=self.can_escalate(),
        )
        if cross_action is not None:
            cross_step = DoomLoopVerdict(
                detector="text-streak",
                action=cross_action,
                streak=streak,
                fingerprint=fingerprint,
                message=(
                    f"Doom loop suspected: the assistant produced identical text for {streak} "
                    "consecutive turns. Stop repeating and take a different approach."
                ),
            )

        return _stronger_verdict(within_response, cross_step)


# endregion
