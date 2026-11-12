"""Doom-loop detection for the tool-execution loop.

Port of upstream ``src/lib/doom-loop.ts``.

A "doom loop" is a run that stops making progress while continuing to spend:
the model re-issues the same tool call with identical arguments (including
repeated *empty* calls and repeated unparseable calls), or emits the same text
tokens over and over. This module provides the deterministic detection
primitives; ``ModelResult`` wires them into the loop behind the ``doom_loop``
option on ``call_model``.

Design principles (see upstream for the full rationale):

- **Deterministic.** Every verdict is a pure function of the recorded sequence
  of tool calls / assistant texts — no wall clocks, no randomness.
- **Cross-port fingerprints.** Key material is canonicalized per RFC 8785 (JCS)
  and hashed with SHA-256 over the UTF-8 bytes. The stdlib ``json.dumps`` is
  NOT JCS (``json.dumps(-0.0)`` is ``-0.0``; it sorts keys by code point, not
  UTF-16 code unit; it does not use ECMAScript number formatting), so this
  module implements the canonicalization itself. The shared vector file
  ``tests/vectors/doom-loop-fingerprints.json`` is the conformance contract.
- **Tool-declared identity.** A tool opts into precise loop identity via
  ``loop_key``: a function computing key material, a field-name list, or
  ``False`` to exempt the tool.
- **Round-scoped streaks.** A streak measures the model *re-issuing a call
  after seeing its result*. N identical calls fanned out in ONE round count
  once. A round's identity for one tool is the *set* of fingerprints it was
  called with; declare the complete set before scoring any of its calls (see
  :meth:`DoomLoopMonitor.declare_round`).
- **Graduated response.** Detection feeds a configurable action ladder
  (observe → steer → escalate → block → stop).
- **Never the cause of failure.** Any internal error degrades to a fallback
  identity or skips detection — it never rejects the call or crashes the run.

Python-specific notes (deliberate divergences):

- Python has no ``undefined``. A ``loop_key`` function returning ``None`` is
  treated like upstream's ``null`` (exempt this call); there is no "returned
  undefined" fallback. ``None`` as the *declaration* means "absent" (full
  arguments), matching upstream's ``undefined`` declaration.
- Python ``int`` is arbitrary precision. JSON numbers are IEEE-754 doubles in
  JS, so integers are canonicalized through ``float`` (exactly as
  ``JSON.parse`` would have rounded them) to keep fingerprints identical across
  ports; integers outside the double range are rejected like non-finite
  numbers. There is no ``bigint`` analog.
- Callable values are dropped from objects and become ``null`` in arrays
  (upstream's function/``undefined`` semantics). Pydantic models are dumped in
  JSON mode first. Other non-JSON types raise ``TypeError``.
- Persisted-state and verdict keys are snake_case (``round_fingerprints``,
  ``call_streaks``, ``escalations_used``, ``stop_verdict``, ``pending_steer``,
  ``tool_name``), matching the rest of this port's ``ConversationState``.
  :meth:`DoomLoopMonitor.restore` additionally accepts upstream's camelCase
  keys so a blob written by the TypeScript SDK restores too.
- Warnings use :func:`warnings.warn` (upstream: ``console.warn``).
"""

from __future__ import annotations

import hashlib
import math
import re
import warnings
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast, overload

from typing_extensions import Literal, NotRequired, TypeAlias, TypedDict

__all__ = [
    "DEFAULT_DOOM_LOOP_LADDER",
    "DEFAULT_MAX_ESCALATIONS",
    "MAX_CANONICALIZE_DEPTH",
    "DoomLoopAction",
    "DoomLoopCallRecord",
    "DoomLoopConfig",
    "DoomLoopDetectorKind",
    "DoomLoopEscalationConfig",
    "DoomLoopLadder",
    "DoomLoopMonitor",
    "DoomLoopOption",
    "DoomLoopRoundCall",
    "DoomLoopSerializedState",
    "DoomLoopStreak",
    "DoomLoopTextOptions",
    "DoomLoopVerdict",
    "LoopKeyResolution",
    "ResolvedDoomLoopConfig",
    "ResolvedEscalationConfig",
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

#: Escalation actions, weakest to strongest. See upstream docs for semantics.
DoomLoopAction: TypeAlias = Literal["observe", "steer", "escalate", "block", "stop"]

#: Which detector produced a verdict.
DoomLoopDetectorKind: TypeAlias = Literal[
    "tool-fingerprint",
    "server-tool-fingerprint",
    "text-repetition",
    "text-streak",
]


class DoomLoopVerdict(TypedDict):
    """A doom-loop detection event.

    ``streak`` is the repetition count that crossed a ladder threshold.
    ``tool_name`` is present for fingerprint verdicts only. ``message`` is used
    verbatim for block outputs and steer messages.
    """

    detector: DoomLoopDetectorKind
    action: DoomLoopAction
    streak: int
    fingerprint: str
    tool_name: NotRequired[str]
    message: str


class DoomLoopCallRecord(TypedDict):
    """Result of recording one tool call with the monitor.

    ``duplicate_in_round`` is True when the same ``(tool_name, fingerprint)``
    was already recorded in this round: the streak did NOT increment and the
    caller should reuse the decision it applied for the first occurrence.
    ``verdict`` is present only when a ladder rung fired.
    """

    fingerprint: str
    streak: int
    duplicate_in_round: bool
    verdict: NotRequired[DoomLoopVerdict]


class DoomLoopLadder(TypedDict, total=False):
    """Streak thresholds per action; ``False`` disables a rung. Strongest wins."""

    observe: Union[int, Literal[False]]
    steer: Union[int, Literal[False]]
    escalate: Union[int, Literal[False]]
    block: Union[int, Literal[False]]
    stop: Union[int, Literal[False]]


class DoomLoopEscalationConfig(TypedDict, total=False):
    """Recovery configuration for the ``escalate`` ladder action."""

    #: Model slug to run the NEXT turn on (one request only).
    model: str
    #: Force an ``openrouter:advisor`` consult on the next turn. ``True`` uses
    #: defaults; a mapping is passed through as the advisor tool's parameters.
    advisor: Union[bool, Dict[str, Any]]
    #: Cap on escalations per run. Default :data:`DEFAULT_MAX_ESCALATIONS`.
    max_escalations: int


class DoomLoopTextOptions(TypedDict, total=False):
    """Options for text-loop detection."""

    #: Master switch for both text detectors. Default True.
    enabled: bool
    #: Longest repeating token block to search for. Default 16.
    max_period_tokens: int
    #: Minimum consecutive repeats of a block to count as repetition. Default 4.
    min_repeats: int
    #: Minimum total tokens covered by the repetition. Default 12.
    min_covered_tokens: int
    #: Only the trailing N tokens of a response are scanned. Default 400.
    max_window_tokens: int


class DoomLoopConfig(TypedDict, total=False):
    """Configuration for doom-loop detection. ``doom_loop=True`` uses all defaults."""

    ladder: DoomLoopLadder
    text: Union[bool, DoomLoopTextOptions]
    escalation: DoomLoopEscalationConfig


#: The ``doom_loop`` option on ``call_model``: ``True`` for defaults, or a config.
DoomLoopOption: TypeAlias = Union[bool, DoomLoopConfig]


class DoomLoopStreak(TypedDict):
    """Consecutive-repetition counter for one identity (a tool, or step text).

    ``round_fingerprints``: the full fingerprint set of the tool's last round,
    present only when that round had more than one distinct call.
    ``call_streaks``: per-call streaks for the tool's last round (fingerprint →
    consecutive rounds), omitted when reconstructible from the set + streak.
    Neither is ever present on text streaks.
    """

    fingerprint: str
    streak: int
    round_fingerprints: NotRequired[List[str]]
    call_streaks: NotRequired[Dict[str, int]]


class DoomLoopSerializedState(TypedDict, total=False):
    """Plain-JSON detector state, persisted inside ``ConversationState.doom_loop``.

    ``tools`` and ``text`` / ``escalations_used`` are produced by
    :meth:`DoomLoopMonitor.get_state`. ``stop_verdict`` and ``pending_steer``
    are engine-owned: the engine merges them into the persisted blob and
    restores them itself (the monitor ignores them).
    """

    tools: Dict[str, DoomLoopStreak]
    text: DoomLoopStreak
    stop_verdict: DoomLoopVerdict
    pending_steer: List[str]
    escalations_used: int


class _ExemptResolution(TypedDict):
    kind: Literal["exempt"]


class _KeyResolution(TypedDict):
    kind: Literal["key"]
    key_material: Any


class _FallbackResolution(TypedDict):
    kind: Literal["fallback"]
    key_material: Any
    warning: str


#: The result of resolving a tool's ``loop_key`` declaration against one call's
#: arguments. See :func:`resolve_loop_key_material`.
LoopKeyResolution: TypeAlias = Union[_ExemptResolution, _KeyResolution, _FallbackResolution]


class DoomLoopRoundCall(TypedDict):
    """One entry of :meth:`DoomLoopMonitor.declare_round` (upstream: inline type)."""

    tool_name: str
    key_material: Any


class TextRepetitionResult(TypedDict):
    """A detected repeating token block at the tail of a response."""

    #: Consecutive repeats of the block (feeds the ladder as the streak).
    repeats: int
    #: Block length in whitespace-delimited tokens.
    period_tokens: int
    #: repeats × period_tokens.
    covered_tokens: int
    #: The repeating block itself.
    sample: str


class ResolvedEscalationConfig(TypedDict):
    """Escalation config with the budget resolved."""

    model: NotRequired[str]
    advisor: NotRequired[Union[bool, Dict[str, Any]]]
    max_escalations: int


class ResolvedDoomLoopConfig(TypedDict):
    """Fully-resolved config used by the monitor (every ladder/text key present)."""

    ladder: DoomLoopLadder
    text: DoomLoopTextOptions
    escalation: Optional[ResolvedEscalationConfig]


# endregion

# region Defaults & Config Resolution

#: Default ladder: observe at 2 consecutive identical rounds, block at 3, stop
#: at 6. Steer and escalate are off by default. Read-only.
DEFAULT_DOOM_LOOP_LADDER: Mapping[str, Union[int, Literal[False]]] = MappingProxyType(
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

#: Ladder rungs ordered weakest → strongest.
_LADDER_ORDER: Tuple[str, ...] = ("observe", "steer", "escalate", "block", "stop")


def _is_number(value: object) -> bool:
    """``typeof value === 'number'`` — bool is NOT a number here."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    if not _is_number(value):
        return False
    try:
        return math.isfinite(cast(float, value))
    except OverflowError:
        return False


def _sanitize_threshold(value: object, fallback: Union[int, Literal[False]]) -> Union[int, Literal[False]]:
    if value is False:
        return False
    if _is_finite_number(value) and cast(float, value) >= 1:
        return int(math.floor(cast(float, value)))
    return fallback


def _warn(message: str) -> None:
    warnings.warn(message, stacklevel=3)


def _warn_on_ladder_hazards(ladder: DoomLoopLadder) -> None:
    """Warn about accepted-but-probably-wrong ladders (block without stop; dead rungs)."""
    if ladder["block"] is not False and ladder["stop"] is False:
        _warn(
            "[DoomLoop] ladder has block enabled with stop disabled: a model that keeps "
            "re-issuing a blocked call loops indefinitely (each blocked round still costs a "
            "model request). Bound the run with stop_when, or enable the stop rung."
        )
    rungs = cast(Mapping[str, Union[int, Literal[False]]], ladder)
    for weak_index, weak_name in enumerate(_LADDER_ORDER):
        weak_threshold = rungs[weak_name]
        if weak_threshold is False:
            continue
        for strong_name in _LADDER_ORDER[weak_index + 1 :]:
            strong_threshold = rungs[strong_name]
            if strong_threshold is not False and weak_threshold >= strong_threshold:
                _warn(
                    f'[DoomLoop] ladder rung "{weak_name}" ({weak_threshold}) can never fire: '
                    f'stronger rung "{strong_name}" ({strong_threshold}) already wins at that streak '
                    "(strongest crossed rung is applied)."
                )
                break  # one warning per dead rung is enough


@overload
def resolve_doom_loop_option(option: Union[DoomLoopConfig, Literal[True]]) -> ResolvedDoomLoopConfig: ...


@overload
def resolve_doom_loop_option(option: Optional[DoomLoopOption]) -> Optional[ResolvedDoomLoopConfig]: ...


def resolve_doom_loop_option(option: Optional[DoomLoopOption]) -> Optional[ResolvedDoomLoopConfig]:
    """Normalize the ``doom_loop`` option. ``None`` / ``False`` → ``None`` (off).

    ``True`` or a config mapping always yields a resolved config, so
    ``DoomLoopMonitor(resolve_doom_loop_option(True))`` is well-typed.
    """
    if option is None or option is False:
        return None
    config: Mapping[str, Any] = {} if option is True else cast(Mapping[str, Any], option)
    ladder_input: Mapping[str, Any] = config.get("ladder") or {}
    text_raw = config.get("text")
    text_input: Union[Literal[False], Mapping[str, Any]] = (
        {} if text_raw is None or text_raw is True else cast(Union[Literal[False], Mapping[str, Any]], text_raw)
    )
    ladder: DoomLoopLadder = {
        "observe": _sanitize_threshold(ladder_input.get("observe"), DEFAULT_DOOM_LOOP_LADDER["observe"]),
        "steer": _sanitize_threshold(ladder_input.get("steer"), DEFAULT_DOOM_LOOP_LADDER["steer"]),
        "escalate": _sanitize_threshold(ladder_input.get("escalate"), DEFAULT_DOOM_LOOP_LADDER["escalate"]),
        "block": _sanitize_threshold(ladder_input.get("block"), DEFAULT_DOOM_LOOP_LADDER["block"]),
        "stop": _sanitize_threshold(ladder_input.get("stop"), DEFAULT_DOOM_LOOP_LADDER["stop"]),
    }
    _warn_on_ladder_hazards(ladder)

    # Escalation recovery: usable only when the config names at least one
    # mechanism. `advisor: False` is an explicit opt-out, not a mechanism —
    # `{"advisor": False}` alone must read the same as `{}`.
    escalation_input = cast(Optional[Mapping[str, Any]], config.get("escalation"))
    advisor = escalation_input.get("advisor") if escalation_input is not None else None
    has_advisor = advisor is not None and advisor is not False
    model = escalation_input.get("model") if escalation_input is not None else None
    has_mechanism = escalation_input is not None and (model is not None or has_advisor)
    escalation: Optional[ResolvedEscalationConfig] = None
    if has_mechanism:
        assert escalation_input is not None
        cap = escalation_input.get("max_escalations")
        escalation = {
            "max_escalations": int(math.floor(cast(float, cap)))
            if _is_finite_number(cap) and cast(float, cap) >= 1
            else DEFAULT_MAX_ESCALATIONS
        }
        if model is not None:
            escalation["model"] = model
        if has_advisor:
            escalation["advisor"] = cast(Union[bool, Dict[str, Any]], advisor)
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

    text: DoomLoopTextOptions
    if text_input is False:
        text = cast(DoomLoopTextOptions, {**_DEFAULT_TEXT_OPTIONS, "enabled": False})
    else:

        def _pick(key: str) -> Any:
            value = text_input.get(key)
            return _DEFAULT_TEXT_OPTIONS[key] if value is None else value

        text = {
            "enabled": _pick("enabled"),
            "max_period_tokens": _pick("max_period_tokens"),
            "min_repeats": _pick("min_repeats"),
            "min_covered_tokens": _pick("min_covered_tokens"),
            "max_window_tokens": _pick("max_window_tokens"),
        }

    return {"ladder": ladder, "escalation": escalation, "text": text}


# endregion

# region loop_key Resolution


def resolve_loop_key_material(loop_key: object, args: Mapping[str, Any]) -> LoopKeyResolution:
    """Resolve a tool's ``loop_key`` declaration against one call's arguments.

    - ``None`` (absent declaration) → the full arguments.
    - ``False`` → the tool is statically exempt.
    - list/tuple of field names → the declarative subset. An empty list, or a
      list whose every field is absent, falls back to full arguments with a
      warning (it would give every call the same identity).
    - callable → called with the arguments. Returning ``None`` exempts THIS
      call (Python has no ``undefined``; see module docstring). A raise falls
      back to the full arguments with a warning.
    - anything else → fallback with a warning.
    """
    if loop_key is None:
        return {"kind": "key", "key_material": args}
    if loop_key is False:
        return {"kind": "exempt"}
    if isinstance(loop_key, (list, tuple)):
        if len(loop_key) == 0:
            return {
                "kind": "fallback",
                "key_material": args,
                "warning": (
                    "loop_key is an empty field list, which would give every call to this tool the same "
                    "identity; falling back to full arguments. Use `False` to exempt the tool."
                ),
            }
        subset: Dict[str, Any] = {}
        for field_name in loop_key:
            if isinstance(field_name, str) and field_name in args:
                subset[field_name] = args[field_name]
        if len(subset) == 0:
            names = ", ".join(f for f in loop_key if isinstance(f, str))
            return {
                "kind": "fallback",
                "key_material": args,
                "warning": (
                    f"loop_key fields [{names}] are all absent from the arguments, which would give every "
                    "call the same identity; falling back to full arguments."
                ),
            }
        return {"kind": "key", "key_material": subset}
    if callable(loop_key):
        try:
            result = loop_key(args)
        except Exception as error:  # detection must never take down a run
            return {
                "kind": "fallback",
                "key_material": args,
                "warning": f"loop_key threw ({error}); falling back to full arguments",
            }
        if result is None:
            return {"kind": "exempt"}
        return {"kind": "key", "key_material": result}
    return {
        "kind": "fallback",
        "key_material": args,
        "warning": f"loop_key has unsupported shape ({type(loop_key).__name__}); falling back to full arguments",
    }


# endregion

# region Canonicalization & Fingerprinting

#: Maximum nesting depth accepted by :func:`canonicalize_key_material`.
MAX_CANONICALIZE_DEPTH = 64

_SHORT_ESCAPES = {
    0x08: "\\b",
    0x09: "\\t",
    0x0A: "\\n",
    0x0C: "\\f",
    0x0D: "\\r",
    0x22: '\\"',
    0x5C: "\\\\",
}


def _quote_json_string(value: str) -> str:
    """ECMAScript ``JSON.stringify`` string quoting (the JCS string form).

    Escapes ``"``, ``\\``, the short control escapes, other C0 controls as
    lowercase ``\\u00xx``, and LONE surrogates as lowercase ``\\udxxx``. A
    surrogate PAIR stored as two code points is a valid pair in UTF-16 (what
    JS sees), so it is combined and emitted raw. Everything else is raw.
    """
    out: List[str] = ['"']
    i = 0
    length = len(value)
    while i < length:
        code = ord(value[i])
        if code in _SHORT_ESCAPES:
            out.append(_SHORT_ESCAPES[code])
        elif code < 0x20:
            out.append(f"\\u{code:04x}")
        elif 0xD800 <= code <= 0xDBFF:
            if i + 1 < length and 0xDC00 <= ord(value[i + 1]) <= 0xDFFF:
                low = ord(value[i + 1])
                out.append(chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)))
                i += 1
            else:
                out.append(f"\\u{code:04x}")
        elif 0xDC00 <= code <= 0xDFFF:
            out.append(f"\\u{code:04x}")
        else:
            out.append(value[i])
        i += 1
    out.append('"')
    return "".join(out)


def _es_number_to_string(value: float) -> str:
    """ECMAScript ``Number::toString`` for a finite double (the JCS number form).

    ``repr(float)`` yields the shortest round-tripping digit string (the same
    digits ECMAScript selects); this only reformats them per ES rules.
    """
    if value == 0:
        return "0"  # covers -0
    if value < 0:
        return "-" + _es_number_to_string(-value)
    sign, digit_tuple, exponent = Decimal(repr(value)).as_tuple()
    digits = list(digit_tuple)
    exp = int(exponent)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exp += 1
    while len(digits) > 1 and digits[0] == 0:
        digits.pop(0)
    text = "".join(str(d) for d in digits)
    k = len(text)
    n = exp + k
    if k <= n <= 21:
        return text + "0" * (n - k)
    if 0 < n <= 21:
        return text[:n] + "." + text[n:]
    if -6 < n <= 0:
        return "0." + "0" * (-n) + text
    e = n - 1
    exp_text = ("+" if e >= 0 else "-") + str(abs(e))
    if k == 1:
        return text + "e" + exp_text
    return text[0] + "." + text[1:] + "e" + exp_text


def _utf16_sort_key(value: str) -> bytes:
    return value.encode("utf-16-be", "surrogatepass")


def canonicalize_key_material(value: Any) -> str:
    """RFC 8785 (JCS) canonical JSON for doom-loop fingerprinting.

    Object keys sorted by UTF-16 code units (recursively), arrays in order,
    strings and finite numbers serialized per ECMAScript ``JSON.stringify``
    (``-0`` → ``0``, ``1e21`` → ``1e+21``, lone surrogates → ``\\udXXX``).
    Insensitive to key insertion order.

    Raises ``ValueError`` on values RFC 8785 cannot represent (non-finite
    numbers, integers outside the double range), circular references, and
    nesting deeper than :data:`MAX_CANONICALIZE_DEPTH`; ``TypeError`` on
    non-JSON types or non-string object keys. Real tool arguments (parsed JSON)
    never trigger these; the engine catches and falls back.
    """
    seen: Set[int] = set()

    def canon(v: Any, depth: int) -> str:
        if depth > MAX_CANONICALIZE_DEPTH:
            raise ValueError(
                f"Cannot canonicalize key material nested deeper than {MAX_CANONICALIZE_DEPTH} levels "
                "for doom-loop fingerprinting"
            )
        if v is None:
            return "null"
        if v is True:
            return "true"
        if v is False:
            return "false"
        if isinstance(v, int):
            try:
                as_float = float(v)
            except OverflowError:
                raise ValueError(
                    "Cannot canonicalize non-finite number (integer outside the IEEE-754 double range) "
                    "for doom-loop fingerprinting: RFC 8785 has no representation"
                ) from None
            return _es_number_to_string(as_float)
        if isinstance(v, float):
            if not math.isfinite(v):
                raise ValueError(
                    "Cannot canonicalize non-finite number (NaN/Infinity) for doom-loop fingerprinting: "
                    "RFC 8785 has no representation"
                )
            return _es_number_to_string(v)
        if isinstance(v, str):
            return _quote_json_string(v)
        if _is_pydantic_model(v):
            return canon(v.model_dump(mode="json"), depth)
        if isinstance(v, (list, tuple, Mapping)):
            marker = id(v)
            if marker in seen:
                raise ValueError("Cannot canonicalize circular key material for doom-loop fingerprinting")
            seen.add(marker)
            try:
                if isinstance(v, (list, tuple)):
                    return "[" + ",".join(canon(item, depth + 1) for item in v) + "]"
                entries: List[Tuple[str, Any]] = []
                for key, entry in v.items():
                    if not isinstance(key, str):
                        raise TypeError(
                            f"Cannot canonicalize object key of type {type(key).__name__} for doom-loop "
                            "fingerprinting: JSON object keys are strings"
                        )
                    if callable(entry) and not _is_pydantic_model(entry):
                        continue
                    entries.append((key, entry))
                entries.sort(key=lambda pair: _utf16_sort_key(pair[0]))
                return (
                    "{"
                    + ",".join(f"{_quote_json_string(key)}:{canon(entry, depth + 1)}" for key, entry in entries)
                    + "}"
                )
            finally:
                seen.discard(marker)
        if callable(v):
            # function — not representable; JSON semantics say null here.
            return "null"
        raise TypeError(
            f"Cannot canonicalize value of type {type(v).__name__} for doom-loop fingerprinting: "
            "RFC 8785 has no representation (convert it in loop_key)"
        )

    return canon(value, 0)


def _is_pydantic_model(value: object) -> bool:
    try:
        from pydantic import BaseModel
    except ImportError:  # pragma: no cover - pydantic is a hard dependency
        return False
    return isinstance(value, BaseModel)


def _sha256_hex(canonical: str) -> str:
    # Lone surrogates never reach the encoder: _quote_json_string escapes them.
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def fingerprint_key_material(key_material: Any) -> str:
    """Fingerprint arbitrary key material: ``sha256(utf8(jcs(key_material)))``, lowercase hex."""
    return _sha256_hex(canonicalize_key_material(key_material))


async def fingerprint_tool_call(tool_name: str, key_material: Any) -> str:
    """Fingerprint a tool call: ``sha256(utf8(tool_name + "\\n" + jcs(key_material)))``.

    This exact construction is the cross-port contract — see
    ``tests/vectors/doom-loop-fingerprints.json``.
    """
    return _sha256_hex(_tool_call_canonical(tool_name, key_material))


def _tool_call_canonical(tool_name: str, key_material: Any) -> str:
    return f"{tool_name}\n{canonicalize_key_material(key_material)}"


# endregion

# region Text Repetition Detection

# ECMAScript `\s` (WhiteSpace + LineTerminator). Python's `\s` is a different
# set (it includes U+001C..U+001F and excludes U+FEFF), and the normalized text
# is fingerprinted, so the class is spelled out for cross-port parity.
_JS_WHITESPACE = "\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_JS_WS_RUN = re.compile(f"[{_JS_WHITESPACE}]+")
_JS_TRIM = re.compile(f"^[{_JS_WHITESPACE}]+|[{_JS_WHITESPACE}]+$")


def _js_trim(text: str) -> str:
    return _JS_TRIM.sub("", text)


def _utf16_tail(text: str, budget: int) -> str:
    """``text.slice(-budget)`` measured in UTF-16 code units, like JS."""
    if len(text) * 2 <= budget:
        return text
    encoded = text.encode("utf-16-le", "surrogatepass")
    if len(encoded) // 2 <= budget:
        return text
    return encoded[-budget * 2 :].decode("utf-16-le", "surrogatepass")


def _option(options: Optional[Mapping[str, Any]], key: str) -> Any:
    value = options.get(key) if options is not None else None
    return _DEFAULT_TEXT_OPTIONS[key] if value is None else value


def detect_text_repetition(
    text: str,
    options: Optional[DoomLoopTextOptions] = None,
) -> Optional[TextRepetitionResult]:
    """Detect a period-p token block repeating at the *tail* of ``text``.

    Whitespace tokenization, suffix comparison, no heuristics beyond the
    thresholds. Only the trailing ``max_window_tokens * 64`` characters are
    tokenized. Ties on covered tokens prefer the smallest period. Returns
    ``None`` when no block meets ``min_repeats`` and ``min_covered_tokens``.

    KNOWN LIMIT: token blocks must repeat *exactly*.
    """
    opts = cast(Optional[Mapping[str, Any]], options)
    max_period_tokens = int(_option(opts, "max_period_tokens"))
    min_repeats = int(_option(opts, "min_repeats"))
    min_covered_tokens = int(_option(opts, "min_covered_tokens"))
    max_window_tokens = int(_option(opts, "max_window_tokens"))

    char_budget = max_window_tokens * 64
    tail = _utf16_tail(text, char_budget)
    all_tokens = [token for token in _JS_WS_RUN.split(_js_trim(tail)) if token]
    tokens = all_tokens[-max_window_tokens:] if max_window_tokens > 0 else []
    if len(tokens) < max(min_covered_tokens, 2):
        return None

    best: Optional[TextRepetitionResult] = None
    count = len(tokens)
    max_period = min(max_period_tokens, count // 2)
    for period in range(1, max_period + 1):
        repeats = 1
        tail_block = tokens[count - period :]
        start = count - 2 * period
        while start >= 0:
            if tokens[start : start + period] != tail_block:
                break
            repeats += 1
            start -= period
        covered_tokens = repeats * period
        if repeats >= min_repeats and covered_tokens >= min_covered_tokens:
            if best is None or covered_tokens > best["covered_tokens"]:
                best = {
                    "repeats": repeats,
                    "period_tokens": period,
                    "covered_tokens": covered_tokens,
                    "sample": " ".join(tail_block),
                }
    return best


# endregion

# region Ladder Resolution


def resolve_ladder_action(
    ladder: DoomLoopLadder,
    streak: float,
    *,
    allow_block: bool,
    allow_escalate: bool = True,
) -> Optional[DoomLoopAction]:
    """Map a streak onto the strongest crossed ladder rung.

    ``allow_block=False`` (text and server-tool verdicts) lets a block-level
    streak fall through to escalate/steer/observe, while stop still stops.
    ``allow_escalate=False`` disables the escalate rung (no mechanism, or the
    budget is exhausted). A missing rung key is treated as disabled.
    """
    rungs = cast(Mapping[str, Any], ladder)

    def meets(name: str) -> bool:
        threshold = rungs.get(name, False)
        return threshold is not False and threshold is not None and streak >= threshold

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
    """In-memory streak entry: serialized shape + run-local round bookkeeping."""

    fingerprint: str
    streak: int
    #: The round's complete declared fingerprint set, sorted.
    round_fingerprints: Optional[List[str]] = None
    #: Current round's per-call streaks (fingerprint → consecutive rounds).
    call_streaks: Optional[Dict[str, int]] = None
    #: Round of the most recent record. Never serialized.
    round: Optional[int] = None
    #: Fingerprints of `round` already recorded (in-round duplicate collapse).
    seen_this_round: Set[str] = field(default_factory=set)
    #: The previous round's set and streak, fixed at the round transition.
    prior_round_fingerprints: Optional[List[str]] = None
    prior_streak: int = 0
    #: The previous round's per-call streaks, fixed at the round transition.
    prior_call_streaks: Dict[str, int] = field(default_factory=dict)


def _get_key(mapping: Mapping[str, Any], snake: str, camel: str) -> Any:
    """Read a persisted field, accepting upstream's camelCase spelling too."""
    if snake in mapping:
        return mapping[snake]
    return mapping.get(camel)


def _is_valid_streak(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("fingerprint"), str)
        and _is_finite_number(value.get("streak"))
    )


def _call_streaks_reconstructible(entry: _StreakEntry) -> bool:
    """True when per-call counts carry nothing beyond the round set + streak."""
    counts = entry.call_streaks
    if counts is None:
        return True
    members = entry.round_fingerprints if entry.round_fingerprints is not None else [entry.fingerprint]
    return len(counts) == len(members) and all(key in members and counts[key] == entry.streak for key in counts)


def _restore_streak_entry(entry: Mapping[str, Any]) -> _StreakEntry:
    """Rebuild one tool's in-memory streak entry from its persisted shape."""
    fingerprint = cast(str, entry["fingerprint"])
    raw_set = _get_key(entry, "round_fingerprints", "roundFingerprints")
    persisted_set: List[str] = (
        sorted(raw_set)
        if isinstance(raw_set, list) and len(raw_set) > 1 and all(isinstance(v, str) for v in raw_set)
        else [fingerprint]
    )
    persisted_call_streaks: Dict[str, int] = {}
    raw_counts = _get_key(entry, "call_streaks", "callStreaks")
    if isinstance(raw_counts, Mapping):
        for call_fingerprint, count in raw_counts.items():
            if isinstance(call_fingerprint, str) and _is_finite_number(count) and count >= 1:
                persisted_call_streaks[call_fingerprint] = int(math.floor(count))
    # Clamp a corrupt streak to a sane count rather than dropping the entry.
    streak = min(max(int(math.floor(cast(float, entry["streak"]))), 1), 1_000_000)
    return _StreakEntry(
        fingerprint=fingerprint,
        streak=streak,
        round_fingerprints=persisted_set,
        call_streaks=(
            persisted_call_streaks if persisted_call_streaks else {member: streak for member in persisted_set}
        ),
    )


_ACTION_STRENGTH: Mapping[str, int] = MappingProxyType({"observe": 0, "steer": 1, "escalate": 2, "block": 3, "stop": 4})


def _stronger_verdict(a: Optional[DoomLoopVerdict], b: Optional[DoomLoopVerdict]) -> Optional[DoomLoopVerdict]:
    if a is None:
        return b
    if b is None:
        return a
    # Within-response wins ties: its message names the repeated block.
    return b if _ACTION_STRENGTH[b["action"]] > _ACTION_STRENGTH[a["action"]] else a


def _summarize_round(fingerprints: Sequence[str]) -> str:
    shown = [value[:8] for value in fingerprints[:3]]
    if len(fingerprints) > 3:
        return f"{'+'.join(shown)}+{len(fingerprints) - 3} more…"
    return f"{'+'.join(shown)}…"


def _build_tool_verdict_message(
    *,
    tool_name: str,
    fingerprint: str,
    call_set: Sequence[str],
    round_streak: int,
    call_streak: int,
    round_declared: bool,
) -> str:
    """Verdict text for a tool-fingerprint detection (see upstream for the dedupe rationale)."""
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


class DoomLoopMonitor:
    """Pure state machine over the recorded transcript.

    Feed it tool calls and assistant texts, get verdicts back. Holds no engine
    concerns (hook emission, blocking, steering, stopping live in
    ``ModelResult``). State is bounded plain JSON and round-trips through
    ``ConversationState.doom_loop`` via :meth:`get_state` / :meth:`restore`.
    """

    def __init__(self, config: ResolvedDoomLoopConfig, initial_state: Any = None) -> None:
        self._config = config
        # Plain dict keyed by tool name: Python dict keys are inert data, so a
        # tool named "__proto__" (or any hostile persisted name) just works.
        self._tools: Dict[str, _StreakEntry] = {}
        self._text: Optional[_StreakEntry] = None
        self._escalations_used = 0
        self._declared_round: Optional[Tuple[int, Dict[str, List[str]]]] = None
        # Per-object fingerprint memo (upstream: WeakMap). Python dicts/lists
        # are not weak-referenceable, so the memo holds a strong reference
        # (keeping `id()` stable) and is scoped to ONE round — it is cleared
        # whenever a different round is declared or recorded, which bounds its
        # size to one round's calls. Within a round it gives upstream's
        # behavior: the same key-material object handed to `declare_round` and
        # then `record_tool_call` is hashed once, with declare-time identity.
        self._memo: Dict[int, Tuple[object, Dict[str, str]]] = {}
        self._memo_round: Optional[int] = None
        if initial_state is not None:
            self.restore(initial_state)

    # -- fingerprint memo -------------------------------------------------

    def _enter_memo_round(self, round: int) -> None:
        if self._memo_round != round:
            self._memo.clear()
            self._memo_round = round

    def _fingerprint_once(self, tool_name: str, key_material: Any) -> str:
        memoizable = isinstance(key_material, (dict, list)) or (
            key_material is not None and not isinstance(key_material, (str, int, float, bool, tuple))
        )
        if memoizable:
            hit = self._memo.get(id(key_material))
            if hit is not None and hit[0] is key_material:
                cached = hit[1].get(tool_name)
                if cached is not None:
                    return cached
        fingerprint = _sha256_hex(_tool_call_canonical(tool_name, key_material))
        if memoizable:
            slot = self._memo.get(id(key_material))
            if slot is None or slot[0] is not key_material:
                slot = (key_material, {})
                self._memo[id(key_material)] = slot
            slot[1][tool_name] = fingerprint
        return fingerprint

    # -- rounds -----------------------------------------------------------

    async def declare_round(self, round: int, calls: Sequence[DoomLoopRoundCall]) -> None:
        """Declare the complete set of calls a round will make, before any is recorded.

        Each entry is one call's ``{"tool_name", "key_material"}``; the monitor
        groups them per tool into that tool's round set. Idempotent per round,
        and safe to skip (per-call detection needs no declaration). Unhashable
        key material is excluded from the set with a warning. Never serialized.
        """
        self._enter_memo_round(round)
        collected: Dict[str, Set[str]] = {}
        for call in calls:
            tool_name = call["tool_name"]
            try:
                fingerprint = self._fingerprint_once(tool_name, call["key_material"])
            except Exception as error:  # fail open per call
                _warn(
                    f'[DoomLoop] could not fingerprint a "{tool_name}" call while declaring round '
                    f"{round}; excluding it from the round set: {error}"
                )
                continue
            collected.setdefault(tool_name, set()).add(fingerprint)
        self._declared_round = (round, {name: sorted(members) for name, members in collected.items()})

    # -- escalation budget ------------------------------------------------

    def can_escalate(self) -> bool:
        """True when a mechanism is configured and the budget is not exhausted."""
        escalation = self._config["escalation"]
        return escalation is not None and self._escalations_used < escalation["max_escalations"]

    def consume_escalation(self) -> None:
        """Consume one escalation. The ENGINE calls this when it applies the recovery."""
        self._escalations_used += 1

    # -- persistence ------------------------------------------------------

    def get_state(self) -> DoomLoopSerializedState:
        """Snapshot the serializable detector state (deep copy; round markers dropped).

        ``stop_verdict`` / ``pending_steer`` are engine-owned and merged by the
        engine alongside this snapshot.
        """
        tools: Dict[str, DoomLoopStreak] = {}
        for name, entry in self._tools.items():
            streak: DoomLoopStreak = {"fingerprint": entry.fingerprint, "streak": entry.streak}
            if entry.round_fingerprints is not None and len(entry.round_fingerprints) > 1:
                streak["round_fingerprints"] = list(entry.round_fingerprints)
            if entry.call_streaks is not None and not _call_streaks_reconstructible(entry):
                streak["call_streaks"] = dict(entry.call_streaks)
            tools[name] = streak
        state: DoomLoopSerializedState = {"tools": tools}
        if self._text is not None:
            state["text"] = {"fingerprint": self._text.fingerprint, "streak": self._text.streak}
        if self._escalations_used > 0:
            state["escalations_used"] = self._escalations_used
        return state

    def restore(self, state: Any) -> None:
        """Restore persisted state (from ``ConversationState.doom_loop``).

        Invalid blobs are ignored with a warning. Engine-owned fields
        (``stop_verdict``, ``pending_steer``) are ignored here.
        """
        if not isinstance(state, Mapping):
            _warn("[DoomLoop] Ignoring invalid persisted doom-loop state")
            return
        tools: Dict[str, _StreakEntry] = {}
        raw_tools = state.get("tools")
        if isinstance(raw_tools, Mapping):
            for name, entry in raw_tools.items():
                if isinstance(name, str) and _is_valid_streak(entry):
                    tools[name] = _restore_streak_entry(entry)
        self._tools = tools
        raw_text = state.get("text")
        self._text = (
            _StreakEntry(fingerprint=raw_text["fingerprint"], streak=raw_text["streak"])
            if isinstance(raw_text, Mapping) and _is_valid_streak(raw_text)
            else None
        )
        used = _get_key(state, "escalations_used", "escalationsUsed")
        self._escalations_used = int(math.floor(used)) if _is_finite_number(used) and used >= 0 else 0

    # -- tool calls -------------------------------------------------------

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

        Per tool: interleaved calls to other tools do not reset a tool's
        streak. Per round: the same ``(tool, fingerprint)`` recorded again in
        the SAME ``round`` is a duplicate. A round's identity is the *set* of
        fingerprints (declare it first via :meth:`declare_round`). The
        stronger of the round-set streak and the per-call streak decides.

        ``allow_block=False`` for post-execution records (server tools).

        Raises whatever :func:`canonicalize_key_material` raises for
        unhashable key material; callers must catch and fall back.
        """
        self._enter_memo_round(round)
        fingerprint = self._fingerprint_once(tool_name, key_material)
        previous = self._tools.get(tool_name)
        is_same_round = previous is not None and previous.round is not None and previous.round == round

        declared: Optional[List[str]] = None
        if self._declared_round is not None and self._declared_round[0] == round:
            declared = self._declared_round[1].get(tool_name)
        declared_member = declared is not None and fingerprint in declared

        seen: Set[str] = previous.seen_this_round if is_same_round and previous is not None else set()
        duplicate_in_round = fingerprint in seen

        prior_set: Optional[List[str]]
        prior_streak: int
        prior_call_streaks: Dict[str, int]
        if is_same_round:
            assert previous is not None
            prior_set = previous.prior_round_fingerprints
            prior_streak = previous.prior_streak
            prior_call_streaks = previous.prior_call_streaks
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
            return prior_streak + 1 if prior_set is not None and list(prior_set) == list(members) else 1

        call_streak = prior_call_streaks.get(fingerprint, 0) + 1

        call_set: List[str] = declared if declared_member and declared is not None else [fingerprint]
        streak = score(call_set)
        round_set: List[str] = declared if declared is not None else [fingerprint]

        if is_same_round and previous is not None and previous.call_streaks is not None:
            previous.call_streaks[fingerprint] = call_streak
            call_streaks = previous.call_streaks
        else:
            call_streaks = {fingerprint: call_streak}
        seen.add(fingerprint)

        self._tools[tool_name] = _StreakEntry(
            fingerprint=(
                previous.fingerprint
                if declared is not None and not declared_member and previous is not None
                else fingerprint
            ),
            round=round,
            seen_this_round=seen,
            round_fingerprints=round_set,
            streak=score(round_set),
            prior_round_fingerprints=prior_set,
            prior_streak=prior_streak,
            prior_call_streaks=prior_call_streaks,
            call_streaks=call_streaks,
        )

        effective_streak = max(streak, call_streak)
        action = resolve_ladder_action(
            self._config["ladder"],
            effective_streak,
            allow_block=allow_block,
            allow_escalate=self.can_escalate(),
        )
        record: DoomLoopCallRecord = {
            "fingerprint": fingerprint,
            "streak": effective_streak,
            "duplicate_in_round": duplicate_in_round,
        }
        if action is None:
            return record
        record["verdict"] = {
            "detector": detector,
            "action": action,
            "streak": effective_streak,
            "fingerprint": fingerprint,
            "tool_name": tool_name,
            "message": _build_tool_verdict_message(
                tool_name=tool_name,
                fingerprint=fingerprint,
                call_set=call_set,
                round_streak=streak,
                call_streak=call_streak,
                round_declared=declared_member,
            ),
        }
        return record

    # -- text -------------------------------------------------------------

    def reset_text_streak(self) -> None:
        """Clear the cross-step text streak (a late async-tool result is progress)."""
        self._text = None

    async def record_assistant_text(self, text: str) -> Optional[DoomLoopVerdict]:
        """Record one step's assistant text; return the strongest text verdict.

        ``text-repetition``: a token block repeating within THIS response.
        ``text-streak``: identical (whitespace-normalized) text across
        consecutive steps. Empty/whitespace-only text is a no-op.
        """
        text_options = self._config["text"]
        if not text_options.get("enabled", True):
            return None
        normalized = _js_trim(_JS_WS_RUN.sub(" ", text))
        if len(normalized) == 0:
            return None

        within_response: Optional[DoomLoopVerdict] = None
        repetition = detect_text_repetition(normalized, text_options)
        if repetition is not None:
            action = resolve_ladder_action(
                self._config["ladder"],
                repetition["repeats"],
                allow_block=False,
                allow_escalate=self.can_escalate(),
            )
            if action is not None:
                within_response = {
                    "detector": "text-repetition",
                    "action": action,
                    "streak": repetition["repeats"],
                    "fingerprint": await fingerprint_key_material(repetition["sample"]),
                    "message": (
                        f'Doom loop suspected: the response repeats "{repetition["sample"]}" '
                        f"{repetition['repeats']} times in a row. Stop repeating and take a different approach."
                    ),
                }

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
            cross_step = {
                "detector": "text-streak",
                "action": cross_action,
                "streak": streak,
                "fingerprint": fingerprint,
                "message": (
                    f"Doom loop suspected: the assistant produced identical text for {streak} "
                    "consecutive turns. Stop repeating and take a different approach."
                ),
            }

        return _stronger_verdict(within_response, cross_step)


# endregion
