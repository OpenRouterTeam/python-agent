"""Result type for error handling - Ok[T] | Err[E]."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")
F = TypeVar("F")


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """Successful result containing a value."""

    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value

    def unwrap_err(self) -> None:
        raise ValueError("Called unwrap_err on Ok value")

    def map(self, fn: Callable[[T], U]) -> Ok[U]:
        """Transform the success value."""
        return Ok(fn(self.value))

    def map_err(self, fn: Callable[..., object]) -> Ok[T]:
        """No-op on Ok; return self unchanged."""
        return self

    def and_then(self, fn: Callable[[T], Ok[U] | Err[E]]) -> Ok[U] | Err[E]:
        """Chain a function that returns a Result on the success value."""
        return fn(self.value)

    def or_else(self, fn: Callable[..., object]) -> Ok[T]:
        """No-op on Ok; return self unchanged."""
        return self

    def unwrap_or(self, default: object) -> T:
        """Return the success value, ignoring the default."""
        return self.value


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    """Error result containing an error."""

    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> None:
        raise ValueError(f"Called unwrap on Err value: {self.error}")

    def unwrap_err(self) -> E:
        return self.error

    def map(self, fn: Callable[..., object]) -> Err[E]:
        """No-op on Err; return self unchanged."""
        return self

    def map_err(self, fn: Callable[[E], F]) -> Err[F]:
        """Transform the error value."""
        return Err(fn(self.error))

    def and_then(self, fn: Callable[..., object]) -> Err[E]:
        """No-op on Err; return self unchanged."""
        return self

    def or_else(self, fn: Callable[[E], Ok[T] | Err[F]]) -> Ok[T] | Err[F]:
        """Chain a function that returns a Result on the error value."""
        return fn(self.error)

    def unwrap_or(self, default: U) -> U:
        """Return the default value since this is an Err."""
        return default


Result = Ok[T] | Err[E]
