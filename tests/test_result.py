"""Tests for the Result type."""

from openrouter_agent import Err, Ok


def test_ok_is_ok():
    result = Ok(42)
    assert result.is_ok()
    assert not result.is_err()
    assert result.unwrap() == 42


def test_err_is_err():
    result = Err("something went wrong")
    assert result.is_err()
    assert not result.is_ok()
    assert result.unwrap_err() == "something went wrong"


def test_ok_unwrap_err_raises():
    result = Ok(42)
    try:
        result.unwrap_err()
        assert False, "Should have raised"
    except ValueError:
        pass


def test_err_unwrap_raises():
    result = Err("error")
    try:
        result.unwrap()
        assert False, "Should have raised"
    except ValueError:
        pass


def test_ok_value_types():
    assert Ok("hello").unwrap() == "hello"
    assert Ok([1, 2, 3]).unwrap() == [1, 2, 3]
    assert Ok({"key": "val"}).unwrap() == {"key": "val"}
    assert Ok(None).unwrap() is None


# --- map ---

def test_ok_map():
    assert Ok(42).map(str) == Ok("42")


def test_err_map():
    assert Err("e").map(str) == Err("e")


# --- map_err ---

def test_ok_map_err():
    assert Ok(42).map_err(str.upper) == Ok(42)


def test_err_map_err():
    assert Err("e").map_err(str.upper) == Err("E")


# --- and_then ---

def test_ok_and_then():
    assert Ok(42).and_then(lambda x: Ok(x + 1)) == Ok(43)


def test_ok_and_then_err():
    assert Ok(42).and_then(lambda x: Err("fail")) == Err("fail")


def test_err_and_then():
    assert Err("e").and_then(lambda x: Ok(x + 1)) == Err("e")


# --- or_else ---

def test_ok_or_else():
    assert Ok(42).or_else(lambda e: Ok(0)) == Ok(42)


def test_err_or_else():
    assert Err("e").or_else(lambda e: Ok(0)) == Ok(0)


def test_err_or_else_err():
    assert Err("e").or_else(lambda e: Err("new")) == Err("new")


# --- unwrap_or ---

def test_ok_unwrap_or():
    assert Ok(42).unwrap_or(0) == 42


def test_err_unwrap_or():
    assert Err("e").unwrap_or(0) == 0
