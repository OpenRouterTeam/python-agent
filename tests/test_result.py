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
