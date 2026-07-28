from __future__ import annotations

import re

from openrouter_agent.hooks_matchers import matches_tool


def test_matches_all_tools_when_matcher_is_none() -> None:
    assert matches_tool(None, "Bash") is True
    assert matches_tool(None, "ReadFile") is True


def test_matches_exact_string() -> None:
    assert matches_tool("Bash", "Bash") is True
    assert matches_tool("Bash", "ReadFile") is False
    assert matches_tool("Bash", "bash") is False


def test_matches_regex_pattern() -> None:
    pattern = re.compile(r"^(Read|Write)File$")
    assert matches_tool(pattern, "ReadFile") is True
    assert matches_tool(pattern, "WriteFile") is True
    assert matches_tool(pattern, "DeleteFile") is False


def test_matches_function_predicate() -> None:
    matcher = lambda name: name.startswith("File")  # noqa: E731
    assert matches_tool(matcher, "FileRead") is True
    assert matches_tool(matcher, "Bash") is False


def test_regex_matcher_is_stateless_across_repeated_calls() -> None:
    # Python compiled patterns have no JS-style lastIndex statefulness, but
    # pin the repeated-call behavior anyway since it mirrors an upstream fix.
    pattern = re.compile(r"^tool_\w+$")
    for _ in range(3):
        assert matches_tool(pattern, "tool_abc") is True
