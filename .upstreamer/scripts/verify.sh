#!/usr/bin/env bash
# Mechanical verification for the Python port. Objective checks only —
# judgment-heavy parity review lives in .upstreamer/eval.md.
set -uo pipefail
cd "$(dirname "$0")/../.."

FAILURES=0
pass() { echo "  PASS: $1"; }
fail() { echo "  FAIL: $1"; FAILURES=$((FAILURES + 1)); }

run() {
  local label="$1"; shift
  if "$@" >/tmp/verify-out 2>&1; then
    pass "$label"
  else
    fail "$label"
    sed 's/^/        /' /tmp/verify-out | tail -40
  fi
}

echo "=== Verification: python-agent ==="
echo

# Coverage ratchet. Must match --cov-fail-under in .github/workflows/ci.yaml.
# Raise when coverage rises; never lower it to make a port run pass.
COVERAGE_FLOOR=83

echo "-- Toolchain"
if command -v uv >/dev/null 2>&1; then
  # --frozen: fail on uv.lock / pyproject.toml drift instead of silently
  # resolving something other than what was reviewed.
  run "uv sync"        uv sync --frozen --all-extras
  run "lockfile in sync" uv lock --check
  run "ruff check"     uv run ruff check .
  run "ruff format"    uv run ruff format --check .
  # tests included: a fake client or payload builder with an unchecked Optional
  # deref is exactly how an assertion silently stops asserting.
  run "mypy"           uv run mypy src tests
  run "pytest + coverage floor ($COVERAGE_FLOOR%)" \
      uv run pytest tests/unit -q --cov --cov-fail-under="$COVERAGE_FLOOR"
else
  fail "uv not installed (required to build and test this package)"
fi
echo

# Required public API. This is the parity floor from .upstreamer/upstreamer.md.
# Presence only — the eval judges behavior.
echo "-- Required public API importable from openrouter_agent"
REQUIRED_SYMBOLS=(
  call_model OpenRouter tool server_tool ModelResult
  create_initial_state append_to_messages update_state partition_tool_calls
  serialize_conversation_state deserialize_conversation_state CONVERSATION_STATE_VERSION
  step_count_is has_tool_call max_tokens_used max_cost finish_reason_is
  HooksManager HookName
  DEFAULT_FINAL_RESPONSE_DIRECTIVE
  from_claude_messages to_claude_message from_chat_messages to_chat_message
  extract_unsupported_content has_unsupported_content get_unsupported_content_summary
  is_claude_style_messages
  ToolContextStore ToolEventBroadcaster SDKHooks
)
if command -v uv >/dev/null 2>&1; then
  missing=$(uv run python - "${REQUIRED_SYMBOLS[@]}" <<'PY' 2>/dev/null
import importlib, sys
mod = importlib.import_module("openrouter_agent")
print(" ".join(n for n in sys.argv[1:] if not hasattr(mod, n)))
PY
)
  status=$?
  if [ $status -ne 0 ]; then
    fail "could not import openrouter_agent to check public API"
  elif [ -n "${missing// /}" ]; then
    fail "missing public API symbols: $missing"
  else
    pass "all ${#REQUIRED_SYMBOLS[@]} required symbols exported"
  fi
fi
echo

echo "-- Version consistency"
declared=$(grep -m1 '^version' pyproject.toml | sed 's/.*"\(.*\)".*/\1/')
upstream_pkg="tmp/upstreamer/upstream/packages/agent/package.json"
if [ -f "$upstream_pkg" ]; then
  target=$(grep -m1 '"version"' "$upstream_pkg" | sed 's/.*"version": *"\([^"]*\)".*/\1/')
  if [ "$declared" = "$target" ]; then
    pass "version $declared matches ported @openrouter/agent $target"
  else
    fail "version drift: pyproject.toml=$declared, upstream @openrouter/agent=$target"
  fi
else
  # Only present during a sync run. Standalone/CI invocations legitimately have no
  # upstream checkout; not a failure, but say so rather than passing silently.
  echo "  SKIP: no upstream checkout — version parity unchecked (declared $declared)"
fi
echo

# The suite is what makes "a version behind on real behavior" visible or
# invisible, so the file-level mapping is mechanically checked. Advisory: which
# gaps are acceptable is a judgment call, and .upstreamer/eval.md makes it. This
# just ensures nobody has to notice the gap on their own.
echo "-- Test parity with upstream (advisory)"
upstream_tests="tmp/upstreamer/upstream/packages/agent/tests/unit"
if [ -d "$upstream_tests" ]; then
  unported=""
  for ts in "$upstream_tests"/*.test.ts; do
    [ -e "$ts" ] || continue
    base=$(basename "$ts" .test.ts | tr '-' '_')
    [ -f "tests/unit/test_${base}.py" ] || unported="$unported ${base}"
  done
  if [ -z "${unported// /}" ]; then
    pass "every upstream tests/unit file has a Python counterpart"
  else
    count=$(printf '%s' "$unported" | wc -w | tr -d ' ')
    echo "  NOTE: $count upstream test file(s) have no tests/unit counterpart:"
    for name in $unported; do echo "        $name.test.ts -> tests/unit/test_$name.py"; done
    echo "        Not a mechanical failure — see the Test Parity section of"
    echo "        .upstreamer/upstreamer.md and let the eval judge severity."
  fi
else
  echo "  SKIP: no upstream checkout — test parity unchecked"
fi
echo

echo "-- No leaked TypeScript artifacts"
leaked=$(find src tests -type f \( -name '*.ts' -o -name '*.js' -o -name 'package.json' \
  -o -name 'tsconfig*.json' -o -name 'pnpm-lock.yaml' \) 2>/dev/null)
[ -z "$leaked" ] && pass "no TS/JS artifacts in src or tests" \
                 || fail "leaked upstream artifacts: $leaked"

echo "-- Repo-owned files present"
for f in LICENSE README.md pyproject.toml scripts/upstream; do
  [ -e "$f" ] && pass "$f present" || fail "$f missing (port must not delete repo-owned files)"
done
echo

if [ "$FAILURES" -eq 0 ]; then
  echo "=== PASS: 0 failures ==="
  exit 0
fi
echo "=== FAIL: $FAILURES failure(s) ==="
exit 1
