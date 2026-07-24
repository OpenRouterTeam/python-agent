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

echo "-- Toolchain"
if command -v uv >/dev/null 2>&1; then
  run "uv sync"        uv sync --all-extras
  run "ruff check"     uv run ruff check .
  run "ruff format"    uv run ruff format --check .
  run "mypy"           uv run mypy src
  run "pytest"         uv run pytest tests/unit -q
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
  echo "  SKIP: no upstream checkout to compare version against"
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
