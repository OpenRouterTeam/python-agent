# Porting

This package is a **port** of the OpenRouter TypeScript Agent SDK
(`@openrouter/agent`). TypeScript is the reference spec; this repo tracks it
automatically using [Upstreamer](https://github.com/mountgram/upstreamer) (MIT).

Behavioral divergence from the TypeScript reference is a **bug**, unless it is
listed in the Idiomatic Divergences section of `.upstreamer/upstreamer.md`.

## How it works

The port tracks upstream's **default-branch HEAD**, not the latest published npm
release.

```
   weekly cron  ·  publish dispatch  ·  manual dispatch
        │              (a nudge,          (optional
        │               not a ref)         explicit ref)
        ▼
.github/workflows/upstreamer-port.yaml
        │  resolve ref: explicit input, else upstream HEAD
        ▼
scripts/upstream
        │  1. fetch upstream, resolve target commit
        │  2. REFUSE if target is behind state.yaml (would revert work)
        │  3. compare against .upstreamer/state.yaml — skip if unchanged
        │  4. opencode runs the port against .upstreamer/upstreamer.md
        │  5. .upstreamer/scripts/verify.sh   (mechanical gate)
        │  6. .upstreamer/eval.md            (parity gate, fresh context)
        │  7. advance state.yaml — ONLY if both gates pass
        ▼
    Pull request, opened by the GitHub App so it gets real
    pull_request-event CI checks  (never a direct push to main)
```

### Why HEAD and not the latest release

Release tracking sounds more conservative and is worse in practice. Upstream can
sit for weeks with large unreleased work on `main` — doom-loop detection (#73) was
~7,500 lines, ~4,700 of it tests, and rewrote a big part of `model-result.ts`. A
release-tracking port stays blind to that, then absorbs the entire delta in one
automated run touching the most load-bearing module in the package. Tracking HEAD
keeps each delta small enough that a human can actually review it.

Two consequences follow, and both are handled rather than ignored:

**The port is routinely ahead of the latest release.** So a release ref is now
*dangerous*: it resolves to an ancestor of what is already ported, and the
converter would faithfully "port" the older tree, reverting landed work.
`scripts/upstream` refuses a target that is behind `state.yaml` (exit 3) unless
`--force` is given, and the workflow ignores the publish dispatch's
`client_payload.ref` for the same reason.

**Its declared version legitimately lags upstream's `package.json`.** Being ahead
of a release means carrying commits upstream has not versioned yet, while
`package.json` still shows the last released number. The verifier reports how many
commits ahead the ported tree is and warns not to publish that version to PyPI —
shipping unreleased upstream work under a released version number is a
misrepresentation, and a PyPI version can never be reused. Publish from a commit
level with a release tag.

The weekly cron is the primary trigger. The publish dispatch still fires on a new
npm release — useful as "something shipped, sync promptly" — but it syncs to HEAD
like everything else. `workflow_dispatch` allows a manual run against any ref.

## The contract is the product

`.upstreamer/upstreamer.md` is the durable artifact — it defines scope, the
required public API, naming maps, permanent idiomatic divergences, and the
substrate pin. The ported source is an *output* of that contract.

So when the port gets something wrong, **fix the contract**, not just the
generated code. A code-only fix gets re-broken on the next sync; a contract fix
holds.

## Files

| Path | What |
|------|------|
| `.upstreamer/upstreamer.md` | The rewrite contract. Binding. |
| `.upstreamer/state.yaml` | Last commit ported *and* verified *and* eval-passed. |
| `.upstreamer/scripts/verify.sh` | Mechanical gate: build, lint, types, tests, required API. |
| `.upstreamer/eval.md` | Parity gate: fresh-context behavioral review. |
| `.upstreamer/eval-report.md` | Latest eval result, or a bankruptcy report. |
| `.upstreamer/skills/upstreamer-converter/` | Execution discipline for the porting agent. |
| `.upstreamer/skills/port-test-quality/` | How to port upstream tests and keep coverage honest. |
| `.upstreamer/port.env` | Local secrets. **Gitignored.** |
| `scripts/upstream` | The wrapper. |

## Two gates, and why state matters

**Mechanical** (`verify.sh`) — objective: does it build, lint, type-check (`src`
*and* `tests`), pass tests, hold the coverage floor, and export every symbol the
contract requires. It also reports which upstream test files have no Python
counterpart — advisory, since severity is the eval's call.

**Parity** (`eval.md`) — judgment, run in a fresh context that reads the upstream
reference directly: does it actually *behave* like upstream. This is the gate that
catches a port which compiles cleanly while sitting a version behind on real
behavior.

If either gate fails the run **declares bankruptcy**: `state.yaml` is left
untouched and `.upstreamer/eval-report.md` explains why. A stale state file is the
correct outcome for a failed port — it makes the next run retry the same delta
instead of skipping past the gap. The workflow labels such a PR `eval-failed` and
marks the title `do not merge`.

Never hand-edit `state.yaml` to make a run look successful.

## Running it locally

```bash
# one-time
bun install -g opencode-ai          # or: npm install -g opencode-ai
cp .upstreamer/port.env.example .upstreamer/port.env
# then edit .upstreamer/port.env and fill in OPENROUTER_API_KEY + OPENCODE_MODEL

./scripts/upstream                  # sync if upstream changed
./scripts/upstream --force           # re-run after editing the contract
./scripts/upstream --ref v0.8.0      # port a specific upstream ref
./scripts/upstream -- --print-logs   # pass args through to opencode
```

`--force` is the escape hatch for a changed contract with unchanged upstream.
Expect to use it often while the contract is still settling.

## Credentials

Two values, same names locally and in CI:

| Name | Where | What |
|------|-------|------|
| `OPENROUTER_API_KEY` | local: `.upstreamer/port.env` · CI: repo **secret** | `sk-or-…` key opencode uses for inference |
| `OPENCODE_MODEL` | local: `.upstreamer/port.env` · CI: repo **variable** | e.g. `openrouter/~anthropic/claude-opus-latest` |

Two more are needed in CI only, for the bot that opens port PRs:

| Name | Kind | What |
|------|------|------|
| `PORT_BOT_APP_ID` | repo **variable** | The GitHub App's App ID |
| `PORT_BOT_PRIVATE_KEY` | repo **secret** | The App's generated private key (full PEM, including the BEGIN/END lines) |

### Why a GitHub App is required, not optional

`main` requires six status checks, and **only `pull_request`-event runs satisfy
them**. GitHub does not trigger workflows from events created with the native
`GITHUB_TOKEN` (its recursion guard), so a PR opened with that token gets no
`pull_request` checks and can never become mergeable.

Measured on PR #24: the commit carried **14 check-runs, while the PR's rollup
showed 7** — a `workflow_dispatch` run of the same workflow on the same commit was
completely invisible to branch protection. That is why "just dispatch `ci.yaml`
at the branch" does not work; it produces green runs that cannot satisfy anything.

An App installation token is not recursion-guarded, so the PR it opens gets real
checks. Preferred over a PAT: scoped to this repo, not tied to anyone's personal
account, and revocable on its own.

**Setup** — create a GitHub App (org Settings → Developer settings → GitHub Apps):

- Repository permissions: **Contents: Read and write**, **Pull requests: Read and
  write**. Nothing else.
- Install it on `OpenRouterTeam/python-agent`.
- Generate a private key, then add `PORT_BOT_APP_ID` (variable) and
  `PORT_BOT_PRIVATE_KEY` (secret).

Until those exist the pipeline still runs and still opens a PR, but emits a
`::warning::` saying the PR will receive no checks and cannot merge as-is. That is
deliberate — an unconfigured bot should degrade loudly, not look healthy while
producing permanently stuck PRs.

The wrapper writes the key into `~/.local/share/opencode/auth.json` so headless
runs work without the interactive `opencode /connect` flow.

This is a load-bearing SDK port behind a strict parity eval — use a strong coding
model. `OPENCODE_MODEL` overrides the `model:` field in the contract, so you can
change models without a code change.

## Releasing to PyPI

The distribution is **`openrouter-agent-sdk`**; the import stays
`openrouter_agent`. The obvious name `openrouter-agent` is taken on PyPI by an
unrelated third-party project, so a sync must not "correct" it — see the Package
Identity table in `.upstreamer/upstreamer.md`.

`.github/workflows/publish.yaml` is manual-only (`workflow_dispatch`), defaults to
a dry run, and uses PyPI **trusted publishing (OIDC)** — no API token is stored in
this repo.

```
1. Land the version bump in pyproject.toml (a port sync does this).
2. Run Publish with target=testpypi to rehearse.
3. Run with target=pypi, dry-run=true — read the summary.
4. Run with target=pypi, dry-run=false to release.
```

Before the first real publish, two things must be set up by hand — the workflow
cannot do them for you:

1. **On PyPI**: add a GitHub trusted publisher (owner `OpenRouterTeam`, repo
   `python-agent`, workflow `publish.yaml`, environment `pypi`). If the project
   does not exist yet, add it as a *pending* publisher.
2. **In repo Settings**: create the `pypi` and `testpypi` environments and set each
   one's deployment branch policy to `main`.

That branch policy is the real ref restriction. PyPI's trusted publisher pins
owner/repo/workflow/environment but carries no branch claim, and
`workflow_dispatch` runs the workflow file from whatever ref is selected — so the
in-file `if:` guard stops accidents, while the environment policy is what actually
binds publishing to `main`.

Publishing is irreversible: a version can never be reused on PyPI, even after a
yank. The workflow re-runs `verify.sh`, checks metadata with `twine check
--strict`, imports the built wheel in isolation, and refuses to upload a version
that already exists on the target index.

## Reviewing a port PR

Review it as a *port*, not a normal diff:

1. Check `.upstreamer/eval-report.md` first. If state did not advance, stop.
2. Read the upstream delta yourself for anything load-bearing — the tool loop,
   state serialization, approval/HITL ordering, hooks, streaming.
3. Confirm new tests assert *upstream behavior*, not merely the port's own shape.
   The contract's Test Parity section lists the patterns that look like coverage
   and are not — membership-only stream assertions, `assert x is not None` as a
   test's only assertion, "a tool ran" where the invariant is *exactly once*.
   Check the verifier's test-parity note for upstream test files left unported.
4. Any new naming mapping the run derived should be promoted into the contract's
   naming table.
5. Check the coverage floor did not move down. It is a ratchet; lowering it needs
   a stated reason.
