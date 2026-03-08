# MVP Acceptance

This document freezes the PR13 MVP acceptance gate for the RL repo.

## Entry Points

Current repo-owned acceptance entry surfaces:

- local/manual runner: `scripts/run_acceptance_gate.py`
- manual GitHub Actions workflow: `.github/workflows/acceptance.yml`

## Gate Coverage

The current acceptance gate runs:

- the full RL test split:
  - `fight_caves_rl/tests/unit`
  - `fight_caves_rl/tests/train`
  - `fight_caves_rl/tests/integration`
  - `fight_caves_rl/tests/determinism`
  - `fight_caves_rl/tests/parity`
  - `fight_caves_rl/tests/smoke`
  - `fight_caves_rl/tests/performance`
- a real train run through `scripts/train.py`
- two deterministic replay-eval runs through `scripts/replay_eval.py`
- the parity matrix through `scripts/run_parity_canary.py`
- the bridge/env/train benchmark entrypoints

## Output Contract

The acceptance runner writes:

- `acceptance_report.json`
- per-command stdout/stderr logs
- train summary output
- replay eval outputs
- parity report output
- bridge/env/train benchmark outputs

The report is a repo-owned JSON summary of:

- commands executed
- command durations
- artifact/category checks
- deterministic replay-eval agreement
- parity matrix status
- benchmark output presence and positive SPS checks

## W&B Mode

The acceptance gate runs with repo-owned offline W&B directories under the acceptance output directory.

This keeps acceptance reproducible and avoids relying on user-global W&B state.

## Current Scope Boundary

The acceptance gate proves the implemented RL integration path is functional, reproducible, instrumented, and benchmarked on the current workspace.

It does not claim:

- that `>= 1,000,000 env steps/sec` has already been reached
- that the manual self-hosted acceptance workflow should run in normal PR CI
- that checkpoint replay-eval parity replaces the existing PR10 determinism contract
