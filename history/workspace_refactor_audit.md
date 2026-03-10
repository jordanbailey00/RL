# Workspace Refactor Audit (Post-PR13)

Date: 2026-03-08

Scope:
- `/home/jordan/code/RL`
- `/home/jordan/code/fight-caves-RL`
- `/home/jordan/code/RSPS`

Method:
- reviewed the current module source-of-truth docs
- reviewed current README/setup guidance and CI/workflow files
- compared the shipped docs against the implemented RL PR13 state
- spot-checked refactor hotspots with targeted code search

This is an audit only.
No runtime behavior changes are proposed here, and no simulator semantics are redefined.

## Executive Summary

Current workspace status is generally strong:
- all three repos are clean on `main`
- the cross-repo role split is now documented
- RL PR13 acceptance is in place and was previously verified end to end

The remaining issues are mostly maintainability and documentation drift rather than correctness blockers.

Highest-signal findings:
1. RL still has one stale manifest doc claim after the benchmark implementation landed.
2. RL has avoidable entrypoint duplication in `scripts/eval.py` and repeated CLI boilerplate across multiple scripts.
3. `fight-caves-RL` still carries stale local-path and stale E2E-step wording in active docs.
4. `RSPS` now has correct source-of-truth docs, but its README still presents only the inherited upstream Void identity and not the workspace oracle/reference role.
5. `fight-caves-RL` and `RSPS` still carry identical inherited release workflows; for `fight-caves-RL` this remains intentionally inherited but not aligned to current repo identity.

## Baseline Facts

- `git status --short --branch` is clean in:
  - `/home/jordan/code/RL`
  - `/home/jordan/code/fight-caves-RL`
  - `/home/jordan/code/RSPS`
- RL CI boundary is now explicit in:
  - `/home/jordan/code/RL/README.md`
  - `/home/jordan/code/RL/.github/workflows/ci.yml`
- RL manual benchmark/acceptance workflows are separate from per-PR CI:
  - `/home/jordan/code/RL/.github/workflows/benchmarks.yml`
  - `/home/jordan/code/RL/.github/workflows/acceptance.yml`

## Findings

### 1. Medium: RL run-manifest doc is stale after PR11 benchmark implementation

Why it matters:
- the doc currently claims benchmarks will later extend the same manifest surface
- the code already ships a distinct benchmark-report path with shared benchmark context
- leaving this stale makes the repo harder to reason about for future artifact/reporting work

Evidence:
- `/home/jordan/code/RL/docs/run_manifest.md:129`
  - says: `Benchmarks will extend the same manifest surface later rather than introducing a second manifest shape.`
- `/home/jordan/code/RL/fight_caves_rl/benchmarks/common.py:29-110`
  - defines the shipped `BenchmarkContext`
- `/home/jordan/code/RL/fight_caves_rl/benchmarks/env_bench.py:38-56`
  - defines `EnvBenchmarkReport`
- `/home/jordan/code/RL/fight_caves_rl/benchmarks/train_bench.py:40-55`
  - defines `TrainBenchmarkReport`

Suggested action:
- update the manifest doc to distinguish:
  - train/eval `run_manifest.json`
  - benchmark JSON reports with shared benchmark context

### 2. Medium: RL keeps a fully duplicated compatibility entrypoint and repeated CLI boilerplate

Why it matters:
- duplicated entrypoints are easy to let drift
- the same argparse/output-writing pattern is repeated across many scripts
- future CLI changes will cost more than necessary

Evidence:
- `/home/jordan/code/RL/scripts/eval.py:1-32`
- `/home/jordan/code/RL/scripts/replay_eval.py:1-32`
  - both parse the same args, call the same function, and write the same JSON payload
- repeated near-identical script structure also appears in:
  - `/home/jordan/code/RL/scripts/benchmark_bridge.py:1-30`
  - `/home/jordan/code/RL/scripts/benchmark_env.py:1-37`
  - `/home/jordan/code/RL/scripts/benchmark_train.py:1-37`
  - `/home/jordan/code/RL/scripts/train.py:1-31`

Suggested action:
- keep `replay_eval.py` canonical
- make `eval.py` delegate instead of duplicating logic
- consider one small shared CLI/output helper for the repo-owned JSON script pattern

### 3. Medium: `fight-caves-RL` still has stale active doc wording from the earlier local environment

Why it matters:
- active docs should not preserve stale machine-local path assumptions
- E2E docs should match the completed plan state if they are meant to be runnable acceptance guidance

Evidence:
- `/home/jordan/code/fight-caves-RL/FCspec.md:581-583`
  - still says the local folder path is `C:\\Users\\jorda\\dev\\personal_projects\\fight-caves-RL`
- `/home/jordan/code/fight-caves-RL/e2e test.md:11-16`
  - still says `FCplan.md steps 0-12 are marked complete`
- `/home/jordan/code/fight-caves-RL/e2e test.md:31-38`
  - still uses `adapt task names to implemented build tasks` wording even though the canonical artifact/build tasks are now known

Suggested action:
- remove the stale local-path closeout note from the active root spec
- update the E2E precondition/task wording to match the now-complete Step 13 state and current canonical task names

### 4. Medium: `RSPS` README is still upstream-oriented and does not reflect the now-documented workspace role

Why it matters:
- `RSPSspec.md` now defines `RSPS` as the workspace oracle/reference module
- the README still onboards readers as if this repo were only a standalone upstream Void server fork
- new contributors can miss the module's intended boundary in this workspace

Evidence:
- `/home/jordan/code/RSPS/RSPSspec.md:5-18`
  - defines `RSPS` as the headed oracle/reference module
- `/home/jordan/code/RSPS/README.md:1-92`
  - presents only the inherited Void identity, upstream badges, upstream setup flow, and upstream GitHub references

Suggested action:
- add a short workspace-role note near the top of the README that points readers to `RSPSspec.md`
- keep the inherited setup instructions, but separate them from the module-role explanation

### 5. Medium: release automation is still duplicated and only partially aligned to current module identity

Why it matters:
- identical duplicated workflows increase maintenance cost
- in `fight-caves-RL`, the workflow still publishes a Void-branded bundle and Docker image even though the repo's runtime/docs identity is now `fight-caves-RL`
- this is already documented as inherited, but the automation surface itself still advertises old naming

Evidence:
- `/home/jordan/code/fight-caves-RL/.github/workflows/create_release.yml:61-74`
  - uses `assembleBundleDist`, uploads `void-${{ env.build_version }}.zip`, and publishes `greghib/void`
- `/home/jordan/code/RSPS/.github/workflows/create_release.yml:61-74`
  - same content
- a direct `diff -u` between the two workflow files produced no output
- `/home/jordan/code/fight-caves-RL/FCspec.md:12-23`
  - explicitly says the workflow remains inherited and is not the source of truth for headless artifact naming

Suggested action:
- keep the current intentionality decision if desired, but record one explicit future owner/decision point:
  - leave inherited indefinitely
  - replace only in `fight-caves-RL`
  - centralize/re-template both workflows later

### 6. Low: RL still carries a clearly placeholder sweep config

Why it matters:
- low risk now, but it is an easy stale artifact for future readers to misinterpret as a real sweep baseline

Evidence:
- `/home/jordan/code/RL/configs/sweep/ppo_sweep_v0.yaml:1-2`
  - contains only `config_id` plus `notes: "placeholder sweep config for later PRs"`

Suggested action:
- either mark it more clearly as inactive in nearby docs or replace it once real sweep work starts

## Recommended Execution Order

1. RL docs cleanup:
   - fix the stale benchmark statement in `docs/run_manifest.md`
   - decide whether to keep `eval.py` as a real alias or turn it into a thin delegation wrapper
2. `fight-caves-RL` doc cleanup:
   - remove the stale Windows-local path note
   - refresh `e2e test.md` preconditions/task wording
3. `RSPS` onboarding cleanup:
   - add a workspace-role note to the README so it matches `RSPSspec.md`
4. Release workflow ownership decision:
   - keep inherited as-is
   - or explicitly schedule identity cleanup for `fight-caves-RL`

## Not Findings

These were checked and are not currently open audit problems:
- RL dev-vs-train test taxonomy and CI split are aligned
- RL per-PR CI vs manual acceptance/benchmark workflow separation is explicit
- canonical headless artifact wording is aligned between RL docs and the current `fight-caves-RL` README
- `RSPSspec.md` and `RSPSplan.md` are no longer placeholders

## Facts vs Assumptions

Facts:
- every finding above is tied to a currently readable file in the workspace
- no runtime or behavior claims are made without a corresponding file reference

Assumptions:
- none were required to establish the findings themselves
- the suggested actions are maintainability recommendations, not source-of-truth decisions
