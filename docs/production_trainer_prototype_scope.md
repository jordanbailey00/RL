# Production Trainer Prototype Scope

Date: 2026-03-10

This document records the `WC-P2-13` decision for the first Phase 2 production-trainer prototype, the local `WC-P2-14` gate outcome, and the now-complete hosted native-Linux rerun.

It answers one narrow question:

- what exact prototype boundary is approved before the next implementation batch starts?

## Approved Direction

The first approved prototype is:

- a project-owned synchronous production trainer path
- benchmarked against `train_benchmark_production_v1`
- still using the current env/vector and policy interfaces where they are already correct

The first approved prototype is not:

- another small `WC-P2-07` suppression slice inside the current `PuffeRL` loop
- a transport-promotion retry
- an actor/learner topology rewrite
- a semantics-changing trainer rewrite

## Current Local Implementation Status

The first local prototype slice now exists in the benchmark surface:

- project-owned trainer module:
  - `/home/jordan/code/RL/fight_caves_rl/puffer/production_trainer.py`
- benchmark runner mode:
  - `prototype_sync_v1`
- current implementation shape:
  - owns the rollout loop in project code
  - owns the PPO update loop in project code
  - omits final evaluate from the prototype production metric entirely
  - supports disabled logging only in the current benchmark slice

This is intentionally a prototype-only benchmark path, not a shipped replacement for `scripts/train.py`.

## Current Local Gate Outcome

The first local post-prototype gate has now run on the current WSL host:

- corrected production fast-path prototype rows:
  - first local prototype gate:
    - `16 env`: `95.74` production SPS
    - `64 env`: `93.06` production SPS
    - `64 vs 16 = 0.9720x`
  - after the follow-on trainer-core slice:
    - `16 env`: `417.36` production SPS
    - `64 env`: `398.80` production SPS
    - `64 vs 16 = 0.9555x`
- learner-ceiling diagnostic companions:
  - `16 env`: `145.70` env-steps/s
  - `64 env`: `144.79` env-steps/s
  - `64 vs 16 = 0.9937x`

Local interpretation:

- the first project-owned synchronous prototype is materially faster than the earlier local shipped-path production band
- the follow-on trainer-core slice materially reduces the padded multi-discrete policy-forward cost on the local production path
- the prototype still does not improve scaling materially
- the learner-ceiling companion remains in the same `~145` env-steps/s band
- the next step should be a native-Linux rerun of the corrected prototype packet before any further local redesign, transport, or topology decision

## Hosted Native-Linux Gate Outcome

The source-of-truth native-Linux rerun is now complete.

Hosted production fast-path rows:

- `16 env`: `469.92` production SPS
- `64 env`: `341.43` production SPS
- `64 vs 16 = 0.7266x`

Hosted learner-ceiling companions:

- `16 env`: `81.64` env-steps/s
- `64 env`: `73.39` env-steps/s
- `64 vs 16 = 0.8989x`

Hosted interpretation:

- the packaging blocker is resolved:
  - local contract testing reproduced the issue under workflow-style env vars
  - the root cause was unsanitized `GITHUB_REF_NAME` in shared Gradle versioning
  - the hosted workflow now validates the packaged artifact with the repo-owned contract harness before benchmarking
- the source-of-truth production row is materially strong at `16 env`
- the source-of-truth `64 env` row regresses sharply enough that topology is not the next move
- the shipped synchronous learner ceiling is both low and negatively scaling, which keeps Phase 2 focused on trainer-path replacement

Explicit restart point:

- next pickup is a deeper trainer-redesign continuation inside the prototype path
- immediate objective:
  - reduce native-Linux `rollout_policy_forward` and `rollout_env_recv` in the prototype path
  - reduce native-Linux `eval_policy_forward`, `train_backward`, and `train_policy_forward` in the shipped synchronous learner path
- Phase 3 decision status:
  - not approved
  - blocked on materially healthier source-of-truth scaling after further trainer redesign
- not tomorrow's objective:
  - transport promotion
  - actor/learner split
  - topology work

## Components To Keep

The first prototype keeps these components in scope:

- the current production observation path and policy tensor meaning
- the current subprocess vecenv backend and transport contract used by the canonical production rows
- the current policy module family, starting with `MultiDiscreteMLPPolicy`
- the current PPO objective family and train-config surface as the source of truth for hyperparameters
- the current benchmark contract:
  - `metric_contract_id = train_benchmark_production_v1`
  - `metric_scope = production_fast_path_v1`
  - canonical native-Linux `16 env` and `64 env` disabled rows

## Components To Bypass Or Replace Immediately

The first prototype is allowed to bypass or replace these pieces immediately in the production fast path:

- the current `ConfigurablePuffeRL.evaluate()` rollout loop
- the current `ConfigurablePuffeRL.train()` update loop
- the current `ConfigurablePuffeRL.mean_and_log()` benchmark-hot-path behavior
- the current `ConfigurablePuffeRL.close()` checkpoint-oriented shutdown path
- the final-evaluate pass as part of the primary production throughput metric
- nested `info` / stats aggregation that is useful for certification or debugging but not required for the production benchmark hot path
- framework-owned rollout-buffer and minibatch-management behavior if a simpler project-owned path preserves the intended PPO semantics

## First Prototype Shape

The first prototype should aim for this shape:

- keep the current subprocess vecenv and policy interfaces
- own the rollout loop in project code rather than treating `PuffeRL` as the production loop
- own the PPO update loop in project code where that is required to remove current framework-structural overhead
- keep the topology synchronous and single-worker for the first prototype so the trainer-path change is isolated cleanly
- keep production benchmarking disabled-logging only until the hot-path shape is validated

## Explicitly Deferred

These items are deferred out of the first prototype:

- actor/learner split or any asynchronous topology work
- renewed transport-promotion decisions
- policy-architecture redesign
- reward, action, observation, or termination semantic changes
- removal of certification, parity, replay, or determinism obligations from the project

Those obligations remain real; they are simply not part of the production benchmark hot path.

## Why This Boundary Was Chosen

The current evidence says:

- the dominant remaining wall-clock buckets are inside the shipped synchronous trainer path
- small control-plane suppressions did not materially raise the learner ceiling
- the benchmark-safe production target is already frozen separately from certification-only responsibilities
- the next implementation move needs cleaner ownership over rollout/update structure before transport or topology are revisited

That means the first prototype should attack trainer-loop ownership directly without mixing in transport, topology, or semantic changes.

## Prototype Success Criteria

The first prototype should be judged first against:

- native-Linux production fast-trainer `16 env`
- native-Linux production fast-trainer `64 env`
- native-Linux production fast-trainer `64 vs 16` scaling
- native-Linux learner-ceiling diagnostic `16 env`
- native-Linux learner-ceiling diagnostic `64 env`

The current planning thresholds still apply:

- continue if the first prototype materially improves the corrected production fast-train rows and scaling
- escalate if the first prototype still leaves the trainer path below about `250` SPS at native-Linux `64 env`
- escalate if the prototype still leaves the same synchronous trainer structure as the dominant wall-clock cost
