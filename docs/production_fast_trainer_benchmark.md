# Production Fast Trainer Benchmark

Date: 2026-03-10

This document freezes the benchmark-safe production fast-trainer target for Phase 2.

It exists to answer one narrow question:

- what exact trainer path should future Phase 2 trainer spikes optimize?

This document does not authorize trainer replacement by itself.
It defines the target path and the rows that count.

## Purpose

The workspace now has two distinct trainer-benchmark families:

1. `train_benchmark_production_v1`
2. `train_ceiling_diagnostic_v1`

These must not be confused.

The production benchmark exists to measure the intended fast path.
The learner-ceiling benchmark exists to diagnose the current shipped synchronous trainer path.

## Production Fast Trainer Responsibilities

The minimum production-mode trainer responsibilities are:

- consume parity-approved training observations from the current production observation path
- perform policy forward and action sampling for rollout collection
- collect on-policy rollout data required for PPO-style updates
- perform the PPO/value update that the current trainer path is meant to represent
- maintain correct `global_step` accounting
- preserve reward/action/observation meaning exactly
- preserve termination/truncation semantics exactly
- preserve seed/config-driven determinism expectations where currently required
- emit enough benchmark metadata to identify the benchmark contract, host class, schema versions, and major runtime versions

The production fast-trainer benchmark may assume:

- replay is disabled
- dashboard output is disabled
- logging mode is disabled
- the benchmark row is concerned with hot-path throughput, not user-facing observability

## Certification-Mode Only Responsibilities

The following responsibilities remain Certification Mode only or otherwise out-of-hot-path for the production fast-trainer benchmark:

- final evaluate after the terminal train loop
- replay export and replay artifact generation
- parity canary execution
- deterministic replay certification work
- full nested `info` / stats aggregation used for debug or audit surfaces
- W&B online/offline logging as part of the benchmarked hot path
- benchmark-visible manifest/artifact/checkpoint write cost in the hot path
- headed demo / RSPS-facing showcase behavior

Important distinction:

- this does not mean checkpoints, manifests, or observability are unimportant in the product
- it means they are not part of the frozen throughput target for the benchmark-safe production fast-trainer path

## What The Benchmark Must Include

The frozen production fast-trainer benchmark must:

- use the current production observation path, not the raw certification projection path
- use the current production vecenv backend
- use the active production trainer candidate path under test
- measure throughput with:
  - `metric_contract_id = train_benchmark_production_v1`
  - `metric_scope = production_fast_path_v1`
  - `production_env_steps_per_second`
- exclude `final_evaluate_seconds` from the primary throughput metric
- still record:
  - `wall_clock_env_steps_per_second`
  - `evaluate_seconds`
  - `train_seconds`
  - `final_evaluate_seconds`
  - runner-stage timing
  - trainer bucket timing when instrumentation is enabled

## What The Benchmark Must Not Include

The frozen production fast-trainer benchmark must not treat the following as part of the primary throughput target:

- final evaluate
- replay export
- parity validation
- W&B online/offline flush cost
- dashboard rendering
- benchmark-only artifact and manifest writes
- certification-only debug metadata aggregation beyond what the hot path truly requires

## Canonical Rows

The canonical Phase 2 production fast-trainer rows are:

- native Linux, `16 env`, disabled logging, production metric
- native Linux, `64 env`, disabled logging, production metric

These rows answer:

- is the fast-trainer path materially faster at the current working set?
- does `16 -> 64` scaling improve?

The canonical diagnostic companion rows are:

- learner ceiling, `16 env`
- learner ceiling, `64 env`

These rows do not define the production target.
They exist to explain why a production row is or is not moving.

## Non-Canonical Rows

The following rows remain useful, but are not the primary success target for the current Phase 2 trainer redesign batch:

- local WSL fast-trainer rows
- local WSL learner-ceiling rows
- `4 env` learner-ceiling rows
- `256 env` and `1024 env` train rows on the current trainer path
- online/offline W&B comparisons
- transport-promotion rows

These remain diagnostic, exploratory, or later-phase rows.

## Current Frozen Interpretation

After `WC-P2-11`, the project should interpret Phase 2 results as follows:

- if the production fast-trainer rows do not move, the active benchmark target did not improve
- if the learner-ceiling rows move but the production fast-trainer rows do not, the remaining blocker is likely still outside the trainer core
- if the production fast-trainer rows move materially and scaling improves, then the trainer redesign path is working and Phase 2 can revisit transport or topology work later

## Relationship To Certification Mode

Certification Mode remains the trust-establishing path.

Production fast-trainer work is allowed only because:

- sim semantics remain authoritative in `fight-caves-RL`
- parity-sensitive fields and contracts remain frozen
- replay/parity/determinism obligations remain covered elsewhere

If a proposed fast-trainer change would require dropping those obligations rather than moving them out of the benchmark hot path, it is not a valid Phase 2 benchmark-safe change.

## Frozen Next-Step Consequence

Future trainer spikes in Phase 2 should be judged first against:

- native-Linux production fast-trainer `16 env`
- native-Linux production fast-trainer `64 env`
- native-Linux production fast-trainer `64 vs 16` scaling

The learner-ceiling benchmark remains required, but only as a diagnostic companion.
