# Performance Plan

This document freezes the staged performance plan for the RL repo.

## Goal

The repo is planned toward `>= 1,000,000 env steps/sec`, but only through staged gates that keep correctness and reproducibility ahead of optimization.

## Official Benchmark Profile v0

Current benchmark profile id:

- `official_profile_v0`

Frozen profile defaults:

- `benchmark_mode = staged_gate_matrix`
- `bridge_mode = recorded_per_run`
- `reward_config = reward_sparse_v0`
- `curriculum_config = curriculum_disabled_v0`
- `replay_mode = disabled`
- `logging_mode = standard`
- `dashboard_mode = disabled`
- env-count ladder: `1`, `16`, `64`, `256`, `1024`

Every benchmark manifest must record:

- benchmark profile id/version
- hardware profile
- RL commit SHA
- sim commit SHA
- sim artifact task/path
- bridge protocol id/version
- observation schema id/version
- action schema id/version
- episode-start contract id/version
- PufferLib distribution/version
- reward config id
- curriculum config id
- replay/logging/dashboard modes
- env count

## Stage Gate A - Correctness Baseline

- measure single-env wrapper throughput and reset/step latency
- profiles: `1 env`, `16 envs`
- exit condition: reproducible baseline numbers, no throughput claims yet

## Stage Gate B - Batched Bridge Baseline

- measure raw bridge throughput
- compare batch vs sequential equivalence on the same hardware/config
- current PR7 benchmark configs:
  - `bridge_1env_v0`
  - `bridge_64env_v0`
- profiles: `1 env` for wrapper-vs-trace comparison, `64 envs` for lockstep multi-slot bridge throughput
- exit condition: batched bridge is faster than correctness mode and semantically equivalent

Current PR7 benchmark split:

- `bridge_1env_v0`
  - compares the current correctness wrapper path against the sim-side `runFightCaveBatch(...)` helper on one slot
- `bridge_64env_v0`
  - measures the lockstep multi-slot bridge against a higher-overhead reference path on the same runtime

## Stage Gate C - Vectorized Env Baseline

- measure VecEnv throughput and stability
- profiles: `64 envs`, `256 envs`
- exit condition: vectorized execution is stable and clearly better than the non-vectorized path

## Stage Gate D - Full Training Baseline

- measure end-to-end training SPS
- compare normal logging, minimized logging, replay disabled, replay periodic
- profiles: `64 envs`, `256 envs`, `1024 envs` where hardware allows
- exit condition: manifests isolate trainer, bridge, wrapper, and W&B overhead

## Stage Gate E - Tuned Production Path

- tune boundary crossings, memory copies, observation packing, worker topology, logging cadence, then policy overhead
- exit condition: reach `>= 1,000,000 env steps/sec` on the defined profile or document the verified remaining blockers

## Mandatory Breakdown

Every stage should keep these benchmark breakdowns visible:

- raw sim batch stepping rate
- Python bridge throughput
- wrapper overhead
- PufferLib VecEnv throughput
- end-to-end training SPS
- W&B overhead at normal cadence
- W&B overhead at aggressive cadence
