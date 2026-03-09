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

Every benchmark report must record:

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

Current PR11 implementation note:

- repo-owned benchmark JSON outputs now carry this metadata under a shared `context` block
- that `context` block is the current benchmark-manifest surface for RL benchmark entrypoints
- the current benchmark entrypoints are:
  - `scripts/benchmark_bridge.py`
  - `scripts/benchmark_env.py`
  - `scripts/benchmark_train.py`

Current measured local baseline gap (2026-03-09 audit packet, Ryzen 5 5600G / WSL):

- canonical audit packet sources:
  - `docs/performance_decomposition_report.md`
  - `docs/benchmark_matrix.md`
  - `docs/python_profiler_report.md`
  - `docs/transport_and_copy_ledger.md`
- direct-JVM repo artifact in `fight-caves-RL/docs/performance_report.md`:
  - throughput benchmark: about `8.9k` ticks/sec
  - important caveat: this is a current-repo artifact from a Windows-native context, not a current-host WSL throughput rerun
- RL local bridge measurements on this workspace host:
  - `bridge_1env_v0` batch trace: about `23.8k` env steps/sec
  - `bridge_64env_v0` lockstep batch: about `1.48k` env steps/sec total
- RL local embedded vecenv measurements on this workspace host:
  - `4 envs`: about `906.6` env steps/sec total
  - `16 envs`: about `1232.6` env steps/sec total
  - `64 envs`: about `1492.1` env steps/sec total
- RL local stable training measurements on this workspace host:
  - `4 envs`, W&B disabled: about `36.4` train SPS
  - `16 envs`, W&B disabled: about `82.8` train SPS
  - `64 envs`, W&B disabled: about `87.9` train SPS
  - `4 envs`, W&B online wall-clock probe: about `11.9` train SPS

Interpretation:

- the current bottleneck is not one thing
- the simulator itself is still far below the end goal even before RL overhead
- the current Python bridge/vector layer collapses throughput further, especially at higher env counts
- the stability-first subprocess training fix adds extra IPC overhead but prevents the reset-boundary segfault in the shipped training path
- online W&B overhead is meaningful, but it is not the primary reason the stack is far from `100,000-1,000,000+` SPS

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
- current PR8/PR11 benchmark/config surfaces:
  - `configs/benchmark/vecenv_256env_v0.yaml`
  - `scripts/benchmark_env.py`
  - `fight_caves_rl/benchmarks/env_bench.py`
- current env benchmark split:
  - `wrapper_sequential`
  - `vecenv_lockstep`
- the wrapper and vecenv measurements currently run in separate child processes because the embedded-JVM lifecycle is process-global and cannot safely benchmark both paths in one Python process
- profiles: `64 envs`, `256 envs`
- exit condition: vectorized execution is stable and clearly better than the non-vectorized path

## Stage Gate D - Full Training Baseline

- measure end-to-end training SPS
- compare normal logging, minimized logging, replay disabled, replay periodic
- current PR11 benchmark/config surfaces:
  - `configs/benchmark/train_1024env_v0.yaml`
  - `scripts/benchmark_train.py`
  - `fight_caves_rl/benchmarks/train_bench.py`
  - `.github/workflows/benchmarks.yml`
- current logging matrix:
  - `disabled`
  - `standard`
  - `aggressive`
- replay remains disabled in the current PR11 training benchmark path so logging overhead can be isolated first
- each training measurement currently runs in a fresh child `train.py` process so logging modes do not share embedded-JVM or W&B process state
- profiles: `64 envs`, `256 envs`, `1024 envs` where hardware allows
- exit condition: benchmark reports isolate trainer, bridge, wrapper, and W&B overhead

## Stage Gate E - Tuned Production Path

- tune boundary crossings, memory copies, observation packing, worker topology, logging cadence, then policy overhead
- exit condition: reach `>= 1,000,000 env steps/sec` on the defined profile or document the verified remaining blockers

Required next optimization queue after the 2026-03-08 remediation:

1. Replace the current subprocess pipe/pickle transport with shared-memory flat numeric buffers.
2. Move multi-slot reset/action/observe work toward truly batched sim-side entrypoints instead of per-slot Python/JVM loops.
3. Re-profile the headless runtime itself and raise the raw sim ceiling well beyond the current single-digit-thousands tick rate.
4. Scale out across multiple worker processes/runtimes only after one-worker transport costs are materially lower.
5. Delay learner-side micro-optimization until environment collection is at least one to two orders of magnitude faster than today.

## Mandatory Breakdown

Every stage should keep these benchmark breakdowns visible:

- raw sim batch stepping rate
- Python bridge throughput
- wrapper overhead
- PufferLib VecEnv throughput
- end-to-end training SPS
- W&B overhead at normal cadence
- W&B overhead at aggressive cadence
