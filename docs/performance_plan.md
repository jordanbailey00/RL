# Performance Plan

This document freezes the staged performance plan for the RL repo.

## Goal

The design-center requirement is `>= 100,000 env steps/sec`.

Longer-term `>= 1,000,000 env steps/sec` remains a stretch goal only. It stays in scope, but it is contingent on evidence from the earlier phases and should not drive premature architectural commitments.

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
- direct-JVM historical repo artifact in `fight-caves-RL/history/performance_report_step11.md`:
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

## Phase 0 Gate Execution Status

Phase 0 measurement infrastructure now exists in repo-owned form:

- `fight-caves-RL`
  - `./gradlew --no-daemon :game:headlessPerformanceReport`
  - `./gradlew --no-daemon :game:headlessPerformanceProfile`
- `RL`
  - `uv run python scripts/refresh_phase0_packet.py --output-dir /tmp/fc_phase0_packet_clean`

Current refreshed WSL packet highlights:

- standalone sim single-slot throughput: about `30.5k` ticks/sec
- standalone sim batched throughput (`16 envs`): about `473.6k` env steps/sec
- bridge:
  - `1 env`: batch about `23.6k` env steps/sec
  - `16 env`: batch about `1.57k` env steps/sec
  - `64 env`: batch about `1.61k` env steps/sec
- vecenv:
  - `1 env`: about `980.3` env steps/sec
  - `16 env`: about `1459.5` env steps/sec
  - `64 env`: about `1426.8` env steps/sec
- train:
  - `4 envs`: about `96.7` SPS
  - `16 envs`: about `96.5` SPS
  - `64 envs`: about `91.6` SPS

Native-Linux source-of-truth gate summary from the hosted Phase 0 packet:

- benchmark host class: `linux_native`
- performance source of truth: `true`
- native-Linux source of truth: `true`
- clean pure-JVM artifact: present
- clean batched sim artifact: present
- bridge / vecenv / train rows on one host class: present
- per-worker ceiling estimate: present
- standalone sim single-slot throughput: about `30.5k` ticks/sec
- standalone sim batched throughput: about `404.6k` env steps/sec
- workers needed for `100k`: `1`
- Phase 1 gate result: `unblocked`

Interpretation:

- the Phase 0 refresh materially improved confidence in the sim-side ceiling
- the refreshed packet reinforces that the RL outer stack is the current dominant bottleneck
- the hosted native-Linux packet now satisfies the approved Phase 0 hard gate
- Phase 1 may begin, but only on the approved raw-vs-flat contract/design scope

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
5. After the Phase 1 flat-path work, do not assume transport-only work is still dominant; use the learner-ceiling diagnostic to decide when trainer/rollout overhead must become the active Phase 2 target.

Current Phase 2 local prototype status:

- the first lower-copy prototype now exists as an opt-in subprocess transport mode:
  - `pipe_pickle_v1` remains the default shipped path
  - `shared_memory_v1` currently uses a `Pipe` control plane plus file-backed `mmap` data plane
- local WSL prototype measurements so far:
  - `16 env / 64 rounds` subprocess transport comparison: about `1.02x`
  - `64 env / 64 rounds` subprocess transport comparison: about `1.29x`
  - `16 env / 128 timesteps` end-to-end train probe: about `1.03x`
- interpretation:
  - the prototype is healthy and measurable
  - the benefit is currently too small to justify a production transport swap on its own
  - Phase 2 should continue only through the approved review/pivot path, not by silently promoting this prototype

Current native-Linux pre-swap gate status:

- workflow: [fight-caves-RL/actions/runs/22883118379](https://github.com/jordanbailey00/fight-caves-RL/actions/runs/22883118379)
- result:
  - transport `64 env`: `0.7674x`
  - disabled train `64 env`: `0.9977x`
  - shared-train scaling `64 vs 16`: `0.9979x`
- blockers:
  - `transport_signal_too_weak`
  - `train_signal_too_weak`
  - `shared_train_scaling_too_weak`
- decision:
  - `WC-P2-03` remains blocked
  - the current blocker is now trainer-bound as well as transport-bound
  - another transport-only iteration is unlikely to clear the gate by itself
  - see [phase2_blocker_diagnosis.md](/home/jordan/code/RL/docs/phase2_blocker_diagnosis.md)
  - these disabled-train rows are legacy pre-`WC-P2-09` rows; future Phase 2 train comparisons must use the corrected production benchmark contract

Current trainer-bound pivot status:

- repo-owned fake-env learner-ceiling diagnostic now exists:
  - `scripts/benchmark_train_ceiling.py`
- current local WSL diagnostic:
  - `4 env`: `154.45` env-steps/s
  - `16 env`: `156.20` env-steps/s
  - `64 env`: `144.43` env-steps/s
- hosted native-Linux confirmation workflow exists:
  - [fight-caves-RL/actions/runs/22886069441](https://github.com/jordanbailey00/fight-caves-RL/actions/runs/22886069441)
  - published native-Linux learner ceiling:
    - `4 env`: `94.97` env-steps/s
    - `16 env`: `74.67` env-steps/s
    - `64 env`: `68.43` env-steps/s
    - `64 vs 16 = 0.9165x`
- next active Phase 2 work:
  - the first local prototype gate is complete
  - local WSL corrected prototype rows after the follow-on trainer-core slice:
    - `16 env`: `417.36` production SPS
    - `64 env`: `398.80` production SPS
    - `64 vs 16 = 0.9555x`
  - latest local interpretation:
    - the prototype is now materially faster than the previous local gate and above the old `250` SPS escalation bar
    - scaling is still flat enough that topology is not yet justified
  - hosted native-Linux rerun status:
    - local contract testing reproduced the packaging bug and traced it to unsanitized shared build versioning
    - the corrected hosted packet completed successfully on `ubuntu-latest`
    - source-of-truth production rows:
      - `16 env`: `469.92` production SPS
      - `64 env`: `341.43` production SPS
      - `64 vs 16 = 0.7266x`
    - source-of-truth learner-ceiling companions:
      - `16 env`: `81.64` env-steps/s
      - `64 env`: `73.39` env-steps/s
      - `64 vs 16 = 0.8989x`
  - next active Phase 2 work:
    - continue the prototype-side trainer redesign on the source-of-truth host evidence
    - target the current native-Linux dominant buckets:
      - production: `rollout_policy_forward`, `rollout_env_recv`
      - learner ceiling: `eval_policy_forward`, `train_backward`, `train_policy_forward`
    - do not revisit transport or topology until those trainer-path costs move materially

Current `WC-P2-10` local instrumentation result:

- current dominant named buckets:
  - `eval_policy_forward`
  - `eval_env_recv`
  - `train_backward`
  - `train_policy_forward`
- smaller buckets like `eval_info_stats`, `eval_tensor_copy`, and optimizer-step time are currently secondary on the benchmark-safe path

Frozen `WC-P2-11` production fast-trainer target:

- canonical target doc:
  - [production_fast_trainer_benchmark.md](/home/jordan/code/RL/docs/production_fast_trainer_benchmark.md)
- canonical production rows:
  - native-Linux `16 env`, disabled logging, `production_fast_path_v1`
  - native-Linux `64 env`, disabled logging, `production_fast_path_v1`
- canonical diagnostic companions:
  - learner ceiling `16 env`
  - learner ceiling `64 env`
- excluded from the primary production throughput target:
  - `final_evaluate_seconds`
  - replay export
  - parity/determinism certification work
  - W&B and dashboard overhead
  - hot-path artifact/manifest/checkpoint costs

Current benchmark contract status after `WC-P2-09`:

- production train benchmark:
  - `metric_contract_id = train_benchmark_production_v1`
  - primary metric: `production_env_steps_per_second`
  - excludes `final_evaluate_seconds`
- learner-ceiling benchmark:
  - `metric_contract_id = train_ceiling_diagnostic_v1`
  - primary metric: `diagnostic_env_steps_per_second`
  - remains diagnostic-only for the shipped synchronous path

Current `WC-P2-07` local status:

- local benchmark-only core runner slice:
  - strips smoke-run artifact/checkpoint/manifest overhead from the disabled-train benchmark path
  - WSL live disabled-train rows: `16 env = 56.65`, `64 env = 57.10`
- local deeper trainer slice:
  - suppresses profile, utilization, and metric-only logging work in the benchmark-only trainer path
  - WSL live disabled-train rows: `16 env = 58.12`, `64 env = 57.71`
- local WSL learner-ceiling rows:
  - first slice: `119.90 / 146.94 / 144.77`
  - deeper slice: `123.60 / 149.19 / 142.83`

Interpretation:

- the obvious benchmark/control-plane overhead is no longer the main limiter
- the deeper trainer slice does not materially raise the learner ceiling locally
- the next native-Linux rerun should be treated as a review-gated decision point, not an assumed unblock for `WC-P2-03`

## Mandatory Breakdown

Every stage should keep these benchmark breakdowns visible:

- raw sim batch stepping rate
- Python bridge throughput
- wrapper overhead
- PufferLib VecEnv throughput
- end-to-end training SPS
- W&B overhead at normal cadence
- W&B overhead at aggressive cadence
