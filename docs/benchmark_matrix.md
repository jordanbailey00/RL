# Benchmark Matrix

Date: 2026-03-10

Repos and SHAs:
- RL: local Phase 2 pivot diagnostics on top of `ee4b205277f3638e2983fc6cf3cb8bfb0dc4a6b4`
- fight-caves-RL: `a433c971a7e24f5bff10fb4e22f740d68e70af73`

This matrix freezes the benchmark packet used in this audit. It separates measured rows from standard-but-not-yet-run rows.

## Phase 0 Gate Refresh

Standardized refresh entrypoint:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python scripts/refresh_phase0_packet.py --output-dir /tmp/fc_phase0_packet_clean
```

Supporting clean sim-side entrypoints:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/fight-caves-RL
./gradlew --no-daemon :game:headlessPerformanceReport
./gradlew --no-daemon :game:headlessPerformanceProfile
```

Current local WSL packet status on this host class:
- host class: `wsl2`
- benchmark source of truth: `false`
- native-Linux gate blocker on this host class: `native_linux_source_of_truth_missing`
- bridge rows complete: `1 / 16 / 64`
- vecenv rows complete: `1 / 16 / 64`
- train rows complete: `4 / 16 / 64`
- clean pure-JVM and clean batched headless sim artifacts: present

Current refreshed rows from `/tmp/fc_phase0_packet_clean`:

| Layer | Command | Env count | Result |
| --- | --- | ---: | --- |
| Standalone sim single-slot | `:game:headlessPerformanceReport` | 1 | `30509.78` ticks/s |
| Standalone sim batched | `:game:headlessPerformanceReport` | 16 | `473574.60` env steps/s |
| Bridge reference vs batch | `scripts/benchmark_bridge.py --config bridge_1env_v0 --env-count 1` | 1 | reference `1074.71`, batch `23615.07` env/s |
| Bridge reference vs batch | `scripts/benchmark_bridge.py --config bridge_64env_v0 --env-count 16` | 16 | reference `1297.72`, batch `1573.98` env/s |
| Bridge reference vs batch | `scripts/benchmark_bridge.py --config bridge_64env_v0 --env-count 64` | 64 | reference `1655.51`, batch `1606.92` env/s |
| Wrapper vs vecenv | `scripts/benchmark_env.py --env-count 1` | 1 | wrapper `751.55`, vecenv `980.31` env/s |
| Wrapper vs vecenv | `scripts/benchmark_env.py --env-count 16` | 16 | wrapper `903.61`, vecenv `1459.51` env/s |
| Wrapper vs vecenv | `scripts/benchmark_env.py --env-count 64` | 64 | wrapper `1038.82`, vecenv `1426.81` env/s |
| Training | `scripts/benchmark_train.py --env-count 4 --total-timesteps 1024 --logging-modes disabled` | 4 | `96.66` SPS |
| Training | `scripts/benchmark_train.py --env-count 16 --total-timesteps 1024 --logging-modes disabled` | 16 | `96.53` SPS |
| Training | `scripts/benchmark_train.py --env-count 64 --total-timesteps 1024 --logging-modes disabled` | 64 | `91.62` SPS |

Hosted native-Linux source-of-truth gate summary:

| Layer | Command | Env count | Result |
| --- | --- | ---: | --- |
| Standalone sim single-slot | hosted `:game:headlessPerformanceReport` | 1 | `30532.58` ticks/s |
| Standalone sim batched | hosted `:game:headlessPerformanceReport` | 16 | `404635.41` env steps/s |
| Phase 0 gate | hosted `scripts/refresh_phase0_packet.py` | `1 / 16 / 64` bridge, `1 / 16 / 64` vecenv, `4 / 16 / 64` train | `phase1_unblocked = true`, `workers_needed_for_100k = 1` |

Updated gate interpretation:
- WSL remains useful for local packet refresh and detailed regression comparison
- native Linux is now the approved performance source of truth
- the hosted native-Linux packet satisfies the Phase 0 hard gate and authorizes Phase 1 design work
- the hosted native-Linux packet remains the source-of-truth baseline that the Phase 1 post-implementation gate must compare against

## Phase 1 Local Preview

Standardized local refresh entrypoint:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python scripts/refresh_phase1_packet.py --output-dir /tmp/fc_phase1_packet_local
```

Current local WSL preview rows from `/tmp/fc_phase1_packet_local`:

| Layer | Command | Env count | Result |
| --- | --- | ---: | --- |
| Bridge | `scripts/benchmark_bridge.py --config bridge_64env_v0 --env-count 16` | 16 | `10809.00` env/s |
| Bridge | `scripts/benchmark_bridge.py --config bridge_64env_v0 --env-count 64` | 64 | `11936.49` env/s in packet gate summary |
| VecEnv | `scripts/benchmark_env.py --config vecenv_256env_v0 --env-count 16` | 16 | `6263.67` env/s |
| VecEnv | `scripts/benchmark_env.py --config vecenv_256env_v0 --env-count 64` | 64 | `7336.43` env/s in packet gate summary |
| Python profile | `scripts/refresh_phase1_packet.py` steady-state profile | 16 | `raw_object_conversion_still_dominant = false` |

Local preview interpretation:

- Phase 1 local rows clear the planning thresholds numerically
- the flat-path implementation appears to have removed the intended bottleneck
- the local packet remains a useful preview, but the native-Linux rerun is the approved decision source

## Phase 1 Native-Linux Packet

Published Phase 1 native-Linux results now exist at:

- [phase1-results/latest](https://github.com/jordanbailey00/fight-caves-RL/tree/codex/phase1-results/phase1-native-linux/latest)

Immutable baseline used for comparison:

- [phase0-results immutable pre-phase1 baseline](https://github.com/jordanbailey00/fight-caves-RL/tree/codex/phase0-results/phase0-native-linux/immutable/pre-phase1/rl-3e557474f3c6b4e44842da82a971c8f97d521b10__sim-216c1fd2ac31f450f8c599f9ec9454330a4e6b3a)

Current final native-Linux gate rows:

| Layer | Result |
| --- | --- |
| Bridge `64 env` | `9148.80` env/s, `6.64x` over immutable baseline |
| VecEnv `64 env` | `10961.11` env/s, `8.01x` over immutable baseline |
| Python profile | `raw_object_conversion_still_dominant = false` |
| Decision | `phase2_unblocked = true` |

Gate interpretation:

- the clean immutable-baseline comparison passes the approved Phase 1 thresholds
- the flat path moved the correct boundary on native Linux
- Phase 2 planning remains active, but the production transport swap is still gated separately by the native-Linux Phase 2 pre-swap packet

## Phase 2 Local Prototype Preview

Local Phase 2 prototype benchmark entrypoint:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python scripts/benchmark_subprocess_transport.py --config configs/benchmark/vecenv_256env_v0.yaml --env-count 64 --rounds 64 --output /tmp/subprocess_transport_bench_64.json
```

Current local WSL prototype rows:

| Layer | Command | Env count | Result |
| --- | --- | ---: | --- |
| Subprocess transport comparison | `scripts/benchmark_subprocess_transport.py --env-count 16 --rounds 64` | 16 | pipe `7041.99`, low-copy `7202.60` env/s (`1.02x`) |
| Subprocess transport comparison | `scripts/benchmark_subprocess_transport.py --env-count 64 --rounds 64` | 64 | pipe `8367.57`, low-copy `10765.14` env/s (`1.29x`) |
| End-to-end train probe | temp `benchmark_train.py` pair, `16 env / 128 timesteps / disabled logging` | 16 | pipe `22.69`, low-copy `23.38` SPS (`1.03x`) |

Interpretation:

- the Phase 2 prototype is real and healthy
- the low-copy data plane helps more at higher env counts than at small env counts on this WSL host
- the current local gain is still too small to justify a production transport swap by itself

## Phase 2 Native-Linux Pre-Swap Gate

Hosted source-of-truth run:

- [fight-caves-RL/actions/runs/22883118379](https://github.com/jordanbailey00/fight-caves-RL/actions/runs/22883118379)

Current native-Linux gate rows:

Benchmark-contract note:

- the disabled-train rows in this table predate `WC-P2-09`
- they are legacy pre-correction rows, not the new frozen production-fast-path metric contract

| Layer | Result |
| --- | --- |
| Transport `64 env` | pipe `10868.61`, `shared_memory_v1` `8340.91` env/s (`0.7674x`) |
| Disabled train `16 env` | pipe `74.05`, `shared_memory_v1` `75.01` SPS |
| Disabled train `64 env` | pipe `75.03`, `shared_memory_v1` `74.85` SPS (`0.9977x`) |
| Shared-train scaling `64 vs 16` | `0.9979x` |
| Decision | `wc_p2_03_unblocked = false` |

Gate interpretation:

- transport-only signal is no longer strong enough on the latest source-of-truth rerun
- end-to-end training signal is too weak
- `16 -> 64` shared-transport scaling is still unhealthy
- `WC-P2-03` remains blocked
- future reruns of this gate must use the corrected production train metric:
  - `metric_contract_id = train_benchmark_production_v1`
  - `production_env_steps_per_second`

## Learner Ceiling Rows

Repo-owned diagnostic:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
WANDB_MODE=disabled uv run python scripts/benchmark_train_ceiling.py --config configs/benchmark/train_1024env_v0.yaml --env-counts 4,16,64 --total-timesteps 4096 --output /tmp/fc_train_ceiling_report.json
```

| Layer | Result |
| --- | --- |
| Fake-env train ceiling `4 env` | `154.45` env-steps/s |
| Fake-env train ceiling `16 env` | `156.20` env-steps/s |
| Fake-env train ceiling `64 env` | `144.43` env-steps/s |
| `64 env` stage split | evaluate `15.83s`, train `24.87s`, final evaluate `16.02s` |

Interpretation:

- with the live sim removed entirely, the current train loop still tops out around `145-156` env-steps/s on this host
- this makes the learner/update path the dominant current blocker for end-to-end training throughput
- this benchmark family is diagnostic-only after `WC-P2-09`

Hosted native-Linux learner-ceiling confirmation:

| Layer | Result |
| --- | --- |
| Fake-env train ceiling `4 env` | `94.97` env-steps/s |
| Fake-env train ceiling `16 env` | `74.67` env-steps/s |
| Fake-env train ceiling `64 env` | `68.43` env-steps/s |
| `64 env` stage split | evaluate `28.64s`, train `62.47s`, final evaluate `28.60s` |
| Scaling | `64 vs 16 = 0.9165x`, `64 vs 4 = 0.7206x` |

Interpretation:

- the source-of-truth host class confirms the trainer-bound diagnosis directly
- this benchmark family now defines the active Phase 2 pivot: do not re-attempt `WC-P2-03` until trainer-bound overhead is reduced or more cleanly isolated

Frozen benchmark distinction after `WC-P2-09`:

- production train benchmark:
  - `metric_contract_id = train_benchmark_production_v1`
  - primary metric excludes `final_evaluate`
- learner-ceiling benchmark:
  - `metric_contract_id = train_ceiling_diagnostic_v1`
  - primary metric remains full shipped synchronous wall-clock and is diagnostic-only

Frozen benchmark distinction after `WC-P2-11`:

- canonical production fast-trainer rows:
  - native-Linux `16 env`, disabled logging
  - native-Linux `64 env`, disabled logging
- canonical production metric:
  - `production_env_steps_per_second`
- canonical diagnostic companions:
  - learner ceiling `16 env`
  - learner ceiling `64 env`
- non-canonical for the current Phase 2 decision path:
  - `4 env` learner ceiling
  - local WSL throughput rows
  - online/offline W&B rows
  - transport-promotion rows
  - `256 / 1024 env` train rows on the current trainer path

## WC-P2-10 Local Trainer Instrumentation

Instrumented local production-fast-path rows:

| Layer | Result |
| --- | --- |
| Production train `16 env` | `58.73` SPS, wall-clock `40.88` SPS |
| Production train `16 env` top buckets | `eval_policy_forward 15.04s`, `eval_env_recv 3.78s`, `train_backward 3.74s`, `train_policy_forward 1.79s` |
| Production train `64 env` | `60.23` SPS, wall-clock `45.46` SPS |
| Production train `64 env` top buckets | `eval_policy_forward 10.26s`, `eval_env_recv 6.41s`, `train_backward 3.40s`, `train_policy_forward 2.01s` |
| Learner ceiling `16 env` | `144.22` env-steps/s |
| Learner ceiling `16 env` top buckets | `eval_policy_forward 30.40s`, `train_backward 15.72s`, `train_policy_forward 10.34s` |
| Learner ceiling `64 env` | `146.87` env-steps/s |
| Learner ceiling `64 env` top buckets | `eval_policy_forward 31.71s`, `train_backward 14.80s`, `train_policy_forward 9.11s` |

Interpretation:

- the current local instrumentation packet points at forward/backward structure as the next active trainer bottleneck
- `eval_info_stats`, `eval_tensor_copy`, and optimizer-step time are currently much smaller on the benchmark-safe path

## WC-P2-14 Local Prototype Gate

Local WSL prototype fast-trainer rows with the subprocess JVM resolver fix in place:

| Layer | Result |
| --- | --- |
| Prototype production train `16 env` | `95.74` SPS |
| Prototype production train `16 env` top buckets | `rollout_policy_forward 15.23s`, `update_backward 14.49s`, `update_policy_forward 9.22s`, `rollout_env_recv 3.43s` |
| Prototype production train `64 env` | `93.06` SPS |
| Prototype production train `64 env` top buckets | `rollout_policy_forward 15.42s`, `update_backward 14.26s`, `update_policy_forward 9.38s`, `rollout_env_recv 4.64s` |
| Prototype production scaling `64 vs 16` | `0.9720x` |
| Learner ceiling `16 env` companion | `145.70` env-steps/s |
| Learner ceiling `64 env` companion | `144.79` env-steps/s |
| Learner ceiling scaling `64 vs 16` | `0.9937x` |

Interpretation:

- the first project-owned prototype materially improves the local production fast-path row versus the earlier local shipped-path band of about `58.73 / 60.23`
- the improvement does not yet produce healthy `16 -> 64` scaling
- the diagnostic learner ceiling remains effectively flat in the same `~145` env-steps/s band, so transport is still not the next active gate
- the next Phase 2 slice should stay inside deeper trainer-loop replacement rather than revisiting transport promotion or actor/learner topology

## PR Batch G Follow-On Local Preview

Local WSL prototype fast-trainer rows after removing the padded multi-discrete sampling/logprob path from the project-owned prototype:

| Layer | Result |
| --- | --- |
| Prototype production train `16 env` | `417.36` SPS |
| Prototype production train `16 env` top buckets | `rollout_policy_forward 4.13s`, `update_backward 1.27s`, `update_policy_forward 0.59s` |
| Prototype production train `64 env` | `398.80` SPS |
| Prototype production train `64 env` top buckets | `rollout_env_recv 4.12s`, `rollout_policy_forward 3.81s`, `update_backward 1.35s`, `update_policy_forward 0.71s` |
| Prototype production scaling `64 vs 16` | `0.9555x` |

Interpretation:

- the follow-on trainer-core slice materially improves the local production fast-path row again
- the biggest local win is removal of the padded multi-discrete policy-forward/logprob path
- the prototype now clears the old local `250` SPS escalation bar
- scaling is still flat enough that the next action should be a native-Linux rerun of this corrected packet, not a transport or topology pivot

## Hosted Native-Linux Prototype Packet

Hosted native-Linux source-of-truth rows:

| Layer | Result |
| --- | --- |
| Prototype production `16 env` | `469.92` SPS |
| Prototype production `64 env` | `341.43` SPS |
| Prototype production scaling `64 vs 16` | `0.7266x` |
| Learner ceiling `16 env` | `81.64` env-steps/s |
| Learner ceiling `64 env` | `73.39` env-steps/s |
| Learner ceiling scaling `64 vs 16` | `0.8989x` |

Hosted prototype top buckets:

- `16 env`:
  - `rollout_policy_forward = 4.89s`
  - `update_backward = 2.02s`
  - `rollout_env_recv = 0.94s`
  - `update_policy_forward = 0.60s`
- `64 env`:
  - `rollout_env_recv = 4.99s`
  - `rollout_policy_forward = 4.36s`
  - `update_backward = 1.95s`
  - `update_policy_forward = 0.52s`

Interpretation:

- the hosted packet now completes successfully end-to-end after fixing the packaged headless-distribution contract
- absolute `16 env` production throughput is strong, but `64 env` regresses materially on the source-of-truth host
- the learner ceiling is both low and negatively scaling, so the next Phase 2 slice stays inside trainer redesign
- Phase 3/topology remains blocked

## WC-P2-07 Local Preview Rows

These rows are local-only and not decision-authoritative, but they are the current review state for `WC-P2-07`.

| Layer | Result |
| --- | --- |
| Disabled train core runner `16 env` | `56.65` SPS |
| Disabled train core runner `64 env` | `57.10` SPS |
| Disabled train deeper trainer slice `16 env` | `58.12` SPS |
| Disabled train deeper trainer slice `64 env` | `57.71` SPS |
| Fake-env learner ceiling after first slice `4 / 16 / 64` | `119.90 / 146.94 / 144.77` env-steps/s |
| Fake-env learner ceiling after deeper trainer slice `4 / 16 / 64` | `123.60 / 149.19 / 142.83` env-steps/s |

Interpretation:

- the benchmark-only runner cleanup helps the short live train row materially relative to the older smoke-driven benchmark path
- the deeper trainer-internal suppression slice produces only marginal movement
- local evidence alone does not justify a source-of-truth claim that `WC-P2-07` has materially raised the trainer ceiling

## Measured Rows

| Layer | Command | Config | Env count | Worker topology | Worker count | Envs / worker | Dashboard | W&B | Replay/artifacts | Result |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- | --- | --- |
| Bridge reference vs batch | `scripts/benchmark_bridge.py` | `bridge_1env_v0` | 1 | embedded JVM | 1 | 1 | disabled | benchmark-minimized | disabled | reference `943.85`, batch `23765.25` env/s |
| Bridge reference vs batch | `scripts/benchmark_bridge.py` | `bridge_64env_v0` | 64 | embedded JVM | 1 | 64 | disabled | benchmark-minimized | disabled | reference `1437.68`, batch `1481.11` env/s |
| Wrapper vs vecenv | `scripts/benchmark_env.py` | `train_baseline_v0` | 4 | embedded JVM | 1 | 4 | enabled in config, benchmark path does not render dashboard | benchmark-standard | disabled | wrapper `498.69`, vecenv `906.64` env/s |
| Wrapper vs vecenv | `scripts/benchmark_env.py` | `vecenv_256env_v0` | 16 | embedded JVM | 1 | 16 | disabled | benchmark-standard | disabled | wrapper `628.68`, vecenv `1232.60` env/s |
| Wrapper vs vecenv | `scripts/benchmark_env.py` | `vecenv_256env_v0` | 64 | embedded JVM | 1 | 64 | disabled | benchmark-standard | disabled | wrapper `854.07`, vecenv `1492.10` env/s |
| Training | `scripts/benchmark_train.py` | `train_baseline_v0` | 4 | parent + subprocess worker | 1 | 4 | disabled | disabled / offline | checkpoint + manifest only | `36.40` disabled, `36.14` offline SPS |
| Training | `scripts/benchmark_train.py` | `train_1024env_v0` | 16 | parent + subprocess worker | 1 | 16 | disabled | disabled / offline | checkpoint + manifest only | `82.84` disabled, `91.66` offline SPS |
| Training | `scripts/benchmark_train.py` | `train_1024env_v0` | 64 | parent + subprocess worker | 1 | 64 | disabled | disabled | checkpoint + manifest only | `87.93` disabled SPS |
| Training online probe | direct `scripts/train.py` | `train_baseline_v0` | 4 | parent + subprocess worker | 1 | 4 | disabled | online | checkpoint + manifest only | `11.87` wall SPS |
| Logging probe | direct `scripts/train.py` | temp baseline variants | 4 | parent + subprocess worker | 1 | 4 | on / off | disabled | checkpoint interval `1` vs `999999` | directional only, not benchmark-harness quality |

## Standard Packet To Reuse

### Raw Sim / Near-Raw

| Status | Command | Env count | Goal |
| --- | --- | ---: | --- |
| Existing artifact only | `fight-caves-RL/history/performance_report_step11.md` | 1 | pure-JVM reference |
| Measured | `scripts/benchmark_bridge.py --config configs/benchmark/bridge_1env_v0.yaml` | 1 | best-case bridge trace |
| Not run in this pass | longer direct-JVM benchmark on current WSL host | 1 | host-local pure-JVM ceiling |

### Bridge Scaling

| Status | Command | Env count | Goal |
| --- | --- | ---: | --- |
| Measured | `benchmark_bridge.py --config bridge_1env_v0` | 1 | single-slot trace ceiling |
| Not run in this pass | `benchmark_bridge.py` with env override | 16 | first batch scaling point |
| Measured | `benchmark_bridge.py --config bridge_64env_v0` | 64 | current batch saturation point |
| Not run in this pass | 256-env bridge run | 256 | larger batch saturation |
| Not run in this pass | 1024-env bridge run | 1024 | only after lower-cost transport exists |

### VecEnv Scaling

| Status | Command | Env count | Goal |
| --- | --- | ---: | --- |
| Not run in this pass | `benchmark_env.py` | 1 | vecenv floor |
| Measured | `benchmark_env.py` | 4 | small correctness baseline |
| Measured | `benchmark_env.py` | 16 | first meaningful batch |
| Measured | `benchmark_env.py` | 64 | current embedded vecenv saturation point |
| Not run in this pass | `benchmark_env.py` | 256 | only after bridge bottleneck relief |
| Not run in this pass | `benchmark_env.py` | 1024 | not worthwhile before transport redesign |

### Training Scaling

| Status | Command | Env count | Goal |
| --- | --- | ---: | --- |
| Measured | `benchmark_train.py --config train_baseline_v0` | 4 | shipped small baseline |
| Measured | `benchmark_train.py --config train_1024env_v0` | 16 | first scale-up |
| Measured | `benchmark_train.py --config train_1024env_v0` | 64 | saturation check |
| Not run in this pass | `benchmark_train.py --config train_1024env_v0` | 256 | defer until transport redesign |
| Not run in this pass | `benchmark_train.py --config train_1024env_v0` | 1024 | defer until earlier gates improve |

## Matrix Interpretation

- `1 env -> 64 env` bridge scaling is currently poor.
- Embedded vecenv throughput tracks bridge throughput closely.
- Training throughput still plateaus by `4-64 envs` in the refreshed Phase 0 packet.
- That makes `256` and `1024` env training runs low-value before the bridge and transport path changes.
- Benchmark hygiene for future reruns:
  - compare only runs from the same host class
  - use `disabled` logging for the core Phase 0 train row
  - treat native Linux as the go/no-go source of truth, not WSL
