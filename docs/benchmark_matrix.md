# Benchmark Matrix

Date: 2026-03-10

Repos and SHAs:
- RL: local Phase 2 pivot diagnostics on top of `ee4b205277f3638e2983fc6cf3cb8bfb0dc4a6b4`
- fight-caves-RL: `a433c971a7e24f5bff10fb4e22f740d68e70af73`

This matrix freezes the benchmark packet used in this audit. It separates measured rows from standard-but-not-yet-run rows.

## Phase 1 Measurement Contract

The active pivot baseline pass now expects WSL/Linux benchmark commands and JSON outputs with explicit stage buckets, env hot-path buckets, and memory snapshots.

Canonical WSL commands from this repo root:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python scripts/benchmark_env.py --config configs/benchmark/vecenv_256env_v0.yaml --env-count 64 --rounds 64 --output /tmp/fc_phase1_env_benchmark.json
uv run python scripts/benchmark_train.py --config configs/benchmark/train_1024env_v0.yaml --env-count 16 --total-timesteps 1024 --logging-modes disabled,standard --output /tmp/fc_phase1_train_benchmark.json
```

Required Phase 1 env benchmark fields:

- `wrapper.env_steps_per_second`
- `measurement.env_steps_per_second`
- `measurement.runner_stage_seconds`
- `measurement.hot_path_bucket_totals`
- `measurement.memory_profile`

Required Phase 1 train benchmark fields:

- `measurements[*].production_env_steps_per_second`
- `measurements[*].wall_clock_env_steps_per_second`
- `measurements[*].runner_stage_seconds`
- `measurements[*].trainer_bucket_totals`
- `measurements[*].memory_profile`

Current interpretation boundary:

- `trainer_bucket_totals` covers trainer-side rollout/evaluate/update work.
- the exact current V1 bridge-first env worker buckets stay in the env-only benchmark, which is the source-of-truth place for Python action decode, JVM apply/tick, flat observe, terminal inference, reward evaluation, and info assembly.
- `memory_profile` is part of the baseline artifact and should be compared only across WSL/Linux-compatible runs.

Active benchmark rule:

- WSL is now the approved source-of-truth benchmark host for this workspace pivot.
- Earlier native-Linux references below are historical snapshots from the pre-WSL rule and should not be used as the current decision gate.

## Phase 4.2 Fast V2 Benchmark Entry Points

The async V2 worker path now has dedicated config entrypoints:

- env-only fast async vecenv:
  - `configs/benchmark/fast_env_v2.yaml`
  - canonical command:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python scripts/benchmark_env.py --config configs/benchmark/fast_env_v2.yaml --mode vecenv --env-count 64 --rounds 256 --output /tmp/fc_fast_env_v2.json
```

- end-to-end fast training:
  - `configs/benchmark/fast_train_v2.yaml`
  - canonical command:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python scripts/benchmark_train.py --config configs/benchmark/fast_train_v2.yaml --env-count 64 --total-timesteps 1024 --logging-modes disabled,standard --output /tmp/fc_fast_train_v2.json
```

Topology fields now expected in Phase 4.2 benchmark outputs:

- `vecenv_topology.backend`
- `vecenv_topology.env_backend`
- `vecenv_topology.transport_mode`
- `vecenv_topology.worker_count`
- `vecenv_topology.worker_env_counts`

Interpretation rule:

- these fields describe the real runner topology used for the measurement
- they do not replace the detailed timing buckets in the benchmark payloads
- settled throughput deltas for PR 4.2 should still be read from the benchmark artifacts, not from this matrix alone

## Phase 0 WSL Source-of-Truth Packet

Standardized refresh entrypoint:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python scripts/refresh_phase0_packet.py --output-dir /tmp/fc_phase0_packet_wsl_20260310
```

Supporting clean sim-side entrypoints:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/fight-caves-RL
./gradlew --no-daemon :game:headlessPerformanceReport
./gradlew --no-daemon :game:headlessPerformanceProfile
```

Captured packet directory:

- `/tmp/fc_phase0_packet_wsl_20260310`
- Frozen immutable pre-Phase-1 WSL comparison baseline:
  - `/home/jordan/code/RL/artifacts/benchmarks/phase0_wsl_pre_phase1_immutable_20260309`
  - reconstructed from the recorded WSL Phase 0 numbers in `docs/performance_decomposition_report.md` and `changelog.md` because the original temp packet directory was not preserved

Gate status from `phase0_packet.json`:
- host class: `wsl2`
- benchmark source of truth: `true`
- bridge rows complete: `1 / 16 / 64`
- vecenv rows complete: `1 / 16 / 64`
- train rows complete: `4 / 16 / 64`
- clean pure-JVM and clean batched headless sim artifacts: present
- per-worker sim env steps per second: `592546.38`
- workers needed for `100k`: `1`
- `phase1_unblocked = true`

Current refreshed rows from `/tmp/fc_phase0_packet_wsl_20260310`:

| Layer | Command | Env count | Result |
| --- | --- | ---: | --- |
| Standalone sim single-slot | `:game:headlessPerformanceReport` | 1 | `27416.25` ticks/s |
| Standalone sim batched | `:game:headlessPerformanceReport` | 16 | `592546.38` env steps/s |
| Bridge reference vs batch | `scripts/benchmark_bridge.py --config bridge_1env_v0 --env-count 1` | 1 | reference `1272.10`, batch `22695.17` env/s |
| Bridge reference vs batch | `scripts/benchmark_bridge.py --config bridge_64env_v0 --env-count 16` | 16 | reference `1576.95`, batch `11245.97` env/s |
| Bridge reference vs batch | `scripts/benchmark_bridge.py --config bridge_64env_v0 --env-count 64` | 64 | reference `1589.07`, batch `9616.06` env/s |
| Wrapper vs vecenv | `scripts/benchmark_env.py --env-count 1` | 1 | wrapper `927.74`, vecenv `2729.12` env/s |
| Wrapper vs vecenv | `scripts/benchmark_env.py --env-count 16` | 16 | wrapper `1011.56`, vecenv `10032.00` env/s |
| Wrapper vs vecenv | `scripts/benchmark_env.py --env-count 64` | 64 | wrapper `1037.96`, vecenv `13182.86` env/s |
| Training | `scripts/benchmark_train.py --env-count 4 --total-timesteps 1024 --logging-modes disabled` | 4 | production `100.94` SPS |
| Training | `scripts/benchmark_train.py --env-count 16 --total-timesteps 1024 --logging-modes disabled` | 16 | production `107.95` SPS |
| Training | `scripts/benchmark_train.py --env-count 64 --total-timesteps 1024 --logging-modes disabled` | 64 | production `102.30` SPS |

Updated gate interpretation:
- the WSL packet now satisfies the active Phase 0 source-of-truth gate directly
- the current V1 bridge-first path saturates around `9.6k` bridge env/s and `13.2k` vecenv env/s at `64 env`
- the current end-to-end train rows remain in the `101-108` SPS band and are still trainer-bound

Frozen immutable pre-Phase-1 WSL baseline rows from `/home/jordan/code/RL/artifacts/benchmarks/phase0_wsl_pre_phase1_immutable_20260309`:

| Layer | Env count | Result |
| --- | ---: | --- |
| Standalone sim single-slot | 1 | `30509.78` ticks/s |
| Standalone sim batched | 16 | `473574.60` env steps/s |
| Bridge reference vs batch | 1 | reference `1074.71`, batch `23615.07` env/s |
| Bridge reference vs batch | 16 | reference `1297.72`, batch `1573.98` env/s |
| Bridge reference vs batch | 64 | reference `1655.51`, batch `1606.92` env/s |
| Wrapper vs vecenv | 1 | wrapper `751.55`, vecenv `980.31` env/s |
| Wrapper vs vecenv | 16 | wrapper `903.61`, vecenv `1459.51` env/s |
| Wrapper vs vecenv | 64 | wrapper `1038.82`, vecenv `1426.81` env/s |
| Training | 4 | production `96.66` SPS |
| Training | 16 | production `96.53` SPS |
| Training | 64 | production `91.62` SPS |

## Phase 1 WSL Source-of-Truth Packet

Standardized local refresh entrypoint:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python scripts/refresh_phase1_packet.py --output-dir /tmp/fc_phase1_packet_wsl_20260310 --phase0-baseline-dir /tmp/fc_phase0_packet_wsl_20260310
```

Captured packet directory:

- `/tmp/fc_phase1_packet_wsl_20260310`

Current refreshed rows from `/tmp/fc_phase1_packet_wsl_20260310`:

| Layer | Command | Env count | Result |
| --- | --- | ---: | --- |
| Bridge | `scripts/benchmark_bridge.py --config bridge_64env_v0 --env-count 16` | 16 | `10997.47` env/s |
| Bridge | `scripts/benchmark_bridge.py --config bridge_64env_v0 --env-count 64` | 64 | `11090.51` env/s |
| VecEnv | `scripts/benchmark_env.py --config vecenv_256env_v0 --env-count 16` | 16 | `8924.95` env/s |
| VecEnv | `scripts/benchmark_env.py --config vecenv_256env_v0 --env-count 64` | 64 | `12402.49` env/s |
| Python profile | `scripts/refresh_phase1_packet.py` steady-state profile | 16 | `raw_object_conversion_still_dominant = false` |

Gate interpretation:

- the WSL Phase 1 packet records the current absolute throughput and profile state on the approved host class
- `raw_object_conversion_still_dominant = false` remains the important qualitative Phase 1 result
- the immutable pre-Phase-1 WSL baseline is now frozen at `/home/jordan/code/RL/artifacts/benchmarks/phase0_wsl_pre_phase1_immutable_20260309`, so the ratio provenance issue is closed
- valid ratios against that frozen baseline are `bridge 64 = 6.9017x` and `vecenv 64 = 8.6925x`
- the Phase 1 gate helper has been updated so this packet is now a valid pass/fail artifact again on WSL, with `phase2_unblocked = true`

## Phase 2 PR 2.1 WSL Rerun Against Frozen Baseline

Rerun artifact directory:

- `/home/jordan/code/RL/artifacts/benchmarks/phase2_pr21_wsl_rerun_20260310`
- comparison summary:
  - `/home/jordan/code/RL/artifacts/benchmarks/phase2_pr21_wsl_rerun_20260310/comparison_vs_phase0_wsl_pre_phase1_immutable.json`

Current PR 2.1 rows vs the frozen immutable pre-Phase-1 WSL baseline:

| Layer | Env count | Frozen baseline | PR 2.1 rerun | Ratio |
| --- | ---: | ---: | ---: | ---: |
| VecEnv | 16 | `1459.51` env/s | `8112.89` env/s | `5.5586x` |
| VecEnv | 64 | `1426.81` env/s | `8675.26` env/s | `6.0802x` |
| Disabled train | 16 | `96.53` SPS | `106.23` SPS | `1.1005x` |
| Disabled train | 64 | `91.62` SPS | `96.59` SPS | `1.0543x` |

Interpretation:

- PR 2.1 is now comparable against a frozen WSL baseline and clears that historical baseline on the measured vecenv and disabled-train rows
- the earlier `--rounds 64` env rerun understated the steady-state path and should not be used for the current-vs-current comparison
- after the follow-up minimal-info vecenv buffer trim, the `vecenv_apply_step_buffers` bucket dropped materially:
  - `16 env`: `0.02645s -> 0.01432s`
  - `64 env`: `0.06603s -> 0.04158s`
- serial post-trim WSL rows now show:
  - `vecenv 16`: `11432.35` env/s, above both the Phase 1 row `8924.95` and the refreshed Phase 0 row `10032.00`
  - `vecenv 64`: serial reruns at `10489.73` and `13702.25` env/s, which shows real host variance and at least one rerun above both the Phase 1 row `12402.49` and the refreshed Phase 0 row `13182.86`
  - disabled train `16`: stable around `106.40-106.73` SPS, still slightly below the refreshed Phase 0 row `107.95`
  - disabled train `64`: stable around `101.40-101.45` SPS, still slightly below the refreshed Phase 0 row `102.30`
- PR 2.1 was later closed for forward progress with an explicit non-blocking disposition on the remaining small-batch `16 env` residuals

Tighter serial attribution set after the follow-up hot-path trim:

- artifact directory:
  - `/home/jordan/code/RL/artifacts/benchmarks/phase2_pr21_wsl_serial_attribution_20260310`
- summary:
  - `/home/jordan/code/RL/artifacts/benchmarks/phase2_pr21_wsl_serial_attribution_20260310/serial_attribution_summary.json`
- contract:
  - `3` serial replicates per row
  - serial-only execution; no concurrent benchmark rows counted

Median results from the tighter serial attribution set:

| Layer | Median | Range | Ratio vs current WSL baseline |
| --- | ---: | ---: | ---: |
| VecEnv `16 env` | `8729.40` env/s | `8454.33 - 11291.75` | `0.8702x` vs refreshed Phase 0, `0.9781x` vs Phase 1 |
| VecEnv `64 env` | `14898.98` env/s | `14749.31 - 15402.48` | `1.1302x` vs refreshed Phase 0, `1.2013x` vs Phase 1 |
| Disabled train `16 env` | `106.59` SPS | `105.75 - 107.51` | `0.9874x` vs refreshed Phase 0 |
| Disabled train `64 env` | `102.14` SPS | `101.88 - 102.45` | `0.9984x` vs refreshed Phase 0 |

Median bucket attribution highlights from that serial set:

- `vecenv 64` now has a strong and stable median improvement over both current WSL baseline packets
- the median `vecenv 64` hot-path buckets improved materially relative to the Phase 1 packet:
  - `client_info_assembly`: `0.07708s -> 0.00171s`
  - `vecenv_apply_step_buffers`: `0.06603s -> 0.02991s`
  - `vecenv_python_action_decode`: `0.10407s -> 0.06638s`
- `vecenv 16` remains noisy enough that its median is still slightly below the refreshed current WSL baseline even though one rerun cleared it comfortably
- disabled-train `64 env` is now effectively at parity with the refreshed current WSL Phase 0 row, while disabled-train `16 env` still carries a modest gap

Current PR 2.1 interpretation after the tighter serial set:

- the stale Phase 1 gate logic is fixed and the Phase 1 packet is valid pass/fail evidence again
- the minimal-info hot-path trims are real and measurable on the env side, especially at `64 env`
- PR 2.1 is retained as a closed no-regret trim: the remaining `16 env` gaps were dispositioned as non-blocking small-batch residuals rather than a reason to delay PR 2.2

## Phase 2 PR 2.2 WSL Batch Bridge Matrix

PR 2.2 artifact directories:

- full matrix:
  - `/home/jordan/code/RL/artifacts/benchmarks/phase2_pr22_wsl_rerun_20260311`
- comparison summary:
  - `/home/jordan/code/RL/artifacts/benchmarks/phase2_pr22_wsl_rerun_20260311/comparison_vs_current_wsl_baselines.json`
- serial vecenv replicate set:
  - `/home/jordan/code/RL/artifacts/benchmarks/phase2_pr22_wsl_vecenv_serial_replicates_20260311`
- serial vecenv replicate summary:
  - `/home/jordan/code/RL/artifacts/benchmarks/phase2_pr22_wsl_vecenv_serial_replicates_20260311/serial_replicate_summary.json`

Full PR 2.2 matrix rows:

| Layer | Env count | Result | Ratio vs current WSL baseline |
| --- | ---: | ---: | ---: |
| Bridge | 16 | `15317.40` env/s | `1.3928x` vs Phase 1 |
| Bridge | 64 | `21120.11` env/s | `1.9043x` vs Phase 1 |
| VecEnv | 16 | `9063.86` env/s | `1.0156x` vs Phase 1 |
| VecEnv | 64 | `9310.34` env/s | `0.7507x` vs Phase 1 |
| Disabled train | 16 | `108.22` SPS | `1.0025x` vs refreshed Phase 0 |
| Disabled train | 64 | `101.34` SPS | `0.9906x` vs refreshed Phase 0 |

Serial vecenv replicate medians used to disposition the initial env-only outlier:

| Layer | Median | Range | Ratio vs current WSL Phase 1 |
| --- | ---: | ---: | ---: |
| VecEnv `16 env` | `10145.80` env/s | `8708.01 - 10560.16` | `1.1368x` |
| VecEnv `64 env` | `12934.54` env/s | `12789.66 - 14483.25` | `1.0429x` |

Median bucket attribution from that PR 2.2 serial vecenv set:

- `vecenv 16` median:
  - `vecenv_send_total = 0.10050s`
  - `vecenv_step_batch_call = 0.09035s`
  - `client_apply_actions_batch_jvm = 0.03862s`
  - `client_flat_observe_batch = 0.02163s`
- `vecenv 64` median:
  - `vecenv_send_total = 0.31535s`
  - `vecenv_step_batch_call = 0.28592s`
  - `client_apply_actions_batch_jvm = 0.15098s`
  - `client_flat_observe_batch = 0.05419s`
  - `vecenv_apply_step_buffers = 0.00815s`

Current PR 2.2 interpretation:

- the interim batch bridge APIs are real and active in the runtime and RL bridge path
- the bridge layer shows a clear overhead reduction at both `16 env` and `64 env`
- the first full-matrix `vecenv 64` run was an outlier; the follow-up serial replicate set restores the expected env-only gain at both `16 env` and `64 env`
- disabled-train throughput stayed effectively flat: `16 env` is slightly above the refreshed current WSL baseline and `64 env` is only about `0.9%` below it
- PR 2.2 is quantitatively good enough to close because its target was bridge/env overhead reduction, not a final V1 training architecture win

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
- that hosted prototype packet did not unblock Phase 3/topology at the time

## Phase 3 PR 3.2 WSL Fast-Kernel Serial Closure

Decision-quality WSL serial fast-kernel artifact:

- summary: `/home/jordan/code/RL/artifacts/benchmarks/phase3_pr32_wsl_fast_serial_replicates_20260311/serial_replicate_summary.json`
- comparison: `/home/jordan/code/RL/artifacts/benchmarks/phase3_pr32_wsl_fast_serial_replicates_20260311/comparison_vs_trimmed_v1_serial_baseline.json`
- trimmed V1 serial baseline: `/home/jordan/code/RL/artifacts/benchmarks/phase2_pr22_wsl_vecenv_serial_replicates_20260311/serial_replicate_summary.json`
- benchmarked path: `FastFightCavesKernelRuntime.resetBatch + stepBatch + flat buffer`
- protocol: `64` warmup rounds, `128` measured rounds, `3` serial replicates per env-count row

| Row | Fast-kernel median | Trimmed V1 median | Ratio |
| --- | ---: | ---: | ---: |
| `16 env` | `41940.63` env-steps/s | `10145.80` env-steps/s | `4.13x` |
| `64 env` | `66856.26` env-steps/s | `12934.54` env-steps/s | `5.17x` |

Fast-kernel median stage totals:

- `16 env`: `reset = 371.26ms`, `apply_actions = 1.37ms`, `tick = 20.78ms`, `observe_flat = 17.61ms`, `projection = 10.64ms`, `total = 48.83ms`
- `64 env`: `reset = 1451.30ms`, `apply_actions = 2.40ms`, `tick = 19.96ms`, `observe_flat = 62.54ms`, `projection = 37.30ms`, `total = 122.53ms`

Interpretation:

- PR 3.2 now clears its quantitative acceptance bar on WSL with an apples-to-apples serial comparison against the accepted trimmed V1 baseline
- the benchmarked path is the real fast-kernel batch reset/step + flat-buffer path, not the older bridge vecenv path
- the remaining dominant kernel costs are `observe_flat` and projection, not action decode or batch apply

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
