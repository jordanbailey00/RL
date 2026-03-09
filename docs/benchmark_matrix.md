# Benchmark Matrix

Date: 2026-03-09

Repos and SHAs:
- RL: `cda7ab4104799be40ffe39f77e5a86c2e6f0eea5`
- fight-caves-RL: `2365506bd3ea5cce515c571f39c24e72a38acc67`

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
- the Phase 1 decision is still pending because the hosted native-Linux packet has not yet been reviewed

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
| Existing artifact only | `fight-caves-RL/docs/performance_report.md` | 1 | pure-JVM reference |
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
