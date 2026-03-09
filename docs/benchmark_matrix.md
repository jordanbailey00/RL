# Benchmark Matrix

Date: 2026-03-09

Repos and SHAs:
- RL: `cda7ab4104799be40ffe39f77e5a86c2e6f0eea5`
- fight-caves-RL: `2365506bd3ea5cce515c571f39c24e72a38acc67`

This matrix freezes the benchmark packet used in this audit. It separates measured rows from standard-but-not-yet-run rows.

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
- Training throughput plateaus by `16-64 envs`.
- That makes `256` and `1024` env training runs low-value before the bridge and transport path changes.
