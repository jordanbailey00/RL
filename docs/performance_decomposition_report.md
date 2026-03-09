# Performance Decomposition Report

Date: 2026-03-09

Repos and SHAs:
- RL: `cda7ab4104799be40ffe39f77e5a86c2e6f0eea5`
- fight-caves-RL: `2365506bd3ea5cce515c571f39c24e72a38acc67`
- RSPS: `ec5f6d0d307fe1072b134693f636e10a22873ce0`

This report decomposes the current stack into separate layers and records the measurements collected in this audit pass. Facts and hypotheses are separated explicitly.

## Topology Used

- Host: WSL2 on Windows, AMD Ryzen 5 5600G, 6 cores / 12 threads, 15 GiB RAM, CPU-only
- Python: `3.11.15`
- Torch: `2.10.0+cpu`
- PufferLib: `pufferlib-core 3.0.17`
- JVM: Temurin `21.0.10+7`
- RL train bridge mode: subprocess worker with embedded JVM
- RL env microbench bridge mode: embedded JVM in-process
- W&B modes used in this packet: `disabled`, `offline`, `online`
- Dashboard modes used in this packet: `disabled` and `enabled` where noted

## Phase 0 Gate Refresh

The Phase 0 measurement gate now has a repo-owned refresh path:

- clean standalone sim benchmark: `./gradlew --no-daemon :game:headlessPerformanceReport`
- clean standalone sim profile: `./gradlew --no-daemon :game:headlessPerformanceProfile`
- unified RL packet refresh: `uv run python scripts/refresh_phase0_packet.py --output-dir /tmp/fc_phase0_packet_clean`

Current-host WSL packet refreshed from those entrypoints:

- sim standalone report:
  - single-slot throughput: `30509.78` ticks/s
  - batched headless throughput (`16` envs): `473574.60` env steps/s
  - per-worker estimate for `100k`: `1` worker on this host-class measurement
- bridge:
  - `1 env`: reference `1074.71`, batch `23615.07` env/s
  - `16 env`: reference `1297.72`, batch `1573.98` env/s
  - `64 env`: reference `1655.51`, batch `1606.92` env/s
- vecenv:
  - `1 env`: wrapper `751.55`, vecenv `980.31` env/s
  - `16 env`: wrapper `903.61`, vecenv `1459.51` env/s
  - `64 env`: wrapper `1038.82`, vecenv `1426.81` env/s
- end-to-end training:
  - `4 env`: disabled `96.66` SPS
  - `16 env`: disabled `96.53` SPS
  - `64 env`: disabled `91.62` SPS

Phase 0 gate status now has both:
- a complete current-host WSL packet at `/tmp/fc_phase0_packet_clean/phase0_packet.json`
- a hosted native-Linux source-of-truth packet published via `codex/phase0-results`

Native-Linux gate summary:
- benchmark host class: `linux_native`
- performance source of truth: `true`
- native-Linux source of truth: `true`
- bridge / vecenv / train rows: complete
- clean pure-JVM and clean batched headless artifacts: present
- per-worker sim estimate: about `404.6k` batched env steps/s, `1` worker needed for `100k`
- Phase 1 gate result: `unblocked`

Interpretation:
- the old `8.9k` direct-sim artifact was materially under-representing the current headless runtime ceiling because it came from the older Step 11 test-harness path
- the new clean standalone harness and native-Linux gate both make `100k+` look plausible from the sim-side ceiling perspective
- the RL outer stack is now even more clearly the dominant current bottleneck because train SPS still sits around `92-97` while the clean batched sim report is in the `473k` env-steps/s range on the same WSL host

## Summary Table

All rows below are single measured runs unless otherwise noted, so the reported throughput is the observed median for `n=1`. The current harnesses do not emit per-round latency distributions, so `p95` is not available in this packet.

| Layer | Benchmark | Command | Config | Env count | Worker topology | Batch / rounds | Throughput | Sample count | p95 |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | --- |
| Direct-JVM sim artifact | Existing pure-JVM throughput artifact | repo artifact only, not rerun numerically under WSL in this pass | `fight-caves-RL/docs/performance_report.md` | 1 | direct JVM | artifact only | `8891.93` ticks/s | 1 artifact | not collected |
| Current-host best-case bridge trace | `benchmark_bridge.py` | `uv run python scripts/benchmark_bridge.py --config configs/benchmark/bridge_1env_v0.yaml --rounds 1024 --output /tmp/fc_perf_audit/bridge_1_run1.json` | `bridge_1env_v0` | 1 | embedded JVM, single trace call | `1024 rounds` | `23765.25` env steps/s | 1 | not collected |
| Bridge lockstep batch | `benchmark_bridge.py` | `uv run python scripts/benchmark_bridge.py --config configs/benchmark/bridge_64env_v0.yaml --rounds 1024 --output /tmp/fc_perf_audit/bridge64_attach_target.json` | `bridge_64env_v0` | 64 | embedded JVM, lockstep batch | `1024 rounds` | `1481.11` env steps/s | 1 | not collected |
| Embedded vecenv | `benchmark_env.py` | `uv run python scripts/benchmark_env.py --config configs/train/train_baseline_v0.yaml --env-count 4 --rounds 512 --output /tmp/fc_perf_audit/env_baseline4_run1.json` | `train_baseline_v0` | 4 | embedded JVM | `512 rounds` | `906.64` env steps/s | 1 | not collected |
| Embedded vecenv | `benchmark_env.py` | `uv run python scripts/benchmark_env.py --config configs/benchmark/vecenv_256env_v0.yaml --env-count 16 --rounds 128 --output /tmp/fc_perf_audit/env_vec16_run1.json` | `vecenv_256env_v0` | 16 | embedded JVM | `128 rounds` | `1232.60` env steps/s | 1 | not collected |
| Embedded vecenv | `benchmark_env.py` | `uv run python scripts/benchmark_env.py --config configs/benchmark/vecenv_256env_v0.yaml --env-count 64 --rounds 64 --output /tmp/fc_perf_audit/env_vec64_run1.json` | `vecenv_256env_v0` | 64 | embedded JVM | `64 rounds` | `1492.10` env steps/s | 1 | not collected |
| End-to-end train | `benchmark_train.py` | `uv run python scripts/benchmark_train.py --config configs/train/train_baseline_v0.yaml --env-count 4 --total-timesteps 512 --logging-modes disabled,standard --output /tmp/fc_perf_audit/train_4_disabled_standard_run1.json` | `train_baseline_v0` | 4 | 1 parent + 1 subprocess worker | `512 timesteps` | `36.40` SPS disabled, `36.14` SPS offline | 1 each | not collected |
| End-to-end train | `benchmark_train.py` | `uv run python scripts/benchmark_train.py --config configs/benchmark/train_1024env_v0.yaml --env-count 16 --total-timesteps 1024 --logging-modes disabled,standard --output /tmp/fc_perf_audit/train_16_disabled_standard_run2.json` | `train_1024env_v0` | 16 | 1 parent + 1 subprocess worker | `1024 timesteps` | `82.84` SPS disabled, `91.66` SPS offline | 1 each | not collected |
| End-to-end train | `benchmark_train.py` | `uv run python scripts/benchmark_train.py --config configs/benchmark/train_1024env_v0.yaml --env-count 64 --total-timesteps 1024 --logging-modes disabled --output /tmp/fc_perf_audit/train_64_disabled_run1.json` | `train_1024env_v0` | 64 | 1 parent + 1 subprocess worker | `1024 timesteps` | `87.93` SPS disabled | 1 | not collected |
| End-to-end train with W&B online | direct `train.py` wall-clock probe | inline subprocess probe to `scripts/train.py` with `WANDB_MODE=online` | `train_baseline_v0` | 4 | 1 parent + 1 subprocess worker | `256 timesteps` | `11.87` wall SPS | 1 | not collected |

The original audit table above remains useful as the pre-Phase-0 packet snapshot. The `Phase 0 Gate Refresh` section is the newer current-host baseline and should be treated as the active measurement reference for optimization work.

## Layer-by-Layer Breakdown

### 1. Direct-JVM Sim Stepping

Purpose:
- establish the pure sim ceiling without Python transport or learner overhead

Evidence:
- Current repo artifact at `/home/jordan/code/fight-caves-RL/docs/performance_report.md`
- Throughput benchmark in that artifact: `8891.93` ticks/s
- Soak benchmark in that artifact: `9186.05` ticks/s

Important limitation:
- That artifact is current repo evidence, but it was generated in a Windows-native context, not in this WSL audit pass.
- In this audit pass, the direct-JVM JUnit benchmark was rerun only as a pass/fail test; it does not emit throughput numerically by itself.

Interpretation:
- The pure sim is not already anywhere near `100000+` SPS even before RL transport and training overhead.
- The direct-JVM artifact says the sim itself still needs major performance work eventually.
- However, the RL stack is currently far slower than the pure sim artifact, so the outer stack is the first dominant gap.

### 2. Python Bridge Throughput

Purpose:
- isolate bridge-layer throughput before PufferLib training overhead

Measured commands:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python scripts/benchmark_bridge.py \
  --config configs/benchmark/bridge_1env_v0.yaml \
  --rounds 1024 \
  --output /tmp/fc_perf_audit/bridge_1_run1.json

uv run python scripts/benchmark_bridge.py \
  --config configs/benchmark/bridge_64env_v0.yaml \
  --rounds 1024 \
  --output /tmp/fc_perf_audit/bridge64_attach_target.json
```

Results:
- `1 env`
  - reference step path: `943.85` env steps/s
  - batch trace path: `23765.25` env steps/s
- `64 env`
  - reference lockstep path: `1437.68` env steps/s
  - batch lockstep path: `1481.11` env steps/s

Interpretation:
- Single-slot batch trace is fast because it amortizes the boundary crossing and only pythonizes the final observation once.
- Multi-slot lockstep throughput collapses hard. Going from `1 env batch trace` to `64 env lockstep batch` does not scale; it drops to roughly `1.5k` total env steps/s.
- The batch bridge is not currently delivering meaningful multi-env leverage. At `64 envs`, batched lockstep is only marginally better than the reference path.

### 3. Wrapper and VecEnv Overhead

Purpose:
- measure how much additional cost is added by the RL wrapper and vecenv shell on top of the embedded bridge

Measured commands:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python scripts/benchmark_env.py \
  --config configs/train/train_baseline_v0.yaml \
  --env-count 4 \
  --rounds 512 \
  --output /tmp/fc_perf_audit/env_baseline4_run1.json

uv run python scripts/benchmark_env.py \
  --config configs/benchmark/vecenv_256env_v0.yaml \
  --env-count 16 \
  --rounds 128 \
  --output /tmp/fc_perf_audit/env_vec16_run1.json

uv run python scripts/benchmark_env.py \
  --config configs/benchmark/vecenv_256env_v0.yaml \
  --env-count 64 \
  --rounds 64 \
  --output /tmp/fc_perf_audit/env_vec64_run1.json
```

Results:
- `4 env`
  - wrapper sequential: `498.69` env steps/s
  - vecenv lockstep: `906.64` env steps/s
- `16 env`
  - wrapper sequential: `628.68` env steps/s
  - vecenv lockstep: `1232.60` env steps/s
- `64 env`
  - wrapper sequential: `854.07` env steps/s
  - vecenv lockstep: `1492.10` env steps/s

Interpretation:
- Vecenv does help versus wrapper-sequential execution, but only by roughly `1.7x` to `2.0x`.
- Absolute throughput still stays in the `0.9k` to `1.5k` range, which is far too low for the project target.
- The embedded vecenv path at `64 envs` is almost identical to the `64 env` bridge lockstep result, so the vecenv shell itself is not the dominant bottleneck. The bridge-plus-observation path is.

### 4. End-to-End Training SPS

Purpose:
- capture real training throughput through the shipped subprocess worker path

Measured commands:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python scripts/benchmark_train.py \
  --config configs/train/train_baseline_v0.yaml \
  --env-count 4 \
  --total-timesteps 512 \
  --logging-modes disabled,standard \
  --output /tmp/fc_perf_audit/train_4_disabled_standard_run1.json

uv run python scripts/benchmark_train.py \
  --config configs/benchmark/train_1024env_v0.yaml \
  --env-count 16 \
  --total-timesteps 1024 \
  --logging-modes disabled,standard \
  --output /tmp/fc_perf_audit/train_16_disabled_standard_run2.json

uv run python scripts/benchmark_train.py \
  --config configs/benchmark/train_1024env_v0.yaml \
  --env-count 64 \
  --total-timesteps 1024 \
  --logging-modes disabled \
  --output /tmp/fc_perf_audit/train_64_disabled_run1.json
```

Results:
- `4 env`
  - disabled: `36.40` SPS
  - offline: `36.14` SPS
- `16 env`
  - disabled: `82.84` SPS
  - offline: `91.66` SPS
- `64 env`
  - disabled: `87.93` SPS

Interpretation:
- Training scales from `4 env` to `16 env`, but only to about `2.3x`, not `4x`.
- Scaling from `16 env` to `64 env` is effectively flat. `82.84 -> 87.93` is only about `1.06x` despite `4x` more envs.
- The shipped subprocess architecture is saturating well before `64 envs`.
- The current architecture is not remotely on track for `100000+` SPS without major redesign.

### 5. W&B, Dashboard, and Artifact Overhead

Purpose:
- determine whether observability is the main reason for low SPS

Measured commands:

Offline / disabled:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python scripts/benchmark_train.py \
  --config configs/train/train_baseline_v0.yaml \
  --env-count 4 \
  --total-timesteps 512 \
  --logging-modes disabled,standard \
  --output /tmp/fc_perf_audit/train_4_disabled_standard_run1.json
```

Online wall-clock probe:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
WANDB_MODE=online WANDB_ENTITY=jordanbaileypmp-georgia-institute-of-technology WANDB_PROJECT=fight-caves-RL \
  uv run python scripts/train.py \
    --config configs/train/train_baseline_v0.yaml \
    --total-timesteps 256 \
    --output /tmp/fc_perf_audit/train_online_probe.json
```

Dashboard / checkpoint probe:
- measured as short direct runs with `WANDB_MODE=disabled`
- these are useful directional signals, but they are noisier than the benchmark harness

Results:
- Offline W&B is effectively noise-level in the benchmark harness:
  - `4 env`: `36.40` disabled vs `36.14` offline
  - `16 env`: `82.84` disabled vs `91.66` offline
- Online W&B is materially harmful in this small baseline:
  - `4 env` online wall-clock: `11.87` SPS
  - compared with the same baseline class at about `36` wall SPS disabled/offline
- Dashboard / checkpoint direct probe, using last logged trainer SPS rather than wall-clock:
  - dashboard off, checkpoint interval `1`: `89.26`
  - dashboard off, checkpoint interval `999999`: `87.17`
  - dashboard on, checkpoint interval `1`: `74.73`
  - dashboard on, checkpoint interval `999999`: `39.37`

Interpretation:
- Offline W&B is definitely not the primary bottleneck.
- Online W&B is a real penalty, around `3x` in the small 4-env wall-clock probe.
- Dashboard output is directionally harmful, but the direct dashboard probe is noisy and should be treated as supporting evidence, not the primary benchmark.
- Observability can hurt, especially in `online` mode, but turning it off does not move the stack anywhere near the target.

### 6. Replay / Artifact / Checkpoint Overhead

Purpose:
- determine whether end-of-run artifact work is materially distorting small-run SPS

Evidence:
- `benchmark_train.py` always produced `3` artifacts in these runs: checkpoint, checkpoint metadata, run manifest.
- `cProfile` on the 4-env train path shows `pufferlib.pufferl.save_checkpoint` with about `1.809s` cumulative time in a short startup-heavy run.
- The short dashboard/checkpoint probe did not show a consistent wall-clock checkpoint benefit large enough to classify checkpoint cadence as the primary bottleneck.

Interpretation:
- Checkpointing and artifact writing are real costs.
- They matter much more in tiny smoke runs than they would in long steady-state production runs.
- They are not the primary reason the system is at `30-90` SPS instead of `100000+`.

## Where Performance Collapses

The collapse happens in this order:

1. Pure or near-pure stepping is in the `8.9k` to `23.8k` range depending on harness.
2. Multi-slot bridge lockstep collapses to roughly `1.5k` total env steps/s.
3. Embedded vecenv remains in the same `1.2k` to `1.5k` total env steps/s range.
4. Shipped subprocess training collapses again to `36` to `88` SPS.
5. Online W&B can push the 4-env baseline down to about `11.9` wall SPS.

## Fact-Based Diagnosis

Facts:
- The current bridge and vecenv path saturates around `1.5k` env steps/s total by `64 envs`.
- The current shipped train path saturates around `88` SPS by `64 envs`.
- Offline W&B is not the main issue.
- Online W&B is a real penalty, but even disabled training is far too slow.

Hypothesis supported by profiling:
- The biggest current order-of-magnitude bottleneck is the Python-side observation conversion and transport design, not the learner math.

## Immediate Audit Conclusion

The current architecture cannot plausibly reach `100000+` SPS without a major transport and observation-path redesign, and it will still need substantial sim-side speedups after that.
