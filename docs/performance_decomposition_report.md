# Performance Decomposition Report

Date: 2026-03-10

Repos and SHAs:
- RL: `ee4b205277f3638e2983fc6cf3cb8bfb0dc4a6b4`
- fight-caves-RL: `a433c971a7e24f5bff10fb4e22f740d68e70af73`
- RSPS: `8ab69d72591030936ae2b4b06372c15fc1653517`

This report decomposes the current stack into separate layers and records the measurements collected in this audit pass. Facts and hypotheses are separated explicitly.

## Phase 1 Implementation Preview

The active Phase 1 implementation batch is no longer design-only.

Current local WSL Phase 1 packet at `/tmp/fc_phase1_packet_local/phase1_packet.json` shows:

- bridge `64 env`: `11936.49` env/s
- vecenv `64 env`: `7336.43` env/s
- steady-state raw object conversion still dominant: `false`

Compared with the active Phase 0 WSL packet:

- bridge `64 env`: `1606.92 -> 11936.49` env/s, about `7.43x`
- vecenv `64 env`: `1426.81 -> 7336.43` env/s, about `5.14x`

Interpretation:

- the flat-path implementation moved the correct boundary locally
- the bridge and vecenv rows now clear the Phase 1 planning thresholds on WSL
- the remaining decision-gate blocker is source-of-truth host class, not local measurement quality

## Native-Linux Phase 1 Gate Execution

The hosted native-Linux Phase 1 packet now runs end to end against an immutable pre-Phase-1 native baseline.

Published native-Linux packet summary:

- bridge `64 env`: `9148.80` env/s
- vecenv `64 env`: `10961.11` env/s
- bridge improvement ratio: `6.6397`
- vecenv improvement ratio: `8.0101`
- `raw_object_conversion_still_dominant = false`
- `phase2_unblocked = true`

Interpretation:

- the Phase 1 implementation behaves correctly on the source-of-truth host class
- the dominant Python bottleneck remains removed in the native-Linux steady-state profile
- the clean immutable-baseline comparison clears the approved Phase 1 thresholds
- Phase 2 is now unblocked

## Native-Linux Phase 2 Pre-Swap Gate

The hosted native-Linux Phase 2 pre-swap gate completed successfully as a workflow and failed only at the decision step, which means the result is diagnostic rather than infrastructural.

Benchmark-contract note:

- the published Phase 2 disabled-train rows below predate `WC-P2-09`
- they remain valid as evidence that the first transport-promotion attempt did not survive end-to-end training
- they are not the frozen production benchmark rows going forward, because `WC-P2-09` now defines production train throughput as `production_fast_path_v1`, excluding `final_evaluate`

Published native-Linux Phase 2 gate summary:

- transport `64 env`: pipe `10868.61`, `shared_memory_v1` `8340.91` env/s, `0.7674x`
- disabled train `16 env`: pipe `74.05`, `shared_memory_v1` `75.01` SPS
- disabled train `64 env`: pipe `75.03`, `shared_memory_v1` `74.85` SPS, `0.9977x`
- shared-train scaling ratio `64 vs 16`: `0.9979x`
- blockers:
  - `transport_signal_too_weak`
  - `train_signal_too_weak`
  - `shared_train_scaling_too_weak`
- `wc_p2_03_unblocked = false`

Interpretation:

- the latest source-of-truth rerun does not show a stable transport gain
- end-to-end training remains effectively flat even after the info-payload trim
- the current production subprocess path must not be replaced yet
- the next Phase 2 move is a trainer-bound escalation path, not `WC-P2-03`

## Learner Ceiling Diagnostic

The new repo-owned fake-env ceiling benchmark removes the live sim, bridge, and transport path while keeping the current policy and `PuffeRL` trainer loop intact.

Local WSL ceiling rows from `scripts/benchmark_train_ceiling.py`:

- `4 env`: `154.45` env-steps/s
- `16 env`: `156.20` env-steps/s
- `64 env`: `144.43` env-steps/s

`64 env` stage breakdown:

- rollout/evaluate: `15.83s`
- PPO train/update: `24.87s`
- final evaluate: `16.02s`

Interpretation:

- the current trainer loop is already mostly env-count invariant on a zero-cost vecenv
- the PPO update path plus rollout/evaluate path now dominate enough wall clock to flatten transport-only wins
- this is the clearest current explanation for why the transport improvement disappears in end-to-end training
- this benchmark family is now explicitly diagnostic-only and must not be confused with the corrected production train benchmark

Native-Linux confirmation:

- the repo-owned hosted learner-ceiling workflow completed successfully at [fight-caves-RL/actions/runs/22886069441](https://github.com/jordanbailey00/fight-caves-RL/actions/runs/22886069441)
- published native-Linux learner-ceiling rows:
  - `4 env`: `94.97` env-steps/s
  - `16 env`: `74.67` env-steps/s
  - `64 env`: `68.43` env-steps/s
  - `64 vs 16` ratio: `0.9165x`
  - `64 vs 4` ratio: `0.7206x`
- `64 env` native-Linux stage split:
  - evaluate: `28.64s`
  - PPO train/update: `62.47s`
  - final evaluate: `28.60s`

Interpretation:

- the source-of-truth host class confirms the same diagnosis as the local WSL fake-env benchmark
- trainer/rollout overhead is now clearly the active Phase 2 blocker
- the next implementation batch should target that trainer-bound ceiling directly

## WC-P2-10 Local Trainer Instrumentation

Local instrumented production-fast-path rows:

- `16 env`:
  - production throughput: `58.73` SPS
  - wall-clock throughput: `40.88` SPS
  - runner stage seconds:
    - evaluate: `11.43s`
    - train: `5.77s`
    - final evaluate: `7.61s`
  - top trainer buckets:
    - `eval_policy_forward = 15.04s`
    - `eval_env_recv = 3.78s`
    - `train_backward = 3.74s`
    - `train_policy_forward = 1.79s`
- `64 env`:
  - production throughput: `60.23` SPS
  - wall-clock throughput: `45.46` SPS
  - runner stage seconds:
    - evaluate: `11.19s`
    - train: `5.51s`
    - final evaluate: `5.53s`
  - top trainer buckets:
    - `eval_policy_forward = 10.26s`
    - `eval_env_recv = 6.41s`
    - `train_backward = 3.40s`
    - `train_policy_forward = 2.01s`

Local instrumented learner-ceiling rows:

- `16 env` diagnostic ceiling:
  - `144.22` env-steps/s
  - top buckets:
    - `eval_policy_forward = 30.40s`
    - `train_backward = 15.72s`
    - `train_policy_forward = 10.34s`
- `64 env` diagnostic ceiling:
  - `146.87` env-steps/s
  - top buckets:
    - `eval_policy_forward = 31.71s`
    - `train_backward = 14.80s`
    - `train_policy_forward = 9.11s`

Interpretation:

- after Phase 1 and the benchmark-validity correction, the dominant named buckets are now inside the current trainer path itself
- `eval_info_stats`, `eval_tensor_copy`, `eval_rollout_write`, and optimizer-step time are all materially smaller than policy-forward and backward costs on the current benchmark path
- the next useful spike should target the current rollout/evaluate/update structure, not more small bookkeeping suppressions

## WC-P2-14 Local Prototype Gate

The subprocess worker startup blocker on this host was runtime resolution, not trainer logic.
The workspace already had a Linux Temurin toolchain at `/home/jordan/code/.workspace-tools/jdk-21`, so the fix was to resolve that toolchain explicitly for both JPype worker startup and benchmark Java-version collection.

Local WSL corrected production-fast-path prototype rows:

- `16 env`:
  - production throughput: `95.74` SPS
  - wall-clock throughput: `95.74` SPS
  - runner stage seconds:
    - evaluate: `18.79s`
    - train: `23.83s`
    - final evaluate: `0.0s`
  - top trainer buckets:
    - `rollout_policy_forward = 15.23s`
    - `update_backward = 14.49s`
    - `update_policy_forward = 9.22s`
    - `rollout_env_recv = 3.43s`
- `64 env`:
  - production throughput: `93.06` SPS
  - wall-clock throughput: `93.06` SPS
  - runner stage seconds:
    - evaluate: `20.09s`
    - train: `23.75s`
    - final evaluate: `0.0s`
  - top trainer buckets:
    - `rollout_policy_forward = 15.42s`
    - `update_backward = 14.26s`
    - `update_policy_forward = 9.38s`
    - `rollout_env_recv = 4.64s`
- production scaling:
  - `64 vs 16 = 0.9720x`

Local WSL learner-ceiling companion rows:

- `16 env` diagnostic ceiling:
  - `145.70` env-steps/s
  - top buckets:
    - `eval_policy_forward = 31.37s`
    - `train_backward = 14.87s`
    - `train_policy_forward = 9.69s`
- `64 env` diagnostic ceiling:
  - `144.79` env-steps/s
  - top buckets:
    - `eval_policy_forward = 32.98s`
    - `train_backward = 13.59s`
    - `train_policy_forward = 9.87s`
- learner-ceiling scaling:
  - `64 vs 16 = 0.9937x`

Interpretation:

- the first project-owned synchronous prototype is real and materially faster on the local production benchmark than the earlier local instrumented shipped-path rows (`58.73 / 60.23`)
- the gain is not coming from improved scaling; `16 -> 64` remains effectively flat
- the learner-ceiling companion stays in the same `~145` env-steps/s band, which means the shipped synchronous trainer diagnosis is unchanged and transport is still not the next active gate
- the correct `WC-P2-14` decision is to continue trainer redesign inside the prototype path, targeting deeper rollout/update ownership before transport promotion or topology work is reconsidered

## PR Batch G Follow-On Local Preview

The next local trainer-core slice targeted the remaining padded multi-discrete action sampling/logprob path inside the project-owned prototype.
The old helper padded every action head up to the largest category count before sampling/logprob, which was especially expensive because two tile heads are `16384`-way categorical outputs.

Local WSL corrected production-fast-path rows after that slice:

- `16 env`:
  - production throughput: `417.36` SPS
  - runner stage seconds:
    - evaluate: `7.44s`
    - train: `1.96s`
  - top trainer buckets:
    - `rollout_policy_forward = 4.13s`
    - `update_backward = 1.27s`
    - `update_policy_forward = 0.59s`
- `64 env`:
  - production throughput: `398.80` SPS
  - runner stage seconds:
    - evaluate: `7.96s`
    - train: `2.15s`
  - top trainer buckets:
    - `rollout_env_recv = 4.12s`
    - `rollout_policy_forward = 3.81s`
    - `update_backward = 1.35s`
    - `update_policy_forward = 0.71s`
- production scaling:
  - `64 vs 16 = 0.9555x`

Interpretation:

- the local prototype no longer looks trainer-forward-bound in the same way as the earlier `95.74 / 93.06` gate
- the production row is now above the old local `250` SPS escalation bar, which is enough to justify a source-of-truth rerun before more local redesign work
- scaling is still flat enough that transport and topology should remain deferred until the native-Linux rerun confirms whether that flatness persists on the source-of-truth host

## WC-P2-07 Local Trainer-Bound Reduction Preview

The current local `WC-P2-07` batch has two benchmark-only slices:

- remove smoke-run artifact/checkpoint/manifest overhead from the disabled-train benchmark path
- suppress profile, utilization, and metric-only logging work inside the benchmark-only `PuffeRL` path

Local WSL live disabled-train rows:

- after the core benchmark runner slice:
  - `16 env`: `56.65` SPS
  - `64 env`: `57.10` SPS
- after the deeper disabled-logging trainer slice:
  - `16 env`: `58.12` SPS
  - `64 env`: `57.71` SPS

Local WSL learner-ceiling rows:

- after the first slice:
  - `4 env`: `119.90`
  - `16 env`: `146.94`
  - `64 env`: `144.77`
- after the deeper slice:
  - `4 env`: `123.60`
  - `16 env`: `149.19`
  - `64 env`: `142.83`

Interpretation:

- the first slice removes real benchmark/control-plane overhead from the short live training path
- the second slice barely changes the learner ceiling
- the current trainer-bound limit is therefore deeper than the obvious benchmark-only logging/profile/control-plane costs
- Phase 2 should keep `WC-P2-03` blocked until either a native-Linux rerun proves otherwise or a deeper trainer-path change moves the learner ceiling materially

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
| Direct-JVM sim artifact | Existing pure-JVM throughput artifact | repo artifact only, not rerun numerically under WSL in this pass | `fight-caves-RL/history/performance_report_step11.md` | 1 | direct JVM | artifact only | `8891.93` ticks/s | 1 artifact | not collected |
| Current-host best-case bridge trace | `benchmark_bridge.py` | `uv run python scripts/benchmark_bridge.py --config configs/benchmark/bridge_1env_v0.yaml --rounds 1024 --output /tmp/fc_perf_audit/bridge_1_run1.json` | `bridge_1env_v0` | 1 | embedded JVM, single trace call | `1024 rounds` | `23765.25` env steps/s | 1 | not collected |
| Bridge lockstep batch | `benchmark_bridge.py` | `uv run python scripts/benchmark_bridge.py --config configs/benchmark/bridge_64env_v0.yaml --rounds 1024 --output /tmp/fc_perf_audit/bridge64_attach_target.json` | `bridge_64env_v0` | 64 | embedded JVM, lockstep batch | `1024 rounds` | `1481.11` env steps/s | 1 | not collected |
| Embedded vecenv | `benchmark_env.py` | `uv run python scripts/benchmark_env.py --config configs/train/train_baseline_v0.yaml --env-count 4 --rounds 512 --output /tmp/fc_perf_audit/env_baseline4_run1.json` | `train_baseline_v0` | 4 | embedded JVM | `512 rounds` | `906.64` env steps/s | 1 | not collected |
| Embedded vecenv | `benchmark_env.py` | `uv run python scripts/benchmark_env.py --config configs/benchmark/vecenv_256env_v0.yaml --env-count 16 --rounds 128 --output /tmp/fc_perf_audit/env_vec16_run1.json` | `vecenv_256env_v0` | 16 | embedded JVM | `128 rounds` | `1232.60` env steps/s | 1 | not collected |
| Embedded vecenv | `benchmark_env.py` | `uv run python scripts/benchmark_env.py --config configs/benchmark/vecenv_256env_v0.yaml --env-count 64 --rounds 64 --output /tmp/fc_perf_audit/env_vec64_run1.json` | `vecenv_256env_v0` | 64 | embedded JVM | `64 rounds` | `1492.10` env steps/s | 1 | not collected |
| End-to-end train | `benchmark_train.py` | `uv run python scripts/benchmark_train.py --config configs/train/train_baseline_v0.yaml --env-count 4 --total-timesteps 512 --logging-modes disabled,standard --output /tmp/fc_perf_audit/train_4_disabled_standard_run1.json` | `train_baseline_v0` | 4 | 1 parent + 1 subprocess worker | `512 timesteps` | `36.40` SPS disabled, `36.14` SPS offline | 1 each | not collected |
| End-to-end train | `benchmark_train.py` | `uv run python scripts/benchmark_train.py --config configs/benchmark/train_1024env_v0.yaml --env-count 16 --total-timesteps 1024 --logging-modes disabled,standard --output /tmp/fc_perf_audit/train_16_disabled_standard_run2.json` | `train_1024env_v0` | 16 | 1 parent + 1 subprocess worker | `1024 timesteps` | `82.84` SPS disabled, `91.66` SPS offline | 1 each | not collected |
| End-to-end train | `benchmark_train.py` | `uv run python scripts/benchmark_train.py --config configs/benchmark/train_1024env_v0.yaml --env-count 64 --total-timesteps 1024 --logging-modes disabled --output /tmp/fc_perf_audit/train_64_disabled_run1.json` | `train_1024env_v0` | 64 | 1 parent + 1 subprocess worker | `1024 timesteps` | `87.93` SPS disabled | 1 | not collected |
| End-to-end train with W&B online | direct `train.py` wall-clock probe | inline subprocess probe to `scripts/train.py` with `WANDB_MODE=online` | `train_baseline_v0` | 4 | 1 parent + 1 subprocess worker | `256 timesteps` | `11.87` wall SPS | 1 | not collected |

The original audit table above remains useful as the pre-Phase-0 packet snapshot. The `Phase 0 Gate Refresh` section is the newer current-host baseline and should be treated as the pre-Phase-1 reference for optimization work.

## Layer-by-Layer Breakdown

### 1. Direct-JVM Sim Stepping

Purpose:
- establish the pure sim ceiling without Python transport or learner overhead

Evidence:
- Current historical repo artifact at `/home/jordan/code/fight-caves-RL/history/performance_report_step11.md`
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
