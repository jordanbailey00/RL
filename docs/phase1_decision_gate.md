# Phase 1 Decision Gate

Date: 2026-03-09

This document defines the exact benchmark and profiling packet that must be rerun after the Phase 1 flat-path implementation lands.

This is the source-of-truth deliverable for optimization `WC-P1-05`.

Current execution status:

- local WSL Phase 1 packet has been executed successfully
- the local packet shows that raw object conversion is no longer the dominant Python hot-path cost
- the native-Linux source-of-truth rerun has now executed successfully end to end
- the remaining blocker is not packet health; it is baseline contamination in the published `phase0-results/latest` comparison source

## Purpose

Phase 1 is successful only if it moves the correct boundary:

- bridge throughput
- vecenv throughput
- Python hot-path attribution

Phase 1 is not judged primarily on end-to-end training SPS yet.
That is a Phase 2 concern after the transport redesign.

## Host And Comparability Rules

The decision gate must use:

- native Linux as the performance source of truth
- the same benchmark host class used by the approved Phase 0 gate
- the same benchmark profile family
- the same logging restrictions used for performance rows
- the same env counts as the Phase 0 comparison packet

Runs are only comparable if they match on:

- host class
- sim artifact build
- RL/fight-caves-RL commit SHAs
- env counts
- logging mode
- dashboard mode
- benchmark profile metadata

## Required Packet After Phase 1 Implementation

Standard automation paths:

- local/manual refresh:
  - `uv run python scripts/refresh_phase1_packet.py --output-dir <phase1-output>`
- hosted native-Linux refresh:
  - `fight-caves-RL/.github/workflows/phase1_native_linux_packet.yml`

### 1. Bridge Packet

Required commands:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python scripts/benchmark_bridge.py \
  --config configs/benchmark/bridge_1env_v0.yaml \
  --env-count 1 \
  --output <phase1-output>/bridge_1env.json

uv run python scripts/benchmark_bridge.py \
  --config configs/benchmark/bridge_64env_v0.yaml \
  --env-count 16 \
  --output <phase1-output>/bridge_16env.json

uv run python scripts/benchmark_bridge.py \
  --config configs/benchmark/bridge_64env_v0.yaml \
  --env-count 64 \
  --output <phase1-output>/bridge_64env.json
```

Required comparison:

- compare against the approved Phase 0 native-Linux packet for the same host class and env counts

### 2. VecEnv Packet

Required commands:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python scripts/benchmark_env.py \
  --config configs/benchmark/vecenv_256env_v0.yaml \
  --env-count 16 \
  --output <phase1-output>/vecenv_16env.json

uv run python scripts/benchmark_env.py \
  --config configs/benchmark/vecenv_256env_v0.yaml \
  --env-count 64 \
  --output <phase1-output>/vecenv_64env.json
```

Optional sanity row:

- `env-count 1` may be rerun for debugging, but it is not the decision row

### 3. Python Profiler Packet

Required steady-state profile:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python - <<'PY'
import cProfile, pstats
from pathlib import Path
import numpy as np
from fight_caves_rl.puffer.factory import load_smoke_train_config, make_vecenv
config = load_smoke_train_config(Path('configs/benchmark/vecenv_256env_v0.yaml'))
config['num_envs'] = 16
vecenv = make_vecenv(config, backend='embedded')
seed = int(config['train']['seed'])
actions = np.zeros((16, len(vecenv.single_action_space.nvec)), dtype=np.int32)
vecenv.async_reset(seed)
vecenv.recv()
for _ in range(8):
    vecenv.send(actions)
    vecenv.recv()
prof = cProfile.Profile()
prof.enable()
for _ in range(64):
    vecenv.send(actions)
    vecenv.recv()
prof.disable()
vecenv.close()
path = '<phase1-output>/python_vec16_steady.prof'
prof.dump_stats(path)
stats = pstats.Stats(path)
stats.sort_stats('cumulative').print_stats(40)
PY
```

Required interpretation:

- dominant time must move materially away from recursive raw object conversion
- if `pythonize_observation`, `_pythonize`, or equivalent raw reconstruction remains dominant, Phase 1 did not remove the intended bottleneck

## Optional Supporting Rows

These are useful but not the primary decision rows:

- `1 env` bridge trace sanity
- `4 env` training sanity row with logging disabled
- raw-vs-flat source-side comparison microbench if a dedicated harness exists by then

## Required Non-Benchmark Gates

Before Phase 1 can pass:

- full raw-vs-flat equivalence gate from [raw_flat_equivalence_plan.md](/home/jordan/code/fight-caves-RL/docs/raw_flat_equivalence_plan.md)
- `fight_caves_rl/tests/integration`
- `fight_caves_rl/tests/determinism`
- `fight_caves_rl/tests/parity`
- replay-eval on a known checkpoint

The benchmark packet is necessary but not sufficient.

## Numeric Continue Thresholds

These are planning thresholds, not guaranteed outcomes.

### Continue to Phase 2 only if:

- `64 env` bridge throughput improves by at least `5x` versus the current baseline
- `64 env` vecenv throughput improves by at least `4x` versus the current baseline
- target range is credibly approaching:
  - bridge `64 env`: about `8k-15k` env/s
  - vecenv `64 env`: about `6k-12k` env/s
- steady-state Python profiles no longer show raw object conversion as the dominant cost center

### Reconsider Phase 1 design immediately if:

- `64 env` bridge remains below about `5k`
- `64 env` vecenv remains below about `4k`
- or the profile still attributes most time to raw object reconstruction / JPype wrapper conversion

## Expected Output Artifacts

After the Phase 1 implementation rerun, update:

- [performance_decomposition_report.md](/home/jordan/code/RL/docs/performance_decomposition_report.md)
- [python_profiler_report.md](/home/jordan/code/RL/docs/python_profiler_report.md)
- [benchmark_matrix.md](/home/jordan/code/RL/docs/benchmark_matrix.md)

The Phase 1 review packet should include:

- the new benchmark rows
- the updated profiler interpretation
- an explicit continue-or-pivot statement

## Local Preview Status

The current local WSL preview packet at `/tmp/fc_phase1_packet_local/phase1_packet.json` shows:

- `bridge_64_env_steps_per_second = 11936.49`
- `vecenv_64_env_steps_per_second = 7336.43`
- `raw_object_conversion_still_dominant = false`

Interpretation:

- the local preview already meets the numeric planning thresholds for bridge and vecenv throughput
- however, the local packet is not the source-of-truth gate because it is WSL and was run without the published native-Linux Phase 0 baseline directory
- therefore the Phase 1 decision remains pending until the hosted native-Linux packet is reviewed

## Native-Linux Gate Status

The current hosted native-Linux Phase 1 packet at:

- [phase1-results/latest/gate_summary.json](https://github.com/jordanbailey00/fight-caves-RL/blob/codex/phase1-results/phase1-native-linux/latest/gate_summary.json)

reports:

- `benchmark_host_class = linux_native`
- `bridge_64_env_steps_per_second = 10076.36`
- `vecenv_64_env_steps_per_second = 12305.08`
- `raw_object_conversion_still_dominant = false`

Those absolute rows are directionally strong and consistent with the intended Phase 1 outcome.

However, the current gate also reports:

- `bridge_64_improvement_ratio = 0.7487`
- `vecenv_64_improvement_ratio = 1.1472`

and blocks Phase 2 with:

- `bridge_threshold_not_met`
- `vecenv_threshold_not_met`

## Current Blocker: Baseline Contamination

The current ratio failure is not a valid pre-vs-post Phase 1 comparison.

Evidence:

- the published `phase0-results/latest` baseline currently points to post-Phase-1 commits:
  - `rl_commit_sha = 290a99a...`
  - `sim_commit_sha = 57bd2b5...`
- those results were republished during Phase 1 infrastructure hardening so the hosted workflow could fetch the full baseline packet files
- that means the current `phase0-results/latest` packet is not a true pre-Phase-1 baseline and should not be used for the final improvement-ratio decision

## Required Remaining Work Before Phase 2

Before Phase 2 can be approved:

1. publish an immutable native-Linux pre-Phase-1 baseline packet using the last pre-Phase-1 RL and sim commits
2. rerun the hosted Phase 1 gate against that immutable baseline
3. record the final continue-vs-pivot decision from that clean comparison

## Output Of WC-P1-05

`WC-P1-05` is complete when the required post-implementation gate is frozen clearly enough that:

- the future Phase 1 implementation batch knows exactly which rows to rerun
- the continue vs pivot rule is unambiguous
- later transport work cannot start on vague or incomparable Phase 1 results
