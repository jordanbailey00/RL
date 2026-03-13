# Phase 1 Decision Gate

## Documentation Status

- Status: archive candidate under `pivot_documentation_triage.md`.
- Current authority: `pivot_plan.md` and `pivot_implementation_plan.md`.
- Retention reason: kept temporarily for historical optimization-gate context; do not treat it as an active pivot gate.

Date: 2026-03-09

This document defines the exact benchmark and profiling packet that must be rerun after the Phase 1 flat-path implementation lands.

This is the source-of-truth deliverable for optimization `WC-P1-05`.

Current execution status:

- local WSL Phase 1 packet has been executed successfully
- the local packet shows that raw object conversion is no longer the dominant Python hot-path cost
- the immutable pre-Phase-1 native-Linux baseline has been published successfully
- the hosted native-Linux Phase 1 rerun has now executed successfully end to end against that immutable baseline
- the final continue-versus-pivot decision is now recorded: `continue to Phase 2`

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
- therefore the local packet remains a preview rather than the final decision artifact

## Immutable Baseline Source

The clean comparison now uses this immutable native-Linux pre-Phase-1 baseline path:

- [phase0-results immutable pre-phase1 baseline](https://github.com/jordanbailey00/fight-caves-RL/tree/codex/phase0-results/phase0-native-linux/immutable/pre-phase1/rl-3e557474f3c6b4e44842da82a971c8f97d521b10__sim-216c1fd2ac31f450f8c599f9ec9454330a4e6b3a)

Published baseline facts:

- `benchmark_host_class = linux_native`
- `rl_commit_sha = 3e557474f3c6b4e44842da82a971c8f97d521b10`
- `sim_commit_sha = 216c1fd2ac31f450f8c599f9ec9454330a4e6b3a`
- bridge `64 env` baseline: `1377.89` env/s
- vecenv `64 env` baseline: `1368.42` env/s

## Native-Linux Gate Status

The final hosted native-Linux Phase 1 packet at:

- [phase1-results/latest/gate_summary.json](https://github.com/jordanbailey00/fight-caves-RL/blob/codex/phase1-results/phase1-native-linux/latest/gate_summary.json)

reports:

- `benchmark_host_class = linux_native`
- `bridge_64_env_steps_per_second = 9148.80`
- `vecenv_64_env_steps_per_second = 10961.11`
- `bridge_64_improvement_ratio = 6.6397`
- `vecenv_64_improvement_ratio = 8.0101`
- `raw_object_conversion_still_dominant = false`
- `phase2_unblocked = true`

Those rows satisfy the approved Phase 1 continue thresholds.

## Final Continue-Versus-Pivot Decision

Decision:

- `continue to Phase 2`

Why:

- bridge `64 env` improved by more than `5x`
- vecenv `64 env` improved by more than `4x`
- both absolute rows land inside the approved target ranges
- the hosted native-Linux steady-state profile confirms that recursive raw object conversion is no longer the dominant cost center

Resolved blocker:

- the earlier ratio failure was caused by a contaminated `phase0-results/latest` baseline
- that blocker is now closed because the gate reran against the immutable pre-Phase-1 baseline path above

## Remaining Work Before Phase 2

None from Phase 1.

Phase 2 is now unblocked on the approved native-Linux source-of-truth host path.

## Output Of WC-P1-05

`WC-P1-05` is now complete because:

- the immutable pre-Phase-1 baseline exists
- the hosted native-Linux rerun completed against that baseline
- the final decision is explicit and evidence-backed
- later transport work no longer depends on vague or contaminated Phase 1 results
