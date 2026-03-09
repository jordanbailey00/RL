# Python Profiler Report

Date: 2026-03-09

Repo and SHA:
- RL: `cda7ab4104799be40ffe39f77e5a86c2e6f0eea5`

## Purpose

Separate Python-side bridge, wrapper, vecenv, and learner costs.

## Profiling Runs

### 1. Embedded VecEnv Steady-State Profile

Command:

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
path = '/tmp/fc_perf_audit/python_vec16_steady.prof'
prof.dump_stats(path)
stats = pstats.Stats(path)
stats.sort_stats('cumulative').print_stats(40)
PY
```

Measured scope:
- `64` lockstep rounds
- `16` envs
- `1024` env steps total

Top cumulative functions:

| Function | Cum time (s) | Notes |
| --- | ---: | --- |
| `HeadlessBatchVecEnv.send` | `0.984` | vecenv shell |
| `HeadlessBatchClient.step_batch` | `0.917` | batch bridge core |
| `HeadlessBatchClient._collect_step_results` | `0.809` | per-slot observe / reward / info |
| `pythonize_observation` | `0.725` | Python dict conversion of JVM observation |
| `_pythonize` | `0.722` | recursive JPype collection walk |
| `observe_jvm` | `0.051` | raw JVM observe call |
| `pythonize_action_result` | `0.041` | Python dict conversion of action result |
| `tick` | `0.038` | raw JVM tick call |
| `build_step_buffers` | `0.038` | batch numpy packing |
| `encode_observation_for_policy` | `0.028` | raw dict -> flat float32 tensor |
| `decode_action_from_policy` | `0.017` | action decode |
| `apply_action_jvm` | `0.019` | raw JVM action call |
| `visible_targets_from_observation` | `0.011` | raw-observation postprocessing |

Strongest fact from this profile:
- `pythonize_observation + _pythonize` accounts for about `0.72s` out of `0.98s` total steady-state profiled time.
- The raw JVM calls themselves are much smaller:
  - `observe_jvm`: `0.051s`
  - `tick`: `0.038s`
  - `apply_action_jvm`: `0.019s`

Interpretation:
- The main Python hot spot is not action decoding, reward math, or buffer stacking.
- The main Python hot spot is recursive JVM-object-to-Python observation conversion.

### 2. Bridge-Only Steady-State Profile

Command:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python - <<'PY'
import cProfile, pstats
from fight_caves_rl.bridge.batch_client import BatchClientConfig, HeadlessBatchClient
from fight_caves_rl.bridge.contracts import HeadlessBootstrapConfig
client = HeadlessBatchClient.create(BatchClientConfig(env_count=16, account_name_prefix='audit_batch16', tick_cap=4096, bootstrap=HeadlessBootstrapConfig(start_world=False)))
actions = [0] * 16
seeds = [70000 + i for i in range(16)]
client.reset_batch(seeds=seeds)
for _ in range(8):
    client.step_batch(actions)
prof = cProfile.Profile()
prof.enable()
for _ in range(64):
    client.step_batch(actions)
prof.disable()
client.close()
path = '/tmp/fc_perf_audit/python_batch16_steady.prof'
prof.dump_stats(path)
stats = pstats.Stats(path)
stats.sort_stats('cumulative').print_stats(40)
PY
```

Key cumulative results:

| Function | Cum time (s) |
| --- | ---: |
| `HeadlessBatchClient.step_batch` | `0.890` |
| `HeadlessBatchClient._collect_step_results` | `0.788` |
| `pythonize_observation` | `0.711` |
| `_pythonize` | `0.705` |
| `observe_jvm` | `0.048` |
| `tick` | `0.040` |
| `pythonize_action_result` | `0.036` |
| `apply_action_jvm` | `0.015` |
| `visible_targets_from_observation` | `0.009` |

Interpretation:
- The bridge-only profile matches the vecenv profile almost exactly.
- This is strong evidence that the bridge observation conversion path, not the vecenv shell, is the main embedded-path bottleneck.

### 3. End-to-End Train Profile

Command:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
WANDB_MODE=disabled uv run python -m cProfile \
  -o /tmp/fc_perf_audit/python_train_disabled.prof \
  scripts/train.py \
  --config configs/train/train_baseline_v0.yaml \
  --total-timesteps 256 \
  --output /tmp/fc_perf_audit/train_profiled_disabled.json
```

Important limitation:
- this captures the parent trainer process
- the live environment stepping occurs in a spawned subprocess worker and is not fully visible to this profile
- short smoke runs are startup-heavy, so import and initialization costs are exaggerated

Key cumulative entries:

| Function | Cum time (s) | Interpretation |
| --- | ---: | --- |
| `run_smoke_training` | `9.683` | total parent-side train path |
| `pufferlib.pufferl.train` | `3.262` | learner/update loop |
| `pufferlib.pufferl.evaluate` | `1.965` | rollout side seen from parent |
| `pufferlib.pufferl.save_checkpoint` | `1.809` | checkpoint overhead visible in short runs |
| `WandbRunLogger._initialize_wandb` | `2.186` | startup/init cost, even with disabled mode plumbing in this short profile |

### 4. Trainer-Internal Timing Buckets

These come from the actual `puffer_logs` emitted by train runs.

Relevant runs:
- `/tmp/fc_train_postfix.json`
- `/tmp/fc_train_postfix_online.json`
- `/tmp/fc_perf_audit/train_online_probe.json`

Observed means:

| Run | Mean `train/performance/env` | Mean `train/performance/learn` | Mean `train/SPS` |
| --- | ---: | ---: | ---: |
| disabled 4-env stable run | `3.221` | `0.238` | `46.69` |
| online 4-env stable run | `1.327` | `0.292` | `44.72` in-loop, but only `13.06` last logged |
| online 4-env wall-clock probe | `0.939` | `0.334` | `34.41` in-loop mean, but `11.87` wall SPS |

Interpretation:
- Environment collection time is much larger than learner-update time in these runs.
- Online W&B hurts wall-clock much more than the in-loop `train/SPS` samples suggest, which means overhead outside the trainer sample window matters too.

## Facts

- Observation pythonization is the largest Python hot spot in steady-state env stepping.
- Action decode, reward logic, and observation flattening are secondary.
- The vecenv shell is not the main embedded-path problem.
- The learner is not the main current bottleneck in the shipped baseline runs.

## What This Rules Out

Definitely not the main issue:
- policy action decode
- reward bookkeeping
- raw `observe_jvm` call latency alone
- raw `tick` call latency alone
- offline W&B mode
