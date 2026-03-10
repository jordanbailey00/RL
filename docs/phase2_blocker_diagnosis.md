# Phase 2 Blocker Diagnosis

Date: 2026-03-10

This document explains why `WC-P2-03` remains blocked after the first low-copy transport iteration.

## Current Native-Linux Source-of-Truth Gate

Latest hosted run:

- [fight-caves-RL/actions/runs/22883118379](https://github.com/jordanbailey00/fight-caves-RL/actions/runs/22883118379)

Published gate summary:

- [codex/phase2-results latest gate summary](https://github.com/jordanbailey00/fight-caves-RL/blob/codex/phase2-results/phase2-native-linux/latest/gate_summary.json)

Latest native-Linux gate numbers:

- transport `64 env`:
  - pipe: `10868.61` env/s
  - `shared_memory_v1`: `8340.91` env/s
  - speedup: `0.7674x`
- disabled train `16 env`:
  - pipe: `74.05` SPS
  - `shared_memory_v1`: `75.01` SPS
- disabled train `64 env`:
  - pipe: `75.03` SPS
  - `shared_memory_v1`: `74.85` SPS
  - speedup: `0.9977x`
- shared-train scaling ratio `64 vs 16`: `0.9979x`
- blockers:
  - `transport_signal_too_weak`
  - `train_signal_too_weak`
  - `shared_train_scaling_too_weak`

Conclusion from the latest source-of-truth gate:

- the first low-copy transport iteration did not produce a stable native-Linux win
- the current production transport must not be swapped
- `WC-P2-03` remains blocked

## Learner Ceiling Benchmark

Repo-owned benchmark:

- `/home/jordan/code/RL/scripts/benchmark_train_ceiling.py`

Purpose:

- hold the policy, `PuffeRL` trainer, and current train config constant
- replace the real sim/bridge/transport path with a fake zero-cost vecenv
- measure how much throughput the current trainer loop can achieve before any live env overhead is added

Local WSL ceiling run:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
WANDB_MODE=disabled uv run python scripts/benchmark_train_ceiling.py \
  --config configs/benchmark/train_1024env_v0.yaml \
  --env-counts 4,16,64 \
  --total-timesteps 4096 \
  --output /tmp/fc_train_ceiling_report.json
```

Measured learner-ceiling rows from `/tmp/fc_train_ceiling_report.json`:

- `4 env`: `154.45` env-steps/s
- `16 env`: `156.20` env-steps/s
- `64 env`: `144.43` env-steps/s

Key scaling result:

- the fake-env ceiling is nearly flat across `4 / 16 / 64` envs
- that means the current train loop is already dominated by trainer-side work rather than live env count

## Stage Breakdown on the Fake Env

The same fake-env benchmark records stage timing for one effective train loop at `64 env / 4096 timesteps`:

- rollout/evaluate: `15.83s`
- PPO train/update: `24.87s`
- final evaluate: `16.02s`
- total elapsed: `56.72s`
- resulting throughput: `144.43` env-steps/s

Interpretation:

- even with a zero-cost env, rollout collection plus trainer updates consume almost the entire wall clock
- the PPO update path is the single largest stage
- the evaluation path is also large enough that modest transport improvements will be masked unless the trainer path changes or the benchmark topology changes

## Why the Transport Win Disappears in Training

Facts:

- Phase 1 already raised the flat-path vecenv row to about `10.96k` env/s at `64 env` on native Linux.
- The latest Phase 2 native-Linux gate shows disabled end-to-end train throughput around `75` SPS at both `16` and `64 envs`.
- The fake-env learner ceiling on this WSL host is only about `144-156` env-steps/s across `4 / 16 / 64 envs`.

Interpretation:

- after Phase 1, the live env path is no longer the dominant end-to-end cost center
- the current `PuffeRL` train loop, rollout structure, and policy update path now dominate wall clock strongly enough to flatten transport improvements
- a transport-only promotion gate is no longer sufficient to justify `WC-P2-03`

## Current Decision

Do not start `WC-P2-03`.

Before the production transport swap can be justified, the workspace needs one of:

- a revised Phase 2 decision path that separates actor/collection gains from learner-bound end-to-end limits, or
- a new optimization batch that reduces the trainer-side ceiling enough for transport improvements to become visible again

This is now a trainer-bound escalation problem, not just a transport implementation problem.
