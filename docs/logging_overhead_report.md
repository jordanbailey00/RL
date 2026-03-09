# Logging Overhead Report

Date: 2026-03-09

Repo and SHA:
- RL: `cda7ab4104799be40ffe39f77e5a86c2e6f0eea5`
- fight-caves-RL: `2365506bd3ea5cce515c571f39c24e72a38acc67`

## Question

Are observability features the main reason training is only reaching `30-90` SPS?

## 1. Disabled vs Offline W&B

Measured benchmark commands:

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
```

Results:

| Env count | Disabled SPS | Offline SPS |
| ---: | ---: | ---: |
| 4 | `36.40` | `36.14` |
| 16 | `82.84` | `91.66` |

Interpretation:
- Offline W&B is not the main bottleneck.
- The 16-env offline run is slightly faster than disabled, which should be treated as normal benchmark noise, not as evidence that logging is beneficial.

## 2. Online W&B

Measured wall-clock probe:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
WANDB_MODE=online WANDB_ENTITY=jordanbaileypmp-georgia-institute-of-technology WANDB_PROJECT=fight-caves-RL \
  uv run python scripts/train.py \
    --config configs/train/train_baseline_v0.yaml \
    --total-timesteps 256 \
    --output /tmp/fc_perf_audit/train_online_probe.json
```

Result:
- wall-clock throughput: `11.87` SPS
- same baseline class in disabled/offline benchmark harness: about `36` SPS

Interpretation:
- Online W&B is a real throughput penalty in short runs.
- It matters, but it is still not the primary reason the stack is far from `100000+` SPS.

## 3. Dashboard Output

Measured directional probe:
- `WANDB_MODE=disabled`
- direct `train.py` runs
- compared dashboard on/off and checkpoint interval `1` vs `999999`

Observed last logged `train/SPS`:

| Dashboard | Checkpoint interval | Last logged `train/SPS` |
| --- | ---: | ---: |
| off | 1 | `89.26` |
| off | 999999 | `87.17` |
| on | 1 | `74.73` |
| on | 999999 | `39.37` |

Important limitation:
- these are not benchmark-harness wall-clock numbers
- they are last logged trainer samples from short runs
- treat them as directional only

Interpretation:
- Dashboard rendering appears harmful
- its exact magnitude is not stable enough from this probe to call it the main issue

## 4. Checkpoint / Artifact Overhead

Evidence:
- all train benchmarks wrote `3` artifacts
- short-run cProfile shows `pufferlib.pufferl.save_checkpoint` at about `1.809s` cumulative

Interpretation:
- checkpoint work is visible
- it matters disproportionately in short smoke runs
- it is not the main reason disabled training still sits in the `36-88` SPS range

## 5. Internal Trainer SPS vs Wall SPS

Online probe:
- mean in-loop `train/SPS`: `34.41`
- wall-clock SPS: `11.87`

Interpretation:
- logging / startup / network / artifact overhead outside the trainer's internal timing buckets is real
- do not treat in-loop `train/SPS` alone as the full user-facing throughput number

## Audit Conclusion

Facts:
- offline W&B is negligible
- online W&B is materially harmful
- dashboard printing is directionally harmful
- checkpointing is visible in short profiles

Strong conclusion:
- observability overhead is real but secondary
- even with observability minimized, the stack is still orders of magnitude too slow
