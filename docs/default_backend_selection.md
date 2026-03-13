# Default Backend Selection

This document records the current default backend choices after PR 8.2.

## Training default

The default RL training path is now:

- config: `configs/train/smoke_fast_v2.yaml`
- env backend: `v2_fast`
- entrypoint: `/home/jordan/code/RL/scripts/train.py`

That means a plain default smoke run now resolves to the V2 fast trainer path:

```bash
cd /home/jordan/code/RL
uv run python scripts/train.py --output /tmp/fc_train_default.json
```

Explicit fallback training selection remains available:

- V1 bridge smoke fallback:
  - `uv run python scripts/train.py --config configs/train/smoke_ppo_v0.yaml --output /tmp/fc_train_v1.json`
- V1 bridge multi-env baseline:
  - `uv run python scripts/train.py --config configs/train/train_baseline_v0.yaml --output /tmp/fc_train_v1_baseline.json`

Those preserved V1 configs pin `env.env_backend: v1_bridge` explicitly so they do not inherit the new `v2_fast` default by accident.

## Demo/replay default

The default headed demo/replay backend is now:

- backend id: `rsps_headed`
- selector entrypoint: `/home/jordan/code/RL/scripts/run_demo_backend.py`

Default dry-run resolution:

```bash
cd /home/jordan/code/RL
uv run python scripts/run_demo_backend.py --dry-run
```

That default resolves to the RSPS-backed headed replay path:

- `/home/jordan/code/RL/scripts/run_headed_trace_replay.py`

Other default headed demo actions remain on the same trusted backend:

- backend smoke:
  - `uv run python scripts/run_demo_backend.py --mode backend_smoke`
- live checkpoint inference:
  - `uv run python scripts/run_demo_backend.py --mode live_inference`

## Preserved fallbacks

The default switch does not remove older paths.

Explicit fallback/reference selection remains:

- frozen lite-demo headed fallback/reference:
  - `uv run python scripts/run_demo_backend.py --backend fight_caves_demo_lite --mode launch_reference --dry-run`
  - real launch command resolves to `cd /home/jordan/code/fight-caves-RL && ./gradlew --no-daemon :fight-caves-demo-lite:run`
- V1 oracle/reference/debug replay path:
  - `uv run python scripts/run_demo_backend.py --backend oracle_v1 --mode replay --dry-run -- --checkpoint /path/to/checkpoint.pt --output /tmp/fc_eval.json`
  - underlying entrypoint: `/home/jordan/code/RL/scripts/replay_eval.py`

## Non-removals

PR 8.2 does not remove:

- `fight-caves-demo-lite`
- V1 oracle/reference/debug replay and parity tools
- explicit config-based selection of older training paths

The point of this document is to make the new defaults deliberate and reversible.
