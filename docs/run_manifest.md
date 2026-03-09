# Run Manifest

PR6 replaces the old bootstrap-only manifest with a full local train/eval run manifest.

Each train/eval run now writes a `run_manifest.json` alongside the run outputs before the same metadata is mirrored into W&B config/artifacts.

## Required Top-Level Fields

- `created_at`
- `run_kind`
- `run_id`
- `config_id`
- `run_output_dir`

## Repo / Runtime Provenance

Every manifest records:

- RL repo root + commit SHA
- headless sim repo root + commit SHA
- RSPS repo root + commit SHA
- Python version + Python baseline
- PufferLib distribution version from distribution metadata
- PufferLib import namespace/version for drift visibility

The canonical PufferLib values come from `importlib.metadata`, not `pufferlib.__version__`, because `pufferlib-core==3.0.17` currently imports as `pufferlib.__version__ == "3.0.3"`.

## W&B Provenance

Every manifest records:

- `wandb_project`
- `wandb_entity`
- `wandb_group`
- `wandb_mode`
- `wandb_resume`
- `wandb_tags`
- `wandb_dir`
- `wandb_data_dir`
- `wandb_cache_dir`

## Frozen Contract / Schema Fields

Every manifest records the PR2/PR5 contract identities:

- benchmark profile id/version
- bridge protocol id/version
- sim observation schema id/version
- sim action schema id/version
- episode-start contract id/version
- policy observation schema id/version
- policy action schema id/version

## Run-Mode Fields

Every manifest records:

- `benchmark_mode`
- `bridge_mode`
- `replay_mode`
- `logging_mode`
- `dashboard_mode`
- `env_count`
- `reward_config_id`
- `curriculum_config_id`
- `policy_id`

Current training-path note:

- train manifests now record `bridge_mode = subprocess_isolated_jvm`
- this indicates that `train.py` used the subprocess-isolated vecenv worker rather than the direct embedded-JVM vecenv
- correctness and direct bridge/env benchmark tools may still record other bridge-mode values

## Sim Artifact Fields

Every manifest records the packaged sim dependency actually consumed by RL:

- `sim_artifact_task`
- `sim_artifact_path`

These values come from the PR2 bridge handshake built from the packaged headless distribution selected by the launcher.

## Checkpoint / Eval Fields

Train/eval manifests record:

- `checkpoint_format_id`
- `checkpoint_format_version`
- `checkpoint_path`
- `checkpoint_metadata_path`

Eval manifests additionally record:

- `seed_pack`
- `seed_pack_version`
- `summary_digest`

## Hardware Profile

Each manifest embeds a `hardware_profile` object with:

- platform
- machine
- processor
- cpu count
- Python implementation

## Artifact Records

Each manifest records artifact records produced by the run. The current artifact set supports:

- checkpoint
- checkpoint metadata
- run manifest
- eval summary
- replay pack
- replay index

Artifact names are repo-owned and versioned in `fight_caves_rl/logging/artifact_naming.py`.

## Current Defaults

- Python baseline: `3.11`
- PufferLib distribution baseline: `pufferlib-core`
- PufferLib distribution baseline version: `3.0.17`
- default W&B mode: `offline`

## File Ownership

- manifest schema + writing: `fight_caves_rl/manifests/run_manifest.py`
- PufferLib version resolution: `fight_caves_rl/manifests/versions.py`
- artifact naming/versioning: `fight_caves_rl/logging/artifact_naming.py`
- train/eval logger integration: `fight_caves_rl/logging/wandb_client.py`

Benchmarks will extend the same manifest surface later rather than introducing a second manifest shape.
