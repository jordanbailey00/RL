# W&B Logging Contract

PR6 standardizes RL-side W&B integration around a repo-owned logger:

- implementation: `fight_caves_rl/logging/wandb_client.py`
- trainer integration point: `fight_caves_rl/puffer/trainer.py`

## Why RL Owns the Logger

RL does not use `pufferlib.pufferl.WandbLogger` directly as the final PR6 logger because the repo needs extra behavior that is part of the RL contract:

- repo-owned run ids
- local run-manifest writing before artifact upload
- explicit artifact naming/versioning
- manifest-to-W&B config synchronization
- stable WSL smoke behavior with console/stat-monitor side effects suppressed

This does not change trainer semantics. `PuffeRL` still calls the logger through the expected logging interface.

## Run Initialization Contract

Every train/eval run must initialize W&B with:

- `project`
- `entity` if configured
- `group`
- `job_type`
- repo-owned `id`
- tags merged from bootstrap config + call-site tags
- `mode`
- local `dir`

Input normalization rules:

- `WANDB_ENTITY` may be configured as either:
  - a bare entity slug
  - a full `https://wandb.ai/<entity>/<project>` URL
- if a full project URL is provided and `WANDB_PROJECT` is still at the repo default (`fight-caves-rl`), RL derives:
  - `entity = <entity>`
  - `project = <project>`
- explicit `WANDB_PROJECT` overrides the project parsed from the URL

This normalization exists because W&B expects the `entity` field to be the entity slug, not the full project URL.

Run ids currently follow:

- `<wandb_run_prefix>-<run_kind>-<unix_timestamp>-<12_hex_suffix>`

## Required Local Directories

The bootstrap config owns three W&B-local paths:

- `WANDB_DIR`
- `WANDB_DATA_DIR`
- `WANDB_CACHE_DIR`

These must point to writable local workspace paths. This is required in WSL because artifact staging and cache writes cannot rely on user-global defaults.

## Runtime Stability Settings

The PR6 logger initializes W&B with settings that reduce subprocess/test instability:

- `console="off"`
- `quiet=True`
- `silent=True`
- `x_disable_stats=True`
- `x_disable_machine_info=True`

These settings are a runtime-stability choice, not a throughput claim.

## Metric Contract

PR6 currently logs two metric families:

- train metrics from `PuffeRL` namespaced as `train/...`
- eval summary metrics from `fight_caves_rl/logging/metrics.py` namespaced as `eval/...`

Current eval summary metrics:

- `eval/episode_count`
- `eval/mean_steps`
- `eval/terminated_rate`
- `eval/truncated_rate`
- `eval/seed_pack_version`

## Artifact Contract

PR6 currently supports these artifact categories:

- `checkpoint`
- `checkpoint_metadata`
- `run_manifest`
- `eval_summary`

Artifact names are versioned and normalized through `fight_caves_rl/logging/artifact_naming.py`.

## Manifest Synchronization Contract

For PR6, local manifests are the canonical run record.

Required behavior:

- write the local manifest first
- update W&B config from the manifest payload
- upload artifacts using the same run id and artifact records recorded in the manifest

This keeps the local `run_manifest.json` and remote W&B config/artifacts aligned.

## Test Harness Contract

Live subprocess tests that exercise train/eval/scripted entrypoints must:

- use isolated offline W&B directories per test
- avoid inheriting the full parent pytest environment
- avoid pytest fd-capture interference around the live subprocess body

The canonical helpers live in:

- `fight_caves_rl/tests/smoke/_helpers.py`
- `fight_caves_rl/tests/conftest.py`
