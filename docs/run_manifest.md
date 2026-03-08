# Run Manifest

Each train/eval/benchmark run must emit a local manifest before later W&B and replay work is added.

## Bootstrap Manifest Fields

The bootstrap manifest records:

- RL repo root
- headless sim repo root
- RSPS repo root
- Python version
- PufferLib distribution name
- PufferLib baseline version
- W&B mode
- timestamp

This is intentionally minimal for PR1. Later PRs will extend it with schema versions, reward/curriculum versions, bridge protocol version, replay schema version, checkpoint metadata, and hardware/benchmark profile details.

## Baseline Values

- Python baseline: `3.11`
- PufferLib distribution: `pufferlib-core`
- PufferLib baseline: `3.0.17`
- default W&B mode: `offline`

PR1 records the pinned PufferLib distribution/version from repo configuration rather than requiring the package to be installed in the default unit-test sync.
Later manifest work should resolve the installed distribution metadata from `importlib.metadata` instead of trusting `pufferlib.__version__`, because the upstream `pufferlib-core==3.0.17` wheel currently imports as `pufferlib.__version__ == "3.0.3"`.
