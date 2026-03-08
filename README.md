# RL

Python training and analytics module for the Fight Caves workspace.

## Scope

- `fight-caves-RL` remains the golden runtime dependency.
- `RSPS` remains the oracle/reference for parity and debugging only.
- `RL` owns Python training, bridge glue, evaluation, replay indexing, analytics, and benchmarking.

## Bootstrap

This repo is managed with `uv` and targets Python `3.11` as the preferred baseline from `RLspec.md`.

From `/home/jordan/code`:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv sync --group dev --python 3.11
uv run pytest fight_caves_rl/tests/unit
```

Training dependencies are pinned in the `train` dependency group:

```bash
source /home/jordan/code/.workspace-env.sh
uv sync --group dev --group train --python 3.11
uv run python -c "import pufferlib, torch, fight_caves_rl"
```

The RL baseline now pins the wheel-backed `pufferlib-core==3.0.17` distribution, which imports as `pufferlib`.
That removes the standard WSL train-bootstrap dependency on the workspace-local GCC sysroot and avoids the broad Ocean/source-build baggage from `pufferlib==3.0.0`.
The RL repo is also configured to resolve `torch` from the CPU wheel index during bootstrap so Linux does not silently pull the CUDA-heavy default wheel set.
The legacy `./scripts/bootstrap_wsl_toolchain.sh` path is retained for source-build comparisons and future native dependencies, but it is no longer part of the standard RL bootstrap flow.
One upstream inconsistency remains: the installed `pufferlib-core==3.0.17` distribution currently imports with `pufferlib.__version__ == "3.0.3"`, so RL manifest/version code must use distribution metadata rather than trusting the import version string.
Official upstream docs still say `pip install pufferlib`; RL intentionally standardizes on `pufferlib-core` because it preserves the RL-facing APIs we need without the import-time `resources` symlink side effect or the source-build footprint.

## Workspace Paths

The default sibling-repo layout is:

- `/home/jordan/code/fight-caves-RL`
- `/home/jordan/code/RSPS`
- `/home/jordan/code/RL`

Override these with `.env` values that match `.env.example` if needed.

## PufferLib Reuse

The RL repo should lean on PufferLib rather than reimplementing surfaces it already provides:

- use PufferLib's trainer stack instead of building a separate trainer abstraction
- use PufferLib's environment integration/emulation layer where it fits the sim wrapper contract
- use PufferLib's vectorized execution path rather than inventing a parallel rollout API
- use PufferLib's dashboard and W&B-facing logging hooks where they satisfy the required metrics contract
- specifically reuse `pufferlib.pufferl.PuffeRL`, `pufferlib.pufferl.WandbLogger`, `pufferlib.vector.make`, and `pufferlib.emulation` unless the sim contract forces a documented wrapper at the RL boundary

Anything simulator-semantic, replay-semantic, or parity-semantic still stays in the sim/oracle boundary and must not be recreated in Python.
