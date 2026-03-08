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
The correctness-mode bridge also depends on `jpype1`, which is now part of the default RL dependency set because PR3 uses it for the embedded JVM path.
The legacy `./scripts/bootstrap_wsl_toolchain.sh` path is retained for source-build comparisons and future native dependencies, but it is no longer part of the standard RL bootstrap flow.
One upstream inconsistency remains: the installed `pufferlib-core==3.0.17` distribution currently imports with `pufferlib.__version__ == "3.0.3"`, so RL manifest/version code must use distribution metadata rather than trusting the import version string.
Official upstream docs still say `pip install pufferlib`; RL intentionally standardizes on `pufferlib-core` because it preserves the RL-facing APIs we need without the import-time `resources` symlink side effect or the source-build footprint.

## Workspace Paths

The default sibling-repo layout is:

- `/home/jordan/code/fight-caves-RL`
- `/home/jordan/code/RSPS`
- `/home/jordan/code/RL`

Override these with `.env` values that match `.env.example` if needed.

## PR3 Runtime Prerequisites

The correctness wrapper currently needs both:

- a packaged headless sim artifact under `/home/jordan/code/fight-caves-RL/game/build/distributions/fight-caves-headless*.zip`
- the checked-out sim workspace root with:
  - `/home/jordan/code/fight-caves-RL/FCspec.md`
  - `/home/jordan/code/fight-caves-RL/config/headless_data_allowlist.toml`
  - `/home/jordan/code/fight-caves-RL/config/headless_manifest.toml`
  - `/home/jordan/code/fight-caves-RL/config/headless_scripts.txt`
  - `/home/jordan/code/fight-caves-RL/data/cache/main_file_cache.dat2`

The packaged distribution alone is not sufficient today because the current sim bootstrap still resolves the checked-out repository root and reads the real cache/config workspace.

Current workspace status:

- the packaged dist exists
- the checked-out sim cache is restored under `/home/jordan/code/fight-caves-RL/data/cache`

That means:

- PR3 unit tests run
- PR3 live integration tests pass against the real sim workspace
- `uv run python scripts/smoke_random.py --max-steps 20000` reaches a full wrapper-managed episode truncation at `max_tick_cap`

Mode A validation note:

- wrapper-vs-raw reset/step equivalence should be validated in separate Python processes
- the embedded-JVM bridge owns one runtime per process, and separate player slots inside one runtime are not a valid proof of identical absolute reset state because each episode gets its own dynamic instance
- PR4 determinism/parity checks therefore compare semantic projections that normalize episode-relative ticks and instance-shifted tiles instead of asserting on raw absolute allocator-dependent values

## PufferLib Reuse

The RL repo should lean on PufferLib rather than reimplementing surfaces it already provides:

- use PufferLib's trainer stack instead of building a separate trainer abstraction
- use PufferLib's environment integration/emulation layer where it fits the sim wrapper contract
- use PufferLib's vectorized execution path rather than inventing a parallel rollout API
- use PufferLib's dashboard and W&B-facing logging hooks where they satisfy the required metrics contract
- specifically reuse `pufferlib.pufferl.PuffeRL`, `pufferlib.pufferl.WandbLogger`, `pufferlib.vector.make`, and `pufferlib.emulation` unless the sim contract forces a documented wrapper at the RL boundary

Anything simulator-semantic, replay-semantic, or parity-semantic still stays in the sim/oracle boundary and must not be recreated in Python.

## PR5 Smoke Path

PR5 now provides the first end-to-end PufferLib smoke path on top of the correctness wrapper:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python scripts/train.py --output /tmp/fc_train.json
uv run python scripts/eval.py --checkpoint "$(python - <<'PY'\nimport json\nprint(json.load(open('/tmp/fc_train.json'))['checkpoint_path'])\nPY)" --output /tmp/fc_eval.json
uv run pytest fight_caves_rl/tests/smoke -q
```

Current PR5 constraints:

- the smoke trainer path is single-env only on Mode A
- PR5 keeps the raw sim semantics and uses a documented RL-local policy encoding:
  - `puffer_policy_observation_v0`
  - `puffer_policy_action_v0`
- PR5 uses a repo-local single-env vecenv shim instead of `pufferlib.vector.Serial` because the stock Serial backend constructs the env twice, and that double-bootstrap is incompatible with the embedded-JVM runtime selected for Mode A
- local dashboard rendering is TTY-aware; smoke and CI subprocesses suppress terminal painting automatically while still preserving the manifest/logging contract
- true batched/vector training remains PR8 scope

## PR6 Run Logging

PR6 adds repo-owned W&B and manifest wiring on top of the PR5 smoke path.

- every `train.py` run now writes:
  - a checkpoint
  - checkpoint metadata
  - a local `run_manifest.json`
  - a `wandb_run_id`
- every `eval.py` run now writes:
  - an `eval_summary.json`
  - a local `run_manifest.json`
  - a `wandb_run_id`

The bootstrap config now owns the local W&B directories:

- `WANDB_DIR`
- `WANDB_DATA_DIR`
- `WANDB_CACHE_DIR`

Those paths matter in WSL because artifact staging and cache writes must stay inside writable workspace-owned directories.

Example:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python scripts/train.py --output /tmp/fc_train.json
python - <<'PY'
import json
payload = json.load(open("/tmp/fc_train.json"))
print(payload["wandb_run_id"])
print(payload["run_manifest_path"])
PY
```

Current PR6 implementation note:

- RL uses a repo-owned `fight_caves_rl.logging.WandbRunLogger` instead of directly using `pufferlib.pufferl.WandbLogger`
- the reason is not trainer semantics; it is manifest/artifact policy and runtime stability:
  - repo-owned run ids
  - explicit artifact naming/versioning
  - local-manifest and W&B config synchronization
  - W&B startup settings that suppress console/stat-monitor side effects in WSL smoke tests

## PR7 Batched Bridge

PR7 adds the first versioned batched bridge baseline:

- bridge contract: `fight_caves_bridge_v1`
- current transport: embedded-JVM lockstep batch
- current slot semantics:
  - one runtime
  - many player slots
  - one shared tick per batch step
  - each slot remains isolated inside its own fight-cave instance

Key files:

- `fight_caves_rl/bridge/protocol.py`
- `fight_caves_rl/bridge/buffers.py`
- `fight_caves_rl/bridge/batch_client.py`
- `fight_caves_rl/benchmarks/bridge_bench.py`

Quick benchmark:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python scripts/benchmark_bridge.py \
  --config configs/benchmark/bridge_1env_v0.yaml \
  --output /tmp/fc_bridge_bench.json
```

Current PR7 note:

- the batch protocol and slot semantics are now explicit
- the final lower-copy subprocess/shared-buffer transport is still future work
- PR8 should build on the PR7 protocol rather than replacing it
