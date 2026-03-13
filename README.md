# RL

Python training and analytics module for the Fight Caves workspace.

## Scope

- `fight-caves-RL` remains the golden runtime dependency.
- `RSPS` now owns the trusted headed demo/replay runtime and still remains the oracle/reference for parity disputes and debugging.
- `RL` owns Python training, bridge glue, evaluation, replay indexing, analytics, and benchmarking.

## Pivot Authority

Current workspace authority is split across:
- `/home/jordan/code/pivot_plan.md`
- `/home/jordan/code/pivot_implementation_plan.md`
- `/home/jordan/code/RL/RLspec.md`

Under that pivot:
- the current simulator-backed path is V1 oracle/reference/debug
- the RSPS-backed headed path is the default demo/replay target
- `fight-caves-demo-lite` is frozen fallback/reference only
- V2 fast headless training is the default training path
- parity is defined as mechanics parity
- agent-driven implementation and validation assume WSL/Linux as the canonical execution environment
- Linux paths and shell commands are canonical; do not author active workflows around Windows-native path semantics or PowerShell

## Bootstrap

This repo is managed with `uv` and targets Python `3.11` as the preferred baseline from `RLspec.md`.

From `/home/jordan/code`:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv sync --group dev --python 3.11
uv run pytest fight_caves_rl/tests/unit
```

This is the default dev-bootstrap contract:
- self-contained unit coverage only
- no `pufferlib`/`torch` requirement
- no live sim workspace requirement

Training dependencies are pinned in the `train` dependency group:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv sync --group dev --group train --python 3.11
uv run python -c "import pufferlib, torch, fight_caves_rl"
uv run pytest fight_caves_rl/tests/train
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

## Validation Boundary

Per-PR CI is intentionally limited to:

- dev-only unit tests under `fight_caves_rl/tests/unit`
- train-bootstrap import smoke
- self-contained train-bootstrap tests under `fight_caves_rl/tests/train`

Local pre-merge validation owns the live runtime suites:

- `fight_caves_rl/tests/integration`
- `fight_caves_rl/tests/determinism`
- `fight_caves_rl/tests/parity`
- `fight_caves_rl/tests/smoke`

Manual or scheduled validation owns:

- `fight_caves_rl/tests/performance`
- `uv run python scripts/benchmark_env.py --config configs/benchmark/fast_env_v2.yaml --env-count 8 --rounds 16 --output /tmp/fc_vecenv_bench.json`
- `uv run python scripts/benchmark_train.py --config configs/benchmark/fast_train_v2.yaml --env-count 2 --total-timesteps 8 --logging-modes disabled,standard --output /tmp/fc_train_bench.json`
- `uv run python scripts/run_acceptance_gate.py --output-dir /tmp/rl-acceptance`
- `.github/workflows/benchmarks.yml` on a self-hosted Linux runner with the sibling workspace repos present
- `.github/workflows/acceptance.yml` on a self-hosted Linux runner with the sibling workspace repos present
- default backend selection is documented in [docs/default_backend_selection.md](./docs/default_backend_selection.md)

## PR3 Runtime Prerequisites

The correctness wrapper currently needs both:

- a packaged headless sim artifact under `/home/jordan/code/fight-caves-RL/game/build/distributions/fight-caves-headless*.zip`
- the checked-out sim workspace root with:
  - `/home/jordan/code/fight-caves-RL/config/headless_data_allowlist.toml`
  - `/home/jordan/code/fight-caves-RL/config/headless_manifest.toml`
  - `/home/jordan/code/fight-caves-RL/config/headless_scripts.txt`
  - `/home/jordan/code/fight-caves-RL/data/cache/main_file_cache.dat2`

The packaged distribution alone is not sufficient today because the current sim bootstrap still resolves the checked-out repository root and reads the real cache/config workspace.

Current workspace status:

- the packaged dist exists
- the checked-out sim cache is restored under `/home/jordan/code/fight-caves-RL/data/cache`

That means:

- PR3 live integration tests can run against the real sim workspace
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

## PR5/PR8 Training Path

PR5 established the first end-to-end PufferLib smoke loop on top of the correctness wrapper, and PR8 replaced the temporary single-env shim with the current batch-backed vectorized env path.

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python scripts/train.py --output /tmp/fc_train.json
uv run python scripts/run_demo_backend.py --mode live_inference --checkpoint "$(python - <<'PY'\nimport json\nprint(json.load(open('/tmp/fc_train.json'))['checkpoint_path'])\nPY)"
uv run pytest fight_caves_rl/tests/smoke -q
```

Current training-path facts:

- RL keeps the raw sim semantics and uses the documented RL-local policy encoding:
  - `puffer_policy_observation_v1`
  - `puffer_policy_action_v0`
- `scripts/train.py` now defaults to the `v2_fast` subprocess-isolated train config so `PuffeRL` training does not share one process with the embedded JPype/JVM runtime
- the shipped embedded vecenv still lives in `fight_caves_rl/envs/vector_env.py` and remains the direct bridge path for correctness tooling and vecenv microbenchmarks
- the subprocess training wrapper currently lives in `fight_caves_rl/envs/subprocess_vector_env.py` and preserves the PR7/PR8 batch semantics while paying Python IPC overhead
- `pufferlib.vector.Serial` is still not used because the stock Serial backend constructs the env twice, and that double-bootstrap is incompatible with the embedded-JVM runtime selected for Mode A
- `configs/train/smoke_fast_v2.yaml` is the current default train-entry config
- `configs/train/train_baseline_v0.yaml` remains the V1 bridge multi-env fallback baseline config and now pins `env_backend = v1_bridge` explicitly
- `configs/benchmark/fast_env_v2.yaml` plus `scripts/benchmark_env.py` are the current default env benchmark entrypoints
- `configs/benchmark/fast_train_v2.yaml` plus `scripts/benchmark_train.py` are the current default train benchmark entrypoints
- local dashboard rendering is TTY-aware; smoke and CI subprocesses suppress terminal painting automatically while still preserving the manifest/logging contract
- live vecenv-only smoke checks use `scripts/run_vecenv_smoke.py` so each smoke run gets a fresh embedded-JVM process
- current local WSL performance is still far below the long-term target; see `/home/jordan/code/pivot_plan.md` for the current hard gates and `docs/performance_decomposition_report.md` for the baseline evidence that motivated the pivot

## PR6 Run Logging

PR6 adds repo-owned W&B and manifest wiring on top of the PR5 smoke path.

- every `train.py` run now writes:
  - a checkpoint
  - checkpoint metadata
  - a local `run_manifest.json`
  - a `wandb_run_id`
- every `replay_eval.py` run now writes:
  - an `eval_summary.json`
  - a `replay_pack.json`
  - a `replay_index.json`
  - a local `run_manifest.json`
  - a `wandb_run_id`

`scripts/eval.py` is still retained as a compatibility alias, but `scripts/replay_eval.py` is now the canonical V1 oracle/reference replay/eval entrypoint rather than the default headed demo path.

The bootstrap config now owns the local W&B directories:

- `WANDB_DIR`
- `WANDB_DATA_DIR`
- `WANDB_CACHE_DIR`

Those paths matter in WSL because artifact staging and cache writes must stay inside writable workspace-owned directories.

Online W&B note:

- `WANDB_ENTITY` may be either a bare entity slug or a full `https://wandb.ai/<entity>/<project>` URL
- if a full project URL is supplied and `WANDB_PROJECT` is still at the repo default, RL normalizes the entity slug and derives the project name automatically before calling `wandb.init(...)`
- this avoids the common failure mode where a full URL is passed straight through as the W&B entity value

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

## Default Backends

Current default backend choices are:

- training default: `v2_fast`
- headed demo/replay default: `rsps_headed`

Preserved explicit fallbacks:

- headed fallback/reference: `fight-caves-demo-lite`
- replay/debug fallback: V1 oracle/reference path

Use [docs/default_backend_selection.md](./docs/default_backend_selection.md) for the exact selector commands.

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

## PR9 Reward and Curriculum

PR9 adds the first versioned reward/curriculum registry layer:

- rewards:
  - `reward_sparse_v0`
  - `reward_shaped_v0`
- curriculum:
  - `curriculum_disabled_v0`
  - `curriculum_wave_progression_v0`

Current defaults remain the parity-safe path:

- train configs default to `reward_sparse_v0`
- train configs default to `curriculum_disabled_v0`
- replay/eval defaults to the checkpoint reward config plus `curriculum_disabled_v0`

See [reward_configs.md](/home/jordan/code/RL/docs/reward_configs.md) for the frozen reward terms and curriculum schedule rules.

## PR10 Replay and Eval Artifacts

PR10 extends eval from summary-only output into replay-grade artifact generation.

Current shipped replay/eval facts:

- the default headed demo/replay selector is `scripts/run_demo_backend.py`
- `scripts/replay_eval.py` remains the canonical V1 oracle/reference replay/eval entrypoint
- every replay eval writes:
  - `eval_summary.json`
  - `replay_pack.json`
  - `replay_index.json`
  - `run_manifest.json`
- replay artifacts are generated from the same fixed-seed eval path that already drives summary digests
- `replay_step_cadence` controls how densely step payloads are captured in `replay_pack.json`
- the thin PR4 canary utilities still exist for scripted-policy determinism checks and parity trace comparisons

See [eval_and_replay.md](/home/jordan/code/RL/docs/eval_and_replay.md) for the frozen replay artifact contract.

## PR11 Performance Hardening

PR11 extends the repo-owned benchmark path without changing simulator semantics.

Current PR11 facts:

- `scripts/benchmark_bridge.py` remains the raw bridge microbenchmark entrypoint
- `scripts/benchmark_env.py` now reports both `wrapper_sequential` and `vecenv_lockstep`
- those env measurements run in separate child processes because the embedded-JVM lifecycle is process-global
- `scripts/benchmark_train.py` now benchmarks end-to-end training SPS across `disabled`, `standard`, and `aggressive` logging modes
- train benchmark measurements run in fresh child `train.py` processes per logging mode
- benchmark reports now carry shared benchmark-context metadata that records the benchmark profile, repo SHAs, sim artifact task/path, schema ids, reward/curriculum ids, PufferLib distribution/version, and hardware profile
- `.github/workflows/benchmarks.yml` is the repo-owned manual benchmark workflow for a self-hosted Linux workspace runner

## PR12 Parity and Oracle Validation

PR12 expands the thin PR4 parity checks into a versioned multi-scenario parity matrix.

Current PR12 facts:

- `scripts/run_parity_canary.py` is the canonical parity entrypoint
- `configs/eval/parity_canary_v0.yaml` now runs the current three-scenario matrix: `parity_single_wave_v0`, `parity_jad_healer_v0`, `parity_tzkek_split_v0`
- each scenario compares wrapper trace, raw sim trace, and the trace-pack-driven scripted replay path in fresh subprocesses
- parity reports are inspectable JSON outputs with per-scenario digests, final ticks, and pass/fail status
- checkpoint replay-eval determinism still lives under the PR10 replay contract; PR12 replay-to-trace equivalence refers to the scripted trace-pack path, not checkpoint eval replay

## PR13 MVP Acceptance Gate

PR13 closes the implementation roadmap with a repo-owned acceptance harness.

Current PR13 facts:

- `scripts/run_acceptance_gate.py` is the canonical local/manual acceptance entrypoint
- `.github/workflows/acceptance.yml` is the repo-owned manual self-hosted acceptance workflow
- the acceptance gate runs the full RL suite split, a real train run, deterministic replay-eval on the produced checkpoint, the parity matrix, and the bridge/env/train benchmark entrypoints
- the acceptance gate writes a single `acceptance_report.json` plus per-command logs and output artifacts
- acceptance runs use offline W&B directories under the acceptance output root so the gate is reproducible and does not depend on user-global W&B state
