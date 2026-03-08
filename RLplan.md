# RLplan.md - RL Module Implementation Plan

## Scope and Source of Truth

This plan is derived from the root `RLspec.md` supplied for the RL module and treats that spec as authoritative.

Planning constraints carried through this document:

- `fight-caves-RL` remains the golden runtime dependency for reset, step, observation, action, determinism, batching, and parity-facing semantics.
- `RSPS` remains the oracle/reference for parity disputes, headed validation, and rare mechanic debugging.
- `RL` owns Python training, PufferLib integration, bridge glue, reward/curriculum configuration, analytics, benchmarking, replay indexing, and CI.
- Dependency direction is one-way: `RL -> fight-caves-RL`, with `RSPS` kept out of the training hot path.
- No work chunk is allowed to change simulator semantics or hide sim/oracle drift behind wrapper logic.
- The plan targets `>= 1,000,000 env steps/sec`, but only through staged performance gates after correctness is locked.
- Canonical RL root filenames are `RLspec.md` and `RLplan.md`.
- If other module specs/plans are referenced, use `FCspec.md` / `FCplan.md` for the headless Fight Caves sim module and `RSPSspec.md` / `RSPSplan.md` for the RSPS module.

Current workspace reality as of 2026-03-08:

- `/home/jordan/code/fight-caves-RL` exists and is the current golden sim repo.
- `/home/jordan/code/RSPS` exists and is the current oracle/reference repo.
- `/home/jordan/code/RL` now exists, has completed the documented PR 1 bootstrap baseline, and is tied to its canonical Git remote.

## 1. Repo Bootstrap Plan for RL

1. Create the RL repo skeleton at `/home/jordan/code/RL` with `uv`-managed Python packaging, a pinned dependency lockfile, baseline docs, and the required top-level directory layout from the spec.
2. Standardize the Python baseline early. The spec prefers Python 3.11 and allows 3.12 only if dependency compatibility is verified; this must be decided in the bootstrap PR and then pinned in `pyproject.toml` and `uv.lock`.
3. Pin the PufferLib baseline to `pufferlib-core==3.0.17` in locked dependencies, while treating the runtime import namespace as `pufferlib` and recording the distribution/version in code-owned manifest/version utilities.
4. Add a minimal package skeleton for `fight_caves_rl`, baseline config loading, manifest generation helpers, and environment-variable based discovery of sibling repos.
5. Add `.env.example` and README setup instructions that explicitly reference:
   - `/home/jordan/code/fight-caves-RL`
   - `/home/jordan/code/RSPS`
   - `/home/jordan/code/RL`
6. Add baseline CI that proves Linux/WSL install, unit test execution, and smoke bootstrap from lockfiles.
7. Normalize root RL filenames immediately to `RLspec.md` and `RLplan.md`, and ensure RL-side cross-module references use `FCspec.md` / `FCplan.md` and `RSPSspec.md` / `RSPSplan.md`.
8. Do not add training logic, reward shaping, or bridge optimization in bootstrap. The first PR should only make the repo reproducible and ready for contract-first work.

## 2. Proposed Directory Structure

```text
RL/
  RLspec.md
  RLplan.md
  README.md
  pyproject.toml
  uv.lock
  .env.example
  .gitignore
  .github/
    workflows/
      ci.yml
      benchmarks.yml
  configs/
    train/
      smoke_ppo_v0.yaml
      train_baseline_v0.yaml
    eval/
      eval_seedpack_v0.yaml
      replay_eval_v0.yaml
      parity_canary_v0.yaml
    sweep/
      ppo_sweep_v0.yaml
    reward/
      reward_sparse_v0.yaml
      reward_shaped_v0.yaml
    curriculum/
      curriculum_disabled_v0.yaml
      curriculum_wave_progression_v0.yaml
    benchmark/
      official_profile_v0.yaml
      bridge_1env_v0.yaml
      bridge_64env_v0.yaml
      vecenv_256env_v0.yaml
      train_1024env_v0.yaml
  docs/
    rl_integration_contract.md
    bridge_contract.md
    hotpath_map.md
    observation_mapping.md
    action_mapping.md
    reward_configs.md
    eval_and_replay.md
    performance_plan.md
    wandb_logging_contract.md
    parity_canaries.md
    run_manifest.md
  scripts/
    train.py
    eval.py
    benchmark_env.py
    benchmark_bridge.py
    replay_eval.py
    smoke_random.py
    smoke_scripted.py
    sweep.py
  fight_caves_rl/
    __init__.py
    bridge/
      __init__.py
      contracts.py
      launcher.py
      debug_client.py
      batch_client.py
      protocol.py
      errors.py
      buffers.py
    envs/
      __init__.py
      schema.py
      action_mapping.py
      observation_mapping.py
      correctness_env.py
      vector_env.py
    puffer/
      __init__.py
      factory.py
      trainer.py
      callbacks.py
    policies/
      __init__.py
      mlp.py
      checkpointing.py
    rewards/
      __init__.py
      registry.py
      reward_sparse_v0.py
      reward_shaped_v0.py
    curriculum/
      __init__.py
      registry.py
      curriculum_disabled_v0.py
      curriculum_wave_progression_v0.py
    replay/
      __init__.py
      seed_packs.py
      trace_packs.py
      replay_export.py
      replay_index.py
      eval_runner.py
    logging/
      __init__.py
      wandb_client.py
      metrics.py
      artifact_naming.py
    manifests/
      __init__.py
      run_manifest.py
      versions.py
    benchmarks/
      __init__.py
      bridge_bench.py
      env_bench.py
      train_bench.py
    utils/
      __init__.py
      config.py
      paths.py
      seeding.py
      timing.py
    tests/
      unit/
      integration/
      determinism/
      smoke/
      performance/
      parity/
```

## 3. Ordered PR / Work-Chunk Plan

### PR 1 - Repo Bootstrap and Lockfile

Goal:

- Create the RL repo skeleton, dependency lockfile, local setup path, baseline config utilities, canonical root filenames, and initial CI.

Expected files/directories:

- `/home/jordan/code/RL/RLspec.md`
- `/home/jordan/code/RL/RLplan.md`
- `/home/jordan/code/RL/README.md`
- `/home/jordan/code/RL/pyproject.toml`
- `/home/jordan/code/RL/uv.lock`
- `/home/jordan/code/RL/.env.example`
- `/home/jordan/code/RL/.gitignore`
- `/home/jordan/code/RL/.github/workflows/ci.yml`
- `/home/jordan/code/RL/configs/`
- `/home/jordan/code/RL/docs/run_manifest.md`
- `/home/jordan/code/RL/fight_caves_rl/__init__.py`
- `/home/jordan/code/RL/fight_caves_rl/utils/config.py`
- `/home/jordan/code/RL/fight_caves_rl/utils/paths.py`
- `/home/jordan/code/RL/fight_caves_rl/manifests/run_manifest.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/unit/test_config_loader.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/unit/test_run_manifest_basics.py`

Dependencies:

- None beyond the source-of-truth spec and the sibling repo paths.

Tests to add:

- Repo bootstrap install smoke from lockfile.
- Repo bootstrap install smoke for the wheel-backed `train` dependency group on the standard Linux/WSL path.
- Config loader unit test.
- Run manifest skeleton unit test.
- CI smoke job that imports the package and runs the unit subset.
- CI train-group smoke job that imports `pufferlib`, `torch`, and `fight_caves_rl`.

Acceptance criteria:

- Linux/WSL bootstrap is reproducible from `uv.lock`.
- Python and PufferLib baselines are pinned.
- Baseline package imports work.
- CI can install dependencies and run the bootstrap test subset.
- RL root filenames and cross-module doc reference conventions are normalized from the start.
- The RL repo documents the chosen bootstrap wheel path for `torch` so Linux does not silently fall back to the wrong package index.
- `uv sync --group dev --group train --python 3.11` succeeds on the standard wheel-backed baseline without requiring the legacy source-build toolchain path.
- CI validates the chosen train-group package path with an import smoke job.

Risks / likely failure modes:

- Python 3.11 vs 3.12 compatibility mismatch.
- PufferLib distribution/import metadata drift causes manifests or diagnostics to record the wrong version source.
- Early path handling that hard-codes local machine assumptions instead of sibling repo discovery.
- Cross-module doc naming drift if canonical filenames are not normalized immediately.
- Torch default Linux resolution may pull CUDA-heavy wheels unless the RL repo explicitly chooses the intended CPU/GPU install path.
- Upstream docs still point users at `pip install pufferlib`, which can reintroduce the heavier legacy package path if the RL docs are not explicit.

PR 1 execution status (2026-03-07 to 2026-03-08):

- [x] Canonical module filenames normalized to `FCspec.md` / `FCplan.md`, `RSPSspec.md` / `RSPSplan.md`, and `RLspec.md` / `RLplan.md`.
- [x] RL bootstrap skeleton created with `pyproject.toml`, `uv.lock`, `.env.example`, README, baseline configs, run-manifest docs, package skeleton, and CI.
- [x] Workspace-local `uv` installed and workspace-local Python `3.11` provisioned for the RL repo.
- [x] Default bootstrap path verified with `uv lock --python 3.11`, `uv sync --group dev --python 3.11`, and unit tests passing.
- [x] Baseline unit tests added for config loading and bootstrap run manifest generation.
- [x] `pufferlib==3.0.0` compatibility issue with `numpy>=2` discovered and corrected by pinning `numpy>=1.26.4,<2.0`.
- [x] Workspace-local compiler/build toolchain installed and exported in `/home/jordan/code/.workspace-env.sh` for WSL-native source builds.
- [x] CPU-only torch bootstrap path selected and documented in `pyproject.toml` so Linux does not silently resolve CUDA-heavy default wheels during PR 1 setup.
- [x] Reduced `NO_OCEAN=1` build path tested and documented as currently broken by an upstream `pufferlib==3.0.0` `setup.py` bug (`c_extension_paths` `NameError`).
- [x] Workspace-local GCC sysroot path validated as the least-risk PR 1 unblock path for `pufferlib==3.0.0` source builds in WSL.
- [x] Re-run bootstrap acceptance with `uv sync --group dev --group train --python 3.11` after the toolchain work is complete.
- [x] Validate whether the plain `pufferlib==3.0.0` build path without `NO_OCEAN=1` succeeds now that the compiler and CPU-only torch path are in place.
- [x] Codify the current WSL toolchain/bootstrap flow in a repo-owned script so the working train-group path is reproducible.
- [x] Initialize `/home/jordan/code/RL` as a Git-backed repository and connect it to `git@github.com:jordanbailey00/RL.git` so RL-side branch hygiene and future manifest SHA capture are unblocked.
- [x] Verify the current upstream package state and correct the candidate baseline to `pufferlib-core==3.0.17` rather than the previously noted `3.0.18`.
- [x] Create an isolated Python `3.11` validation matrix for `pufferlib==3.0.0` versus `pufferlib-core==3.0.17`.
- [x] Confirm that `pufferlib-core==3.0.17` preserves the required RL-facing surfaces:
  - `pufferlib.pufferl.PuffeRL`
  - `pufferlib.pufferl.WandbLogger`
  - `pufferlib.vector.make`
  - `pufferlib.emulation`
  - compiled `pufferlib._C`
- [x] Record the observed upstream drift that matters to RL integration:
  - `pufferlib-core==3.0.17` imports as `pufferlib` but reports `pufferlib.__version__ == "3.0.3"`
  - `pufferlib==3.0.0` creates an import-time `resources` symlink in the current working directory
  - `pufferlib-core==3.0.17` has a materially smaller dependency footprint than `pufferlib==3.0.0`
  - core modules still differ from the current `pufferlib==3.0.0` sources, especially in `pufferl.py`, `vector.py`, and `emulation.py`
- [x] Adopt `pufferlib-core==3.0.17` as the RL baseline and update the repo pins, manifest/version utilities, docs, and CI to match.

Resolved PR 1 support work (2026-03-08):

- [x] Replace the standard train-group baseline with `pufferlib-core==3.0.17`, keeping the old workspace-local GCC sysroot path only as a legacy fallback for source-built comparisons or future native dependencies.
- [x] Keep the default dev bootstrap path independent of the train group, while making the train-group path simple enough to validate in CI.
- [x] Preserve the CPU-only torch package source selection so Linux does not silently resolve CUDA-heavy default wheels.
- [x] Document the official-docs mismatch and the imported-version mismatch so later manifest code uses distribution metadata rather than `pufferlib.__version__`.

Newly discovered follow-up now queued into PR 2:

- Keep the module-drift observations for `pufferl.py`, `vector.py`, and `emulation.py` visible when PR 2 freezes the RL/sim bridge and env contract docs.
- Keep the imported-version mismatch visible when PR 2 builds the version/schema registry, because `pufferlib-core==3.0.17` still imports as `pufferlib.__version__ == "3.0.3"` in the validated RL environment.

Immediate next chunk to resume:

1. Start PR 3 and build the first correctness-first Python wrapper around the headless sim.
2. Implement the PR 2-selected Mode A bridge path around the current direct-runtime surface.
3. Mirror the sim's player provisioning flow exactly before reset/step/observe calls.
4. Keep wrapper behavior semantically transparent while adding reset/step/close integration tests and smoke scripts.

Stopping condition for the completed PR 2 chunk:

- RL freezes the sim-aligned integration contract in repo-owned docs.
- RL defines the action/observation/episode-start/bridge/benchmark identities in one place.
- RL documents the selected artifact strategy and Mode A bridge path early enough to steer PR 3.

### PR 2 - RL/Sim Contract Docs, Episode Start Contract, Bridge Strategy, Artifact Strategy, Benchmark Profile, and Version Registry

Goal:

- Freeze the integration contract in repo-owned docs before any wrapper code starts drifting, including the episode-start-state contract, the early bridge strategy, the official sim artifact consumption strategy, and the official benchmark profile v0.

Expected files/directories:

- `/home/jordan/code/RL/docs/rl_integration_contract.md`
- `/home/jordan/code/RL/docs/bridge_contract.md`
- `/home/jordan/code/RL/docs/observation_mapping.md`
- `/home/jordan/code/RL/docs/action_mapping.md`
- `/home/jordan/code/RL/docs/hotpath_map.md`
- `/home/jordan/code/RL/docs/performance_plan.md`
- `/home/jordan/code/RL/configs/benchmark/official_profile_v0.yaml`
- `/home/jordan/code/RL/fight_caves_rl/manifests/versions.py`
- `/home/jordan/code/RL/fight_caves_rl/envs/schema.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/unit/test_contract_version_registry.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/unit/test_required_docs_exist.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/unit/test_episode_start_contract_registry.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/unit/test_official_benchmark_profile_registry.py`

Dependencies:

- PR 1.
- Current headless sim contract in `/home/jordan/code/fight-caves-RL`.
- Current headless episode initialization contract and packaged artifact surfaces in `/home/jordan/code/fight-caves-RL`, including the Step 5 episode initialization artifacts and the Step 10 packaging artifacts.

Tests to add:

- Version registry unit test.
- Required docs presence test.
- Schema constant consistency test between docs-facing code constants and manifest registry.
- Episode-start contract registry test.
- Official benchmark profile registry/config presence test.

Existing bootstrap artifacts to extend rather than recreate:

- `/home/jordan/code/RL/fight_caves_rl/manifests/versions.py`
- `/home/jordan/code/RL/configs/benchmark/official_profile_v0.yaml`

Acceptance criteria:

- Observation schema ID/version, action schema ID/version, bridge protocol version, and episode-start contract version are defined in one place.
- The version registry treats `pufferlib-core==3.0.17` distribution metadata as canonical and does not rely on `pufferlib.__version__` as the source of truth for manifests or W&B config.
- `docs/rl_integration_contract.md` explicitly freezes the constant episode-start-state contract used for all training episodes, including:
  - equipped items
  - inventory
  - skills/stats
  - full HP and prayer at episode start
  - run energy 100%
  - run toggle ON by default
  - no XP gain during episodes
  - no stat changes during episodes
- The episode-start-state contract is explicitly aligned to the headless sim contract rather than invented locally in RL.
- `docs/bridge_contract.md` documents bridge candidate solutions and decision criteria early enough to steer PR 3 correctly, including at minimum:
  - a debug-friendly Mode A path for correctness bring-up
  - a provisional coarse-grained Mode B/C direction for batching/high-throughput
  - selection criteria covering semantic transparency, copy count, instrumentation quality, failure reporting, batching fitness, and implementation complexity
- `docs/bridge_contract.md` also freezes the official sim artifact consumption strategy early. The default development/test artifact should be the packaged headless distribution from `fight-caves-RL` (for example `headlessDistZip`), with any jar/local-build fallback rules documented explicitly.
- `docs/performance_plan.md` and `configs/benchmark/official_profile_v0.yaml` define the official benchmark profile v0 early, including:
  - reward config
  - curriculum mode
  - replay mode
  - logging mode
  - dashboard mode
  - env-count ladder
  - manifest fields required for benchmark runs
- The repo has a clear hot-path map before transport work begins.

Risks / likely failure modes:

- Copying RL-local assumptions into docs instead of reflecting the current sim.
- Leaving version constants scattered across modules.
- Freezing the wrong episode-start-state contract or letting it drift from the headless sim initializer.
- Deferring bridge/artifact/benchmark decisions too long and forcing PR 3 onto a dead-end path.

PR 2 execution status (2026-03-08):

- [x] Read the current sim-side Step 5, Step 6, Step 7, and Step 10 artifacts directly from `/home/jordan/code/fight-caves-RL`.
- [x] Added RL-side integration/bridge/action/observation/hot-path/performance docs rooted in the verified sim contract.
- [x] Added `fight_caves_rl/envs/schema.py` as the single RL-side registry for observation schema, action schema, episode-start contract, bridge protocol, and official benchmark profile identities.
- [x] Froze the default sim artifact boundary to the packaged headless distribution from `:game:headlessDistZip`, with `:game:packageHeadless` as the documented build/validation fallback.
- [x] Selected a concrete PR 3 Mode A direction:
  - embedded JVM direct-runtime bridge
  - packaged headless distribution as the input artifact
  - `HeadlessMain.bootstrap(...)` as the runtime entrypoint
  - player provisioning aligned to the sim's headless test-support path
- [x] Froze the provisional Mode B/C direction around a dedicated batched subprocess bridge and lower-copy vector backend path.
- [x] Expanded `configs/benchmark/official_profile_v0.yaml` into a real benchmark profile contract with env ladder and required manifest fields.
- [x] Added PR 2 unit coverage for:
  - contract version registry
  - required docs presence
  - episode-start contract registry
  - official benchmark profile registry/config consistency
- [x] Re-ran RL unit tests after PR 2 contract work: `13 passed`.

### PR 3 - Correctness Wrapper Bring-Up

Goal:

- Build the first correctness-first Python wrapper around the headless sim, with exact reset/step/close handling and no hot-path optimization yet, using the bridge and artifact strategy frozen in PR 2.

Expected files/directories:

- `/home/jordan/code/RL/fight_caves_rl/bridge/contracts.py`
- `/home/jordan/code/RL/fight_caves_rl/bridge/launcher.py`
- `/home/jordan/code/RL/fight_caves_rl/bridge/debug_client.py`
- `/home/jordan/code/RL/fight_caves_rl/bridge/errors.py`
- `/home/jordan/code/RL/fight_caves_rl/envs/action_mapping.py`
- `/home/jordan/code/RL/fight_caves_rl/envs/observation_mapping.py`
- `/home/jordan/code/RL/fight_caves_rl/envs/correctness_env.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/integration/test_wrapper_reset_matches_sim_contract.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/integration/test_wrapper_step_matches_sim_trace.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/unit/test_action_schema_version_compatibility.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/unit/test_observation_flattening_determinism.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/unit/test_bridge_launcher_preflight.py`
- `/home/jordan/code/RL/scripts/smoke_random.py`

Dependencies:

- PR 1.
- PR 2.
- The PR 2-selected official sim artifact strategy.
- The PR 2-selected Mode A bridge strategy.
- A runnable headless sim entrypoint or packaged artifact from `/home/jordan/code/fight-caves-RL`.

Tests to add:

- `WrapperResetMatchesSimContractTest` equivalent.
- `WrapperStepMatchesSimTraceTest` equivalent.
- Observation dtype/shape correctness test.
- Action mapping compatibility test.
- Random-policy episode smoke.

Acceptance criteria:

- Reset/step/close execute without wrapper ambiguity.
- Terminal reasons, rejection metadata, and action application metadata pass through cleanly.
- Observation flattening is deterministic and versioned.
- Reset behavior explicitly matches the PR 2 episode-start-state contract.
- A random policy can run full episodes through the correctness wrapper.

Risks / likely failure modes:

- Bridging to the sim using a surface that is too coupled to debug output and too weak for later batching.
- Accidentally reinterpreting action semantics in the wrapper.
- Observation flattening drift between debug and train modes.
- Wrapper reset behavior drifting from the frozen episode-start-state contract.
- The packaged headless artifact still depends on the checked-out sim workspace plus `data/cache/main_file_cache.dat2`, so live PR 3 acceptance can remain blocked by missing workspace prerequisites even when the Python bridge code is correct.

PR 3 execution status (2026-03-08):

- [x] Added `jpype1` to the RL dependency baseline for the embedded JVM bridge path.
- [x] Implemented `fight_caves_rl/bridge/contracts.py`.
- [x] Implemented `fight_caves_rl/bridge/launcher.py` with dist-glob discovery, zip extraction, workspace preflight, and handshake generation.
- [x] Implemented `fight_caves_rl/bridge/debug_client.py` for correctness-mode runtime bootstrap, player provisioning, reset, observe, action application, and single-step calls.
- [x] Implemented `fight_caves_rl/envs/action_mapping.py`.
- [x] Implemented `fight_caves_rl/envs/observation_mapping.py`.
- [x] Implemented `fight_caves_rl/envs/correctness_env.py`.
- [x] Added PR3 unit coverage for action normalization, observation flattening determinism, and bridge launcher preflight.
- [x] Added PR3 integration tests for reset/step contract alignment; they currently skip cleanly until the sim cache prerequisite is restored in the workspace.
- [x] Added `scripts/smoke_random.py` with fail-fast runtime preflight.
- [x] Corrected the sim artifact assumption from a single fixed zip path to the verified `fight-caves-headless*.zip` distribution glob.
- [ ] Restore `/home/jordan/code/fight-caves-RL/data/cache/main_file_cache.dat2` and rerun PR3 live integration acceptance.
- [ ] Verify live terminal-reason handling against the real sim runtime and remove the provisional inferred-only note if the selected Mode A surface is sufficient.
- [ ] Run a full random-policy episode through `scripts/smoke_random.py` after the sim cache is restored.

### PR 4 - Determinism, Equivalence Validation, and Early Parity Canaries

Goal:

- Prove that the wrapper is semantically transparent relative to the headless sim on fixed seeds and fixed traces, and introduce the first thin RL-side parity canaries early enough to catch drift before training complexity increases.

Expected files/directories:

- `/home/jordan/code/RL/docs/eval_and_replay.md`
- `/home/jordan/code/RL/docs/parity_canaries.md`
- `/home/jordan/code/RL/fight_caves_rl/replay/seed_packs.py`
- `/home/jordan/code/RL/fight_caves_rl/replay/trace_packs.py`
- `/home/jordan/code/RL/fight_caves_rl/utils/seeding.py`
- `/home/jordan/code/RL/configs/eval/eval_seedpack_v0.yaml`
- `/home/jordan/code/RL/configs/eval/parity_canary_v0.yaml`
- `/home/jordan/code/RL/fight_caves_rl/tests/determinism/test_fixed_seed_reset_reproducibility.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/determinism/test_wrapper_vs_raw_sim_trajectory_agreement.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/determinism/test_deterministic_eval_same_checkpoint_same_seed_pack.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/parity/test_parity_canary_smoke.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/parity/test_replay_to_trace_equivalence_smoke.py`

Dependencies:

- PR 3.
- Stable seed/trace input surfaces from the sim.

Tests to add:

- Fixed-seed reset reproducibility.
- Wrapper-vs-raw-sim trajectory agreement on fixed action traces.
- Deterministic eval equivalence for fixed checkpoint and seed packs.
- Early parity canary smoke.
- Replay-to-trace equivalence smoke.

Acceptance criteria:

- The wrapper shows no semantic drift on the determinism matrix required by the spec.
- Seed packs and trace packs are versioned and reproducible.
- Deterministic eval behavior is documented and testable.
- Early parity canaries exist and can run on thin, versioned trace/seed subsets before heavier full-canary workflows land later.

Risks / likely failure modes:

- Hidden wrapper-local caches contaminating determinism.
- Schema drift causing false deterministic mismatches.
- Seed pack definitions that are not portable across machines or commit SHAs.
- Waiting too long to introduce canaries and discovering wrapper drift only after training plumbing is layered on top.

### PR 5 - PufferLib Smoke Integration

Goal:

- Connect the correctness wrapper to the first minimal PufferLib training and evaluation flow.

Expected files/directories:

- `/home/jordan/code/RL/fight_caves_rl/puffer/factory.py`
- `/home/jordan/code/RL/fight_caves_rl/puffer/trainer.py`
- `/home/jordan/code/RL/fight_caves_rl/puffer/callbacks.py`
- `/home/jordan/code/RL/fight_caves_rl/policies/mlp.py`
- `/home/jordan/code/RL/fight_caves_rl/policies/checkpointing.py`
- `/home/jordan/code/RL/configs/train/smoke_ppo_v0.yaml`
- `/home/jordan/code/RL/configs/eval/replay_eval_v0.yaml`
- `/home/jordan/code/RL/scripts/train.py`
- `/home/jordan/code/RL/scripts/eval.py`
- `/home/jordan/code/RL/scripts/smoke_scripted.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/smoke/test_puffer_smoke_train_loop.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/smoke/test_checkpoint_save_load_smoke.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/smoke/test_eval_loop_smoke.py`

Dependencies:

- PR 3.
- PR 4.

Tests to add:

- Minimal PuffeRL train loop smoke.
- Checkpoint save/load smoke.
- Eval loop smoke.
- Scripted baseline smoke run.

Acceptance criteria:

- A minimal training loop completes using PufferLib.
- Checkpoints are saved and can be reloaded.
- Local dashboard output is optional and functional.
- Early parity canaries from PR 4 still pass after smoke-training integration is added.
- The first trainer path reuses `pufferlib.pufferl.PuffeRL` unless a missing contract requirement forces a documented wrapper around it.

Risks / likely failure modes:

- Early policy wiring depending on wrapper-only behavior that will not survive batching.
- PufferLib API mismatches at the pinned version.
- Checkpoint metadata missing schema/version information.

### PR 6 - W&B Integration and Run Manifests

Goal:

- Make every train/eval run observable, resumable, and reproducible through manifests and W&B.

Expected files/directories:

- `/home/jordan/code/RL/docs/wandb_logging_contract.md`
- `/home/jordan/code/RL/docs/run_manifest.md`
- `/home/jordan/code/RL/fight_caves_rl/logging/wandb_client.py`
- `/home/jordan/code/RL/fight_caves_rl/logging/metrics.py`
- `/home/jordan/code/RL/fight_caves_rl/logging/artifact_naming.py`
- `/home/jordan/code/RL/fight_caves_rl/manifests/run_manifest.py`
- `/home/jordan/code/RL/.env.example`
- `/home/jordan/code/RL/fight_caves_rl/tests/integration/test_wandb_run_manifest_completeness.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/integration/test_wandb_offline_smoke.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/unit/test_artifact_naming_versioning.py`

Dependencies:

- PR 5.

Tests to add:

- W&B dry-run/offline smoke.
- Manifest completeness test.
- Artifact naming/versioning test.

Acceptance criteria:

- Every train/eval run writes a complete manifest locally.
- W&B receives the required run metadata, core metric families, and artifact categories.
- Resume-safe metadata is recorded.
- Run manifests include the episode-start contract version, sim artifact metadata, bridge protocol version, and benchmark profile reference defined earlier in PR 2.
- Run manifests and W&B config record PufferLib distribution plus version, using distribution metadata rather than `pufferlib.__version__`.
- Prefer `pufferlib.pufferl.WandbLogger` as the baseline logger integration point; any RL-local logger wrapper should exist only to satisfy missing manifest/artifact requirements.

Risks / likely failure modes:

- Logging overhead leaking into the hot path too early.
- Incomplete commit/version metadata.
- Divergent local-manifest and W&B run metadata.

### PR 7 - Batched Bridge

Goal:

- Replace correctness-only wrapper crossings with a coarse-grained batched bridge suitable for production vectorization, building on the strategy frozen in PR 2 and the correctness evidence from PRs 3-6.

Expected files/directories:

- `/home/jordan/code/RL/fight_caves_rl/bridge/batch_client.py`
- `/home/jordan/code/RL/fight_caves_rl/bridge/protocol.py`
- `/home/jordan/code/RL/fight_caves_rl/bridge/buffers.py`
- `/home/jordan/code/RL/fight_caves_rl/benchmarks/bridge_bench.py`
- `/home/jordan/code/RL/configs/benchmark/bridge_1env_v0.yaml`
- `/home/jordan/code/RL/configs/benchmark/bridge_64env_v0.yaml`
- `/home/jordan/code/RL/scripts/benchmark_bridge.py`
- `/home/jordan/code/RL/docs/bridge_contract.md`
- `/home/jordan/code/RL/docs/hotpath_map.md`
- `/home/jordan/code/RL/fight_caves_rl/tests/integration/test_bridge_batch_step_parity.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/integration/test_bridge_schema_fail_fast.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/performance/test_bridge_benchmark_smoke.py`

Dependencies:

- PR 3.
- PR 4.
- PR 6.
- The PR 2-selected artifact strategy and batched/high-throughput bridge direction.
- A sim-side surface that can support batch stepping without changing semantics.

Tests to add:

- Batch-vs-sequential equivalence.
- Bridge batch stability test.
- Schema/version fail-fast behavior.
- Bridge benchmark smoke.

Acceptance criteria:

- Batched reset/step works correctly.
- Batched mode outperforms the correctness wrapper on the same benchmark profile.
- The bridge protocol is versioned and explicit about errors.
- The bridge/vector path reuses `pufferlib.emulation` and `pufferlib.vector` primitives where they fit the contract instead of duplicating generic flattening or vector-execution logic.
- The bridge implementation remains inside the guardrails frozen in PR 2 rather than introducing an incompatible late-stage transport shape.

Risks / likely failure modes:

- The chosen transport is too verbose or too allocation-heavy.
- Batch layout hides per-env failures instead of classifying them explicitly.
- Sim-side batching assumptions require upstream changes in `fight-caves-RL`.

### PR 8 - Vectorized PufferLib Backend

Goal:

- Connect the batched bridge to PufferLib's vectorized training path and stabilize multi-env execution.

Expected files/directories:

- `/home/jordan/code/RL/fight_caves_rl/envs/vector_env.py`
- `/home/jordan/code/RL/fight_caves_rl/puffer/factory.py`
- `/home/jordan/code/RL/fight_caves_rl/puffer/trainer.py`
- `/home/jordan/code/RL/configs/benchmark/vecenv_256env_v0.yaml`
- `/home/jordan/code/RL/configs/train/train_baseline_v0.yaml`
- `/home/jordan/code/RL/scripts/benchmark_env.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/smoke/test_vecenv_reset_step_smoke.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/smoke/test_multi_worker_smoke.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/smoke/test_long_run_vector_stability.py`

Dependencies:

- PR 5.
- PR 7.

Tests to add:

- VecEnv reset/step smoke.
- Multi-worker smoke.
- Long-run stability smoke.

Acceptance criteria:

- Vectorized env execution is stable and benchmarkable.
- Worker-aware indexing and deterministic slot seeding are implemented.
- The production training path no longer depends on one Python call per env per tick.
- The vectorized training path is built around `pufferlib.vector.make` or a thin compatible wrapper rather than a wholly separate rollout framework.
- The official benchmark profile v0 from PR 2 can be executed end-to-end through the vectorized path.

Risks / likely failure modes:

- Worker topology amplifies JVM startup or transport overhead.
- Async collection creates determinism ambiguity.
- Env slot indexing drifts from replay and artifact indexing.

### PR 9 - Reward and Curriculum System

Goal:

- Add versioned reward configs and curriculum scaffolding without contaminating simulator semantics or debug equivalence.

Expected files/directories:

- `/home/jordan/code/RL/docs/reward_configs.md`
- `/home/jordan/code/RL/configs/reward/reward_sparse_v0.yaml`
- `/home/jordan/code/RL/configs/reward/reward_shaped_v0.yaml`
- `/home/jordan/code/RL/configs/curriculum/curriculum_disabled_v0.yaml`
- `/home/jordan/code/RL/configs/curriculum/curriculum_wave_progression_v0.yaml`
- `/home/jordan/code/RL/fight_caves_rl/rewards/registry.py`
- `/home/jordan/code/RL/fight_caves_rl/rewards/reward_sparse_v0.py`
- `/home/jordan/code/RL/fight_caves_rl/rewards/reward_shaped_v0.py`
- `/home/jordan/code/RL/fight_caves_rl/curriculum/registry.py`
- `/home/jordan/code/RL/fight_caves_rl/curriculum/curriculum_disabled_v0.py`
- `/home/jordan/code/RL/fight_caves_rl/curriculum/curriculum_wave_progression_v0.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/unit/test_reward_reproducibility.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/unit/test_reward_no_future_leakage.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/unit/test_curriculum_config_loading.py`

Dependencies:

- PR 6.
- PR 8.

Tests to add:

- Reward reproducibility tests.
- No-future-leakage reward tests.
- Curriculum config loader tests.

Acceptance criteria:

- `reward_sparse_v0` and `reward_shaped_v0` exist and are versioned.
- Reward/curriculum versions are recorded in manifests and W&B.
- Training core does not need code changes to swap reward or curriculum configs.
- Reward/curriculum defaults remain compatible with the official benchmark profile and parity-sensitive canary modes.

Risks / likely failure modes:

- Reward shaping sneaking in fields that are unavailable in strict equivalence mode.
- Reward implementations depending on debug-only diagnostics.
- Curriculum introducing hidden distribution shift into deterministic eval.

### PR 10 - Replay and Eval Artifacts

Goal:

- Make checkpoint evaluation deterministic and artifact-rich, with replay packs suitable for debugging and future integrations.

Expected files/directories:

- `/home/jordan/code/RL/docs/eval_and_replay.md`
- `/home/jordan/code/RL/fight_caves_rl/replay/replay_export.py`
- `/home/jordan/code/RL/fight_caves_rl/replay/replay_index.py`
- `/home/jordan/code/RL/fight_caves_rl/replay/eval_runner.py`
- `/home/jordan/code/RL/scripts/replay_eval.py`
- `/home/jordan/code/RL/configs/eval/replay_eval_v0.yaml`
- `/home/jordan/code/RL/fight_caves_rl/tests/integration/test_replay_generation_smoke.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/integration/test_replay_manifest_integrity.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/determinism/test_replay_eval_equivalence.py`

Dependencies:

- PR 5.
- PR 6.
- PR 8.
- PR 9.

Tests to add:

- Replay generation smoke.
- Replay manifest integrity.
- Deterministic replay-eval equivalence.

Acceptance criteria:

- Checkpoints can be evaluated on fixed seed packs.
- Replay artifacts are generated deterministically for fixed checkpoint plus seed pack.
- Replay indexing is usable for W&B artifact linking and manual debugging.
- Replay/eval flows continue to support the thin canary path introduced in PR 4.

Risks / likely failure modes:

- Replay exports becoming too heavy for routine evaluation cadence.
- Divergence between eval manifests and replay artifact metadata.
- Confusion between training-time logs and replay-grade artifacts.

### PR 11 - Performance Hardening

Goal:

- Benchmark every layer independently, remove avoidable overhead, and document the path toward the final SPS target.

Expected files/directories:

- `/home/jordan/code/RL/docs/performance_plan.md`
- `/home/jordan/code/RL/docs/hotpath_map.md`
- `/home/jordan/code/RL/configs/benchmark/train_1024env_v0.yaml`
- `/home/jordan/code/RL/fight_caves_rl/benchmarks/env_bench.py`
- `/home/jordan/code/RL/fight_caves_rl/benchmarks/train_bench.py`
- `/home/jordan/code/RL/scripts/benchmark_env.py`
- `/home/jordan/code/RL/scripts/benchmark_bridge.py`
- `/home/jordan/code/RL/.github/workflows/benchmarks.yml`
- `/home/jordan/code/RL/fight_caves_rl/tests/performance/test_env_benchmark_smoke.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/performance/test_train_benchmark_smoke.py`

Dependencies:

- PR 7.
- PR 8.
- PR 10.

Tests to add:

- Bridge microbenchmark smoke.
- Env SPS benchmark smoke.
- Training SPS benchmark smoke.

Acceptance criteria:

- Benchmarks isolate raw sim, bridge, wrapper, VecEnv, full training, and W&B overhead.
- Benchmark profiles exist for `1`, `16`, `64`, `256`, and `1024` envs where hardware allows.
- Benchmark manifests record hardware, logging mode, bridge mode, reward config, and replay mode.
- The repo shows credible progress toward `>= 1,000,000 env steps/sec`.
- Performance reporting extends and refines the benchmark profile that was already frozen in PR 2 rather than inventing a new late-stage benchmark definition.

Risks / likely failure modes:

- Benchmark claims that change semantics or skip required work.
- Logging, replay, or artifact generation quietly dominating SPS.
- Tuning effort focusing on trainer internals before boundary crossings and copies are fixed.

### PR 12 - Expanded Parity Canaries and Oracle-Reference Validation

Goal:

- Expand the earlier RL-side parity canaries into a broader oracle-reference validation layer that proves the wrapper, replay path, and eval path are not introducing semantic drift relative to the sim and oracle.

Expected files/directories:

- `/home/jordan/code/RL/docs/parity_canaries.md`
- `/home/jordan/code/RL/configs/eval/parity_canary_v0.yaml`
- `/home/jordan/code/RL/fight_caves_rl/replay/trace_packs.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/parity/test_parity_canary_smoke.py`
- `/home/jordan/code/RL/fight_caves_rl/tests/parity/test_replay_to_trace_equivalence_smoke.py`

Dependencies:

- PR 4.
- PR 10.
- PR 11.
- Versioned seed/trace packs from sim/oracle validation workflows.

Tests to add:

- Expanded parity canary smoke.
- Replay-to-trace equivalence smoke.

Acceptance criteria:

- RL integration does not mask or introduce semantic drift.
- Canary runs use versioned seed/trace references and produce inspectable outputs.
- The oracle remains separate from the production training path.
- The early parity canary path introduced in PR 4 is expanded rather than replaced, so drift is checked continuously from early integration through late validation.

Risks / likely failure modes:

- Canary packs drifting from the actual sim/oracle validation packs.
- Debug-only wrapper features contaminating parity mode.
- Treating infrastructure failures as episode outcomes in parity runs.

### PR 13 - MVP Acceptance Gate

Goal:

- Execute the full RL acceptance gate and close the repo at MVP-complete integration status.

Expected files/directories:

- `/home/jordan/code/RL/README.md`
- `/home/jordan/code/RL/docs/performance_plan.md`
- `/home/jordan/code/RL/docs/wandb_logging_contract.md`
- `/home/jordan/code/RL/docs/parity_canaries.md`
- `/home/jordan/code/RL/.github/workflows/ci.yml`
- `/home/jordan/code/RL/.github/workflows/benchmarks.yml`
- `/home/jordan/code/RL/fight_caves_rl/tests/`
- `/home/jordan/code/RL/configs/`

Dependencies:

- PR 1 through PR 12.

Tests to add:

- Full RL-side CI suite.
- Deterministic eval smoke on a release-candidate checkpoint.
- End-to-end train/eval/checkpoint/replay flow smoke.

Acceptance criteria:

- Repo installs reproducibly from lockfiles.
- Correctness wrapper matches the headless sim contract.
- PuffeRL training/eval runs complete.
- W&B emits the required metrics and artifacts.
- Checkpoints resume and evaluate deterministically.
- Replay artifacts are generated successfully.
- Batch/vector integration is implemented and benchmarked.
- RL-side parity canaries pass.
- Benchmarking documents progress toward `>= 1,000,000 env steps/sec`.

Risks / likely failure modes:

- Last-minute contract mismatches between manifests, bridge versions, and schema versions.
- Acceptance running on hardware that differs from the documented benchmark profile.
- Treating benchmark progress as sufficient without proving correctness and reproducibility.

## 4. Bridge / Integration Plan

The bridge plan should follow the three-mode progression already required by the spec and the current sim runtime surfaces, but the strategy must be pulled forward early enough that the wrapper is not built on a dead-end transport.

1. Artifact dependency model
   - RL should consume `fight-caves-RL` as a sibling-repo dependency and treat its packaged headless runtime as the authoritative runtime artifact.
   - The preferred artifact boundary is the sim's packaged headless distribution from `/home/jordan/code/fight-caves-RL` (for example `headlessDistZip`) or an equivalent reproducible local build output explicitly blessed in PR 2.
   - RL should record the exact sim commit SHA and artifact/version metadata in every manifest.
   - Artifact strategy must be frozen in PR 2, not deferred until wrapper implementation.

2. Early bridge strategy decision point
   - PR 2 must compare bridge candidate solutions early and choose a concrete Mode A path plus a provisional Mode B/C direction.
   - Candidate solution families the agent should evaluate include:
     - subprocess launch with a debug-friendly request/response channel for correctness mode
     - local binary IPC over socket/pipe for batched transport
     - shared-memory/direct-buffer style transport with a lightweight control channel for high throughput
     - any direct in-process/native bridge only if it preserves debugging, reproducibility, and implementation simplicity well enough to justify the added complexity
   - Selection criteria must include:
     - semantic transparency to the sim contract
     - ease of fail-fast version/schema validation
     - suitability for batched stepping
     - copy count / allocation pressure
     - instrumentation and debugging quality
     - operational complexity on Linux/WSL
   - The intent is not to over-engineer PR 2, but to prevent PR 3 from baking in a transport shape that blocks later batching.

3. Process topology
   - One Python worker should manage one JVM runtime process in production-oriented modes.
   - Each JVM process should own many active Fight Caves episodes so one Python call can submit a batch of actions and receive a batch of observations/results.
   - RSPS should remain out-of-band and only participate in parity/debug workflows, never in the rollout hot path.

4. Startup handshake
   - On startup, Python must validate bridge protocol version, observation schema ID/version, action schema ID/version, episode-start contract version, benchmark profile reference where applicable, and sim commit/build metadata.
   - Any mismatch must fail fast before training begins.

5. Mode A: correctness wrapper
   - Use a debug-friendly launch path with explicit reset/step/close sequencing and rich failure reporting.
   - Favor traceability over throughput.
   - Use this mode for wrapper correctness, determinism checks, and smoke training only.

6. Mode B: batched bridge
   - Replace per-env stepping with batched reset/step calls.
   - Batch observations and results into explicit numeric buffers with deterministic layout.
   - Keep diagnostics separate from the hot-path result struct so normal training does not pay debug costs.

7. Mode C: high-throughput vector backend
   - Plug the batched bridge into a vectorized PufferLib backend.
   - Minimize Python/JVM crossings, memory copies, schema lookups, and per-step object churn.
   - Prefer flat numeric buffers and a low-copy transport shape suitable for high env counts.

8. Deterministic replay and parity use
   - Deterministic eval should use the same bridge stack, but with fixed seed packs and replay-focused artifact capture.
   - Early parity canaries should reuse the same wrapper/bridge path as soon as PR 4 lands so correctness is measured on the real integration boundary.
   - Later oracle-reference canaries should expand on the same path instead of introducing a separate validation-only wrapper.

## 5. Testing Plan

Testing should be layered exactly as required by the spec and should mature with the PR plan.

1. Wrapper correctness
   - Validate reset/step/close behavior against the headless sim contract.
   - Verify action mapping, rejection reasons, terminal reasons, and observation shape/dtype handling.
   - Keep strict equivalence tests separate from performance tests.
   - Include explicit reset-state validation against the frozen episode-start-state contract.

2. Deterministic replay equivalence
   - Use fixed seed packs and fixed action traces.
   - Compare wrapper trajectories against raw headless sim trajectories.
   - Verify fixed checkpoint plus fixed seed pack generates identical eval summaries and replay artifacts.

3. Parity canaries
   - Introduce thin RL-side canaries in PR 4 on versioned trace/seed subsets that map back to sim/oracle expectations.
   - Expand those canaries later once replay/eval/performance plumbing exists.
   - Ensure RL-local features and reward logic are disabled or isolated when parity correctness is being measured.

4. Batch / vector env validation
   - Compare batch vs sequential equivalence before claiming throughput.
   - Add VecEnv reset/step smoke, multi-worker smoke, and long-run stability smoke.
   - Keep worker-slot seeding and episode indexing under direct test coverage.

5. Training smoke tests
   - Run random-policy and scripted-policy smoke episodes.
   - Run a minimal PuffeRL train/eval loop.
   - Verify checkpoint save/load and deterministic eval after restore.

6. CI execution model
   - Per-PR CI should run unit tests, integration tests, smoke training tests, deterministic eval smoke, and lint/type checks if enabled.
   - Heavy benchmarks should live in a scheduled or manually triggered workflow.
   - Parity canaries should begin as early smoke coverage once PR 4 lands and can expand into heavier/nightly jobs later if they depend on larger external packs.

## 6. Performance Plan

The throughput plan should follow staged gates that match the spec's performance ladder and benchmark breakdown requirements.

### Official Benchmark Profile v0 (frozen in PR 2)

- The official benchmark profile must be defined early in `configs/benchmark/official_profile_v0.yaml` and documented in `docs/performance_plan.md`.
- The profile should lock, at minimum:
  - benchmark mode / bridge mode
  - reward config
  - curriculum mode
  - replay mode
  - W&B logging mode
  - local dashboard mode
  - env-count ladder
  - required manifest metadata for benchmark runs
- If a permanent benchmark host is not yet fixed, the actual machine profile used for each benchmark must still be recorded in the manifest and treated as part of the benchmark identity.

### Stage Gate A - Correctness baseline

- Measure single-env wrapper throughput and reset/step latency.
- Benchmark profiles: `1 env`, `16 envs`.
- Exit condition: baselines are recorded and reproducible; no performance claims yet.

### Stage Gate B - Batched bridge baseline

- Measure raw bridge throughput and compare batch vs sequential equivalence on the same hardware/config.
- Benchmark profiles: `16 envs`, `64 envs`.
- Exit condition: batched bridge is faster than the correctness wrapper and remains semantically equivalent.

### Stage Gate C - Vectorized env baseline

- Measure VecEnv throughput and stability under realistic worker/env-per-worker settings.
- Benchmark profiles: `64 envs`, `256 envs`.
- Exit condition: vectorized execution is stable, benchmarked, and clearly better than the non-vectorized path.

### Stage Gate D - Full training baseline

- Measure end-to-end training SPS with normal logging, minimized logging, replay disabled, and replay enabled periodically.
- Benchmark profiles: `64 envs`, `256 envs`, `1024 envs` if hardware allows.
- Exit condition: benchmark manifests isolate trainer, bridge, wrapper, and W&B overhead.

### Stage Gate E - Tuned production path

- Tune boundary crossings, memory copies, observation packing, worker topology, logging cadence, and policy inference overhead in that order.
- Benchmark profiles: official tuned profile plus comparison profiles for sync vs async and logging/replay variants.
- Exit condition: the repo either reaches `>= 1,000,000 env steps/sec` on the defined benchmark profile or documents the remaining blocking bottlenecks with verified measurements.

Mandatory benchmark breakdowns across the stages:

- Raw sim batch stepping rate.
- Python bridge throughput.
- Wrapper overhead.
- PufferLib VecEnv throughput.
- End-to-end training SPS.
- W&B overhead at normal logging cadence.
- W&B overhead at aggressive logging cadence.

## 7. W&B + PufferLib Analytics / Dashboard Plan

1. W&B run initialization
   - Every training and evaluation run should start with project, run name, tags, run group, RL commit, sim commit, hardware profile, reward config ID, curriculum config ID, observation schema ID/version, action schema ID/version, episode-start contract version, benchmark profile ID/version, and the canonical PufferLib distribution/version pair.

2. Required metric families
   - Throughput: `env_steps_total`, `env_steps_per_sec`, `learner_updates_per_sec`, `batch_collection_time_ms`, `train_time_ms`, `bridge_time_ms`, `obs_pack_time_ms`, `action_unpack_time_ms`.
   - Training quality: `episode_return_mean`, `episode_return_max`, `episode_length_mean`, `value_loss`, `policy_loss`, `entropy`, `explained_variance`, `clip_fraction` or equivalent.
   - Environment/game progress: `wave_reached_mean`, `jad_reach_rate`, `completion_rate`, `death_rate`, `invalid_action_rate`, `rejection_rate_by_reason`, `prayer_potion_used_mean`, `shark_used_mean`, `truncation_rate`.
   - Stability: `reset_failures`, `bridge_failures`, `env_crashes`, `replay_generation_failures`, `parity_canary_failures`.

3. Artifact logging
   - Log checkpoints, eval summaries, replay packs, benchmark reports, and config manifests as first-class W&B artifacts or artifact-linked outputs.
   - Keep replay-grade artifacts out of the per-step hot path and generate them on configured cadence.

4. Run manifests
   - Persist a local run manifest for every run.
   - Keep the local manifest and W&B config payload aligned.
   - Include reward/curriculum versions, bridge protocol version, replay schema version, seed policy, official benchmark profile reference, episode-start contract version, sim artifact metadata, and the canonical PufferLib distribution/version pair.

5. PufferLib dashboard alignment
   - Local dashboard printing should be optional and config-driven.
   - Dashboard summaries should mirror the W&B metric families closely enough that local and remote monitoring agree.
   - Dashboard output must not become a hot-path bottleneck.

6. Offline and resume support
   - W&B offline/dry-run mode should be supported for smoke tests and constrained environments.
   - Resume-safe manifest and checkpoint metadata should allow interrupted runs to restart without ambiguity.

## 8. Open Assumptions and Ambiguities to Resolve Before Coding

1. RL repo bootstrap
   - Resolved in PR 1.
   - `/home/jordan/code/RL` now contains the repo skeleton, baseline configs, package scaffolding, CI, and bootstrap docs.

2. Cross-module doc filename alignment outside RL
   - Resolved in PR 1.
   - RL naming is fixed to `RLspec.md` / `RLplan.md`.
   - The sibling module roots now align to `FCspec.md` / `FCplan.md` and `RSPSspec.md` / `RSPSplan.md`.

3. Python baseline
   - Resolved in PR 1.
   - RL standardizes on Python `3.11` via workspace-local `uv` provisioning and locked dependencies.

4. Sim artifact consumption strategy
   - Resolved in PR 2.
   - RL now treats the packaged headless distribution from `/home/jordan/code/fight-caves-RL` as the canonical dev/test artifact boundary.
   - `:game:headlessDistZip` is the default artifact task, with `:game:packageHeadless` as the documented build/validation fallback.

5. Final bridge transport mechanism
   - Partially resolved in PR 2.
   - PR 2 froze Mode A to the embedded-JVM direct-runtime path and froze the provisional Mode B/C direction around a batched subprocess bridge with lower-copy payloads.
   - The exact production transport implementation may still evolve after empirical testing, but it must stay inside the coarse-grained, low-copy, non-JSON hot-path constraints now frozen by PR 2.

6. Sim-side batching surface
   - The current sim repo already exposes runtime reset/action/observe APIs and a batch stepping helper, but the exact multi-episode interface needed by RL may require explicit sim-side support. Any such dependency must be tracked as an external prerequisite on the sim repo, not hidden inside RL.

7. Action-space encoding
   - Partially resolved in PR 2.
   - `docs/action_mapping.md` now freezes the sim-aligned action ids, parameters, and rejection reasons.
   - The final PufferLib/Gymnasium space encoding still needs implementation in PR 3/PR 5, but it must stay inside the PR 2 contract.

8. Reward shaping details
   - `reward_sparse_v0` and `reward_shaped_v0` are required, but the exact shaped terms, coefficients, and default curriculum behavior are not fully specified. These need to be explicitly documented before training comparisons begin.

9. Seed/trace/parity pack ownership
   - The source, storage location, and versioning policy for seed packs, trace packs, and oracle-reference parity packs must be nailed down so RL does not invent its own drifting copies.

10. W&B operational settings
   - Project name, entity/team, artifact retention policy, and default online/offline behavior are operational details that should be fixed before PR 6 lands.

11. Official benchmark hardware/profile
   - Partially resolved in PR 2.
   - Benchmark profile v0 is now frozen in RL config/docs.
   - If a permanent benchmark host is not yet selected, benchmark manifests must still record the exact machine profile so `>= 1,000,000 env steps/sec` claims remain comparable and auditable.

12. WSL toolchain baseline
   - `scripts/bootstrap_wsl_toolchain.sh` remains available for legacy source-build comparisons and future native dependencies, but it is no longer required for the standard RL train-group bootstrap after the move to `pufferlib-core==3.0.17`.
   - If a future RL dependency reintroduces source builds, re-evaluate whether the workspace-local sysroot remains the right fallback path for this workspace.

13. Permanent PufferLib package choice (resolved 2026-03-08)
   - RL now standardizes on `pufferlib-core==3.0.17` as the baseline distribution, imported as `pufferlib`.
   - Decision basis: wheel-backed install path, required RL-facing surfaces present, compiled `_C` available, no import-time `resources` symlink side effect, and a materially smaller dependency footprint than `pufferlib==3.0.0`.
   - Known upstream drift to carry forward:
     - official docs still say `pip install pufferlib`
     - the installed `pufferlib-core==3.0.17` distribution currently imports with `pufferlib.__version__ == "3.0.3"`
     - RL manifests must therefore capture distribution metadata instead of trusting the imported version string

14. RL Git/repo initialization (resolved 2026-03-07)
   - `/home/jordan/code/RL` is now a Git-backed repository in this workspace.
   - The canonical RL remote is `git@github.com:jordanbailey00/RL.git`.
   - RL branch cleanliness and push workflows are now available; later manifest code still needs to record the RL commit SHA at runtime.
