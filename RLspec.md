# RuneScape RL - PufferLib Module Spec (Source of Truth)

## 0) Purpose

Build the **RL/training module** for the RuneScape Fight Caves project as a separate repository that integrates with the **headless Fight Caves sim** and references the **headed RSPS oracle** for parity and debugging.

This spec defines the complete source of truth for the RL-side system:
- what the RL module owns
- what it must not own
- how it integrates with the headless sim
- how it references the headed RSPS/oracle
- how training, evaluation, benchmarking, replay, and analytics must work
- what must be delivered before the RL module is considered MVP-complete

This is a strict spec:
- the **headless sim is the source of truth for environment semantics**
- the **headed RSPS remains the oracle/reference implementation**
- the RL module may optimize throughput, batching, logging, storage, and training infrastructure, but it **must not silently change game semantics**
- if this spec conflicts with the headless sim spec or verified implementation, **the verified sim contract wins** and this spec must be updated

Primary acceptance target:
- a fully functioning RL module that can train against the headless Fight Caves sim through PufferLib, with deterministic evaluation, full experiment analytics, replay artifacts, parity canaries, and a performance path designed to reach **>= 1,000,000 env steps/sec minimum** under the defined performance configuration

---

## 1) Repository Reality and Workspace Layout

The workspace is intentionally multi-repo. The repositories are peers, not nested ownership domains.

### 1.1 Workspace layout

```text
\wsl$\Ubuntu\home\jordan\code
  fight-caves-RL/          # headless simulator (golden runtime)
  RSPS/                    # headed/oracle RSPS
  RL/                      # RL module (this spec)
```

### 1.2 Ownership boundaries

#### `fight-caves-RL/` owns:
- episode reset semantics
- action semantics
- observation semantics
- deterministic replay semantics
- batch stepping semantics
- parity harness against oracle
- simulator packaging/builds
- simulator-side benchmark harnesses

#### `RSPS/` owns:
- headed/oracle runtime behavior
- oracle parity reference
- headed debugging and manual validation
- source-of-truth gameplay behavior where parity disputes arise

#### `RL/` owns:
- Python training stack
- PufferLib integration
- Python<->sim bridge glue
- vectorized env adapters
- reward config and curriculum config
- training/eval/checkpoint pipelines
- replay artifact indexing on RL side
- experiment logging and dashboards
- throughput benchmarking for end-to-end RL rollouts
- CI for RL integration correctness

### 1.3 Dependency direction

Dependency direction is one-way:
- `RL` depends on `fight-caves-RL`
- `fight-caves-RL` does **not** depend on `RL`
- `RSPS` is referenced by parity/debug flows and remains external to RL runtime hot path

The RL repo must never duplicate core Fight Caves semantics that properly belong in the sim.

---

## 2) Version and Tooling Contract

### 2.1 PufferLib baseline

The RL module must target **`pufferlib-core==3.0.17`** as the current baseline distribution.
At runtime, that distribution imports under the **`pufferlib`** namespace.

Current project state as of 2026-03-08:
- the live RL bootstrap in `/home/jordan/code/RL` is pinned to `pufferlib-core==3.0.17`
- that distribution ships wheels, preserves the `pufferlib` import namespace, and includes the required RL-facing surfaces currently planned for this repo:
  - `pufferlib.pufferl.PuffeRL`
  - `pufferlib.pufferl.WandbLogger`
  - `pufferlib.vector.make`
  - `pufferlib.emulation`
  - compiled `pufferlib._C`
- the older `pufferlib==3.0.0` package remains the legacy source-only reference path, but it is no longer the RL baseline because it drags Ocean/source-build overhead into standard installs and creates an import-time `resources` symlink in the current working directory
- official upstream docs still recommend `pip install pufferlib`
- the installed `pufferlib-core==3.0.17` distribution currently imports with `pufferlib.__version__ == "3.0.3"`, so RL manifests must record distribution metadata instead of trusting the imported version string

Rules:
- implementation may move to a newer PufferLib release only if the agent verifies the newer release and updates this spec + compatibility notes
- the repo must pin a concrete PufferLib distribution and version in lockfiles
- the repo must record the PufferLib distribution and version in every run manifest and W&B run config

### 2.2 Python baseline

Preferred Python baseline:
- Python 3.11

Allowed:
- Python 3.12 if verified compatible with all training dependencies

The repo must use:
- `uv` for environment and dependency management
- lockfile-based dependency pinning
- reproducible local setup on Linux/WSL first

### 2.3 Core dependencies

Required baseline dependencies:
- `pufferlib-core` (runtime import namespace: `pufferlib`)
- `torch`
- `wandb`
- `numpy`
- `pytest`
- `hypothesis`
- `pyyaml` or equivalent config parsing library
- serialization/manifest utilities as needed

Optional but likely:
- `rich`
- `orjson`
- `msgpack`
- `pyarrow`
- `tensorboard` only if explicitly desired for local debugging (W&B remains primary)

### 2.4 Platform baseline

Primary development/runtime platform:
- Linux / WSL

Primary performance validation platform:
- native Linux preferred

Windows-native support is not a performance target and must not drive architecture decisions.

---

## 3) Non-Negotiable RL-Side Contracts

### 3.1 Sim authority contract

The headless sim is authoritative for:
- reset behavior
- tick semantics
- action application rules
- target mapping rules
- lockout/timer behavior
- observation contents and ordering
- done/termination conditions
- deterministic replay semantics

The RL module may:
- flatten or repackage observations
- batch actions
- batch episodes
- vectorize stepping
- cache static metadata
- optimize transport and logging

The RL module may **not**:
- reinterpret action meaning
- alter reward-relevant state transitions in the sim
- synthesize future-leakage fields not explicitly allowed
- patch semantic mismatches by changing wrapper behavior to hide sim/oracle divergence

### 3.2 Oracle reference contract

The headed RSPS remains the oracle reference for:
- parity disputes
- intermittent manual debugging
- validation of new trace packs
- investigation of rare mismatches or mechanic regressions

The RL runtime hot path must not depend on the headed RSPS.

### 3.3 No semantic drift through performance work

Any throughput optimization that changes simulator semantics is rejected.

Performance work may optimize:
- allocation patterns
- batching
- memory copies
- process topology
- serialization format
- logging rate
- checkpoint cadence
- evaluation cadence

But performance work may not alter:
- episode state transitions
- tick order assumptions exported by the sim
- observation meanings
- action meanings
- deterministic evaluation outcomes for the same seed + trace

### 3.4 Reproducibility contract

Every run must be reproducible from recorded metadata.

At minimum, each run must record:
- RL repo commit SHA
- headless sim commit SHA
- headed RSPS commit SHA if relevant to parity pack used
- PufferLib version
- Python version
- Torch version
- W&B version
- observation schema ID/version
- action schema ID/version
- reward config ID/version
- curriculum config ID/version
- replay schema version
- bridge protocol version
- seed policy
- benchmark profile / hardware profile

Current workspace note as of 2026-03-08:
- `/home/jordan/code/RL` is now a Git-backed repository in this workspace snapshot
- the canonical RL remote is `git@github.com:jordanbailey00/RL.git`
- RL-side commit SHA capture is no longer blocked by repo initialization, though later manifests still need code-path implementation to record it consistently

---

## 4) Product Goal for This Repo

The RL module’s product is:
- training
- evaluation
- analytics
- checkpointing
- replay artifacts
- deterministic debugging support
- performance benchmarking

It is **not** the simulator itself.

The RL repo is complete when:
1. it can train policies against the headless Fight Caves sim through PufferLib
2. it can evaluate checkpoints deterministically on fixed trace/seed packs
3. it logs full run metadata, metrics, artifacts, and replay assets to W&B
4. it supports terminal-local monitoring via the PufferLib dashboard path
5. it maintains integration correctness against the headless sim contract
6. it contains a performance path toward >= 1,000,000 env steps/sec minimum

---

## 5) Repository Structure Contract

The RL repo must be organized so the environment bridge, training core, analytics, configs, and tests are clearly separated.

Required top-level layout:

```text
RL/
  RLspec.md
  README.md
  pyproject.toml
  uv.lock
  .env.example
  configs/
    train/
    eval/
    sweep/
    reward/
    curriculum/
    benchmark/
  docs/
  scripts/
  fight_caves_rl/
    bridge/
    envs/
    puffer/
    policies/
    rewards/
    curriculum/
    replay/
    logging/
    manifests/
    benchmarks/
    utils/
    tests/
```

### 5.1 Required docs in `docs/`

Required docs:
- `docs/rl_integration_contract.md`
- `docs/bridge_contract.md`
- `docs/hotpath_map.md`
- `docs/observation_mapping.md`
- `docs/action_mapping.md`
- `docs/reward_configs.md`
- `docs/eval_and_replay.md`
- `docs/performance_plan.md`
- `docs/wandb_logging_contract.md`
- `docs/parity_canaries.md`
- `docs/run_manifest.md`

### 5.2 Required code ownership areas

#### `fight_caves_rl/bridge/`
Owns Python<->sim transport and protocol handling.

#### `fight_caves_rl/envs/`
Owns correctness wrappers and vectorized env adapters.

#### `fight_caves_rl/puffer/`
Owns PufferLib-specific trainer/env/policy wiring.

#### `fight_caves_rl/policies/`
Owns model definitions and policy loading.

#### `fight_caves_rl/rewards/`
Owns reward definitions/configurable shaping.

#### `fight_caves_rl/curriculum/`
Owns curriculum schedules and progression logic.

#### `fight_caves_rl/replay/`
Owns RL-side replay packaging, indexing, and eval helpers.

#### `fight_caves_rl/logging/`
Owns W&B integration, run manifests, artifact naming, and structured logs.

#### `fight_caves_rl/benchmarks/`
Owns RL-side SPS benchmarks and transport microbenchmarks.

#### `fight_caves_rl/tests/`
Owns integration, determinism, regression, and smoke tests.

---

## 6) Integration Architecture Contract

### 6.1 Integration modes

The RL repo must support three integration modes in order.

#### Mode A: correctness wrapper
Purpose:
- prove contract correctness
- support smoke training
- validate action/observation/done handling

Characteristics:
- may be slower
- favors clarity and debuggability
- used for deterministic equivalence tests and early bring-up
- uses an embedded JVM bridge from Python, with `jpype1` as the baseline candidate

#### Mode B: batched bridge
Purpose:
- reduce boundary overhead
- step many episodes per bridge call
- prepare for vectorized PufferLib rollouts

Characteristics:
- actions passed in batches
- observations returned in contiguous batched buffers
- minimal per-step object allocation

#### Mode C: high-throughput vector backend
Purpose:
- production training path
- maximize SPS
- support async or quasi-async rollout collection as appropriate

Characteristics:
- many envs per worker
- coarse-grained Python<->sim crossings
- direct use of PufferLib vectorization features
- zero- or low-copy observation handling wherever feasible

### 6.2 Mandatory architectural rule

The final high-throughput path must **not** rely on one Python call per env per tick.

The intended throughput architecture is:
- one worker controls many sim episodes
- one batched action submission per step cycle
- one batched observation/result return per step cycle
- contiguous memory layout where possible
- no JSON or verbose object serialization in the hot path

### 6.3 Bridge protocol goals

The bridge must optimize for:
- low call count
- stable schemas
- stable dtypes/shapes
- deterministic error reporting
- minimal allocation churn
- easy benchmark instrumentation

The bridge must not optimize for human readability in the hot path.

Current verified runtime note:
- the packaged headless distribution remains the canonical RL artifact input
- the current sim bootstrap still requires the checked-out sibling `fight-caves-RL` workspace root plus `data/cache/main_file_cache.dat2`
- the RL bridge must therefore preflight both the packaged artifact and the checked-out sim workspace before attempting correctness-mode bring-up
- fresh-runtime wrapper-vs-raw equivalence checks must use separate Python processes in Mode A because the embedded JVM is process-global and episode resets allocate per-player dynamic instances
- reset/determinism validation must compare a semantic projection rather than raw absolute instance ids, raw absolute ticks, or instance-shifted absolute tiles

Human-readable artifacts belong in:
- debug mode
- replay export
- diagnostics persistence
- tests

---

## 7) Environment Interface Contract

The RL module must mirror the headless sim contract exactly.

### 7.1 Reset contract

Required reset semantics:
- reset accepts explicit seed or seed policy
- reset returns initial observation in the current observation schema version
- reset records episode metadata required for logging/debugging
- reset clears wrapper-side episode-local caches and counters

Reset must not:
- mutate global trainer state unexpectedly
- silently auto-advance beyond the initial contract-defined reset state

### 7.1.1 Episode-start state contract

At the start of every training episode, the agent must begin from the same constant initial state unless a future spec revision explicitly versions and changes that baseline.

This starting state is required to remain constant across the training set so episode initialization is controlled and comparable.

#### Equipped items
- Weapon: Rune crossbow
- Ammo: Adamant bolts
  - Default ammo count is controlled by the headless reset contract input `ammo`
  - Current default: `1000`
- Helm: Coif
- Body: Black dragonhide body
- Legs: Black dragonhide chaps
- Hands: Black dragonhide vambraces
- Feet: Snakeskin boots
- No neck item is part of the current canonical episode-start contract

#### Inventory
- Prayer potion x8
  - Required starting form: 4-dose potions
  - Potion consumption must degrade to the dose-minus-1 variant exactly as the game/fork normally does
- Shark x20

#### Skills and resource state
- Attack: 1
- Strength: 1
- Defence: 70
- Prayer: 43
  - Each episode starts with full prayer points
  - Prayer drains normally during the episode
- Hitpoints: 70
  - Internal engine scale: Constitution `700`
  - Each episode starts with full hitpoints
  - Hitpoints drain normally during the episode
- Ranged: 70
- Magic: 1
- All other skills: 1

#### Other required starting conditions
- Run energy: 100%
- Run mode toggle: ON at episode start
  - The agent may toggle run on/off at any time during the episode
- No XP gain during episodes
- Episode-start stats/loadout are fixed by the headless sim contract and must not be loosened through wrapper-side substitutions

This episode-start state must be implemented in the headless sim contract and treated by the RL module as authoritative environment initialization state, not as wrapper-invented local state.

### 7.2 Step contract

Required step semantics:
- `step(action)` for correctness wrapper
- batched `step(actions)` or equivalent for vectorized backend
- outputs must contain at minimum:
  - observation
  - reward
  - done / terminated / truncated mapping as applicable
  - info metadata
  - action application metadata when available
  - rejection metadata when available
  - terminal reason when episode ends

### 7.3 Action contract

The RL repo must not redefine action meaning.

Required action rules:
- stable action enum/id mapping versioned on the RL side
- mapping must correspond exactly to the sim’s action contract
- invalid/rejected actions must preserve explicit reasons when surfaced by the sim
- if action masking is added later, masks must be advisory and must not change base action semantics

### 7.4 Observation contract

Observation handling rules:
- the RL repo must support the current headless observation schema version
- flattening/unflattening must be deterministic
- feature order must be versioned
- dtype and shape contracts must be explicit
- any wrapper-added fields must be clearly marked as RL-local derived features
- RL-local derived features are forbidden in parity/debug equivalence mode unless explicitly enabled

### 7.5 Termination contract

The RL repo must preserve simulator terminal reasons.

At minimum, it must distinguish:
- success / full cave completion
- player death
- invalid cave exit if enabled by policy/runtime
- max tick cap / truncation
- infrastructure failure (bridge/env failure), which must be clearly separated from normal game termination

### 7.6 Reward contract

Reward is owned by the RL repo, not by the simulator, unless the sim explicitly exports authoritative reward components.

Rules:
- base reward configs must be versioned and named
- sparse reward baseline must exist
- shaped reward configs must be pluggable, not hard-coded into core trainer logic
- reward computation must never require future leakage
- reward terms must be reproducible from logged inputs

### 7.7 Info contract

`info` payloads must be small on the hot path.

Rules:
- per-step info in hot mode must be minimal and structured
- verbose diagnostic payloads belong behind debug flags or sampled logging
- terminal info must contain enough detail for W&B summary stats and replay indexing

---

## 8) PufferLib Integration Contract

### 8.1 Supported integration layers

The RL repo must support both:
- a compatibility/emulation path for early correctness validation
- a high-throughput vectorized path for production training

### 8.2 Required PufferLib surfaces

The repo must support the core PufferLib training flow:
- load config
- construct vecenv
- construct policy
- create PuffeRL trainer
- evaluate rollouts
- train updates
- aggregate/log metrics
- save checkpoints
- print local dashboard
- close cleanly

### 8.3 VecEnv expectations

The final production training path must use vectorized env execution.

Required capabilities:
- reset
- step
- optional async send/recv path where beneficial
- close
- deterministic seeding policy across vectorized env slots
- worker-aware env indexing

PR5 implementation note:
- the correctness/smoke path may use a single-env compatibility shim if an upstream vecenv backend constructs the Mode A env more than once during bootstrap
- this does not satisfy the final production vector path
- PR8 must replace any such shim with a true batched/vector backend

### 8.4 Policy contract

Policies are normal PyTorch modules.

Required policy support:
- simple MLP baseline
- optional recurrent policy path if needed later
- observation flatten/unflatten compatibility with PufferLib expectations
- checkpoint save/load contract
- deterministic eval mode

### 8.5 No framework lock-in below the integration layer

The bridge and environment contract code must remain largely trainer-agnostic below the PufferLib-specific wiring so future experimentation is possible without rewriting core transport.

---

## 9) Bridge Contract (Python <-> Headless Sim)

### 9.1 Principle

The Python/JVM boundary is the primary throughput risk.

Therefore the RL module must optimize boundary crossings before optimizing minor trainer details.

### 9.2 Required bridge modes

#### Debug mode
- verbose diagnostics allowed
- lower throughput acceptable
- friendly payloads acceptable

#### Benchmark mode
- precise timing instrumentation
- minimal logging
- fixed seeds and benchmark harness controls

#### Train mode
- minimal hot-path overhead
- batched transport only
- structured compact buffers

### 9.3 Prohibited hot-path transport patterns

The following are prohibited in the final production path:
- per-env per-tick JSON serialization
- per-step creation of large nested Python dicts/lists for full batch state
- repeated schema introspection in hot loop
- per-action bridge calls for single envs in production training
- replay-grade full snapshot persistence every step during normal training

### 9.4 Preferred transport properties

Preferred properties:
- flat numeric buffers
- compact enums/ids
- explicit offsets and lengths
- shared memory or direct-buffer style transport where feasible
- stable schema versioning
- batch result structs that separate hot-path essentials from optional diagnostics

### 9.5 Error handling

Bridge failures must be explicit and classifiable.

Required error classes:
- protocol/schema mismatch
- sim process startup failure
- sim process crash
- timeout / deadlock suspicion
- invalid payload size/dtype
- unsupported schema version

Bridge failures must never be reported as ordinary episode terminations.

---

## 10) Observation Mapping Contract

### 10.1 Observation versions

The RL repo must explicitly support observation schema versions from the headless sim.

Rules:
- observation schema ID/version must be recorded in code and manifests
- any incompatible schema bump must fail fast
- compatibility shims, if ever added, must be documented and tested

### 10.2 Flattening contract

The RL repo must define:
- exact field order
- exact dtype per field/group
- exact flattened tensor layout
- batch layout semantics
- mask layout semantics if used

### 10.3 No hidden feature drift

Feature engineering must be explicit.

If RL-local derived features are introduced, they must:
- be listed in `docs/observation_mapping.md`
- be versioned
- be benchmarked for cost
- be disabled in strict equivalence/debug mode unless explicitly allowed

### 10.4 Debug observation support

Debug-only extra fields may be supported, but they must not contaminate the production policy input pipeline by accident.

### 10.5 PR5 policy-observation baseline

PR5 freezes the first RL-local policy input schema as `puffer_policy_observation_v0`.

Requirements:
- raw simulator observations must still be validated against `headless_observation_v1` before encoding
- the policy-input layout must be explicitly documented
- any cap or categorical dictionary used by the policy-input encoding must be versioned
- later production encodings may replace this baseline only via a schema/version bump

---

## 11) Action Mapping Contract

### 11.1 Action schema

The RL repo must maintain an explicit action schema document with:
- action IDs
- action names
- parameterization rules if any
- target indexing assumptions
- rejection reason mapping

### 11.2 Action compatibility

If the sim changes action IDs or semantics, the RL repo must fail fast and update the versioned mapping.

### 11.3 Optional masking

Action masking is optional and out of scope for initial correctness bring-up unless needed for basic integration.

If added later:
- masks must be versioned
- masks must be advisory only
- invalid actions must still be handled correctly by the environment path

### 11.4 PR5 policy-action baseline

PR5 freezes the first RL-local action encoding as `puffer_policy_action_v0`.

Requirements:
- the RL-local encoding must decode back into the frozen sim action contract without renumbering or repurposing action ids
- inactive parameter heads must be ignored rather than overloaded with alternate semantics
- later production encodings may replace this baseline only via a schema/version bump

---

## 12) Reward and Curriculum Contract

### 12.1 Reward ownership

Reward logic belongs to the RL repo.

Reward configs must be declarative and versioned.

### 12.2 Required reward configs

The repo must include at minimum:
- `reward_sparse_v0`
- `reward_shaped_v0`

Sparse baseline should prioritize:
- cave completion
- wave progression
- terminal success/failure distinctions

Shaped baseline may include carefully bounded terms for:
- survival progress
- resource efficiency
- damage avoidance / prayer correctness if derivable without leakage
- anti-stall incentives

### 12.3 Reward safety rules

Reward shaping must not:
- encode future knowledge
- depend on headed/oracle-only fields
- reward simulator glitches
- create hidden dependence on verbose debug data not present in production

### 12.4 Curriculum contract

Curriculum is optional for initial correctness, but the system must be designed to support it.

Required curriculum support surfaces:
- curriculum config files
- curriculum version recording in run manifests
- deterministic eval that can disable curriculum entirely

---

## 13) Training Pipeline Contract

### 13.1 Required entrypoints

Required scripts/entrypoints:
- `train.py`
- `eval.py`
- `benchmark_env.py`
- `benchmark_bridge.py`
- `replay_eval.py`
- `smoke_random.py`
- `smoke_scripted.py`
- `sweep.py`

### 13.2 Train loop requirements

The train loop must support:
- config-driven startup
- deterministic seed handling
- checkpoint save/resume
- periodic evaluation
- W&B logging
- graceful shutdown and resume-safe manifests
- optional local dashboard printing

### 13.3 Checkpoint contract

Each checkpoint must be associated with:
- model weights
- optimizer state as needed
- trainer step/update counters
- config snapshot
- schema versions
- reward config version
- sim commit SHA
- RL commit SHA

### 13.4 Eval contract

Evaluation must support:
- fixed seed packs
- fixed trace packs if applicable
- deterministic checkpoint replay
- replay artifact generation
- summary metric emission
- evaluation without reward-shaping drift from training config ambiguity

---

## 14) Replay and Artifact Contract

### 14.1 Replay goals

The RL repo must generate replay artifacts suitable for:
- debugging failures
- checkpoint evaluation review
- future website/replay integration

### 14.2 Required artifact classes

Required artifact classes:
- run manifests
- checkpoints
- evaluation summaries
- replay packs
- benchmark outputs
- trace packs / seed packs references
- parity canary outputs if generated by RL-side validation

### 14.3 Artifact naming/versioning

Artifacts must have stable naming conventions and schema versions.

### 14.4 Replay integrity

Replay generation must be deterministic for fixed:
- checkpoint
- seed pack
- action policy
- schema versions

---

## 15) Weights & Biases Logging Contract

W&B is mandatory for this repo.

### 15.1 W&B baseline requirements

The RL repo must:
- initialize a run with config metadata
- log training metrics continuously
- log checkpoints as artifacts or artifact-linked outputs
- log replay/eval artifacts
- log benchmark results
- record grouping keys to make runs comparable

### 15.2 Required run metadata

Every W&B run must include at minimum:
- project name
- run name
- tags
- run group
- W&B mode
- W&B resume mode
- RL repo commit
- sim repo commit
- RSPS repo commit
- oracle/parity pack version if relevant
- hardware profile
- trainer config snapshot
- reward config ID
- curriculum config ID
- observation schema ID/version
- action schema ID/version
- policy observation schema ID/version
- policy action schema ID/version
- bridge protocol ID/version
- episode-start contract ID/version
- benchmark profile ID/version
- PufferLib distribution and distribution version
- PufferLib import namespace/version when it differs from distribution metadata
- local run-manifest path and artifact records

### 15.3 Required metric families

At minimum log:

#### throughput
- env_steps_total
- env_steps_per_sec
- learner_updates_per_sec
- batch_collection_time_ms
- train_time_ms
- bridge_time_ms
- obs_pack_time_ms
- action_unpack_time_ms

#### training quality
- episode_return_mean
- episode_return_max
- episode_length_mean
- value_loss
- policy_loss
- entropy
- explained_variance
- clip_fraction or equivalent PPO diagnostics

#### environment/game progress
- wave_reached_mean
- jad_reach_rate
- completion_rate
- death_rate
- invalid_action_rate
- rejection_rate_by_reason
- prayer_potion_used_mean
- shark_used_mean
- truncation_rate

#### stability
- reset_failures
- bridge_failures
- env_crashes
- replay_generation_failures
- parity_canary_failures

### 15.4 Required artifact logging

Required artifact categories in W&B:
- checkpoints
- checkpoint metadata
- eval summaries
- replay packs
- benchmark reports
- config manifests

### 15.5 Offline and resume support

The repo should support offline logging / later sync where practical, but online W&B is the default target.

PR6 implementation note:
- a repo-owned RL logger is acceptable at the trainer boundary when the stock PufferLib logger does not satisfy manifest/artifact/version requirements
- if RL owns the logger, it must still keep the `PuffeRL` logging contract intact and avoid redefining trainer semantics
- local W&B directories for run files, artifact staging, and cache data must be configurable and owned by the repo bootstrap config
- W&B console/system-monitor features that destabilize WSL subprocess smoke tests may be disabled by default

---

## 16) PufferLib Dashboard / Local Monitoring Contract

The RL repo must support local terminal monitoring using the PufferLib trainer dashboard path where available.

Rules:
- local dashboard output must be optional/config-driven
- local dashboard output must degrade to a no-op in non-interactive subprocess and CI contexts even when config requests it
- dashboard printing must not become a hot-path bottleneck
- dashboard summaries must align with W&B metrics as closely as practical

---

## 17) Performance Contract

### 17.1 Top-line goal

The end-state goal is **>= 1,000,000 env steps/sec minimum**.

This is a real acceptance target for the mature training path, not for the first correctness wrapper.

### 17.2 Performance ladder

The RL repo must track performance in stages:

1. single-env correctness wrapper baseline
2. batched bridge baseline
3. vectorized env baseline
4. full training baseline
5. tuned production training baseline

### 17.3 Mandatory benchmark breakdowns

Benchmarks must isolate at least:
- raw sim batch stepping rate
- Python bridge throughput
- wrapper overhead
- PufferLib vecenv throughput
- end-to-end training SPS
- W&B overhead with normal logging cadence
- W&B overhead with aggressive logging cadence

### 17.4 Mandatory benchmark profiles

At minimum benchmark:
- 1 env
- 16 envs
- 64 envs
- 256 envs
- 1024 envs if hardware allows

And where applicable:
- multiple envs per worker settings
- sync vs async vectorization path
- logging on vs logging minimized
- replay generation disabled vs periodic replay enabled

### 17.5 Hot-path optimization priorities

Optimization priority order:
1. boundary crossings
2. memory copies
3. observation packing/unpacking
4. process topology / envs per worker
5. trainer logging overhead
6. policy inference overhead
7. secondary tooling overhead

### 17.6 Forbidden benchmark practice

Do not claim throughput using a setup that changes simulator semantics or drops required environment work.

Benchmark configs must document:
- exact config
- hardware
- env count
- worker count
- bridge mode
- logging mode
- reward config
- replay mode

---

## 18) Testing and Validation Contract

### 18.1 Test layers

The RL repo must include all of the following test layers.

#### Layer A: unit tests
For:
- self-contained dev-bootstrap validation only
- no `pufferlib`/`torch` imports
- no live sim runtime prerequisite
- config loading
- observation flatten/unflatten
- action mapping
- reward math
- manifest generation
- artifact naming

#### Layer A2: train-bootstrap tests
For:
- self-contained tests that require the `train` dependency group
- train bootstrap/import-path validation
- trainer-wrapper behavior that does not require the live sim runtime

#### Layer B: integration tests
For:
- wrapper reset/step correctness
- bridge startup/shutdown
- batched env stepping
- checkpoint save/load
- W&B dry-run integration

#### Layer C: determinism tests
For:
- fixed seed reset reproducibility
- fixed checkpoint + fixed seeds => identical eval summaries
- wrapper-vs-raw-sim trajectory agreement for fixed action traces
- semantic-projection equivalence when raw absolute reset state drifts only because of global tick/instance allocation details

#### Layer D: smoke training tests
For:
- random policy runs
- scripted baseline runs
- minimal PuffeRL train/eval loop completes

#### Layer E: performance tests
For:
- bridge microbenchmarks
- env SPS benchmarks
- training SPS benchmarks

#### Layer F: parity canaries
For:
- fixed trace pack subsets against sim/oracle expectations where appropriate
- wrapper does not introduce semantic drift
- versioned per-tick RL trace packs expanded from sim replay traces where the sim source uses `ticksAfter > 1`

### 18.2 Required correctness tests

Required named tests/artifacts should include equivalents of:
- `WrapperResetMatchesSimContractTest`
- `WrapperStepMatchesSimTraceTest`
- `ObservationFlatteningDeterminismTest`
- `ActionSchemaVersionCompatibilityTest`
- `BridgeBatchStepParityTest`
- `CheckpointResumeRoundTripTest`
- `WandbRunManifestCompletenessTest`
- `DeterministicEvalSameCheckpointSameSeedPackTest`
- `RandomPolicySmokeEpisodeTest`
- `PuffeRLSmokeTrainLoopTest`

### 18.3 CI requirements

Per-PR CI must at minimum run:
- dev-only unit tests
- train-bootstrap import smoke
- train-bootstrap self-contained tests
- lint/type checks if configured

Local pre-merge validation must at minimum run:
- live integration tests
- determinism tests
- smoke training/eval tests
- parity canaries

Heavy benchmarks may run on a separate scheduled or manual job.

---

## 19) Delivery Plan (Required Order)

The RL repo must be built in the following order.

### Step 0 - Repo Bootstrap and Lockfile

Required action items:
1. Create repo skeleton and dependency management.
2. Pin Python baseline and PufferLib version.
3. Add `.env.example` and local setup docs.
4. Add baseline config loader and manifest utilities.

Required artifacts:
- `pyproject.toml`
- `uv.lock`
- `README.md`
- `docs/run_manifest.md`

Exit criteria:
- clean install and test bootstrap succeed on Linux/WSL.

---

### Step 1 - RL/Sim Integration Contract Docs

Required action items:
1. Write `docs/rl_integration_contract.md`.
2. Write `docs/bridge_contract.md`.
3. Write `docs/observation_mapping.md`.
4. Write `docs/action_mapping.md`.
5. Write `docs/hotpath_map.md` using current sim/runtime knowledge.

Exit criteria:
- exact env/bridge/schema contracts are documented and versioned.

---

### Step 2 - Correctness Wrapper Bring-Up

Required action items:
1. Implement a thin correctness-first env wrapper.
2. Support reset/step/close.
3. Map observations/actions exactly to the headless sim contract.
4. Surface terminal reasons and rejection metadata.

Current implementation note:
- if the selected Mode A sim surface does not expose a dedicated terminal-reason envelope, the RL wrapper may record clearly-labeled provisional inferences for death/completion/truncation only until a more direct sim surface is available
- this does not relax the requirement to preserve simulator outcomes; it only documents the bring-up constraint that must be closed before PR3 is fully accepted
- correctness-mode wrapper-vs-raw trace checks may use subprocess-isolated fresh runtimes; they must not assume that two player slots inside one embedded JVM runtime begin from identical absolute tiles

Required tests:
1. wrapper reset correctness
2. wrapper step correctness
3. action mapping correctness
4. observation shape/dtype correctness

Exit criteria:
- random policy can execute full episodes without wrapper ambiguity.

---

### Step 3 - Determinism and Equivalence Validation

Required action items:
1. Compare wrapper trajectories to raw headless sim trajectories on fixed traces.
2. Validate fixed-seed reset reproducibility.
3. Validate deterministic eval for fixed seeds/checkpoints.

Required artifacts:
- deterministic eval docs
- seed pack definitions
- sample equivalence logs

Exit criteria:
- wrapper introduces no semantic drift on required validation matrix.

---

### Step 4 - PufferLib Smoke Integration

Required action items:
1. Construct basic vecenv integration.
   - A single-env shim is acceptable for PR5 if the stock vecenv bootstrap pattern conflicts with the current Mode A runtime lifecycle.
2. Build minimal policy and trainer wiring.
3. Add train/eval entrypoints.
4. Save checkpoints.
5. Print local dashboard output.

Required tests:
1. minimal trainer smoke loop
2. checkpoint save/load smoke
3. eval loop smoke

Exit criteria:
- minimal PuffeRL training loop completes and logs structured outputs.

---

### Step 5 - W&B Integration and Run Manifests

Required action items:
1. Add mandatory W&B run initialization.
2. Log config, metrics, and artifacts.
3. Persist run manifests locally and in W&B-linked outputs.
4. Add group/tag conventions.

Required tests:
1. W&B dry-run / offline smoke
2. manifest completeness test
3. artifact naming/versioning test

Exit criteria:
- every training/eval run emits complete metadata and core metrics.

---

### Step 6 - Batched Bridge

Required action items:
1. Implement batched action submission.
2. Implement batched observation/result retrieval.
3. Freeze the versioned batch protocol and slot-lockstep semantics before any later transport swap.
4. Replace verbose hot-path payloads with compact transport where it does not change the frozen semantics.
5. Add benchmark harness for bridge throughput.

Required tests:
1. batch vs sequential equivalence
2. bridge batch stability
3. schema/version fail-fast behavior

Exit criteria:
- batched bridge works correctly and outperforms correctness wrapper path.

Implementation note:
- the first shipped PR7 bridge may remain in-process on the embedded JVM runtime if that is the shortest correct path to freezing batch semantics against the golden sim
- a later subprocess/shared-buffer transport is acceptable only if it preserves the PR7 protocol/semantics and bumps the bridge version when needed

---

### Step 7 - Vectorized PufferLib Backend

Required action items:
1. Integrate batched bridge with PufferLib vectorization.
2. Tune envs-per-worker and worker topology.
3. Support sync and async collection experiments where useful.
4. Add throughput benchmark configs.

Required tests:
1. vecenv reset/step smoke
2. multi-worker smoke
3. long-run stability smoke

Exit criteria:
- vectorized training path is stable and benchmarkable.

---

### Step 8 - Reward and Curriculum System

Required action items:
1. Implement versioned reward configs.
2. Add sparse and shaped baselines.
3. Add optional curriculum scaffolding.
4. Record reward/curriculum versions in manifests and W&B.

Required tests:
1. reward reproducibility tests
2. no-future-leakage reward tests
3. config loading tests

Exit criteria:
- reward and curriculum are configurable without changing trainer core.

---

### Step 9 - Replay and Eval Artifacts

Required action items:
1. Generate evaluation replay artifacts from checkpoints.
2. Add replay indexing/manifesting.
3. Add fixed evaluation seed packs.
4. Make replay generation configurable by cadence.

Required tests:
1. replay generation smoke
2. replay manifest integrity
3. deterministic replay-eval equivalence

Exit criteria:
- checkpoints can be evaluated reproducibly and produce inspectable replay artifacts.

---

### Step 10 - Performance Hardening

Required action items:
1. benchmark each layer separately
2. reduce copies/allocations
3. tune worker/env topology
4. reduce logging overhead where needed
5. document bottlenecks and improvements

Required artifacts:
- `docs/performance_plan.md`
- benchmark result logs
- hardware-specific benchmark manifests

Exit criteria:
- production training path demonstrates credible progress toward >= 1,000,000 SPS.

---

### Step 11 - Parity Canaries and Oracle-Reference Validation

Required action items:
1. add RL-side parity canaries using fixed trace/seed packs
2. verify wrapper and replay paths do not drift from sim/oracle expectations
3. document investigation workflow for rare divergences

Required tests:
1. parity canary smoke
2. replay-to-trace equivalence smoke

Exit criteria:
- RL-side integration does not mask or introduce semantic drift.

---

### Step 12 - MVP Acceptance Gate

Required action items:
1. run full RL-side test suite
2. run end-to-end train/eval/checkpoint/replay flow
3. verify W&B artifact generation
4. verify benchmark output generation
5. verify deterministic eval pack success

Exit criteria:
- RL module is functional, reproducible, instrumented, and integration-correct.

---

## 20) Required Acceptance Criteria

The RL module is accepted only when all of the following are true:

1. The repo installs reproducibly from lockfiles on the target Linux/WSL environment.
2. The correctness wrapper matches the headless sim contract for reset/step/action/observation/termination.
3. PuffeRL-based training runs complete successfully.
4. W&B logging is integrated and emits required metrics/artifacts.
5. Checkpoints can be saved, resumed, and evaluated deterministically.
6. Replay artifacts can be produced from evaluation runs.
7. Batch/vector integration is implemented and benchmarked.
8. RL-side parity canaries do not show semantic drift introduced by the wrapper/bridge.
9. Performance benchmarks isolate major bottlenecks and document progress toward the SPS target.
10. The production path is architected for >= 1,000,000 env steps/sec minimum and benchmark work is active against that target.

---

## 21) Explicit Non-Goals (for Initial RL Module)

Out of scope unless explicitly added later:
- human-play browser runtime in this repo
- modifying headed RSPS gameplay semantics from RL repo
- embedding full simulator logic into Python
- supporting every RL library under the sun
- building a generic MMO RL platform before Fight Caves training is stable
- mixing oracle parity code into the production training hot path

---

## 22) Change Control Rules

Any change that affects one of the following must update this spec and the relevant contract docs:
- PufferLib version baseline
- observation schema mapping
- action schema mapping
- bridge protocol
- reward versioning rules
- W&B artifact policy
- benchmark methodology
- acceptance criteria

If implementation reality diverges from this spec:
- verified implementation + sim contract win
- this spec must be updated immediately
- silent drift is not allowed

---

## 23) First Implementation Priorities (Immediate Next Work)

The first coding-agent work after adopting this spec must be:
1. repo bootstrap
2. integration contract docs
3. correctness wrapper
4. deterministic equivalence tests
5. PufferLib smoke train/eval wiring
6. W&B run manifests/logging
7. batched bridge
8. vectorized backend

Do not begin with reward tinkering or large hyperparameter sweeps before the correctness and bridge layers are in place.

---

## 24) Practical Guidance for the Coding Agent

When in doubt:
- keep the sim authoritative
- keep the oracle separate
- keep the bridge coarse-grained
- keep hot-path payloads compact
- version every contract
- benchmark every layer separately
- prefer simple correctness first, then speed
- never trade away parity for throughput
