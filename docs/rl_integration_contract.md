# RL/Sim Integration Contract

This document freezes the RL-side integration contract against the current headless simulator in `/home/jordan/code/fight-caves-RL`.

## Pivot Status

Current workspace authority is:
- `/home/jordan/code/pivot_plan.md`
- `/home/jordan/code/pivot_implementation_plan.md`

This document remains an active reference for the current simulator-backed V1 oracle/reference path.
It does not replace the pivot architecture.

Agent execution environment:
- WSL/Linux is the canonical environment for RL-side integration work, validation commands, and runtime launch instructions.
- Linux paths and shells are canonical; do not author the active integration contract around Windows-native path semantics or PowerShell.

## Authority

- `fight-caves-RL` is authoritative for reset behavior, step semantics, observation contents, target indexing, deterministic replay, and packaging.
- `RSPS` remains the oracle/reference for parity disputes and headed validation only.
- `RL` may flatten, batch, and log around the simulator contract, but it must not reinterpret or hide simulator semantics.

## Mechanics Parity Scope

Under the pivot, parity is defined as **mechanics parity** between:
- the future V2 fast trainer
- the RSPS-backed headed demo path
- the oracle/reference path

`fight-caves-demo-lite` remains a frozen headed fallback/reference module only.

It is not defined as full engine/runtime parity.

## Phase 0 V2 Contract Freeze

Phase 0 freezes the portable RL-facing V2 contract surfaces in:
- `fight_caves_rl/contracts/mechanics_contract.py`
- `fight_caves_rl/contracts/terminal_codes.py`
- `fight_caves_rl/contracts/reward_feature_schema.py`
- `fight_caves_rl/contracts/parity_trace_schema.py`

Those files freeze the mechanics boundary, terminal codes, reward features, and parity trace fields before the fast kernel exists.

## Canonical Sim Sources

PR 2 is aligned to the current verified simulator sources:

- Episode initialization contract:
  - `fight-caves-RL/docs/episode_init_contract.md`
  - `fight-caves-RL/game/src/main/kotlin/FightCaveEpisodeInitializer.kt`
- Observation schema:
  - `fight-caves-RL/docs/observation_schema.md`
  - `fight-caves-RL/game/src/main/kotlin/HeadlessObservationBuilder.kt`
- Stable action surface:
  - `fight-caves-RL/game/src/main/kotlin/HeadlessActionAdapter.kt`
  - `fight_caves_rl/envs/schema.py` (`HEADLESS_ACTION_SCHEMA` and `HEADLESS_ACTION_DEFINITIONS`)
- Headless artifact/package boundary:
  - `fight-caves-RL/docs/runtime_pruning.md`
  - `fight-caves-RL/game/build.gradle.kts`
  - `fight-caves-RL/game/src/main/kotlin/HeadlessMain.kt`

## Canonical Runtime Artifact

RL treats the packaged headless distribution as the default runtime dependency.

- Default build task: `:game:headlessDistZip`
- Validation/build aggregator: `:game:packageHeadless`
- Distribution glob: `fight-caves-RL/game/build/distributions/fight-caves-headless*.zip`
- Current verified dev artifact: `fight-caves-RL/game/build/distributions/fight-caves-headless-dev.zip`
- Expected runtime jar inside the extracted distribution: `fight-caves-headless.jar`
- Headless JVM entrypoint class: `HeadlessMain`

Fallback rules:

- First choice: consume an existing extracted or zipped `fight-caves-headless` distribution.
- Second choice: build `:game:headlessDistZip` from the sibling sim repo.
- Third choice: build `:game:packageHeadless` when packaging validation or deletion-candidate generation is also desired.
- Local jar/classpath launch is a manual-debug fallback only. It is not the canonical dev/test artifact boundary for RL.

Current runtime invariant verified during PR 3 bring-up:

- the current headless bootstrap still locates the checked-out sim repository root dynamically
- RL therefore needs both the packaged headless distribution and the checked-out sibling sim workspace
- the current workspace-required files are:
  - `fight-caves-RL/config/headless_data_allowlist.toml`
  - `fight-caves-RL/config/headless_manifest.toml`
  - `fight-caves-RL/config/headless_scripts.txt`
  - `fight-caves-RL/data/cache/main_file_cache.dat2`
- the current workspace now satisfies those prerequisites, and PR3 live reset/step/smoke validation has run successfully against them

## Episode Start Contract

RL freezes the current episode-start contract to the sim implementation behind:

- `HeadlessRuntime.resetFightCaveEpisode(...)`
- `FightCaveEpisodeInitializer.reset(player, config)`

Required config inputs and defaults:

- `seed: Long` is required.
- `startWave`: default `1`, valid `1..63`
- `ammo`: default `1000`, must be `> 0`
- `prayerPotions`: default `8`, must be `>= 0`
- `sharks`: default `20`, must be `>= 0`

Reset guarantees:

- Shared RNG is seeded from `seed`, and `player["episode_seed"]` is set to the same value.
- Transient queues, timers, facing/watch state, animation/graphics state, and known lockout clocks are cleared.
- Fight Caves variables and prayer variables are reset before the new episode starts.
- Any prior dynamic instance is cleared before a new Fight Caves instance is created.
- XP gain is blocked during the episode reset contract.

Fixed stats/resources:

- Attack `1`
- Strength `1`
- Defence `70`
- Constitution `700`
- Ranged `70`
- Prayer `43`
- Magic `1`
- all other skills `1`
- run energy `100%`
- run toggle `ON`

Fixed equipment:

- `coif`
- `rune_crossbow`
- `black_dragonhide_body`
- `black_dragonhide_chaps`
- `black_dragonhide_vambraces`
- `snakeskin_boots`
- `adamant_bolts x ammo`

Fixed inventory item ids:

- `prayer_potion_4 x prayerPotions`
- `shark x sharks`

Instance/wave startup:

- a new small dynamic instance is created
- the player is teleported/walked to the Fight Caves start positions in that instance
- `fightCave.startWave(player, startWave, start = false)` is invoked
- `start = false` is intentional and part of the contract for headless episodes

Player provisioning precondition:

- A correctness-mode RL wrapper must provision a valid player the same way the headless sim tests do:
  - `AccountManager.setup(...)`
  - `AccountManager.spawn(...)`
  - `player["creation"] = -1`
  - `player["skip_level_up"] = true`
  - `player.viewport?.loaded = true`
- RL must not treat a raw uninitialized `Player` object as a valid episode slot.

Fresh-runtime comparison note:

- PR3 wrapper-vs-raw equivalence tests must use separate Python processes for fresh-runtime comparisons
- Mode A owns one embedded JVM runtime per process, and different player slots inside one runtime are not expected to share identical absolute reset tiles because each reset creates a new Fight Caves dynamic instance

## Action and Observation Authority

Action contract authority:

- The simulator exposes stable append-only `HeadlessActionType.id` values.
- The supported action set is:
  - `Wait`
  - `WalkToTile`
  - `AttackVisibleNpc`
  - `ToggleProtectionPrayer`
  - `EatShark`
  - `DrinkPrayerPotion`
  - `ToggleRun`
- One intent is allowed per tick.
- Rejected actions must preserve explicit rejection reasons and action metadata from the simulator.

Observation contract authority:

- `schema_id = headless_observation_v1`
- `schema_version = 1`
- `compatibility_policy = v1_additive_only`
- RL must preserve the simulator's deterministic NPC ordering and target-index alignment.
- `debug_future_leakage` is opt-in only and must not appear in the default training payload.

## Fail-Fast Handshake Requirements

Before stepping begins, the RL bridge/wrapper must validate:

- sim artifact task/path identity
- sim commit/build metadata when available
- observation schema id/version
- action schema id/version
- episode-start contract id/version
- bridge protocol id/version
- benchmark profile id/version when benchmarking
- PufferLib distribution/version

Any mismatch must fail fast before training or evaluation starts.

## Termination/Truncation

RL does not redefine simulator episode outcomes.

- terminal and truncation reasons must be surfaced from the simulator/runtime policy
- wrapper logic may annotate outcomes, but it must not translate simulator failures into fake gameplay outcomes

PR 3 implementation note:

- the current sim runtime does not yet expose a dedicated terminal-reason envelope through the selected Mode A surface
- the correctness env currently records explicit inferred labels only for:
  - `player_death`
  - `cave_complete`
  - `max_tick_cap`
- this is tracked as a follow-up validation point before PR 3 can be considered fully complete
- the portable V2 terminal-code freeze now lives in `fight_caves_rl/contracts/terminal_codes.py`
