# Reward Configs

This document freezes the first versioned reward and curriculum surfaces owned by the RL repo.

## Principles

- reward lives in `RL`, not in the simulator
- reward functions must use only current-step inputs plus the previous observation already available to the wrapper
- reward must not depend on future leakage, headed-only state, or RSPS-only oracle data
- curriculum may change reset options, but it must not change simulator semantics

## Phase 0 V2 Reward Feature Freeze

Phase 0 freezes the portable V2 reward-feature schema in
`fight_caves_rl/contracts/reward_feature_schema.py`.

The fast kernel will own reward-feature emission.
Python will continue to own config-driven weighting over that emitted vector.

Frozen V2 reward features:
- `damage_dealt`
- `damage_taken`
- `npc_kill`
- `wave_clear`
- `jad_damage_dealt`
- `jad_kill`
- `player_death`
- `cave_complete`
- `food_used`
- `prayer_potion_used`
- `correct_jad_prayer_on_resolve`
- `wrong_jad_prayer_on_resolve`
- `invalid_action`
- `movement_progress`
- `idle_penalty_flag`
- `tick_penalty_base`

Reward rule:
- V2 does not reward prayer toggling by itself.
- V2 only rewards the mechanically correct Jad prayer state at hit resolution through `correct_jad_prayer_on_resolve` and penalizes the wrong state through `wrong_jad_prayer_on_resolve`.

## Reward Configs

V0 reward configs remain the observation-based V1/oracle reference path.
V2 reward configs are direct kernel-feature weighting configs for `env_backend = v2_fast`.

### `reward_sparse_v0`

Purpose:

- parity-safe baseline
- official benchmark/profile default
- minimal shaping for correctness-first training bring-up

Terms:

- `wave_progress = +1.0` per completed wave transition
- `cave_complete = +10.0` on successful cave completion
- `player_death = -1.0` on terminal player death

### `reward_shaped_v0`

Purpose:

- optional richer training signal while staying reproducible from logged inputs

Terms:

- all `reward_sparse_v0` terms
- `npc_damage = +0.02` per visible-NPC hitpoint removed between consecutive observations
- `player_damage = -0.02` per player hitpoint lost between consecutive observations
- `shark_use = -0.05` per shark consumed
- `prayer_potion_use = -0.05` per prayer-potion dose consumed
- `ammo_use = -0.001` per ammo consumed
- `step_penalty = -0.0005` per step

Notes:

- shaped reward still uses only observation/action-result data already available in the RL wrapper
- if later shaping needs additional terms, the config id must change

### `reward_sparse_v2`

Purpose:

- sparse evaluation default for the V2 fast path
- canary-safe baseline that avoids Python observation reconstruction

Terms:

- `cave_complete = +1.0`
- `player_death = -1.0`

### `reward_shaped_v2`

Purpose:

- main shaped-reward surface for the V2 fast path
- direct weighting over emitted kernel reward features

Terms:

- positive:
  - `damage_dealt = +0.02`
  - `npc_kill = +0.1`
  - `wave_clear = +1.0`
  - `jad_damage_dealt = +0.03`
  - `cave_complete = +1.0`
  - `correct_jad_prayer_on_resolve = +0.25`
- negative:
  - `damage_taken = -0.02`
  - `wrong_jad_prayer_on_resolve = -0.25`
  - `invalid_action = -0.02`
  - `food_used = -0.05`
  - `prayer_potion_used = -0.05`
  - `tick_penalty_base = -0.0005`
  - `player_death = -1.0`
- reserved but currently zero-weighted:
  - `movement_progress = 0.0`

Notes:

- V2 reward configs must map directly onto `fight_caves_rl/contracts/reward_feature_schema.py`
- V2 reward configs must not resolve through the V1 observation-based reward-function path
- V2 still does not reward prayer toggling by itself; it rewards the mechanically correct Jad prayer outcome at hit resolution

## Curriculum Configs

### `curriculum_disabled_v0`

- default parity-safe mode
- emits no reset overrides

### `curriculum_wave_progression_v0`

- deterministic scaffolding only
- changes reset `start_wave` by per-slot episode index

Current schedule:

- episodes `0-7` => `start_wave = 1`
- episodes `8-15` => `start_wave = 8`
- episodes `16+` => `start_wave = 31`

## Eval Rule

Deterministic replay/eval defaults to:

- checkpoint reward config
- `curriculum_disabled_v0`

That avoids reward ambiguity while keeping eval-start conditions fixed.
