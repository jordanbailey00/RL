# Reward Configs

This document freezes the first versioned reward and curriculum surfaces owned by the RL repo.

## Principles

- reward lives in `RL`, not in the simulator
- reward functions must use only current-step inputs plus the previous observation already available to the wrapper
- reward must not depend on future leakage, headed-only state, or RSPS-only oracle data
- curriculum may change reset options, but it must not change simulator semantics

## Reward Configs

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
