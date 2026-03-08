# Observation Mapping

This document freezes the RL-side interpretation of the current headless simulator observation contract.

## Source of Truth

The canonical simulator observation source is `HeadlessObservationV1` in `fight-caves-RL`.

Schema identity:

- `schema_id = headless_observation_v1`
- `schema_version = 1`
- `compatibility_policy = v1_additive_only`

## Top-Level Order

RL must preserve the simulator's top-level field order:

1. `schema_id`
2. `schema_version`
3. `compatibility_policy`
4. `tick`
5. `episode_seed`
6. `player`
7. `wave`
8. `npcs`

Optional debug field:

- `debug_future_leakage`

## Player Block

RL must preserve the simulator meanings and units for:

- `tile.{x,y,level}`
- `hitpoints_current`
- `hitpoints_max`
- `prayer_current`
- `prayer_max`
- `run_energy`
- `run_energy_max`
- `run_energy_percent`
- `running`
- `protection_prayers.*`
- `lockouts.*`
- `consumables.*`

Important notes:

- Constitution values are surfaced in the sim's native numeric scale.
- `run_energy_percent` is an integer derived from the simulator's current/max run energy.
- `ammo_id` is an empty string when no ammo item is equipped.
- `prayer_potion_dose_count` is the summed dose count across 4/3/2/1-dose items.

## Wave Block

RL must preserve:

- current wave index
- Fight Caves rotation
- remaining NPC count

These fields come directly from the headless Fight Caves state variables.

## NPC Block

RL must preserve the simulator NPC ordering exactly.

Each entry includes:

- `visible_index`
- `npc_index`
- `id`
- `tile`
- `hitpoints_current`
- `hitpoints_max`
- `hidden`
- `dead`
- `under_attack`

Ordering contract:

- the NPC list is deterministically ordered
- the ordering source is the same visible-NPC mapping used by the headless action adapter
- `AttackVisibleNpc.visible_npc_index` must continue to align with `npcs[*].visible_index`

## RL Flattening Guardrails

RL may flatten or vectorize the observation later, but it may not:

- remove required fields from the strict payload
- reorder fields in a way that changes semantic interpretation
- change the visible-NPC target alignment
- include `debug_future_leakage` by default

Allowed later wrapper behavior:

- booleans may be converted to `0/1` at the last possible wrapper boundary
- strings such as `ammo_id` may be mapped to stable encoded forms, but only through an explicit documented dictionary/versioned mapping
