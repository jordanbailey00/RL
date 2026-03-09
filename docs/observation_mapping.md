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
- `jad_telegraph_state`

Ordering contract:

- the NPC list is deterministically ordered
- the ordering source is the same visible-NPC mapping used by the headless action adapter
- `AttackVisibleNpc.visible_npc_index` must continue to align with `npcs[*].visible_index`
- `jad_telegraph_state` is a semantic cue, not an oracle:
  - `0 = idle`
  - `1 = magic_windup`
  - `2 = ranged_windup`
  - only the real Jad NPC may surface a non-zero value
  - RL must not reinterpret this as a countdown or a direct prayer answer

## RL Flattening Guardrails

RL may flatten or vectorize the observation later, but it may not:

- remove required fields from the strict payload
- reorder fields in a way that changes semantic interpretation
- change the visible-NPC target alignment
- include `debug_future_leakage` by default

Future flat-path rule:

- the future training-flat path must remain a semantically equivalent projection of this raw contract
- decision-critical combat cues already present in the raw contract, including `jad_telegraph_state`, must retain identical onset window and meaning in any future flat layout
- Certification Mode must prove raw-vs-flat equivalence before RL trusts the flat path in Production Training Mode

Allowed later wrapper behavior:

- booleans may be converted to `0/1` at the last possible wrapper boundary
- strings such as `ammo_id` may be mapped to stable encoded forms, but only through an explicit documented dictionary/versioned mapping

## PR5 Policy Input Encoding

The current Gym/Puffer policy-input encoding is:

- `policy_observation_schema_id = puffer_policy_observation_v1`
- `policy_observation_schema_version = 1`

This is an RL-local encoding for trainer input only.
The raw sim payload above remains authoritative and is still validated before encoding.

Phase 1 design note:

- the first future sim-owned flat training schema is intentionally designed to mirror this current trainer layout so RL can remove Python raw-object reconstruction without simultaneously redesigning the policy feature set
- see [flat_training_observation_schema.md](/home/jordan/code/fight-caves-RL/docs/flat_training_observation_schema.md) and [flat_observation_ingestion.md](/home/jordan/code/RL/docs/flat_observation_ingestion.md)

### Constants dropped after validation

The PR5 policy vector does not include:

- `schema_id`
- `compatibility_policy`

Those fields are constant after the raw payload passes contract validation, so PR5 excludes them from the policy tensor and records the schema identity separately in checkpoint metadata.

### Current flat vector order

Base prefix order:

1. `schema_version`
2. `tick`
3. `episode_seed`
4. `player.tile.{x,y,level}`
5. `player.hitpoints_current`
6. `player.hitpoints_max`
7. `player.prayer_current`
8. `player.prayer_max`
9. `player.run_energy`
10. `player.run_energy_max`
11. `player.run_energy_percent`
12. `player.running`
13. `player.protection_prayers.{magic,missiles,melee}`
14. `player.lockouts.{attack,food,drink,combo,busy}`
15. `player.consumables.{shark_count,prayer_potion_dose_count,ammo_id_code,ammo_count}`
16. `wave.{wave,rotation,remaining}`
17. `npcs.visible_count`

Per-NPC slot order:

1. `present`
2. `visible_index`
3. `npc_index`
4. `id_code`
5. `tile.{x,y,level}`
6. `hitpoints_current`
7. `hitpoints_max`
8. `hidden`
9. `dead`
10. `under_attack`
11. `jad_telegraph_state`

### PR5 visible-NPC cap

PR5 policy encoding fixes `max_visible_npcs = 8`.

Rationale from the current verified sim closure:

- static wave data peaks at `6` declared NPC entries
- `tz_kek` split behavior can raise live count to `7`
- Jad healer scenarios peak lower than that

Unused slots are zero-padded in deterministic order.
If the sim grows beyond this cap, RL must bump the policy-observation schema rather than silently truncating.

### PR5 categorical dictionaries

`ammo_id_code`:

- `0` => `""`
- `1` => `adamant_bolts`

`npc_id_code` order:

1. `tz_kih`
2. `tz_kih_spawn_point`
3. `tz_kek`
4. `tz_kek_spawn_point`
5. `tz_kek_spawn`
6. `tok_xil`
7. `tok_xil_spawn_point`
8. `yt_mej_kot`
9. `yt_mej_kot_spawn_point`
10. `ket_zek`
11. `ket_zek_spawn_point`
12. `tztok_jad`
13. `yt_hur_kot`
