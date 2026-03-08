# Action Mapping

This document freezes the RL-side view of the current headless simulator action surface.

## Source of Truth

The canonical simulator action surface is `HeadlessActionType` and related types in `fight-caves-RL/game/src/main/kotlin/HeadlessActionAdapter.kt`.

Important note:

- the simulator currently guarantees stable append-only numeric action IDs
- the simulator does not currently export a separate action schema id/version constant
- RL therefore freezes the current action surface as `headless_action_v1` for manifest/handshake purposes
- this naming does not change simulator semantics; it only versions the current stable interface

## Action Schema Identity

- `action_schema_id = headless_action_v1`
- `action_schema_version = 1`
- `compatibility_policy = append_only_ids`

## Supported Actions

`0` `wait`
- sim type: `Wait`
- params: none

`1` `walk_to_tile`
- sim type: `WalkToTile`
- params: `tile.x`, `tile.y`, `tile.level`
- note: pathfinder-backed

`2` `attack_visible_npc`
- sim type: `AttackVisibleNpc`
- params: `visible_npc_index`
- note: targets the deterministic visible-NPC list exported by the simulator observation/action adapter

`3` `toggle_protection_prayer`
- sim type: `ToggleProtectionPrayer`
- params: `prayer`
- allowed values:
  - `protect_from_magic`
  - `protect_from_missiles`
  - `protect_from_melee`

`4` `eat_shark`
- sim type: `EatShark`
- params: none

`5` `drink_prayer_potion`
- sim type: `DrinkPrayerPotion`
- params: none

`6` `toggle_run`
- sim type: `ToggleRun`
- params: none

## One-Intent-Per-Tick Rule

The simulator enforces one action intent per tick.

- repeated actions in the same tick are rejected
- RL must not hide this rule with client-side action coalescing or queueing

## Rejection Reasons

The simulator currently exports these rejection reasons:

- `AlreadyActedThisTick`
- `InvalidTargetIndex`
- `TargetNotVisible`
- `PlayerBusy`
- `MissingConsumable`
- `ConsumptionLocked`
- `PrayerPointsDepleted`
- `InsufficientRunEnergy`
- `NoMovementRequired`

RL must preserve these reasons and action metadata rather than collapsing them into generic invalid-action buckets too early.

## Wrapper Guardrails

RL may encode/decode actions for vectorization later, but it may not:

- renumber existing action IDs
- repurpose an existing action ID
- reorder visible-NPC target indices
- hide simulator rejection metadata behind wrapper-local heuristics
