# eval_and_replay.md

This document freezes the PR4 deterministic evaluation and pre-replay validation surface.

## Current PR4 Scope

PR4 does not yet generate full replay artifacts.
That lands later when checkpoint/eval artifact plumbing is added.

PR4 does establish the deterministic inputs that later replay flows must consume:

- versioned seed packs
- versioned per-tick trace packs
- deterministic serialized rollout payloads suitable for equivalence checks

## Seed Packs

Seed packs live in `fight_caves_rl/replay/seed_packs.py`.

Rules:

- seed pack ids are versioned
- seed lists are stable and explicit
- adding seeds is append-only unless semantics require a version bump
- PR4 uses packs aligned to the current sim parity harness seed set

## Trace Packs

Trace packs live in `fight_caves_rl/replay/trace_packs.py`.

Rules:

- RL trace packs are expressed in per-tick env action space
- sim-side replay traces that use `ticksAfter > 1` are expanded into repeated per-tick RL actions
- each trace pack records its sim source reference

This avoids hiding timing semantics inside RL-local transport glue.

## Determinism Projection

The raw sim observation contains fields that legitimately change across resets without violating the sim contract:

- absolute `tick`
- dynamic `instance_id`
- absolute instance-shifted player/NPC tile coordinates

PR4 determinism checks therefore compare a semantic projection, not the raw absolute payload.

The current projection rules are:

- top-level `tick` becomes episode-relative tick
- episode-state `instance_id` is excluded
- episode-state/player/NPC tiles are normalized relative to the episode start tile

This keeps determinism checks aligned to simulator semantics rather than allocator details.

## Deterministic Eval Smoke

PR4 deterministic eval smoke uses a scripted checkpoint fixture instead of learned weights.

Current checkpoint fixture shape:

- JSON file with `checkpoint_schema = scripted_policy_checkpoint_v0`
- `policy_id` selects a deterministic scripted policy

This is intentionally thin and exists only so seed-pack eval determinism can be tested before PuffeRL checkpointing lands.

## Pre-Replay Equivalence

Until replay pack generation lands, PR4 uses deterministic serialized trajectory payloads as the pre-replay contract.

Those payloads must preserve:

- seed pack / trace pack identity
- per-step action sequence
- semantic trajectory digest
- contract-aligned final summary fields
