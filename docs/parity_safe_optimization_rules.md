# Parity-Safe Optimization Rules

Date: 2026-03-09

This document defines what future performance work is allowed to change and what it must preserve.

## Non-Negotiable Invariants

- Episode-start contract must remain identical to the sim-owned reset contract.
- Action semantics must remain identical.
- Observation semantics must remain identical unless the observation schema version is intentionally changed and revalidated.
- Decision-critical combat cues already present in the raw contract, including `jad_telegraph_state`, must retain identical onset, duration, and meaning unless intentionally versioned and revalidated.
- Fight-caves headless runtime remains the golden runtime dependency.
- RSPS remains the oracle/reference path, not the RL hot path.
- Deterministic replay expectations must be preserved for the same schema / contract / seed version.

## Performance Refactors That Are Allowed

- lower-copy transport changes
- shared-memory or flat-buffer IPC
- moving observation flattening closer to the sim boundary
- replacing nested map/list emission with semantically equivalent typed layouts
- reducing redundant conversions and copies
- batching reset/action/observe calls
- changing worker topology
- changing logging cadence defaults
- disabling or deferring expensive non-semantic bookkeeping in benchmark mode

## Changes That Are Risky And Must Be Treated As Parity-Sensitive

- changing visible-NPC ordering
- changing target indexing semantics
- changing reset loadout/stats/consumables
- changing tick stride or action timing
- changing terminal conditions
- changing observation field meaning, even if the final tensor shape stays the same
- changing the meaning or timing window of `jad_telegraph_state` or any equivalent combat telegraph cue
- changing any logic that could alter headless-to-headed transfer behavior

## Determinism Expectations

- same seed and same contract version must remain replay-stable where the existing determinism suite requires it
- if transport changes but semantics do not, deterministic replay and parity canaries must still pass

## Required Re-Validation After Optimization Batches

At minimum:
- `fight_caves_rl/tests/integration`
- `fight_caves_rl/tests/determinism`
- `fight_caves_rl/tests/parity`
- `fight_caves_rl/tests/smoke`
- relevant `fight_caves_rl/tests/performance`
- `scripts/run_parity_canary.py`
- `scripts/replay_eval.py` on a known checkpoint
- benchmark packet rows affected by the optimization

For the future flat training path specifically:

- source-side raw-vs-flat equivalence gate from [raw_flat_equivalence_plan.md](/home/jordan/code/fight-caves-RL/docs/raw_flat_equivalence_plan.md)
- RL-side consumer equivalence and fail-fast checks from [flat_observation_ingestion.md](/home/jordan/code/RL/docs/flat_observation_ingestion.md)

## Acceptance Rule

An optimization is acceptable only if:

1. it improves the targeted benchmark layer with real measurements
2. it does not regress the parity / determinism gates
3. it does not silently change schema or contract meaning

## Audit Position

The current highest-value future optimizations are transport and observation-path changes, but they must be executed as performance-only refactors until proven otherwise.
