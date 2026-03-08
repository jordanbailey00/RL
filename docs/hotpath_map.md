# Hot Path Map

This document identifies the intended runtime path for correctness and throughput work.

## Mode A - Correctness Bring-Up

1. `PuffeRL` / smoke policy loop
2. RL env wrapper
3. action mapping
4. embedded-JVM bridge launcher
5. `HeadlessMain.bootstrap(...)`
6. player provisioning
7. `resetFightCaveEpisode(...)` / `applyFightCaveAction(...)` / `tick(...)` / `observeFightCave(...)`
8. observation mapping
9. RL env returns observation/reward/done/info

Primary costs to measure, not optimize away semantically:

- Python/JVM boundary calls
- player-slot provisioning/reset
- observation packing
- per-step Python object churn

## Mode B - PR7 Batched Bridge

1. policy produces batch of actions
2. vector env batches env slots
3. batch client normalizes actions and validates the versioned batch protocol
4. JVM runtime applies actions for many player slots
5. bridge advances one shared runtime tick
6. bridge returns packed batch results
6. vector env scatters results back to slots

Primary costs:

- action normalization/building
- per-slot apply cost before the shared tick
- per-slot error classification
- observation/result packing

Current PR7 implementation notes:

- transport stays in-process for now
- visible targets are derived from `observation.npcs` in the batch path instead of requiring a second runtime query
- single-slot trace benchmarking can bypass per-step Python loops by reusing the sim-side `runFightCaveBatch(...)` helper

## Mode C - High-Throughput Vector Target

1. policy inference on batched observations
2. low-copy batch transport
3. JVM batched step/reset
4. low-copy observation/result buffers
5. vectorized trainer collection/update

Primary costs:

- copy count
- buffer layout conversions
- logging/replay overhead
- worker topology/JVM startup amortization

## Optimization Order

Optimize in this order:

1. semantic correctness and fail-fast handshake
2. boundary crossing count
3. observation/action packing
4. worker/JVM topology / transport swap
5. logging cadence and replay cadence
6. policy/trainer overhead

## Guardrail

No optimization is allowed to change:

- episode reset semantics
- action semantics
- observation semantics
- deterministic replay behavior
