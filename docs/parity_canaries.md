# parity_canaries.md

This document freezes the PR12 expanded parity-canary layer.

## Purpose

PR12 parity canaries are the repo-owned oracle-reference validation layer for fixed trace/seed packs.

They are intentionally:

- versioned
- inspectable
- tied back to known sim parity scenarios from `/home/jordan/code/fight-caves-RL`
- separate from the production training hot path

## Current Entry Point

Current PR12 entry surfaces:

- config: `configs/eval/parity_canary_v0.yaml`
- runner: `scripts/run_parity_canary.py`
- seed pack: `parity_reference_v0`

The current config expands the parity matrix across:

- `parity_single_wave_v0` with seed `11001`
- `parity_action_rejection_v0` with seed `33003`
- `parity_prayer_toggle_timing_v0` with seed `11001`
- `parity_jad_healer_v0` with seed `33003`
- `parity_tzkek_split_v0` with seed `44004`
- `parity_terminal_tick_cap_v0` with seed `11001`

Each trace pack remains a per-tick RL expansion of the sim-side replay-style parity traces.
The PR 6.2-only packs pin the new V2 mechanics gate for rejection codes, prayer timing, and
tick-cap terminal codes on top of the original wave/Jad-era parity scenarios.

## Current Comparison Surfaces

Each parity scenario now compares three RL-facing semantic surfaces:

- wrapper trace via `scripts/collect_trajectory_trace.py --mode wrapper`
- raw sim trace via `scripts/collect_trajectory_trace.py --mode raw`
- trace-pack-driven scripted replay path via `scripts/smoke_scripted.py`

It also compares two mechanics-parity surfaces:

- oracle mechanics trace via `scripts/collect_mechanics_parity_trace.py --mode oracle`
- V2 fast mechanics trace via `scripts/collect_mechanics_parity_trace.py --mode v2_fast`

The configured comparison mode remains `semantic_digest`, but the parity runner also checks:

- semantic episode-state agreement
- semantic initial-observation agreement
- per-step action agreement
- per-step semantic-observation agreement
- per-step action-result agreement
- per-step semantic visible-target agreement
- oracle-vs-`v2_fast` mechanics digest agreement
- first-mismatch field comparison on the shared mechanics trace schema
- final relative tick agreement
- completed-all-steps agreement

The semantic digest intentionally ignores allocator-specific fields such as dynamic instance ids and absolute instance-shifted tiles.
The mechanics digest is frozen per trace pack and covers the shared parity trace fields from
`fight_caves_rl/contracts/parity_trace_schema.py`, including action acceptance, rejection codes,
visible-target ordering, Jad telegraph fields, and terminal codes.

## Process Model

Fresh-runtime parity comparisons must run in separate Python processes.

Reasons:

- the embedded JVM is process-global in Mode A
- multiple bootstraps in one Python process are not a valid fresh-runtime comparison path
- multiple player slots in one runtime do not guarantee identical absolute reset state

## Oracle Boundary

The trace packs and seed pack are sourced from sim-side parity scenarios and remain the reference inputs for RL-side drift checks.

`RSPS` remains an oracle/reference module for parity disputes and headed debugging, but PR12 does not pull `RSPS` into the RL runtime hot path.

## Current Limitations

The PR12 "replay-to-trace equivalence" path refers to the trace-pack-driven scripted replay path from `scripts/smoke_scripted.py`, not the checkpoint replay-eval path from `scripts/replay_eval.py`.

Checkpoint replay determinism remains covered by the PR10 replay artifact contract and the determinism suite.

The Mode A surface still does not expose a dedicated simulator terminal-reason envelope.

Parity canaries therefore preserve the PR3 documented inferred-only labels for:

- `player_death`
- `cave_complete`
- `max_tick_cap`

If the sim later exposes direct terminal reasons, the canary digest version must be bumped accordingly.

## Failure Artifacts

When oracle-vs-`v2_fast` mechanics parity fails, the runner writes a first-divergence artifact under:

- `artifacts/parity/pr62_canary_failures/<config_id>/<scenario_id>.json`

Those artifacts store the reference trace, candidate trace, and the first mismatched field so parity
drift can be debugged without rerunning the full canary matrix immediately.
