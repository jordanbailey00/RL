# parity_canaries.md

This document freezes the PR4 thin parity-canary layer.

## Purpose

PR4 parity canaries are early smoke checks that ensure the RL wrapper/bridge path does not introduce semantic drift before training complexity is added.

They are intentionally thin:

- versioned
- fast enough to run in normal PR validation
- tied back to known sim parity scenarios where possible

## Current Canary Inputs

The first RL canaries are sourced from the sim parity harness scenarios in `/home/jordan/code/fight-caves-RL`:

- single-wave trace
- Jad healer trace
- Tz-Kek split trace

In RL, those traces are expanded from replay-style `ticksAfter` steps into per-tick env actions.

## Current Comparison Mode

The current canary comparison mode is `semantic_digest`.

That digest is computed from:

- semantic initial observation
- per-step action sequence
- per-step semantic observations
- action results
- semantic visible-target payloads
- terminal/truncation labels currently available through the PR3 surface

It intentionally ignores allocator-specific fields such as dynamic instance ids and absolute instance-shifted tiles.

## Process Model

Fresh-runtime wrapper-vs-raw comparisons must run in separate Python processes.

Reasons:

- the embedded JVM is process-global in Mode A
- multiple bootstraps in one Python process are not a valid fresh-runtime comparison path
- multiple player slots in one runtime do not guarantee identical absolute reset state

## Current Limitation

The Mode A surface still does not expose a dedicated simulator terminal-reason envelope.

Parity canaries therefore preserve the PR3 documented inferred-only labels for:

- `player_death`
- `cave_complete`
- `max_tick_cap`

If the sim later exposes direct terminal reasons, the canary digest version must be bumped accordingly.
