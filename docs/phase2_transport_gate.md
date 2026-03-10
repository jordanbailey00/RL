# Phase 2 Transport Promotion Gate

Date: 2026-03-10

This document defines the source-of-truth gate used to decide whether `WC-P2-03` may begin.

`WC-P2-03` is the production transport swap. It must not start only because a local prototype exists. It starts only after the low-copy transport shows a meaningful advantage on the approved native-Linux host class.

## Purpose

Phase 2 currently has two different questions:

1. Is the low-copy transport technically valid?
2. Is it strong enough to justify replacing the shipped default training path?

`WC-P2-02` answered the first question locally.

This gate answers the second question on the source-of-truth host class.

## Required Packet

Standard refresh entrypoint:

```bash
source /home/jordan/code/.workspace-env.sh
cd /home/jordan/code/RL
uv run python scripts/refresh_phase2_packet.py --output-dir /tmp/fc_phase2_packet
```

Hosted native-Linux source-of-truth path:

- `fight-caves-RL/.github/workflows/phase2_native_linux_packet.yml`

## Required Rows

The packet must include:

- subprocess transport comparison rows:
  - `16 env`
  - `64 env`
- train benchmark rows with disabled logging for both transport modes:
  - `pipe_pickle_v1`, `16 env`
  - `shared_memory_v1`, `16 env`
  - `pipe_pickle_v1`, `64 env`
  - `shared_memory_v1`, `64 env`

## Gate Meaning

This is not the final Phase 2 continue-versus-pivot gate.

This is the narrower pre-swap gate for `WC-P2-03`.

It answers:

- does the low-copy transport beat the current shipped transport enough to justify making it the default training path?

## Promotion Thresholds

These are planning thresholds, not guaranteed outcomes.

`WC-P2-03` may start only if all of the following are true on native Linux:

- transport comparison rows are complete
- train comparison rows are complete
- `shared_memory_v1` transport benchmark at `64 env` is at least `1.20x` the `pipe_pickle_v1` row
- `shared_memory_v1` disabled training row at `64 env` is at least `1.10x` the `pipe_pickle_v1` row
- `shared_memory_v1` disabled training row at `64 env` is at least `1.25x` the `shared_memory_v1` `16 env` row

Interpretation:

- the first threshold proves the transport itself is materially better
- the second threshold proves the benefit survives end-to-end training
- the third threshold proves scaling is actually improving instead of staying flat

## Block Conditions

Keep `WC-P2-03` blocked if any of the following are true:

- source-of-truth host class is missing
- transport packet is incomplete
- train packet is incomplete
- transport speedup signal is weak
- end-to-end training speedup signal is weak
- end-to-end `16 -> 64` scaling signal is weak

## What Happens Next

If this gate passes:

- start `WC-P2-03`
- make the low-copy path the default subprocess training transport
- keep the old path accessible for Certification Mode and rollback/debug comparisons

If this gate fails:

- do not promote the low-copy path
- keep `WC-P2-03` blocked
- continue Phase 2 work only through another transport iteration or a justified escalation path

## Current Source-of-Truth Result

Hosted native-Linux run:

- [fight-caves-RL/actions/runs/22882424149](https://github.com/jordanbailey00/fight-caves-RL/actions/runs/22882424149)

Published gate summary:

- [codex/phase2-results latest gate summary](https://github.com/jordanbailey00/fight-caves-RL/blob/codex/phase2-results/phase2-native-linux/latest/gate_summary.json)

Current result:

- benchmark host class: `linux_native`
- transport `64 env`:
  - pipe: `7793.12` env/s
  - `shared_memory_v1`: `10906.45` env/s
  - speedup: `1.3995x`
- disabled train `16 env`:
  - pipe: `49.44` SPS
  - `shared_memory_v1`: `50.74` SPS
- disabled train `64 env`:
  - pipe: `48.15` SPS
  - `shared_memory_v1`: `48.38` SPS
  - speedup: `1.0048x`
- shared-train scaling ratio `64 vs 16`: `0.9534x`
- gate result: `wc_p2_03_unblocked = false`

Blockers:

- `train_signal_too_weak`
- `shared_train_scaling_too_weak`

Interpretation:

- the low-copy transport materially improves the transport microbenchmark on native Linux
- that benefit does not currently survive end-to-end training
- `WC-P2-03` remains blocked for real performance reasons, not workflow/plumbing reasons
