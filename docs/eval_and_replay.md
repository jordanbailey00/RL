# eval_and_replay.md

This document freezes the shipped PR10 replay/eval contract.

## Canonical Entry Points

Canonical replay/eval entrypoint:

- `scripts/replay_eval.py`

Compatibility alias retained for earlier smoke/tests:

- `scripts/eval.py`

Both entrypoints execute the same `fight_caves_rl/replay/eval_runner.py` path.

## Fixed Inputs

Replay eval consumes:

- a real RL checkpoint plus checkpoint metadata
- a versioned seed pack from `fight_caves_rl/replay/seed_packs.py`
- the fixed semantic projection rules from `fight_caves_rl/replay/trace_packs.py`
- the eval config in `configs/eval/replay_eval_v0.yaml`

Current defaults:

- `seed_pack = bootstrap_smoke`
- `reward_config = use_checkpoint`
- `curriculum_config = curriculum_disabled_v0`
- `policy_mode = greedy`
- `replay_step_cadence = 1`

The shipped PR10 path evaluates real RL checkpoints.
The older PR4 scripted checkpoint fixture remains only in the thin canary utilities such as `scripts/collect_seedpack_eval.py`.

## Replay Artifacts

Every replay-eval run now writes:

- `eval_summary.json`
- `replay_pack.json`
- `replay_index.json`
- `run_manifest.json`

Schema ids:

- `replay_pack_v0`
- `replay_index_v0`

Artifact intent:

- `eval_summary.json` is the top-level human/script summary for the run
- `replay_pack.json` is the replay-grade payload with per-seed captured step traces
- `replay_index.json` is the stable deterministic lookup layer for W&B/manual debugging
- `run_manifest.json` remains the provenance contract for repo/runtime/schema metadata

## Step Cadence

`replay_step_cadence` controls replay-pack density, not simulator behavior.

Rules:

- it must be `>= 1`
- cadence `1` captures every step
- cadence `N` captures every `N`th step plus the final step
- `summary_digest` still comes from the full per-seed semantic summaries, not the downsampled replay payload

This keeps replay generation configurable without changing eval semantics.

## Determinism Projection

Replay determinism still compares semantic projections rather than allocator-dependent raw payloads.

The raw sim observation contains fields that legitimately vary across resets:

- absolute `tick`
- dynamic `instance_id`
- absolute instance-shifted player/NPC tile coordinates

The current projection rules are:

- top-level `tick` becomes episode-relative tick
- episode-state `instance_id` is excluded
- player/NPC tiles are normalized relative to the episode start tile

That projection is what drives:

- per-seed `trajectory_digest`
- replay-pack `replay_digest`
- run-level `summary_digest`

## Thin Canary Continuity

PR10 does not replace the PR4 thin canary path.

The following still remain valid:

- versioned seed packs
- versioned trace packs
- replay-to-trace parity canaries
- `scripts/collect_seedpack_eval.py` for scripted-policy deterministic checks

PR10 extends the real checkpoint eval path with replay artifacts instead of replacing those thinner tools.
