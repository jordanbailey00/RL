# Observation And Action Cost Report

Date: 2026-03-09

Repo and SHA:
- RL: `cda7ab4104799be40ffe39f77e5a86c2e6f0eea5`
- fight-caves-RL: `2365506bd3ea5cce515c571f39c24e72a38acc67`

## Observation Schema Cost

### Policy Observation Tensor

Source:
- `fight_caves_rl/envs/puffer_encoding.py`

Facts:
- flattened observation size: `126`
- dtype: `float32`
- fixed tensor bytes per env per step: `504`

Layout:
- base fields: `30`
- NPC slot fields: `12`
- max visible NPC slots: `8`
- total = `30 + 8 * 12 = 126`

### Raw Observation Structure

Current raw observation path:
- JVM returns `HeadlessObservationV1`
- `HeadlessObservationV1.toOrderedMap()` creates nested ordered maps and lists
- Python `_pythonize(...)` recursively rebuilds that into Python dict/list objects
- `encode_observation_for_policy(...)` then flattens the raw dict into the final tensor

Variable-size region:
- `npcs` list is variable-size in the raw observation
- policy encoding pads to `8` slots

### Ordering / Sorting Work

Evidence:
- `HeadlessActionAdapter.visibleNpcTargets(...)` sorts visible NPCs every time via:
  - level
  - x
  - y
  - id
  - index

Code path:
- `fight-caves-RL/game/src/main/kotlin/HeadlessActionAdapter.kt`

Interpretation:
- visible target ordering is parity-safe and intentional
- it is also per-step work that cannot be ignored when optimizing the observation path

### Python-Side Observation Costs

From steady-state `16 env` profiles over `1024` env steps:

| Function | Cum time (s) |
| --- | ---: |
| `pythonize_observation` | `0.711 - 0.725` |
| `_pythonize` | `0.705 - 0.722` |
| `encode_observation_for_policy` | `0.028` |
| `visible_targets_from_observation` | `0.009 - 0.011` |

Interpretation:
- flattening to the final tensor is not the main cost
- converting the nested JVM object graph into Python dict/list objects is the main cost

## Action Schema Cost

### Policy Action Tensor

Source:
- `fight_caves_rl/envs/puffer_encoding.py`

Facts:
- action lanes: `6`
- current discrete factors: `(7, 16384, 16384, 4, 8, 3)`
- if stored as contiguous `int32`, raw payload is about `24` bytes per env
- if cast to `int64`, raw payload is `48` bytes per env

### Action Decode / Validation Cost

From steady-state `16 env` profiles over `1024` env steps:

| Function | Cum time (s) |
| --- | ---: |
| `decode_action_from_policy` | `0.017 - 0.019` |
| `normalize_action` | `0.005 - 0.007` |
| `_validate_action_vector` | `0.004` |
| `build_action` | `0.005` |
| `_build_jvm_action` | `0.002 - 0.004` |

Interpretation:
- action decode and validation are real but small
- they are not the main bottleneck

### Visible Target Indexing Cost

Per-step behavior:
- the policy uses a visible-NPC index
- the sim side re-derives visible target ordering every step
- Python also rebuilds a `visible_targets` list from the observation in `visible_targets_from_observation(...)`

Cost:
- visible-target postprocessing in Python is small relative to `_pythonize`
- the sim-side visible-target sort still matters, but this audit did not obtain a clean JVM-only sample for its exact share

## Main Cost Conclusions

Facts:
- observation tensor size itself is small: `504` bytes per env
- action tensor size itself is very small
- action handling is not the main cost
- policy flattening is not the main cost

Strong conclusion:
- the expensive part of the schema is the current raw observation representation and conversion path, not the final fixed-size policy tensor
