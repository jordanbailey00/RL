# Transport And Copy Ledger

Date: 2026-03-10

Repo and SHA:
- RL: local Phase 2 prototype on top of `ea2b1e115c3149a2a0769dc094f92368774849bb`
- fight-caves-RL: `2365506bd3ea5cce515c571f39c24e72a38acc67`

This document records what crosses boundaries on reset and step, what format it takes, and how many material copies occur.

## Current Training Transport

Current shipped train path:
- parent process: `scripts/train.py` -> `ConfigurablePuffeRL`
- worker process: `SubprocessHeadlessBatchVecEnv`
- worker-internal env: `HeadlessBatchVecEnv`
- worker-internal bridge: `HeadlessBatchClient`
- JVM boundary: JPype embedded JVM
- IPC boundary: Python `multiprocessing.Pipe`
- IPC serialization: pickle of numpy arrays plus Python `infos` list
- shared memory: none
- zero-copy flat buffers: none

Current local Phase 2 prototype path:
- same parent/worker topology as the shipped train path
- control plane: Python `multiprocessing.Pipe`
- data plane: file-backed `mmap` arrays in `fight_caves_rl/envs/shared_memory_transport.py`
- transport mode id: `shared_memory_v1`
- Production Training Mode info payload mode:
  - `full` for Certification/debug paths
  - `minimal` for current train/benchmark configs
- status: opt-in only for local review; not the default production training path
- reason for file-backed `mmap` instead of POSIX shared memory:
  - the current host rejects `multiprocessing.shared_memory`
  - file-backed `mmap` remains within the approved "shared-memory or equivalent low-copy IPC" direction

## Reset Path Ledger

### Parent -> Worker

Boundary:
- Python parent process -> Python subprocess worker

Payload:
- command name: `"reset"`
- payload: seed integer only

Format:
- Python tuple sent through `Pipe.send(...)`
- pickled by multiprocessing

Batching:
- one reset command per lockstep reset
- all envs in the worker batch reset together

Expected frequency:
- once per initial episode start
- once again whenever an episode boundary is crossed in training

### Worker -> Embedded Bridge

Boundary:
- Python worker -> embedded JVM through JPype

Per-env reset sequence:
1. build `HeadlessEpisodeConfig`
2. `reset_episode(...)`
3. `observe(...)` or `observe_jvm(...)` to produce the first observation
4. convert the raw observation to Python structures
5. encode the first observation to the flat policy tensor
6. attach reset `info` metadata, including:
   - `episode_state`
   - `bridge_handshake`
   - `batch_protocol`

Format:
- Python calls on JPype proxies
- raw observation enters Python as JVM object graph
- then becomes Python dict/list structure

### Worker -> Parent

Boundary:
- Python worker -> Python parent process

Serialized reset transition shape:
- same tensor/flag arrays as the step path
- plus a materially larger `infos` payload because reset `info` includes contract and episode metadata

Format:
- `_serialize_transition(...)` copies every array with `np.array(..., copy=True)`
- `infos` are omitted entirely when `info_payload_mode = minimal`
- the resulting payload is then pickled through the pipe

Important implication:
- reset payload bytes are not huge, but they are larger than step payload bytes because the `infos` list contains protocol and episode metadata for every slot
- long training runs amortize reset overhead, but frequent short episodes or tiny tick caps make it more visible

### Parent -> Worker

Boundary:
- Python parent process -> Python subprocess worker

Payload:
- command name: `"step"`
- payload: contiguous numpy joint action array

Format:
- Python object sent through `Pipe.send(...)`
- pickled by multiprocessing

Batching:
- one command per lockstep step
- all envs in the worker batch move together

Expected frequency:
- once per `vecenv.send(...)`

### Worker -> Embedded Bridge

Boundary:
- Python worker -> embedded JVM through JPype

Per-env step sequence:
1. decode policy action to `NormalizedAction`
2. build JVM action object
3. `apply_action_jvm(...)`
4. shared `tick(...)`
5. `observe_jvm(...)`
6. recursively pythonize observation
7. infer reward / terminal state / info
8. encode observation to flat policy array

Format:
- Python calls on JPype proxies
- raw observation enters Python as JVM object graph
- then becomes Python dict/list structure

### Worker -> Parent

Boundary:
- Python worker -> Python parent process

Serialized transition shape:
- `observations`: numpy `float32`, shape `(env_count, 134)`
- `rewards`: numpy `float32`, shape `(env_count,)`
- `terminals`: numpy `bool`, shape `(env_count,)`
- `truncations`: numpy `bool`, shape `(env_count,)`
- `teacher_actions`: numpy `int32`, shape `(env_count,)`
- `agent_ids`: numpy `int64`, shape `(env_count,)`
- `masks`: numpy `bool`, shape `(env_count,)`
- `infos`: Python list of Python dicts

Format:
- `_serialize_transition(...)` copies every array with `np.array(..., copy=True)`
- `SharedMemoryTransportWorker.publish_transition(...)` omits `infos` entirely when they are empty
- the resulting payload is then pickled through the pipe only for control-plane fields

## Step Path Ledger

## Actual Serialized Payload Sizes

Measurement note:
- step payload sizes came from an inline pickle-size probe on the current subprocess worker payload
- reset payload sizes were measured from a temporary standalone Python script, because the subprocess worker uses `spawn` and cannot reliably re-import a `<stdin>` main module
- the byte tables below are still useful directionally, but they were captured before the additive Jad telegraph field increased the flat observation width to `134`
- treat the exact byte counts as pre-Phase-2 directional evidence rather than current precise totals

Measured reset results:

| Env count | Observation batch bytes | Rewards bytes | Terminals bytes | Truncations bytes | Teacher-action bytes | Agent-id bytes | Mask bytes | Pickled `infos` bytes | Total pickled reset payload bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `504` | `4` | `1` | `1` | `4` | `8` | `1` | `1131` | `2125` |
| 4 | `2016` | `16` | `4` | `4` | `16` | `32` | `4` | `1723` | `4286` |
| 16 | `8064` | `64` | `16` | `16` | `64` | `128` | `16` | `4087` | `12926` |

Measured step results:

| Env count | Observation batch bytes | Rewards bytes | Terminals bytes | Truncations bytes | Teacher-action bytes | Agent-id bytes | Mask bytes | Pickled `infos` bytes | Total pickled step payload bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `504` | `4` | `1` | `1` | `4` | `8` | `1` | `267` | `1261` |
| 4 | `2016` | `16` | `4` | `4` | `16` | `32` | `4` | `484` | `3047` |
| 16 | `8064` | `64` | `16` | `16` | `64` | `128` | `16` | `1348` | `10187` |

Important interpretation:
- the raw numeric payload sizes are not large enough by themselves to explain the current SPS collapse
- reset payloads are larger than step payloads mainly because reset `infos` include protocol and episode metadata for every slot
- the expensive part is the object construction and repeated copying before and during serialization

## Material Copy / Reconstruction Points

Current step path copies or reconstructs data at least here:

1. JVM creates `HeadlessObservationV1` object graph.
2. JVM builds nested `LinkedHashMap` / list structures via `toOrderedMap()`.
3. JPype iteration reconstructs those values into Python dict/list objects in `_pythonize`.
4. `encode_observation_for_policy` builds a Python list of floats and then a numpy array.
5. `build_step_buffers` stacks per-slot arrays into a batch numpy array.
6. `_serialize_transition` copies each numpy array again with `copy=True`.
7. multiprocessing pickles the arrays and Python `infos` list.
8. parent process unpickles those arrays and Python dicts.
9. PufferLib copies from received arrays into its own buffers.

Current minimal-info Production Training Mode removes one part of that story:

- empty per-step `infos` no longer cross the control plane at all
- richer per-step metadata stays on the `full` path for Certification/debug use

The current design is therefore low-bytes but high-copy and high-object-churn.

## Why This Matters

Fact:
- the payload bytes are modest

Fact:
- the Python profiler shows the dominant time is in observation pythonization, not in raw numpy byte movement alone

Conclusion:
- the fundamental transport problem is not "messages are too large"
- it is "too many object graphs are built and copied before the parent ever sees a flat tensor"

## High-Confidence Implication

To move toward `100000+` SPS, the current nested observation object path needs to be replaced by a much flatter transport:
- typed flat buffers
- shared memory or other lower-copy IPC
- direct batched flat observation emission from the sim side
