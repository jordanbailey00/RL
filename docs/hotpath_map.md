# Hot Path Map

Date: 2026-03-09

Repos and SHAs:
- RL: `cda7ab4104799be40ffe39f77e5a86c2e6f0eea5`
- fight-caves-RL: `2365506bd3ea5cce515c571f39c24e72a38acc67`
- RSPS: `ec5f6d0d307fe1072b134693f636e10a22873ce0`

This is the concrete runtime hot path for one training step on the current shipped RL stack.

Measured training topology for this map:
- parent process: `scripts/train.py` + `ConfigurablePuffeRL`
- child process: `SubprocessHeadlessBatchVecEnv`
- env worker count: `1`
- envs per worker: all envs in the run
- JVM topology: one embedded JPype/JVM inside the child process
- transport: `multiprocessing.Pipe` with pickle

## Current Shipped Train Path

```text
scripts/train.py
-> fight_caves_rl.puffer.trainer.run_smoke_training(...)
-> ConfigurablePuffeRL
-> SubprocessHeadlessBatchVecEnv (parent process)
-> multiprocessing Pipe send/recv
-> SubprocessHeadlessBatchVecEnv worker process
-> HeadlessBatchVecEnv
-> HeadlessBatchClient.step_batch(...)
-> HeadlessDebugClient.apply_action_jvm(...) / tick(...) / observe_jvm(...)
-> embedded JVM / HeadlessMain runtime
-> JVM builds HeadlessObservationV1
-> JVM builds ordered map/list object graph
-> Python worker recursively pythonizes observation/action result
-> Python worker encodes observation to float32 policy vector
-> Python worker copies arrays in _serialize_transition(...)
-> Pipe pickles payload back to parent
-> parent unpickles numpy arrays + infos
-> PuffeRL evaluate/train/update/logging
```

## Boundary Ledger

| Stage | File / function | Process boundary | Thread boundary | Sync / blocking point | Transport | Serialization | Batching | Data shape | Material copies | Expected frequency | Current cost signal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| CLI entry | `scripts/train.py` -> `run_smoke_training` | none | main thread | config load | n/a | n/a | run-level | yaml / json / manifest | low | once per run | not hot |
| Trainer setup | `ConfigurablePuffeRL.__init__` | none | main thread | blocks on vecenv reset | Python calls | Python objects | run-level | initial reset batch | moderate | once per run | startup-heavy |
| Parent vecenv send | `SubprocessHeadlessBatchVecEnv.send` | parent -> worker | main thread -> child proc | `Pipe.send` | pipe | pickle | lockstep batch | joint action array | 1+ | every step | moderate |
| Worker receive | `_subprocess_vecenv_worker` | same as above | child proc | `conn.recv` | pipe | pickle | lockstep batch | action array | 1 | every step | moderate |
| Vecenv shell | `HeadlessBatchVecEnv.send` | none | worker main thread | blocks through bridge path | Python calls | Python objects | all envs in one batch | `(env_count, 6)` joint action | low | every step | not dominant |
| Action decode | `_decode_joint_action` / `decode_action_from_policy` | none | worker main thread | none | n/a | numpy -> Python scalars | per env | 6-lane vector | low | every env every step | small |
| Bridge step | `HeadlessBatchClient.step_batch` | Python -> JVM | worker main thread -> embedded JVM threads | apply/tick/observe calls | JPype JNI | JVM proxy objects | lockstep across envs | per-env action + shared tick | medium | every step | dominant bridge entry |
| Action apply | `HeadlessDebugClient.apply_action_jvm` | Python -> JVM | same | JPype call | JNI | typed action object | per env | one action | low | every env every step | small |
| Shared tick | `HeadlessDebugClient.tick` | Python -> JVM | same | JPype call | JNI | primitive int | one call per batch | tick count | low | every batch step | small |
| Observe | `HeadlessDebugClient.observe_jvm` | Python -> JVM | same | JPype call | JNI | JVM object return | per env | `HeadlessObservationV1` | medium | every env every step | raw call not dominant |
| JVM observation build | `HeadlessObservationBuilder.build` and `toOrderedMap` | inside JVM | JVM threads | synchronous | in-memory | Kotlin object graph + ordered maps/lists | per env | nested structured observation | unknown | every env every step | unresolved clean JVM sample |
| Pythonize observation | `pythonize_observation` / `_pythonize` | JVM -> Python | worker main thread | blocks on recursive iteration | JPype collection iteration | nested dict/list reconstruction | per env | raw nested observation | high | every env every step | dominant |
| Reward / done bookkeeping | `_collect_step_results` | none | worker main thread | synchronous | Python objects | dict/list ops | per env | reward, terminal info | low | every env every step | small |
| Policy encode | `encode_observation_for_policy` | none | worker main thread | synchronous | n/a | Python list -> numpy float32 | per env then batch | 126 float32s | medium | every env every step | secondary |
| Batch buffer pack | `build_step_buffers` | none | worker main thread | synchronous | n/a | numpy stack | batch | `(env_count, 126)` | medium | every batch step | secondary |
| Worker serialize | `_serialize_transition` | worker -> parent | child proc | `conn.send` | pipe | pickle of numpy arrays + `infos` | batch | arrays + Python infos | high | every batch step | major |
| Parent receive | `SubprocessHeadlessBatchVecEnv.recv` | worker -> parent | child proc -> parent thread | `conn.recv` | pipe | pickle | batch | arrays + Python infos | high | every batch step | major |
| Trainer rollout | `pufferlib.pufferl.evaluate` | none | parent main thread | waits on vecenv | Python + torch | numpy / tensor conversion | batch | observations/actions | medium | every rollout chunk | secondary |
| Learner update | `pufferlib.pufferl.train` | none | parent main thread | synchronous | torch | tensor ops | minibatch | model activations and grads | medium | every update | smaller than env collection in current runs |
| Logging / artifacts | `WandbRunLogger`, checkpoints, manifests | optional external network | extra threads in W&B | blocking at init / flush / upload | files / HTTP | JSON + artifact writes | run-level and periodic | manifests, checkpoints | medium | periodic | real but secondary |

## Current Data Shapes

- policy observation tensor: `126 x float32 = 504 bytes per env`
- policy action tensor: `6` lanes, typically contiguous `int32` or cast to `int64`
- serialized reset payloads observed on the shipped subprocess path:
  - `1 env`: `2125` pickled bytes
  - `4 env`: `4286` pickled bytes
  - `16 env`: `12926` pickled bytes
- subprocess step payload examples:
  - `1 env`: `1261` pickled bytes
  - `4 env`: `3047` pickled bytes
  - `16 env`: `10187` pickled bytes

## Where Time Is Currently Going

Strong facts from steady-state Python profiles:

- observation pythonization dominates worker-side Python time
- raw `observe_jvm`, `tick`, and `apply_action_jvm` are much smaller than `_pythonize`
- embedded vecenv throughput closely tracks embedded bridge throughput
- training throughput collapses again after adding subprocess IPC and trainer lifecycle work

## Current Interpretation

- `PufferLib` vectorization topology is not the primary embedded-path problem
- nested observation construction and conversion is the primary hot-path cost on the current stack
- the subprocess transport adds another major layer of cost on top of that
- the current architecture is stability-first, not throughput-first
