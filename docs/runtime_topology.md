# Runtime Topology

Date: 2026-03-11

Repos and SHAs:
- RL: `cda7ab4104799be40ffe39f77e5a86c2e6f0eea5`
- fight-caves-RL: `2365506bd3ea5cce515c571f39c24e72a38acc67`
- RSPS: `ec5f6d0d307fe1072b134693f636e10a22873ce0`

## Host

- OS: `Linux DESKTOP-61IQ04T 6.6.87.2-microsoft-standard-WSL2`
- Topology: WSL2 guest on Windows host
- Benchmark host class: `wsl2`
- Performance source of truth on this host: `true`
- CPU: `AMD Ryzen 5 5600G with Radeon Graphics`
- Physical cores: `6`
- Logical CPUs: `12`
- RAM: `15 GiB`
- Available RAM at audit start: about `14 GiB`
- Storage for workspace root: `/dev/sdd`, about `911 GiB` free

## Language Runtimes

- Python: `3.11.15`
- Torch: `2.10.0+cpu`
- CUDA: unavailable
- Torch threads: `6`
- JVM: Temurin `21.0.10+7`
- PufferLib distribution: `pufferlib-core 3.0.17`

## RL Runtime Topologies Used

### Embedded Correctness / Env Benchmark Topology

- processes: `1`
- Python process embeds the JVM through JPype
- no IPC between env wrapper and bridge
- batching lives inside one `HeadlessBatchClient`
- used for:
  - bridge benchmarks
  - env benchmarks
  - correctness tooling

### Embedded Fast V2 Serial Topology

- processes: `1`
- Python process embeds the JVM through JPype
- no IPC between the trainer/env wrapper and the fast kernel
- batching lives inside one `FastFightCavesKernelRuntime`
- used for:
  - serial fast vecenv smoke
  - early V2 wrapper bring-up
  - direct kernel bring-up / debug

### Shipped Training Topology

- parent process:
  - PufferLib trainer
  - policy
  - logging
  - artifacts
- child process:
  - `SubprocessHeadlessBatchVecEnv`
  - embedded JPype/JVM
  - `HeadlessBatchVecEnv`
  - `HeadlessBatchClient`
- worker count: `1`
- envs per worker: all envs in the run
- IPC mechanism: `multiprocessing.Pipe`
- serialization: pickle of numpy arrays plus Python `infos` list
- shared memory: none
- affinity / pinning: none configured
- subprocess start method: `spawn`

### Fast V2 Subprocess Training Topology

- parent process:
  - PufferLib trainer
  - policy
  - logging
  - artifacts
- child processes:
  - `SubprocessHeadlessBatchVecEnv` worker shards
  - embedded JPype/JVM per worker
  - worker-local `FastKernelVecEnv`
  - worker-local `FastFightCavesKernelRuntime`
- worker count: configurable through `env.subprocess_worker_count`
- envs per worker: balanced partition of the joint batch
- IPC mechanism:
  - control plane via `multiprocessing.Pipe`
  - actions/results via either pickle or per-worker shared-memory slabs
- active shared-memory mode:
  - `shared_memory_v1`
- subprocess start method: `spawn`
- benchmark/smoke topology outputs now record:
  - `backend`
  - `env_backend`
  - `transport_mode`
  - `worker_count`
  - `worker_env_counts`

## Current Headless Sim Artifact Boundary

- canonical artifact task: `:game:headlessDistZip`
- canonical fallback / validation task: `:game:packageHeadless`
- current artifact path seen by RL benchmarks:
  - `/home/jordan/code/fight-caves-RL/game/build/distributions/fight-caves-headless-dev.zip`

## Machine Interpretation

Facts:
- this machine is not obviously memory-starved
- this machine is not using GPU acceleration
- the current RL runtime surface now includes both:
  - single-process embedded bring-up paths
  - configurable multi-worker subprocess paths
- the current benchmark metadata now records:
  - `host_class`
  - `is_wsl`
  - `performance_source_of_truth`
  - `java_runtime_version`
  - `java_vm_name`

Hypothesis:
- WSL2 adds some overhead, but the measured bridge and vecenv saturation points are low enough that the main issue is the current architecture, not merely host under-provisioning

Reason:
- bridge and vecenv top out around `1.5k` env steps/s before the learner dominates
- training plateaus around `88` SPS by `64 envs`
- those numbers are too low to blame primarily on minor host effects

## Phase 0 Gate Note

- clean standalone sim and clean standalone JFR artifacts now exist on this host class
- the refreshed Phase 0 packet is complete for:
  - bridge `1 / 16 / 64`
  - vecenv `1 / 16 / 64`
  - train `4 / 16 / 64`
- WSL is now the approved source-of-truth host class for the active pivot benchmark contract
