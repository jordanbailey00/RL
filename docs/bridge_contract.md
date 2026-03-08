# Bridge Contract

This document freezes the early bridge strategy so PR 3 does not build the correctness wrapper on a dead-end transport.

## Constraints

- `fight-caves-RL` is the golden runtime dependency.
- The default RL artifact boundary is the packaged headless distribution, not an IDE run config.
- RSPS is out-of-band for parity/debug only and must stay out of the training hot path.
- Bridge work must preserve exact simulator semantics and fail fast on contract/version drift.

## Selected Artifact Strategy

Default dev/test artifact:

- packaged headless distribution from `fight-caves-RL`
- build task: `:game:headlessDistZip`
- validation/aggregator task: `:game:packageHeadless`
- distribution glob: `game/build/distributions/fight-caves-headless*.zip`
- current verified dev output: `game/build/distributions/fight-caves-headless-dev.zip`

Expected extracted contents:

- `fight-caves-headless.jar`
- `game.properties`
- `config/headless_data_allowlist.toml`
- `config/headless_manifest.toml`
- `config/headless_scripts.txt`
- allowlisted headless `data/*.toml`
- `run-headless.sh`
- `run-headless.bat`

Current runtime invariant:

- the packaged distribution is the classpath/artifact boundary for RL
- the current sim bootstrap still requires the checked-out `fight-caves-RL` workspace root at runtime
- specifically, PR3 correctness mode requires:
  - `FCspec.md`
  - `config/headless_data_allowlist.toml`
  - `config/headless_manifest.toml`
  - `config/headless_scripts.txt`
  - populated `data/cache/main_file_cache.dat2`
- packaged dist alone is therefore not sufficient today

Fallback order:

1. Use an existing extracted/zipped distribution if it matches the intended sim commit/worktree.
2. Build `:game:headlessDistZip`.
3. Build `:game:packageHeadless` if packaging validation is also needed.
4. Use a jar/classpath fallback only for manual debugging, never as the default CI/runtime artifact boundary.

## Selected Mode A - Correctness Bring-Up

Selected Mode A direction:

- embedded JVM direct-runtime bridge from Python
- primary implementation target: a JNI-style Python/JVM bridge library, with `jpype1` as the default candidate
- artifact input: the packaged `fight-caves-headless` distribution, using the jar inside that distribution as the JVM classpath root

Why Mode A is selected:

- the sim already exposes a stable in-process runtime surface through `HeadlessMain.bootstrap(...)` and `FightCaveSimulationRuntime`
- the packaged artifact does not currently expose a dedicated external bridge server entrypoint
- direct runtime calls preserve simulator exceptions, return types, and method boundaries for correctness-first bring-up
- this keeps PR 3 focused on semantic transparency instead of inventing a request protocol too early

Mode A launch shape:

- one Python worker owns one embedded JVM runtime
- PR3 launches the JVM with the jar extracted from the packaged dist while setting the process cwd inside the checked-out sim workspace so the current repo-root discovery code succeeds
- JPype/JVM startup is process-global for correctness mode, so independent fresh-runtime equivalence checks must use separate Python processes rather than multiple bootstraps inside one process
- the wrapper provisions player slots using the same setup path as the sim's headless tests
- correctness bring-up should default to `loadContentScripts = true`
- correctness bring-up should default to `startWorld = true` for full reset/step/observation parity, while `startWorld = false` remains valid for narrower bootstrap/perf micro-cases explicitly validated by the sim tests
- shutdown hooks should be disabled in managed RL test/process lifecycles

Mode A guarantees:

- exact use of the current runtime methods:
  - `resetFightCaveEpisode(...)`
  - `visibleFightCaveNpcTargets(...)`
  - `applyFightCaveAction(...)`
  - `observeFightCave(...)`
  - `tick(...)`
  - `shutdown()`
- raw JVM exceptions and contract mismatches are surfaced directly to Python
- no wrapper-local reinterpretation of action or observation semantics

## Provisional Mode B - Batched Production Bridge

Mode B direction:

- dedicated JVM subprocess per Python worker
- coarse-grained batch reset/step boundaries
- explicit schema/version handshake before env creation
- binary/control channel preferred over ad hoc text transport

Target topology:

- one Python worker controls one JVM runtime process
- one JVM runtime process owns many active episode/player slots
- one Python bridge call submits/receives a batch for many slots

## Provisional Mode C - High-Throughput Vector Backend

Mode C direction:

- keep the batched subprocess topology from Mode B
- move payloads toward flat numeric buffers with minimal Python/JVM crossings
- use a lightweight control channel plus low-copy batch payloads
- optimize for the staged path to `>= 1,000,000 env steps/sec`, not just correctness-mode convenience

## Selection Criteria

Every bridge implementation must be judged on:

- semantic transparency to the sim contract
- explicit version/schema handshake quality
- copy count and allocation pressure
- failure visibility and debuggability
- ability to batch many env slots cleanly
- operational simplicity on Linux/WSL

## Frozen Handshake Fields

The bridge must validate these before stepping:

- `observation_schema_id`
- `observation_schema_version`
- `action_schema_id`
- `action_schema_version`
- `episode_start_contract_id`
- `episode_start_contract_version`
- `bridge_protocol_id`
- `bridge_protocol_version`
- `benchmark_profile_id`
- `benchmark_profile_version`
- `sim_artifact_task`
- `sim_artifact_path`
- `pufferlib_distribution`
- `pufferlib_version`

## Versioning Rule

- PR 3 correctness mode records `bridge_protocol_v0`
- if PR 7 changes the batch envelope, transport semantics, or handshake surface materially, the bridge protocol version must increment rather than drifting silently
