# Flat Observation Ingestion Design

Date: 2026-03-09

This document defines the Phase 1 RL-side ingestion design for the future sim-owned flat training observation schema.

## Purpose

The goal is to consume the future flat training path without reconstructing raw objects in Python.

This design is for `WC-P1-03`.
It is design-frozen, not implemented yet.

The corresponding sim-owned schema design is:

- [flat_training_observation_schema.md](/home/jordan/code/fight-caves-RL/docs/flat_training_observation_schema.md)

## Input Contract

RL will consume:

- `schema_id = headless_training_flat_observation_v1`
- `schema_version = 1`
- dtype `float32`
- shape `[env_count, 134]`
- row-major contiguous layout

This initial flat schema intentionally mirrors the current trainer tensor layout documented in:

- [observation_mapping.md](/home/jordan/code/RL/docs/observation_mapping.md)

## Consumer-Side Design

### Hot-path rule

The production training hot path must not:

- call `_pythonize`
- reconstruct nested dict/list observation objects
- remap string ids in Python
- restack per-env vectors into a second equivalent batch array

### Target ingestion path

RL should ingest the batch as:

- one validated contiguous batch buffer
- exposed to Python as a direct `numpy.ndarray` view
- passed through vecenv/trainer code as the already-flat observation tensor

The intended steady-state shape is:

- `obs.shape == (env_count, 134)`
- `obs.dtype == np.float32`

### Transport independence

This design is transport-agnostic.

The same consumer contract should work whether the data plane later uses:

- shared memory
- mmap
- a custom ring buffer
- another lower-copy batch transport

Phase 1 does not choose transport.
It chooses the consumer shape that later transport must satisfy.

## Schema Validation

RL must validate the flat schema at startup/handshake time before trusting Production Training Mode.

Required handshake/consumer checks:

- `flat_observation_schema_id`
- `flat_observation_schema_version`
- `flat_observation_dtype`
- `flat_observation_feature_count`
- `flat_observation_max_visible_npcs`

Validation failure must be fail-fast.
RL must not silently coerce incompatible flat layouts.

## Relationship To Current RL Policy Schema

Current shipped trainer layout:

- `policy_observation_schema_id = puffer_policy_observation_v1`

Phase 1 design decision:

- the first sim-owned flat schema intentionally mirrors that current field order and categorical code mapping
- RL should therefore be able to treat `headless_training_flat_observation_v1` as a zero-transform producer for the existing `puffer_policy_observation_v1` tensor shape

Implication:

- the current Python encoder remains useful in Certification Mode as a reference projection from raw observations
- Production Training Mode should bypass that encoder once the flat path is implemented

## Certification Mode Vs Production Training Mode

### Certification Mode

Certification Mode keeps the raw path active and uses it to prove equivalence.

RL responsibilities:

- run the existing raw observation validation
- project raw observations through the current encoder/reference path
- compare that result against the flat batch rows from the sim
- fail if field meaning, ordering, padding, or categorical mapping drift

### Production Training Mode

Production Training Mode consumes only the flat batch observation payload in the hot loop.

Production hot-path restrictions:

- no raw object reconstruction
- no raw-to-flat Python projection
- no debug/replay-only metadata in the observation batch

Debug, replay, parity, and inspection surfaces remain available through the raw/debug path and separate trace/control surfaces.

## VecEnv Contract

The future vecenv path should treat the flat batch as the canonical observation payload.

Target behavior:

- worker returns one contiguous observation batch
- parent process receives or maps that batch without rebuilding equivalent arrays
- per-env slicing should be a view or a single unavoidable batch-level reshape, not a reconstruction pass

The vecenv should continue to keep reward, terminal flags, and non-hot-path infos outside the observation batch.

## Manifest And Provenance Requirements

When the flat path lands, RL manifests should additionally record:

- `observation_path_mode` (`raw` or `flat`)
- `flat_observation_schema_id`
- `flat_observation_schema_version`
- `flat_observation_dtype`
- `flat_observation_feature_count`
- `flat_observation_max_visible_npcs`

This is in addition to the raw sim observation schema id/version, which remains the semantic reference.

## Jad Cue Constraint

RL must treat `jad_telegraph_state` in the flat path as semantically identical to the raw contract.

RL may not reinterpret it as:

- a direct prayer answer
- a countdown
- a future attack oracle

Equivalence checks must prove that the flat row carries the same `idle / magic_windup / ranged_windup` signal on the same ticks as the raw path.

## Exit Condition For WC-P1-03

This design chunk is complete when:

- RL has a frozen consumer shape for the first flat schema
- startup/handshake validation requirements are explicit
- Certification Mode and Production Training Mode responsibilities are explicit
- the design guarantees that the target state removes `_pythonize` and raw dict reconstruction from the training hot path
