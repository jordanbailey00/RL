from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import json
from typing import Any

import numpy as np

from fight_caves_rl.bridge.contracts import HeadlessBootstrapConfig
from fight_caves_rl.bridge.launcher import (
    assert_sim_runtime_ready,
    build_headless_settings_overrides,
    discover_headless_runtime_paths,
)
from fight_caves_rl.contracts.parity_trace_schema import (
    MECHANICS_PARITY_TRACE_FIELD_NAMES,
    MECHANICS_PARITY_TRACE_SCHEMA,
    coerce_mechanics_parity_trace_records,
    mechanics_parity_trace_digest,
)
from fight_caves_rl.envs.action_mapping import NormalizedAction
from fight_caves_rl.envs_fast.fast_trace_adapter import (
    adapt_fast_parity_traces,
    extract_fast_parity_traces,
)
from fight_caves_rl.envs_fast.fast_vector_env import (
    FastKernelVecEnvConfig,
    _create_fast_kernel_runtime,
    _ensure_fast_jvm,
    _java_episode_configs,
    _java_int_array,
)
from fight_caves_rl.replay.trace_packs import TracePack, resolve_trace_pack

ORACLE_RUNTIME_PATH_ID = "oracle"
FAST_RUNTIME_PATH_ID = "v2_fast"
SUPPORTED_MECHANICS_TRACE_PATHS = (ORACLE_RUNTIME_PATH_ID, FAST_RUNTIME_PATH_ID)


def collect_mechanics_parity_trace(
    runtime_path: str,
    trace_pack: str | TracePack,
    *,
    seed: int | None = None,
    tick_cap: int = 20_000,
) -> dict[str, Any]:
    resolved_trace_pack = (
        resolve_trace_pack(trace_pack) if isinstance(trace_pack, str) else trace_pack
    )
    resolved_seed = int(
        resolved_trace_pack.default_seed if seed is None else seed
    )
    if runtime_path == ORACLE_RUNTIME_PATH_ID:
        return collect_oracle_mechanics_parity_trace(
            resolved_trace_pack,
            seed=resolved_seed,
            tick_cap=tick_cap,
        )
    if runtime_path == FAST_RUNTIME_PATH_ID:
        return collect_fast_mechanics_parity_trace(
            resolved_trace_pack,
            seed=resolved_seed,
            tick_cap=tick_cap,
        )
    raise ValueError(
        f"Unsupported mechanics parity runtime path: {runtime_path!r}. "
        f"Expected one of {SUPPORTED_MECHANICS_TRACE_PATHS!r}."
    )


def collect_oracle_mechanics_parity_trace(
    trace_pack: TracePack,
    *,
    seed: int,
    tick_cap: int = 20_000,
) -> dict[str, Any]:
    paths = discover_headless_runtime_paths()
    assert_sim_runtime_ready(paths)
    classes = _ensure_fast_jvm(paths.headless_jar, paths.launch_cwd)
    overrides = classes["HashMap"]()
    for key, value in _parity_harness_settings_overrides(paths).items():
        overrides.put(str(key), str(value))
    parity_harness = classes["jpype"].JClass("ParityHarness")()
    packed_actions = _pack_trace_actions(trace_pack)
    traces = parity_harness.collectMechanicsParityTrace(
        ORACLE_RUNTIME_PATH_ID,
        int(seed),
        _java_int_array(packed_actions),
        int(trace_pack.start_wave),
        int(tick_cap),
        "rl-oracle-mechanics-parity",
        overrides,
    )
    records = adapt_fast_parity_traces(traces)
    return _build_trace_export_payload(
        runtime_path=ORACLE_RUNTIME_PATH_ID,
        trace_pack=trace_pack,
        seed=seed,
        tick_cap=tick_cap,
        records=records,
    )


def collect_fast_mechanics_parity_trace(
    trace_pack: TracePack,
    *,
    seed: int,
    tick_cap: int = 20_000,
) -> dict[str, Any]:
    config = FastKernelVecEnvConfig(
        env_count=1,
        account_name_prefix="rl_fast_mechanics_parity",
        start_wave=int(trace_pack.start_wave),
        tick_cap=int(tick_cap),
    )
    runtime, classes = _create_fast_kernel_runtime(config)
    try:
        episode_configs = _java_episode_configs(
            classes,
            [
                (
                    int(seed),
                    int(trace_pack.start_wave),
                    int(config.ammo),
                    int(config.prayer_potions),
                    int(config.sharks),
                )
            ],
        )
        reset = runtime.resetBatch(_java_int_array((0,)), episode_configs, True, None, None)
        records = list(extract_fast_parity_traces(reset))
        for packed_action in _pack_trace_actions_per_step(trace_pack):
            response = runtime.stepBatch(
                _java_int_array((0,)),
                _java_int_array(packed_action),
                True,
                None,
                None,
            )
            records.extend(extract_fast_parity_traces(response))
        return _build_trace_export_payload(
            runtime_path=FAST_RUNTIME_PATH_ID,
            trace_pack=trace_pack,
            seed=seed,
            tick_cap=tick_cap,
            records=records,
        )
    finally:
        runtime.close()


def compare_mechanics_parity_traces(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    reference_records = coerce_mechanics_parity_trace_records(reference["records"])
    candidate_records = coerce_mechanics_parity_trace_records(candidate["records"])
    comparison: dict[str, Any] = {
        "reference_runtime_path": str(reference["runtime_path"]),
        "candidate_runtime_path": str(candidate["runtime_path"]),
        "reference_digest": str(reference["semantic_digest"]),
        "candidate_digest": str(candidate["semantic_digest"]),
        "schema_id": str(reference["schema_id"]),
        "schema_version": int(reference["schema_version"]),
        "record_count_match": len(reference_records) == len(candidate_records),
        "digests_match": str(reference["semantic_digest"]) == str(candidate["semantic_digest"]),
        "first_mismatch": None,
    }
    first_mismatch = _first_trace_mismatch(reference_records, candidate_records)
    if first_mismatch is not None:
        comparison["first_mismatch"] = first_mismatch
    return comparison


def write_first_divergence_artifact(
    output_path: Path,
    *,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference": reference,
        "candidate": candidate,
        "comparison": comparison,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _build_trace_export_payload(
    *,
    runtime_path: str,
    trace_pack: TracePack,
    seed: int,
    tick_cap: int,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    coerced_records = coerce_mechanics_parity_trace_records(records)
    return {
        "runtime_path": runtime_path,
        "schema_id": MECHANICS_PARITY_TRACE_SCHEMA.contract_id,
        "schema_version": MECHANICS_PARITY_TRACE_SCHEMA.version,
        "field_names": list(MECHANICS_PARITY_TRACE_FIELD_NAMES),
        "trace_pack": trace_pack.identity.contract_id,
        "trace_pack_version": trace_pack.identity.version,
        "seed": int(seed),
        "start_wave": int(trace_pack.start_wave),
        "tick_cap": int(tick_cap),
        "record_count": len(coerced_records),
        "records": list(coerced_records),
        "semantic_digest": mechanics_parity_trace_digest(coerced_records),
    }


def _first_trace_mismatch(
    reference_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if len(reference_records) != len(candidate_records):
        return {
            "field_path": "records.size",
            "reference_value": len(reference_records),
            "candidate_value": len(candidate_records),
        }
    for index, (reference_record, candidate_record) in enumerate(
        zip(reference_records, candidate_records, strict=True)
    ):
        for field_name in MECHANICS_PARITY_TRACE_FIELD_NAMES:
            if reference_record[field_name] != candidate_record[field_name]:
                return {
                    "record_index": index,
                    "field_path": field_name,
                    "reference_value": reference_record[field_name],
                    "candidate_value": candidate_record[field_name],
                }
    return None


def _pack_trace_actions(trace_pack: TracePack) -> np.ndarray:
    if not trace_pack.steps:
        return np.zeros((0,), dtype=np.int32)
    return np.concatenate(_pack_trace_actions_per_step(trace_pack)).astype(np.int32, copy=False)


def _pack_trace_actions_per_step(trace_pack: TracePack) -> tuple[np.ndarray, ...]:
    return tuple(_pack_normalized_action(step.action) for step in trace_pack.steps)


def _pack_normalized_action(action: NormalizedAction) -> np.ndarray:
    packed = np.zeros((4,), dtype=np.int32)
    packed[0] = int(action.action_id)
    if action.action_id == 1:
        tile = action.tile
        if tile is None:
            raise ValueError("walk_to_tile parity actions require tile coordinates.")
        packed[1] = int(tile.x)
        packed[2] = int(tile.y)
        packed[3] = int(tile.level)
    elif action.action_id == 2:
        if action.visible_npc_index is None:
            raise ValueError("attack_visible_npc parity actions require visible_npc_index.")
        packed[1] = int(action.visible_npc_index)
    elif action.action_id == 3:
        prayer_to_index = {
            "protect_from_magic": 0,
            "protect_from_missiles": 1,
            "protect_from_melee": 2,
        }
        if action.prayer not in prayer_to_index:
            raise ValueError(f"Unsupported protection prayer for parity packing: {action.prayer!r}.")
        packed[1] = prayer_to_index[str(action.prayer)]
    return packed


def _parity_harness_settings_overrides(paths: Any) -> dict[str, str]:
    overrides = build_headless_settings_overrides(paths, HeadlessBootstrapConfig())
    overrides.update(
        {
            "runtime.mode": "headed",
            "headless.map.regions": "",
            "bots.numberedNames": "true",
            "bots.names": str((paths.sim_repo / "data" / "bot_names.txt").resolve()),
            "spawns.npcs": "tzhaar_city.npc-spawns.toml",
            "spawns.items": "tzhaar_city.items.toml",
            "spawns.objects": "tzhaar_city.objs.toml",
        }
    )
    return overrides
