from pathlib import Path
import json

from fight_caves_rl.contracts.parity_trace_schema import (
    MECHANICS_PARITY_TRACE_FIELD_NAMES,
    MECHANICS_PARITY_TRACE_SCHEMA,
    mechanics_parity_trace_digest,
)
from fight_caves_rl.replay.mechanics_parity import (
    compare_mechanics_parity_traces,
    write_first_divergence_artifact,
)


def test_compare_mechanics_parity_traces_flags_visible_target_order_drift():
    reference = _trace_payload(visible_target_order=[1], rejection_code=0)
    candidate = _trace_payload(visible_target_order=[2], rejection_code=0)

    comparison = compare_mechanics_parity_traces(reference, candidate)

    assert comparison["digests_match"] is False
    assert comparison["first_mismatch"] == {
        "record_index": 0,
        "field_path": "visible_target_order",
        "reference_value": [1],
        "candidate_value": [2],
    }


def test_compare_mechanics_parity_traces_flags_rejection_code_drift():
    reference = _trace_payload(visible_target_order=[1], rejection_code=2)
    candidate = _trace_payload(visible_target_order=[1], rejection_code=3)

    comparison = compare_mechanics_parity_traces(reference, candidate)

    assert comparison["digests_match"] is False
    assert comparison["first_mismatch"] == {
        "record_index": 0,
        "field_path": "rejection_code",
        "reference_value": 2,
        "candidate_value": 3,
    }


def test_write_first_divergence_artifact_persists_reference_candidate_and_comparison(
    tmp_path: Path,
):
    reference = _trace_payload(visible_target_order=[1], rejection_code=2)
    candidate = _trace_payload(visible_target_order=[1], rejection_code=3)
    comparison = compare_mechanics_parity_traces(reference, candidate)

    output = write_first_divergence_artifact(
        tmp_path / "divergence.json",
        reference=reference,
        candidate=candidate,
        comparison=comparison,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["reference"]["semantic_digest"] == reference["semantic_digest"]
    assert payload["candidate"]["semantic_digest"] == candidate["semantic_digest"]
    assert payload["comparison"]["first_mismatch"]["field_path"] == "rejection_code"


def _trace_payload(*, visible_target_order: list[int], rejection_code: int) -> dict[str, object]:
    record = {
        "tick_index": 1,
        "action_name": "attack_visible_npc",
        "action_accepted": False,
        "rejection_code": rejection_code,
        "player_hitpoints": 700,
        "player_prayer_points": 43,
        "run_enabled": True,
        "inventory_ammo": 1000,
        "inventory_sharks": 20,
        "inventory_prayer_potions": 32,
        "wave_id": 63,
        "remaining_npcs": 1,
        "visible_target_order": visible_target_order,
        "visible_npc_type": ["tztok_jad"],
        "visible_npc_hitpoints": [250],
        "visible_npc_alive": [True],
        "jad_telegraph_state": 0,
        "jad_hit_resolve_outcome": "pending",
        "damage_dealt": 0.0,
        "damage_taken": 0.0,
        "terminal_code": 0,
    }
    return {
        "runtime_path": "oracle",
        "schema_id": MECHANICS_PARITY_TRACE_SCHEMA.contract_id,
        "schema_version": MECHANICS_PARITY_TRACE_SCHEMA.version,
        "field_names": list(MECHANICS_PARITY_TRACE_FIELD_NAMES),
        "trace_pack": "unit_test_trace_pack",
        "trace_pack_version": 0,
        "seed": 33003,
        "start_wave": 63,
        "tick_cap": 20_000,
        "record_count": 1,
        "records": [record],
        "semantic_digest": mechanics_parity_trace_digest([record]),
    }
