import pytest

from fight_caves_rl.replay.mechanics_parity import (
    compare_mechanics_parity_traces,
    collect_mechanics_parity_trace,
)
from fight_caves_rl.replay.trace_packs import resolve_trace_pack
from fight_caves_rl.tests.smoke._helpers import require_live_runtime


@pytest.mark.parametrize(
    ("trace_pack_id", "seed"),
    (
        ("parity_single_wave_v0", 11_001),
        ("parity_action_rejection_v0", 33_003),
        ("parity_prayer_toggle_timing_v0", 11_001),
        ("parity_jad_healer_v0", 33_003),
        ("parity_terminal_tick_cap_v0", 11_001),
    ),
)
def test_v2_fast_matches_oracle_mechanics_trace(trace_pack_id: str, seed: int):
    require_live_runtime()

    trace_pack = resolve_trace_pack(trace_pack_id)
    tick_cap = int(trace_pack.tick_cap if trace_pack.tick_cap is not None else 20_000)
    oracle = collect_mechanics_parity_trace(
        "oracle",
        trace_pack,
        seed=seed,
        tick_cap=tick_cap,
    )
    fast = collect_mechanics_parity_trace(
        "v2_fast",
        trace_pack,
        seed=seed,
        tick_cap=tick_cap,
    )
    comparison = compare_mechanics_parity_traces(oracle, fast)

    assert comparison["reference_runtime_path"] == "oracle"
    assert comparison["candidate_runtime_path"] == "v2_fast"
    assert comparison["record_count_match"] is True
    assert comparison["digests_match"] is True
    assert comparison["first_mismatch"] is None
