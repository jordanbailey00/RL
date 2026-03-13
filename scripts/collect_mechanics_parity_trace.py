from __future__ import annotations

import argparse
import json
from pathlib import Path

from fight_caves_rl.replay.mechanics_parity import (
    FAST_RUNTIME_PATH_ID,
    ORACLE_RUNTIME_PATH_ID,
    SUPPORTED_MECHANICS_TRACE_PATHS,
    collect_mechanics_parity_trace,
    compare_mechanics_parity_traces,
    write_first_divergence_artifact,
)
from fight_caves_rl.replay.trace_packs import resolve_trace_pack


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect shared mechanics-parity traces from oracle or V2 fast runtimes."
    )
    parser.add_argument(
        "--mode",
        choices=(*SUPPORTED_MECHANICS_TRACE_PATHS, "compare"),
        required=True,
    )
    parser.add_argument("--trace-pack", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tick-cap", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace_pack = resolve_trace_pack(args.trace_pack)
    tick_cap = int(
        20_000
        if args.tick_cap is None and trace_pack.tick_cap is None
        else (trace_pack.tick_cap if args.tick_cap is None else args.tick_cap)
    )

    if args.mode == "compare":
        oracle = collect_mechanics_parity_trace(
            ORACLE_RUNTIME_PATH_ID,
            args.trace_pack,
            seed=args.seed,
            tick_cap=tick_cap,
        )
        fast = collect_mechanics_parity_trace(
            FAST_RUNTIME_PATH_ID,
            args.trace_pack,
            seed=args.seed,
            tick_cap=tick_cap,
        )
        comparison = compare_mechanics_parity_traces(oracle, fast)
        payload = {
            "mode": "compare",
            "trace_pack": oracle["trace_pack"],
            "trace_pack_version": oracle["trace_pack_version"],
            "seed": oracle["seed"],
            "oracle": oracle,
            "v2_fast": fast,
            "comparison": comparison,
        }
        if comparison["first_mismatch"] is not None:
            divergence_path = args.output.with_name(args.output.stem + ".first_divergence.json")
            write_first_divergence_artifact(
                divergence_path,
                reference=oracle,
                candidate=fast,
                comparison=comparison,
            )
            payload["first_divergence_artifact"] = str(divergence_path)
    else:
        payload = collect_mechanics_parity_trace(
            args.mode,
            args.trace_pack,
            seed=args.seed,
            tick_cap=tick_cap,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
