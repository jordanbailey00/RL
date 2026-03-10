from __future__ import annotations

import argparse
import json
from pathlib import Path

from fight_caves_rl.benchmarks.subprocess_transport_bench import (
    parse_transport_modes,
    run_subprocess_transport_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 2 subprocess transport benchmark.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark/vecenv_256env_v0.yaml"),
    )
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--env-count", type=int, default=None)
    parser.add_argument("--transport-modes", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run_subprocess_transport_benchmark(
        args.config,
        rounds_override=args.rounds,
        env_count_override=args.env_count,
        transport_modes_override=parse_transport_modes(args.transport_modes),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
