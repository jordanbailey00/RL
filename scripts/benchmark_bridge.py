from __future__ import annotations

import argparse
import json
from pathlib import Path

from fight_caves_rl.benchmarks.bridge_bench import run_bridge_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PR7 bridge benchmark.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark/bridge_1env_v0.yaml"),
    )
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run_bridge_benchmark(args.config, rounds_override=args.rounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
