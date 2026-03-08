from __future__ import annotations

import argparse
import json
from pathlib import Path

from fight_caves_rl.benchmarks.train_bench import parse_logging_modes, run_train_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PR11 training benchmark matrix.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark/train_1024env_v0.yaml"),
    )
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--env-count", type=int, default=None)
    parser.add_argument("--logging-modes", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run_train_benchmark(
        args.config,
        total_timesteps_override=args.total_timesteps,
        env_count_override=args.env_count,
        logging_modes_override=parse_logging_modes(args.logging_modes),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
