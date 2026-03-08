from __future__ import annotations

import argparse
import json
from pathlib import Path

from fight_caves_rl.benchmarks.vector_env_bench import run_vecenv_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PR8 vectorized env benchmark.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark/vecenv_256env_v0.yaml"),
    )
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--env-count", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run_vecenv_benchmark(
        args.config,
        rounds_override=args.rounds,
        env_count_override=args.env_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
