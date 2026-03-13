from __future__ import annotations

import argparse
import json
from pathlib import Path

from fight_caves_rl.defaults import DEFAULT_TRAIN_CONFIG_PATH
from fight_caves_rl.puffer.trainer import run_smoke_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the default Fight Caves training loop.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAIN_CONFIG_PATH,
    )
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--benchmark-instrumentation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run_smoke_training(
        config_path=args.config,
        total_timesteps=args.total_timesteps,
        data_dir=args.data_dir,
        instrumentation_enabled=bool(args.benchmark_instrumentation),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
