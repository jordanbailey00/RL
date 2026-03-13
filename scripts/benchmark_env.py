from __future__ import annotations

import argparse
import json
from pathlib import Path

from fight_caves_rl.defaults import DEFAULT_ENV_BENCHMARK_CONFIG_PATH
from fight_caves_rl.benchmarks.env_bench import (
    _run_vecenv_measurement,
    _run_wrapper_measurement,
    run_env_benchmark,
)
from fight_caves_rl.puffer.factory import load_smoke_train_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PR11 env benchmark.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_ENV_BENCHMARK_CONFIG_PATH,
    )
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--env-count", type=int, default=None)
    parser.add_argument("--wrapper-env-count", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=("combined", "wrapper", "vecenv"),
        default="combined",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "combined":
        payload = run_env_benchmark(
            args.config,
            rounds_override=args.rounds,
            env_count_override=args.env_count,
            wrapper_env_count_override=args.wrapper_env_count,
        ).to_dict()
    else:
        config = load_smoke_train_config(args.config)
        if args.env_count is not None:
            config["num_envs"] = int(args.env_count)
        benchmark_config = dict(config.get("benchmark", {}))
        rounds = int(args.rounds if args.rounds is not None else benchmark_config.get("rounds", 128))
        wrapper_env_count = int(
            args.wrapper_env_count
            if args.wrapper_env_count is not None
            else benchmark_config.get("wrapper_env_count", 1)
        )
        if args.mode == "wrapper":
            payload = _run_wrapper_measurement(
                config,
                rounds=rounds,
                env_count=wrapper_env_count,
            ).to_dict()
        else:
            payload = _run_vecenv_measurement(
                config,
                rounds=rounds,
                warmup_rounds=int(benchmark_config.get("warmup_rounds", 0)),
            ).to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
