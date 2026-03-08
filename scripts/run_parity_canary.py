from __future__ import annotations

import argparse
import json
from pathlib import Path

from fight_caves_rl.replay.parity_canaries import run_parity_canary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PR12 parity canary matrix.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/parity_canary_v0.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run_parity_canary(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
