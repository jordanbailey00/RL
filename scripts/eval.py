from __future__ import annotations

import argparse
import json
from pathlib import Path

from fight_caves_rl.puffer.trainer import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PR5 replay-eval smoke loop.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/replay_eval_v0.yaml"),
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        max_steps=args.max_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
