from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from fight_caves_rl.defaults import (
    DEMO_BACKEND_FIGHT_CAVES_DEMO_LITE,
    DEMO_BACKEND_ORACLE_V1,
    DEMO_BACKEND_RSPS_HEADED,
    DEFAULT_DEMO_BACKEND,
    DEFAULT_REPLAY_BACKEND,
    ORACLE_REPLAY_CONFIG_PATH,
    backend_selection_registry,
)
from fight_caves_rl.utils.paths import repo_root, workspace_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the current default demo/replay backend without removing fallback/reference paths."
    )
    parser.add_argument(
        "--backend",
        choices=tuple(backend_selection_registry().keys()),
        default=DEFAULT_DEMO_BACKEND,
    )
    parser.add_argument(
        "--mode",
        choices=("replay", "live_inference", "backend_smoke", "launch_reference"),
        default="replay",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = build_command(
        backend=args.backend,
        mode=args.mode,
        checkpoint=args.checkpoint,
        extra_args=tuple(args.extra_args),
    )
    payload = {
        "schema_id": "fight_caves_demo_backend_selection_v1",
        "default_demo_backend": DEFAULT_DEMO_BACKEND,
        "default_replay_backend": DEFAULT_REPLAY_BACKEND,
        "selected_backend": args.backend,
        "selected_mode": args.mode,
        "command": list(command),
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    raise SystemExit(subprocess.call(command, cwd=str(repo_root())))


def build_command(
    *,
    backend: str,
    mode: str,
    checkpoint: Path | None,
    extra_args: tuple[str, ...],
) -> tuple[str, ...]:
    root = repo_root()
    workspace = workspace_root()
    if backend == DEMO_BACKEND_RSPS_HEADED:
        if mode == "replay":
            return (
                sys.executable,
                str(root / "scripts" / "run_headed_trace_replay.py"),
                *extra_args,
            )
        if mode == "live_inference":
            command = [
                sys.executable,
                str(root / "scripts" / "run_headed_checkpoint_inference.py"),
            ]
            if checkpoint is not None:
                command.extend(("--checkpoint", str(checkpoint)))
            command.extend(extra_args)
            return tuple(command)
        if mode == "backend_smoke":
            return (
                sys.executable,
                str(root / "scripts" / "run_headed_backend_smoke.py"),
                *extra_args,
            )
        raise SystemExit(
            "rsps_headed supports --mode replay, --mode live_inference, or --mode backend_smoke."
        )
    if backend == DEMO_BACKEND_ORACLE_V1:
        if mode != "replay":
            raise SystemExit("oracle_v1 fallback supports only --mode replay.")
        command = [
            sys.executable,
            str(root / "scripts" / "replay_eval.py"),
        ]
        if checkpoint is not None:
            command.extend(("--checkpoint", str(checkpoint)))
        elif "--checkpoint" not in extra_args:
            command.extend(("--config", str(root / ORACLE_REPLAY_CONFIG_PATH)))
        command.extend(extra_args)
        return tuple(command)
    if backend == DEMO_BACKEND_FIGHT_CAVES_DEMO_LITE:
        if mode != "launch_reference":
            raise SystemExit("fight_caves_demo_lite fallback supports only --mode launch_reference.")
        return (
            "bash",
            "-lc",
            (
                "source /home/jordan/code/.workspace-env.sh && "
                f"cd {workspace / 'fight-caves-RL'} && "
                "./gradlew --no-daemon :fight-caves-demo-lite:run"
            ),
        )
    raise SystemExit(f"Unsupported backend: {backend!r}")


if __name__ == "__main__":
    main()
