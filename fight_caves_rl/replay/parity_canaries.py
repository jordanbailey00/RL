from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import yaml

from fight_caves_rl.replay.mechanics_parity import (
    compare_mechanics_parity_traces,
    collect_mechanics_parity_trace,
    write_first_divergence_artifact,
)
from fight_caves_rl.replay.seed_packs import SeedPack, resolve_seed_pack
from fight_caves_rl.replay.trace_packs import TracePack, resolve_trace_pack
from fight_caves_rl.utils.paths import repo_root


@dataclass(frozen=True)
class ParityCanaryScenario:
    scenario_id: str
    trace_pack_id: str
    seed: int
    description: str | None = None
    tick_cap: int | None = None


@dataclass(frozen=True)
class ParityCanaryConfig:
    config_id: str
    seed_pack: SeedPack
    comparison_mode: str
    scenarios: tuple[ParityCanaryScenario, ...]


@dataclass(frozen=True)
class ParityScenarioReport:
    scenario_id: str
    trace_pack: str
    trace_pack_version: int
    seed: int
    tick_cap: int
    wrapper_semantic_digest: str
    raw_semantic_digest: str
    scripted_semantic_digest: str
    expected_semantic_digest: str | None
    oracle_mechanics_digest: str
    v2_fast_mechanics_digest: str
    expected_mechanics_digest: str | None
    wrapper_matches_raw: bool
    scripted_matches_wrapper: bool
    oracle_matches_v2_fast: bool
    expected_digest_matches: bool | None
    expected_mechanics_digest_matches: bool | None
    final_relative_tick: int
    expected_final_relative_tick: int | None
    final_relative_tick_matches: bool | None
    completed_all_steps: bool
    passed: bool
    mismatches: tuple[str, ...]
    mechanics_first_mismatch: dict[str, Any] | None
    mechanics_divergence_artifact: str | None
    wrapper_summary: dict[str, Any]
    raw_summary: dict[str, Any]
    scripted_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParityCanaryReport:
    config_id: str
    seed_pack: str
    seed_pack_version: int
    comparison_mode: str
    scenarios: tuple[ParityScenarioReport, ...]
    all_passed: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scenarios"] = [scenario.to_dict() for scenario in self.scenarios]
        return payload


def load_parity_canary_config(path: str | Path) -> ParityCanaryConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config_id = str(payload["config_id"])
    seed_pack = resolve_seed_pack(str(payload["seed_pack"]))
    raw_scenarios = payload.get("scenarios")
    if not raw_scenarios:
        trace_pack = resolve_trace_pack(str(payload["trace_pack"]))
        raw_scenarios = [
            {
                "scenario_id": trace_pack.identity.contract_id,
                "trace_pack": trace_pack.identity.contract_id,
                "seed": int(trace_pack.default_seed),
                "description": "Back-compat single-scenario parity config.",
            }
        ]

    scenarios = tuple(_parse_scenario(entry, seed_pack=seed_pack) for entry in raw_scenarios)
    return ParityCanaryConfig(
        config_id=config_id,
        seed_pack=seed_pack,
        comparison_mode=str(payload.get("comparison_mode", "semantic_digest")),
        scenarios=scenarios,
    )


def run_parity_canary(config_path: str | Path) -> ParityCanaryReport:
    config = load_parity_canary_config(config_path)
    scenario_reports = tuple(
        _run_scenario(config_id=config.config_id, scenario=scenario)
        for scenario in config.scenarios
    )
    return ParityCanaryReport(
        config_id=config.config_id,
        seed_pack=config.seed_pack.identity.contract_id,
        seed_pack_version=config.seed_pack.identity.version,
        comparison_mode=config.comparison_mode,
        scenarios=scenario_reports,
        all_passed=all(report.passed for report in scenario_reports),
    )


def _parse_scenario(entry: dict[str, Any], *, seed_pack: SeedPack) -> ParityCanaryScenario:
    trace_pack = resolve_trace_pack(str(entry["trace_pack"]))
    seed = int(entry.get("seed", trace_pack.default_seed))
    if seed not in seed_pack.seeds:
        raise ValueError(
            f"Seed {seed} for trace pack {trace_pack.identity.contract_id!r} is not present in "
            f"seed pack {seed_pack.identity.contract_id!r}."
        )
    return ParityCanaryScenario(
        scenario_id=str(entry.get("scenario_id", trace_pack.identity.contract_id)),
        trace_pack_id=trace_pack.identity.contract_id,
        seed=seed,
        description=None if entry.get("description") is None else str(entry["description"]),
        tick_cap=None if entry.get("tick_cap") is None else int(entry["tick_cap"]),
    )


def _run_scenario(*, config_id: str, scenario: ParityCanaryScenario) -> ParityScenarioReport:
    trace_pack = resolve_trace_pack(scenario.trace_pack_id)
    tick_cap = int(
        trace_pack.tick_cap
        if scenario.tick_cap is None
        else scenario.tick_cap
    ) if (scenario.tick_cap is not None or trace_pack.tick_cap is not None) else 20_000
    wrapper = _collect_trajectory("wrapper", trace_pack.identity.contract_id, scenario.seed)
    raw = _collect_trajectory("raw", trace_pack.identity.contract_id, scenario.seed)
    scripted = _collect_scripted_trace(trace_pack.identity.contract_id, scenario.seed)
    oracle_mechanics = collect_mechanics_parity_trace(
        "oracle",
        trace_pack,
        seed=scenario.seed,
        tick_cap=tick_cap,
    )
    v2_fast_mechanics = collect_mechanics_parity_trace(
        "v2_fast",
        trace_pack,
        seed=scenario.seed,
        tick_cap=tick_cap,
    )
    mechanics_comparison = compare_mechanics_parity_traces(
        oracle_mechanics,
        v2_fast_mechanics,
    )

    mismatches = _compare_wrapper_vs_raw(wrapper=wrapper, raw=raw)
    _compare_scripted_to_trace(
        mismatches,
        scripted=scripted,
        wrapper=wrapper,
        trace_pack=trace_pack,
        seed=scenario.seed,
    )

    wrapper_digest = str(wrapper["summary"]["semantic_digest"])
    raw_digest = str(raw["summary"]["semantic_digest"])
    scripted_digest = str(scripted["semantic_digest"])
    expected_digest = trace_pack.expected_semantic_digest
    oracle_mechanics_digest = str(oracle_mechanics["semantic_digest"])
    v2_fast_mechanics_digest = str(v2_fast_mechanics["semantic_digest"])
    expected_mechanics_digest = trace_pack.expected_mechanics_digest
    expected_tick = trace_pack.expected_final_relative_tick

    expected_digest_matches = None
    if expected_digest is not None:
        expected_digest_matches = (
            wrapper_digest == expected_digest
            and raw_digest == expected_digest
            and scripted_digest == expected_digest
        )
        if not expected_digest_matches:
            mismatches.append(
                "Expected semantic digest mismatch across wrapper/raw/scripted parity paths."
            )

    if not bool(mechanics_comparison["record_count_match"]):
        mismatches.append("Oracle/v2_fast mechanics trace record count mismatch.")
    if not bool(mechanics_comparison["digests_match"]):
        mismatches.append("Oracle/v2_fast mechanics trace digest mismatch.")
    if mechanics_comparison["first_mismatch"] is not None:
        mismatches.append("Oracle/v2_fast mechanics trace diverged on a shared parity field.")

    expected_mechanics_digest_matches = None
    if expected_mechanics_digest is not None:
        expected_mechanics_digest_matches = (
            oracle_mechanics_digest == expected_mechanics_digest
            and v2_fast_mechanics_digest == expected_mechanics_digest
        )
        if not expected_mechanics_digest_matches:
            mismatches.append(
                "Expected mechanics digest mismatch across oracle/v2_fast parity paths."
            )

    final_relative_tick_matches = None
    if expected_tick is not None:
        final_relative_tick_matches = (
            int(wrapper["summary"]["final_relative_tick"]) == expected_tick
            and int(raw["summary"]["final_relative_tick"]) == expected_tick
            and int(scripted["final_relative_tick"]) == expected_tick
        )
        if not final_relative_tick_matches:
            mismatches.append("Expected final relative tick mismatch in parity scenario.")

    mechanics_divergence_artifact = None
    if (
        not bool(mechanics_comparison["record_count_match"])
        or not bool(mechanics_comparison["digests_match"])
        or mechanics_comparison["first_mismatch"] is not None
        or expected_mechanics_digest_matches is False
    ):
        artifact_path = _parity_failure_artifact_path(config_id, scenario.scenario_id)
        mechanics_divergence_artifact = str(
            write_first_divergence_artifact(
                artifact_path,
                reference=oracle_mechanics,
                candidate=v2_fast_mechanics,
                comparison=mechanics_comparison,
            )
        )

    return ParityScenarioReport(
        scenario_id=scenario.scenario_id,
        trace_pack=trace_pack.identity.contract_id,
        trace_pack_version=trace_pack.identity.version,
        seed=int(scenario.seed),
        tick_cap=int(tick_cap),
        wrapper_semantic_digest=wrapper_digest,
        raw_semantic_digest=raw_digest,
        scripted_semantic_digest=scripted_digest,
        expected_semantic_digest=expected_digest,
        oracle_mechanics_digest=oracle_mechanics_digest,
        v2_fast_mechanics_digest=v2_fast_mechanics_digest,
        expected_mechanics_digest=expected_mechanics_digest,
        wrapper_matches_raw=(wrapper_digest == raw_digest),
        scripted_matches_wrapper=(scripted_digest == wrapper_digest),
        oracle_matches_v2_fast=(
            bool(mechanics_comparison["record_count_match"])
            and bool(mechanics_comparison["digests_match"])
            and mechanics_comparison["first_mismatch"] is None
        ),
        expected_digest_matches=expected_digest_matches,
        expected_mechanics_digest_matches=expected_mechanics_digest_matches,
        final_relative_tick=int(wrapper["summary"]["final_relative_tick"]),
        expected_final_relative_tick=expected_tick,
        final_relative_tick_matches=final_relative_tick_matches,
        completed_all_steps=bool(wrapper["summary"]["completed_all_steps"]),
        passed=not mismatches,
        mismatches=tuple(mismatches),
        mechanics_first_mismatch=(
            None
            if mechanics_comparison["first_mismatch"] is None
            else dict(mechanics_comparison["first_mismatch"])
        ),
        mechanics_divergence_artifact=mechanics_divergence_artifact,
        wrapper_summary=dict(wrapper["summary"]),
        raw_summary=dict(raw["summary"]),
        scripted_summary={
            "completed_all_steps": bool(scripted["completed_all_steps"]),
            "final_relative_tick": int(scripted["final_relative_tick"]),
            "semantic_digest": scripted_digest,
        },
    )


def _compare_wrapper_vs_raw(
    *,
    wrapper: dict[str, Any],
    raw: dict[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    if wrapper["semantic_episode_state"] != raw["semantic_episode_state"]:
        mismatches.append("Semantic episode-state mismatch between wrapper and raw parity paths.")
    if wrapper["semantic_initial_observation"] != raw["semantic_initial_observation"]:
        mismatches.append(
            "Semantic initial-observation mismatch between wrapper and raw parity paths."
        )
    if wrapper["summary"] != raw["summary"]:
        mismatches.append("Summary mismatch between wrapper and raw parity paths.")
    if len(wrapper["steps"]) != len(raw["steps"]):
        mismatches.append("Step-count mismatch between wrapper and raw parity paths.")
        return mismatches

    for index, (wrapper_step, raw_step) in enumerate(
        zip(wrapper["steps"], raw["steps"], strict=True)
    ):
        for field in (
            "action",
            "semantic_observation",
            "action_result",
            "semantic_visible_targets",
            "terminated",
            "truncated",
            "terminal_reason",
        ):
            if wrapper_step[field] != raw_step[field]:
                mismatches.append(
                    f"Wrapper/raw mismatch at step {index} for field {field!r}."
                )
                break
    return mismatches


def _compare_scripted_to_trace(
    mismatches: list[str],
    *,
    scripted: dict[str, Any],
    wrapper: dict[str, Any],
    trace_pack: TracePack,
    seed: int,
) -> None:
    if str(scripted["trace_pack"]) != trace_pack.identity.contract_id:
        mismatches.append("Scripted parity path reported the wrong trace-pack id.")
    if int(scripted["seed"]) != int(seed):
        mismatches.append("Scripted parity path reported the wrong seed.")
    if bool(scripted["completed_all_steps"]) != bool(wrapper["summary"]["completed_all_steps"]):
        mismatches.append("Scripted parity path disagrees on completed-all-steps.")
    if int(scripted["final_relative_tick"]) != int(wrapper["summary"]["final_relative_tick"]):
        mismatches.append("Scripted parity path disagrees on final relative tick.")
    if str(scripted["semantic_digest"]) != str(wrapper["summary"]["semantic_digest"]):
        mismatches.append("Scripted parity path semantic digest does not match wrapper trace.")


def _collect_trajectory(mode: str, trace_pack: str, seed: int) -> dict[str, Any]:
    return _run_json_script(
        "collect_trajectory_trace.py",
        "--mode",
        mode,
        "--trace-pack",
        trace_pack,
        "--seed",
        str(int(seed)),
    )


def _collect_scripted_trace(trace_pack: str, seed: int) -> dict[str, Any]:
    return _run_json_script(
        "smoke_scripted.py",
        "--trace-pack",
        trace_pack,
        "--seed",
        str(int(seed)),
    )


def _run_json_script(script_name: str, *args: str) -> dict[str, Any]:
    root = repo_root()
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as handle:
        output_path = Path(handle.name)
    try:
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / script_name), *args, "--output", str(output_path)],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
            timeout=180.0,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{script_name} failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return json.loads(output_path.read_text(encoding="utf-8"))
    finally:
        output_path.unlink(missing_ok=True)


def _parity_failure_artifact_path(config_id: str, scenario_id: str) -> Path:
    return (
        repo_root()
        / "artifacts"
        / "parity"
        / "pr62_canary_failures"
        / config_id
        / f"{scenario_id}.json"
    )
