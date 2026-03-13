from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from fight_caves_rl.envs.action_mapping import NormalizedAction, normalize_action
from fight_caves_rl.envs.schema import VersionedContract


@dataclass(frozen=True)
class TraceStep:
    action: NormalizedAction


@dataclass(frozen=True)
class TracePack:
    identity: VersionedContract
    start_wave: int
    default_seed: int
    steps: tuple[TraceStep, ...]
    description: str
    source_ref: str
    expected_semantic_digest: str | None = None
    expected_final_relative_tick: int | None = None
    expected_mechanics_digest: str | None = None
    tick_cap: int | None = None


WAIT_ACTION = normalize_action(0)
ATTACK_FIRST_VISIBLE = normalize_action({"action_id": 2, "visible_npc_index": 0})
ATTACK_INVALID_VISIBLE = normalize_action({"action_id": 2, "visible_npc_index": 7})
TOGGLE_PROTECT_FROM_MAGIC = normalize_action(
    {"action_id": 3, "prayer": "protect_from_magic"}
)


def _expand_replay_steps(
    *segments: tuple[int | Mapping[str, object] | NormalizedAction, int],
) -> tuple[TraceStep, ...]:
    expanded: list[TraceStep] = []
    for action, ticks_after in segments:
        if ticks_after <= 0:
            raise ValueError(f"ticks_after must be > 0, got {ticks_after}.")
        expanded.append(TraceStep(normalize_action(action)))
        for _ in range(ticks_after - 1):
            expanded.append(TraceStep(WAIT_ACTION))
    return tuple(expanded)


WAIT_ONLY_16_TRACE_PACK = TracePack(
    identity=VersionedContract(
        contract_id="wait_only_16_v0",
        version=0,
        compatibility_policy="append_only_or_bump",
    ),
    start_wave=1,
    default_seed=11_001,
    steps=tuple(TraceStep(WAIT_ACTION) for _ in range(16)),
    description="Sixteen per-tick wait actions for thin wrapper-vs-raw trajectory agreement checks.",
    source_ref="RL-local deterministic trace pack",
)

PARITY_SINGLE_WAVE_TRACE_PACK = TracePack(
    identity=VersionedContract(
        contract_id="parity_single_wave_v0",
        version=0,
        compatibility_policy="replace_on_semantic_change",
    ),
    start_wave=1,
    default_seed=11_001,
    steps=_expand_replay_steps(
        (WAIT_ACTION, 2),
        (ATTACK_FIRST_VISIBLE, 8),
        (WAIT_ACTION, 10),
        (ATTACK_FIRST_VISIBLE, 8),
        (WAIT_ACTION, 12),
    ),
    description="Per-tick RL env expansion of the sim parity harness single-wave trace.",
    source_ref="fight-caves-RL paritySingleWaveTrace expanded from replay ticks to per-tick RL env actions",
    expected_semantic_digest="50f696569ed20307aa247a29aa84bf29ddeb3ba2a4886d561813cdb650a504f3",
    expected_final_relative_tick=40,
    expected_mechanics_digest="05f002f3dab119d2df2ee78e2551fbc6951f190d70eb812d2ad0cb2bb4f9b792",
)

PARITY_JAD_HEALER_TRACE_PACK = TracePack(
    identity=VersionedContract(
        contract_id="parity_jad_healer_v0",
        version=0,
        compatibility_policy="replace_on_semantic_change",
    ),
    start_wave=63,
    default_seed=33_003,
    steps=_expand_replay_steps(
        (WAIT_ACTION, 5),
        (WAIT_ACTION, 3),
        (WAIT_ACTION, 4),
    ),
    description="Per-tick RL env expansion of the sim parity harness Jad healer trace.",
    source_ref="fight-caves-RL parityJadHealerTrace expanded from replay ticks to per-tick RL env actions",
    expected_semantic_digest="f0f7b30d5fea1f9e73ce756003795579dcae88990330a5f25b90af10bc097727",
    expected_final_relative_tick=12,
    expected_mechanics_digest="f90376f2d7030fa29bcdab6d80a4db061182a281a61b8e6e22279fa30a6ea5da",
)

PARITY_TZKEK_SPLIT_TRACE_PACK = TracePack(
    identity=VersionedContract(
        contract_id="parity_tzkek_split_v0",
        version=0,
        compatibility_policy="replace_on_semantic_change",
    ),
    start_wave=3,
    default_seed=44_004,
    steps=_expand_replay_steps(
        (WAIT_ACTION, 5),
        (WAIT_ACTION, 7),
        (WAIT_ACTION, 2),
    ),
    description="Per-tick RL env expansion of the sim parity harness Tz-Kek split trace.",
    source_ref="fight-caves-RL parityTzKekSplitTrace expanded from replay ticks to per-tick RL env actions",
    expected_semantic_digest="6237b862364cffa99fa76b6618e81c79ee4dca0795edb5d3ee991914dcee9b00",
    expected_final_relative_tick=14,
    expected_mechanics_digest="781ae05ec34dfcb6e501ebf76ee83a0b226c55240244f330adcf42c2e991b6dc",
)

PARITY_ACTION_REJECTION_TRACE_PACK = TracePack(
    identity=VersionedContract(
        contract_id="parity_action_rejection_v0",
        version=0,
        compatibility_policy="replace_on_semantic_change",
    ),
    start_wave=63,
    default_seed=33_003,
    steps=_expand_replay_steps(
        (WAIT_ACTION, 10),
        (ATTACK_INVALID_VISIBLE, 1),
    ),
    description=(
        "Wave-63 parity trace that waits for Jad visibility, then submits an invalid "
        "visible-npc target index to pin rejection-code and target-order parity."
    ),
    source_ref="RL-local mechanics parity scenario for action rejection and visible-target ordering",
    expected_semantic_digest="53837f567e2f771f4baea1c89adb7ea1127b06630bdd6b5bc3ac1d9219113cc5",
    expected_final_relative_tick=11,
    expected_mechanics_digest="00f40d88e2824a92a0ae80a6206c7c3631071305a4b81b89f302bdd0089f09ef",
)

PARITY_PRAYER_TOGGLE_TIMING_TRACE_PACK = TracePack(
    identity=VersionedContract(
        contract_id="parity_prayer_toggle_timing_v0",
        version=0,
        compatibility_policy="replace_on_semantic_change",
    ),
    start_wave=1,
    default_seed=11_001,
    steps=_expand_replay_steps(
        (TOGGLE_PROTECT_FROM_MAGIC, 1),
        (WAIT_ACTION, 8),
        (TOGGLE_PROTECT_FROM_MAGIC, 1),
        (WAIT_ACTION, 4),
    ),
    description=(
        "Protection-prayer toggle trace that pins shared prayer-drain timing on the "
        "mechanics parity surface."
    ),
    source_ref="RL-local mechanics parity scenario for protection-prayer timing",
    expected_semantic_digest="da5142bef18cec2f4c5f6576a6337e8e7041044f3547eeb17e98707ad846436e",
    expected_final_relative_tick=14,
    expected_mechanics_digest="cef28b739a3e0034a078cc29311b2d941e21406b0bf08028e5f9dba97720b506",
)

PARITY_TERMINAL_TICK_CAP_TRACE_PACK = TracePack(
    identity=VersionedContract(
        contract_id="parity_terminal_tick_cap_v0",
        version=0,
        compatibility_policy="replace_on_semantic_change",
    ),
    start_wave=1,
    default_seed=11_001,
    steps=tuple(TraceStep(WAIT_ACTION) for _ in range(8)),
    description="Low tick-cap wait trace that pins terminal-code parity on episode truncation.",
    source_ref="RL-local mechanics parity scenario for tick-cap terminal code parity",
    expected_semantic_digest="513eb47cd9d34d426c295b41e280d1ffe96310378455afeb1b213b136c0851cb",
    expected_final_relative_tick=4,
    expected_mechanics_digest="c7b280384684bac187c2700d2745ad9e3c20c91a44e745de63310e7a73b102c9",
    tick_cap=4,
)


TRACE_PACKS = {
    WAIT_ONLY_16_TRACE_PACK.identity.contract_id: WAIT_ONLY_16_TRACE_PACK,
    PARITY_SINGLE_WAVE_TRACE_PACK.identity.contract_id: PARITY_SINGLE_WAVE_TRACE_PACK,
    PARITY_JAD_HEALER_TRACE_PACK.identity.contract_id: PARITY_JAD_HEALER_TRACE_PACK,
    PARITY_TZKEK_SPLIT_TRACE_PACK.identity.contract_id: PARITY_TZKEK_SPLIT_TRACE_PACK,
    PARITY_ACTION_REJECTION_TRACE_PACK.identity.contract_id: PARITY_ACTION_REJECTION_TRACE_PACK,
    PARITY_PRAYER_TOGGLE_TIMING_TRACE_PACK.identity.contract_id: PARITY_PRAYER_TOGGLE_TIMING_TRACE_PACK,
    PARITY_TERMINAL_TICK_CAP_TRACE_PACK.identity.contract_id: PARITY_TERMINAL_TICK_CAP_TRACE_PACK,
}


def resolve_trace_pack(pack_id: str) -> TracePack:
    try:
        return TRACE_PACKS[pack_id]
    except KeyError as exc:
        raise KeyError(f"Unknown trace pack id: {pack_id!r}") from exc


def trace_pack_ids() -> tuple[str, ...]:
    return tuple(TRACE_PACKS.keys())


def serialize_action(action: NormalizedAction) -> dict[str, object]:
    payload: dict[str, object] = {
        "action_id": action.action_id,
        "name": action.name,
    }
    if action.tile is not None:
        payload["tile"] = {
            "x": action.tile.x,
            "y": action.tile.y,
            "level": action.tile.level,
        }
    if action.visible_npc_index is not None:
        payload["visible_npc_index"] = action.visible_npc_index
    if action.prayer is not None:
        payload["prayer"] = action.prayer
    return payload


def project_episode_state_for_determinism(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(state["seed"]),
        "wave": int(state["wave"]),
        "rotation": int(state["rotation"]),
        "remaining": int(state["remaining"]),
        "player_tile": {"x": 0, "y": 0, "level": 0},
    }


def project_observation_for_determinism(
    observation: Mapping[str, Any],
    *,
    episode_start_tick: int,
    episode_start_tile: Mapping[str, Any],
) -> dict[str, Any]:
    projected = deepcopy(dict(observation))
    projected["tick"] = int(observation["tick"]) - int(episode_start_tick)
    projected["player"] = deepcopy(dict(observation["player"]))
    projected["player"]["tile"] = _tile_delta(
        observation["player"]["tile"],
        episode_start_tile,
    )
    projected["npcs"] = [
        {
            **deepcopy(dict(npc)),
            "tile": _tile_delta(npc["tile"], episode_start_tile),
        }
        for npc in observation["npcs"]
    ]
    return projected


def project_visible_targets_for_determinism(
    visible_targets: list[dict[str, Any]],
    *,
    episode_start_tile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for target in visible_targets:
        projected.append(
            {
                **deepcopy(target),
                "tile": _tile_delta(target["tile"], episode_start_tile),
            }
        )
    return projected


def semantic_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tile_delta(tile: Mapping[str, Any], origin: Mapping[str, Any]) -> dict[str, int]:
    return {
        "x": int(tile["x"]) - int(origin["x"]),
        "y": int(tile["y"]) - int(origin["y"]),
        "level": int(tile["level"]) - int(origin["level"]),
    }
