from __future__ import annotations

from dataclasses import dataclass

from fight_caves_rl.envs.schema import VersionedContract


REWARD_FEATURE_SCHEMA = VersionedContract(
    contract_id="fight_caves_v2_reward_features_v1",
    version=1,
    compatibility_policy="append_only_features",
)


@dataclass(frozen=True)
class RewardFeatureDefinition:
    index: int
    name: str
    description: str


REWARD_FEATURE_DEFINITIONS = (
    RewardFeatureDefinition(0, "damage_dealt", "Positive damage applied to Fight Caves NPCs."),
    RewardFeatureDefinition(1, "damage_taken", "Damage received by the player."),
    RewardFeatureDefinition(2, "npc_kill", "Count of NPC kills resolved on this step."),
    RewardFeatureDefinition(3, "wave_clear", "Wave clear event emitted on this step."),
    RewardFeatureDefinition(4, "jad_damage_dealt", "Positive damage applied specifically to Jad."),
    RewardFeatureDefinition(5, "jad_kill", "Jad kill event emitted on this step."),
    RewardFeatureDefinition(6, "player_death", "Player death event emitted on this step."),
    RewardFeatureDefinition(7, "cave_complete", "Fight Caves completion event emitted on this step."),
    RewardFeatureDefinition(8, "food_used", "Food-consumption event emitted on this step."),
    RewardFeatureDefinition(
        9,
        "prayer_potion_used",
        "Prayer-potion-consumption event emitted on this step.",
    ),
    RewardFeatureDefinition(
        10,
        "correct_jad_prayer_on_resolve",
        "Correct Jad prayer state at the hit-resolution boundary.",
    ),
    RewardFeatureDefinition(
        11,
        "wrong_jad_prayer_on_resolve",
        "Incorrect Jad prayer state at the hit-resolution boundary.",
    ),
    RewardFeatureDefinition(
        12,
        "invalid_action",
        "Action rejection or invalid-intent event emitted on this step.",
    ),
    RewardFeatureDefinition(
        13,
        "movement_progress",
        "Movement/progress feature reserved for validated shaping only.",
    ),
    RewardFeatureDefinition(
        14,
        "idle_penalty_flag",
        "Idle-step indicator for optional shaping penalties.",
    ),
    RewardFeatureDefinition(
        15,
        "tick_penalty_base",
        "Per-tick shaping baseline emitted every environment step.",
    ),
)

REWARD_FEATURE_NAMES = tuple(definition.name for definition in REWARD_FEATURE_DEFINITIONS)
REWARD_FEATURE_INDEX = {definition.name: definition.index for definition in REWARD_FEATURE_DEFINITIONS}
