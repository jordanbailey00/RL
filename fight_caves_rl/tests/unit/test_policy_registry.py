from __future__ import annotations

import pytest

from fight_caves_rl.envs.puffer_encoding import build_policy_action_space, build_policy_observation_space
from fight_caves_rl.policies.checkpointing import CheckpointMetadata
from fight_caves_rl.policies.lstm import MultiDiscreteLSTMPolicy
from fight_caves_rl.policies.mlp import MultiDiscreteMLPPolicy
from fight_caves_rl.policies.registry import build_policy_from_config, build_policy_from_metadata


def test_policy_registry_builds_mlp_from_train_config():
    policy = build_policy_from_config(
        {
            "policy": {"id": "mlp_v0", "hidden_size": 128},
            "train": {"use_rnn": False},
        },
        build_policy_observation_space(),
        build_policy_action_space(),
    )

    assert isinstance(policy, MultiDiscreteMLPPolicy)


def test_policy_registry_builds_lstm_from_checkpoint_metadata():
    metadata = CheckpointMetadata(
        checkpoint_format_id="rl_checkpoint_v0",
        checkpoint_format_version=0,
        train_config_id="train_fast_v2",
        policy_id="lstm_v0",
        reward_config_id="reward_shaped_v2",
        curriculum_config_id="curriculum_wave_progression_v2",
        sim_observation_schema_id="obs",
        sim_observation_schema_version=1,
        sim_action_schema_id="act",
        sim_action_schema_version=1,
        episode_start_contract_id="episode",
        episode_start_contract_version=1,
        policy_observation_schema_id="policy_obs",
        policy_observation_schema_version=1,
        policy_action_schema_id="policy_action",
        policy_action_schema_version=1,
        pufferlib_distribution="pufferlib-core",
        pufferlib_version="3.0.17",
        policy_hidden_size=128,
        policy_use_rnn=True,
    )

    policy = build_policy_from_metadata(
        metadata,
        build_policy_observation_space(),
        build_policy_action_space(),
    )

    assert isinstance(policy, MultiDiscreteLSTMPolicy)


def test_policy_registry_rejects_inconsistent_rnn_flags():
    with pytest.raises(ValueError, match="requires use_rnn=true"):
        build_policy_from_config(
            {
                "policy": {"id": "lstm_v0", "hidden_size": 128},
                "train": {"use_rnn": False},
            },
            build_policy_observation_space(),
            build_policy_action_space(),
        )
