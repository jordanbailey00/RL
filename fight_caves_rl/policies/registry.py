from __future__ import annotations

from typing import Any, Mapping

from gymnasium import spaces

from fight_caves_rl.policies.checkpointing import CheckpointMetadata
from fight_caves_rl.policies.lstm import MultiDiscreteLSTMPolicy
from fight_caves_rl.policies.mlp import MultiDiscreteMLPPolicy


def build_policy_from_config(
    config: Mapping[str, Any],
    observation_space: spaces.Box,
    action_space: spaces.MultiDiscrete,
):
    policy_config = dict(config.get("policy", {}))
    policy_id = str(policy_config.get("id", "mlp_v0"))
    hidden_size = int(policy_config.get("hidden_size", 128))
    use_rnn = bool(dict(config.get("train", {})).get("use_rnn", False))
    return build_policy(
        policy_id=policy_id,
        observation_space=observation_space,
        action_space=action_space,
        hidden_size=hidden_size,
        use_rnn=use_rnn,
    )


def build_policy_from_metadata(
    metadata: CheckpointMetadata,
    observation_space: spaces.Box,
    action_space: spaces.MultiDiscrete,
):
    return build_policy(
        policy_id=str(metadata.policy_id),
        observation_space=observation_space,
        action_space=action_space,
        hidden_size=int(metadata.policy_hidden_size),
        use_rnn=bool(metadata.policy_use_rnn),
    )


def build_policy(
    *,
    policy_id: str,
    observation_space: spaces.Box,
    action_space: spaces.MultiDiscrete,
    hidden_size: int,
    use_rnn: bool,
):
    if policy_id == "mlp_v0":
        if use_rnn:
            raise ValueError("policy_id='mlp_v0' is incompatible with use_rnn=true.")
        return MultiDiscreteMLPPolicy.from_spaces(
            observation_space,
            action_space,
            hidden_size=hidden_size,
        )
    if policy_id == "lstm_v0":
        if not use_rnn:
            raise ValueError("policy_id='lstm_v0' requires use_rnn=true.")
        return MultiDiscreteLSTMPolicy.from_spaces(
            observation_space,
            action_space,
            hidden_size=hidden_size,
        )
    raise ValueError(f"Unsupported policy id: {policy_id!r}")
