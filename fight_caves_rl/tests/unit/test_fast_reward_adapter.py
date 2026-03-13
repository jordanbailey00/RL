from __future__ import annotations

import numpy as np
import pytest

from fight_caves_rl.contracts.reward_feature_schema import REWARD_FEATURE_INDEX, REWARD_FEATURE_NAMES
from fight_caves_rl.envs_fast.fast_reward_adapter import FastRewardAdapter


def test_fast_reward_adapter_sparse_maps_reward_features_directly():
    adapter = FastRewardAdapter.from_config_id("reward_sparse_v2")
    features = np.zeros((2, len(REWARD_FEATURE_NAMES)), dtype=np.float32)
    features[0, REWARD_FEATURE_INDEX["cave_complete"]] = 1.0
    features[1, REWARD_FEATURE_INDEX["cave_complete"]] = 1.0
    features[1, REWARD_FEATURE_INDEX["player_death"]] = 1.0

    weighted = adapter.weight_batch(features)

    assert weighted.tolist() == pytest.approx([1.0, 0.0])


def test_fast_reward_adapter_shaped_v2_supports_direct_feature_weighting():
    adapter = FastRewardAdapter.from_config_id("reward_shaped_v2")
    features = np.zeros((1, len(REWARD_FEATURE_NAMES)), dtype=np.float32)
    features[0, REWARD_FEATURE_INDEX["damage_dealt"]] = 10.0
    features[0, REWARD_FEATURE_INDEX["damage_taken"]] = 2.0
    features[0, REWARD_FEATURE_INDEX["npc_kill"]] = 1.0
    features[0, REWARD_FEATURE_INDEX["wrong_jad_prayer_on_resolve"]] = 1.0
    features[0, REWARD_FEATURE_INDEX["tick_penalty_base"]] = 1.0

    weighted = adapter.weight_batch(features)

    assert weighted.tolist() == pytest.approx([0.0095], abs=1.0e-7)


def test_fast_reward_adapter_rejects_legacy_unsupported_nonzero_coefficients():
    adapter = FastRewardAdapter.from_config_id("reward_shaped_v0")

    with pytest.raises(ValueError, match="cannot reconstruct unsupported coefficients"):
        adapter.validate_supported()
