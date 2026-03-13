from __future__ import annotations

import numpy as np
from gymnasium import spaces

from fight_caves_rl.envs.puffer_encoding import POLICY_ACTION_NVECS
from fight_caves_rl.envs.schema import HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA

FAST_OBSERVATION_FEATURE_COUNT = HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA.feature_count
FAST_OBSERVATION_MAX_VISIBLE_NPCS = HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA.max_visible_npcs
FAST_ACTION_NVECS = np.asarray(POLICY_ACTION_NVECS, dtype=np.int64)


def build_fast_observation_space() -> spaces.Box:
    return spaces.Box(
        low=np.float32(-1_000_000.0),
        high=np.float32(1_000_000.0),
        shape=(FAST_OBSERVATION_FEATURE_COUNT,),
        dtype=np.float32,
    )


def build_fast_action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete(FAST_ACTION_NVECS)
