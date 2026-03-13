"""Fast-kernel vector environment surfaces for the V2 trainer path."""

from fight_caves_rl.envs_fast.fast_reward_adapter import FastRewardAdapter
from fight_caves_rl.envs_fast.fast_trace_adapter import (
    adapt_fast_parity_trace,
    adapt_fast_parity_traces,
    extract_fast_parity_traces,
)
from fight_caves_rl.envs_fast.fast_vector_env import FastKernelVecEnv, FastKernelVecEnvConfig

__all__ = [
    "FastKernelVecEnv",
    "FastKernelVecEnvConfig",
    "FastRewardAdapter",
    "adapt_fast_parity_trace",
    "adapt_fast_parity_traces",
    "extract_fast_parity_traces",
]
