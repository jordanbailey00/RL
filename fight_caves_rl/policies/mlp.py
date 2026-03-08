from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

import pufferlib.pytorch


class MultiDiscreteMLPPolicy(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_nvec: Sequence[int],
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.observation_size = int(observation_size)
        self.action_nvec = tuple(int(value) for value in action_nvec)

        self.encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.observation_size, hidden_size)),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.GELU(),
        )
        self.action_heads = nn.ModuleList(
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, size), std=0.01)
            for size in self.action_nvec
        )
        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1.0)

    @classmethod
    def from_spaces(
        cls,
        observation_space: spaces.Box,
        action_space: spaces.MultiDiscrete,
        hidden_size: int = 128,
    ) -> "MultiDiscreteMLPPolicy":
        if not isinstance(observation_space, spaces.Box):
            raise TypeError(f"Expected Box observation space, got {type(observation_space)!r}.")
        if not isinstance(action_space, spaces.MultiDiscrete):
            raise TypeError(
                f"Expected MultiDiscrete action space, got {type(action_space)!r}."
            )
        observation_size = int(np.prod(observation_space.shape))
        return cls(
            observation_size=observation_size,
            action_nvec=tuple(int(value) for value in action_space.nvec),
            hidden_size=hidden_size,
        )

    def forward_eval(self, observations: torch.Tensor, state: dict[str, object] | None = None):
        hidden = self.encode_observations(observations, state=state)
        return self.decode_actions(hidden)

    def forward(self, observations: torch.Tensor, state: dict[str, object] | None = None):
        return self.forward_eval(observations, state=state)

    def encode_observations(
        self,
        observations: torch.Tensor,
        state: dict[str, object] | None = None,
    ) -> torch.Tensor:
        batch_size = observations.shape[0]
        flattened = observations.view(batch_size, -1).float()
        return self.encoder(flattened)

    def decode_actions(self, hidden: torch.Tensor):
        logits = tuple(head(hidden) for head in self.action_heads)
        values = self.value_head(hidden)
        return logits, values

