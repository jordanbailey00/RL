from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

import pufferlib.pytorch


class MultiDiscreteLSTMPolicy(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_nvec: Sequence[int],
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.observation_size = int(observation_size)
        self.action_nvec = tuple(int(value) for value in action_nvec)

        self.encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.observation_size, self.hidden_size)),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, self.hidden_size)),
            nn.GELU(),
        )
        self.rollout_cell = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.training_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.action_heads = nn.ModuleList(
            pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, size), std=0.01)
            for size in self.action_nvec
        )
        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(self.hidden_size, 1), std=1.0)

    @classmethod
    def from_spaces(
        cls,
        observation_space: spaces.Box,
        action_space: spaces.MultiDiscrete,
        hidden_size: int = 128,
    ) -> "MultiDiscreteLSTMPolicy":
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

    def forward_eval(
        self,
        observations: torch.Tensor,
        state: dict[str, object] | None = None,
    ):
        batch_size = observations.shape[0]
        flattened = observations.view(batch_size, -1).float()
        encoded = self.encoder(flattened)
        lstm_h, lstm_c = self._rollout_state(batch_size, encoded.device, state)

        done_mask = self._done_mask(batch_size, encoded.device, state)
        if done_mask is not None:
            keep = (~done_mask).unsqueeze(1).to(dtype=lstm_h.dtype)
            lstm_h = lstm_h * keep
            lstm_c = lstm_c * keep

        next_h, next_c = self.rollout_cell(encoded, (lstm_h, lstm_c))
        if state is not None:
            state["lstm_h"] = next_h
            state["lstm_c"] = next_c
        return self.decode_actions(next_h)

    def forward(
        self,
        observations: torch.Tensor,
        state: dict[str, object] | None = None,
    ):
        if observations.ndim == 2:
            return self.forward_eval(observations, state=state)

        batch_size, horizon = observations.shape[:2]
        flattened = observations.reshape(batch_size * horizon, -1).float()
        encoded = self.encoder(flattened).reshape(batch_size, horizon, self.hidden_size)
        h0, c0 = self._training_state(batch_size, encoded.device, state)
        outputs, (hn, cn) = self.training_lstm(encoded, (h0, c0))
        if state is not None:
            state["lstm_h"] = hn.squeeze(0)
            state["lstm_c"] = cn.squeeze(0)
        return self.decode_actions(outputs.reshape(batch_size * horizon, self.hidden_size))

    def decode_actions(self, hidden: torch.Tensor):
        logits = tuple(head(hidden) for head in self.action_heads)
        values = self.value_head(hidden)
        return logits, values

    def _rollout_state(
        self,
        batch_size: int,
        device: torch.device,
        state: dict[str, object] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None or state.get("lstm_h") is None or state.get("lstm_c") is None:
            zeros = torch.zeros(batch_size, self.hidden_size, device=device)
            return zeros, zeros.clone()
        return (
            torch.as_tensor(state["lstm_h"], device=device).view(batch_size, self.hidden_size),
            torch.as_tensor(state["lstm_c"], device=device).view(batch_size, self.hidden_size),
        )

    def _training_state(
        self,
        batch_size: int,
        device: torch.device,
        state: dict[str, object] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None or state.get("lstm_h") is None or state.get("lstm_c") is None:
            zeros = torch.zeros(1, batch_size, self.hidden_size, device=device)
            return zeros, zeros.clone()
        return (
            torch.as_tensor(state["lstm_h"], device=device).view(1, batch_size, self.hidden_size),
            torch.as_tensor(state["lstm_c"], device=device).view(1, batch_size, self.hidden_size),
        )

    def _done_mask(
        self,
        batch_size: int,
        device: torch.device,
        state: dict[str, object] | None,
    ) -> torch.Tensor | None:
        if state is None:
            return None
        done = state.get("done")
        if done is None:
            return None
        return torch.as_tensor(done, device=device).view(batch_size).bool()
