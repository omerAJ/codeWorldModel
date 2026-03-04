from __future__ import annotations

import torch
from torch import nn


class ConcatLatentPredictor(nn.Module):
    """Predicts concatenated [next_state_latent || next_action_latent]."""

    def __init__(
        self,
        hidden_size: int,
        *,
        hidden_multiplier: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = hidden_size * 3
        mlp_dim = hidden_size * max(1, hidden_multiplier)
        output_dim = hidden_size * 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
