"""Traditional feed-forward MLP baseline for voltage regression."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


def _build_activation(activation_config: Optional[dict]) -> nn.Module:
    if activation_config is None:
        activation_config = {"type": "GELU", "params": {}}
    activation_type = activation_config.get("type", "GELU")
    params = activation_config.get("params", {})
    if activation_type == "GELU":
        return nn.GELU(**params)
    if activation_type == "ReLU":
        return nn.ReLU(**params)
    if activation_type == "LeakyReLU":
        return nn.LeakyReLU(**params)
    if activation_type == "ELU":
        return nn.ELU(**params)
    raise ValueError(f"Unsupported activation function type: {activation_type}")


class MLPBaseline(nn.Module):
    """Flattened encoder + decoder (+ structure) -> MLP -> scalar voltage."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rate: float,
        activation_config: Optional[dict] = None,
    ):
        super().__init__()
        if activation_config is None:
            activation_config = {"type": "GELU", "params": {}}
        self.activation_config = activation_config

        layers: List[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    _build_activation(activation_config),
                    nn.Dropout(dropout_rate),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout_rate = dropout_rate

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        return {
            "model_type": "MLPBaseline",
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
        }
