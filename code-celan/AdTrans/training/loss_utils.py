"""Training loss decomposition (voltage MSE + L1)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from utils.config import TRAINING_CONFIG


@dataclass
class LossBreakdown:
    """Per-batch training loss components."""
    voltage_mse: torch.Tensor
    l1: torch.Tensor
    total: torch.Tensor

    def detached_dict(self) -> dict:
        return {
            'voltage_mse': float(self.voltage_mse.detach()),
            'l1': float(self.l1.detach()),
            'total': float(self.total.detach()),
        }


def compute_l1_penalty(model: nn.Module) -> torch.Tensor:
    if not TRAINING_CONFIG['l1_regularization']['enabled']:
        device = next(model.parameters()).device
        return torch.zeros((), device=device)
    l1_lambda = TRAINING_CONFIG['l1_regularization']['lambda']
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return l1_lambda * l1_norm


def compute_training_loss(
    voltage_pred: torch.Tensor,
    voltage_target: torch.Tensor,
    model: nn.Module,
    voltage_criterion: nn.Module,
    **_ignored,
) -> LossBreakdown:
    voltage_mse = voltage_criterion(voltage_pred, voltage_target)
    l1 = compute_l1_penalty(model)
    total = voltage_mse + l1
    return LossBreakdown(voltage_mse=voltage_mse, l1=l1, total=total)


def format_train_epoch_summary(breakdown: dict, fallback_total: float = 0.0) -> str:
    total = breakdown.get('total', fallback_total)
    return (
        f"Train[optimize] total={total:.6f} "
        f"(voltage_mse={breakdown.get('voltage_mse', 0):.6f}, "
        f"l1={breakdown.get('l1', 0):.6f})"
    )


def format_val_epoch_summary(val_voltage_mse: float, val_breakdown: dict | None = None) -> str:
    return f"Val voltage_mse={val_voltage_mse:.6f}"


def average_breakdown_dict(accum: dict, count: int) -> dict:
    if count <= 0:
        return {k: 0.0 for k in accum}
    return {k: v / count for k, v in accum.items()}


def default_loss_accum() -> dict:
    return {
        'voltage_mse': 0.0,
        'l1': 0.0,
        'total': 0.0,
    }
