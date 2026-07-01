"""Dual-state CHGNet backbone and voltage regression head."""

from __future__ import annotations

from typing import List, Literal, Sequence

import torch
import torch.nn as nn
from chgnet.graph.crystalgraph import CrystalGraph
from chgnet.model.functions import find_activation
from chgnet.model.model import CHGNet

from .config import FusionMode, validate_chgnet_model_name
from .device_utils import move_graphs_to_device

__all__ = [
    "DualStateVoltageModel",
    "configure_freeze",
    "load_chgnet_backbone",
    "fuse_crystal_features",
]


def fuse_crystal_features(
    charge_fea: torch.Tensor,
    discharge_fea: torch.Tensor,
    fusion_mode: FusionMode,
) -> torch.Tensor:
    if fusion_mode == "discharge":
        return discharge_fea
    if fusion_mode == "concat":
        # [charge_64 | discharge_64] -> chg_struct_emb_0..63 / _64..127
        return torch.cat([charge_fea, discharge_fea], dim=-1)
    if fusion_mode == "diff":
        return discharge_fea - charge_fea
    if fusion_mode == "mean":
        return 0.5 * (charge_fea + discharge_fea)
    raise ValueError(f"Unknown fusion_mode: {fusion_mode}")


def _backbone_activation_module(chgnet: CHGNet) -> nn.Module:
    name = "silu"
    model_args = getattr(chgnet, "model_args", None)
    if model_args:
        name = model_args.get("non_linearity", name)
    return find_activation(name)


class DualStateVoltageModel(nn.Module):
    def __init__(
        self,
        chgnet: CHGNet,
        *,
        fusion_mode: FusionMode = "concat",
        head_hidden: int = 0,
    ):
        super().__init__()
        self.chgnet = chgnet
        self.fusion_mode = fusion_mode
        fea_dim = 128 if fusion_mode == "concat" else 64
        if head_hidden and head_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(fea_dim, head_hidden),
                _backbone_activation_module(chgnet),
                nn.Linear(head_hidden, 1),
            )
        else:
            self.head = nn.Linear(fea_dim, 1)

    def encode_graphs(self, graphs: Sequence[CrystalGraph]) -> torch.Tensor:
        device = next(self.parameters()).device
        graph_list = move_graphs_to_device(graphs, device)
        out = self.chgnet.forward(graph_list, task="e", return_crystal_feas=True)
        return out["crystal_fea"]

    def encode_dual(
        self,
        charge_graphs: Sequence[CrystalGraph],
        discharge_graphs: Sequence[CrystalGraph],
    ) -> torch.Tensor:
        batch_size = len(charge_graphs)
        if batch_size != len(discharge_graphs):
            raise ValueError("charge/discharge batch sizes must match")
        if batch_size == 0:
            raise ValueError("empty graph batch")
        if self.fusion_mode == "discharge":
            return self.encode_graphs(discharge_graphs)
        all_fea = self.encode_graphs(list(charge_graphs) + list(discharge_graphs))
        charge_fea = all_fea[:batch_size]
        discharge_fea = all_fea[batch_size:]
        return fuse_crystal_features(charge_fea, discharge_fea, self.fusion_mode)

    def forward(
        self,
        charge_graphs: Sequence[CrystalGraph],
        discharge_graphs: Sequence[CrystalGraph],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        crystal_fea = self.encode_dual(charge_graphs, discharge_graphs)
        voltage_pred = self.head(crystal_fea).squeeze(-1)
        return voltage_pred, crystal_fea

    def save_backbone(self, path: str) -> None:
        payload = {"model": self.chgnet.as_dict()}
        torch.save(payload, path)


def load_chgnet_backbone(
    model_name: str = "0.3.0",
    checkpoint: str | None = None,
    device: str = "cpu",
) -> CHGNet:
    if checkpoint:
        model = CHGNet.from_file(checkpoint)
    else:
        model = CHGNet.load(
            model_name=validate_chgnet_model_name(model_name),
            use_device=device,
            verbose=True,
        )
    return model.to(device)


def configure_freeze(model: DualStateVoltageModel, level: str) -> List[str]:
    for param in model.chgnet.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    trainable: List[str] = ["head"]
    level = level.upper()

    if level == "L0":
        return trainable

    if level in {"L1", "L2"}:
        for param in model.chgnet.pooling.parameters():
            param.requires_grad = True
        trainable.append("pooling")

    if level in {"L1", "L2"}:
        for param in model.chgnet.atom_conv_layers[-1].parameters():
            param.requires_grad = True
        trainable.append("atom_conv_layers[-1]")

    if level == "L2" and len(model.chgnet.atom_conv_layers) >= 2:
        for param in model.chgnet.atom_conv_layers[-2].parameters():
            param.requires_grad = True
        trainable.append("atom_conv_layers[-2]")

    if level == "L3":
        for param in model.chgnet.parameters():
            param.requires_grad = True
        trainable = ["chgnet", "head"]

    if level not in {"L0", "L1", "L2", "L3"}:
        raise ValueError(f"Unknown freeze_level: {level}")

    return trainable
