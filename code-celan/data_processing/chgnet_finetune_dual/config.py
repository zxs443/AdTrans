"""CHGNet charge/discharge finetune configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Optional

FusionMode = Literal["discharge", "concat", "diff", "mean"]
ExportEmbeddingMode = Literal["discharge", "concat", "diff", "mean"]
LRSchedulerType = Literal["reduce_on_plateau", "cosine"]
PreloadDevice = Literal["auto", "cpu", "cuda"]
ChgnetModelName = Literal["0.3.0", "0.2.0", "r2scan"]

CHGNET_PRETRAINED_MODELS: tuple[str, ...] = ("0.3.0", "0.2.0", "r2scan")


def default_graph_cache_dir(model_name: str) -> str:
    safe = str(model_name).replace(".", "_")
    return f"graph_cache/chgnet_{safe}_dual"


def validate_chgnet_model_name(model_name: str) -> str:
    name = str(model_name).strip()
    if name not in CHGNET_PRETRAINED_MODELS:
        raise ValueError(
            f"chgnet_model_name={name!r} unsupported; "
            f"choose from {list(CHGNET_PRETRAINED_MODELS)}"
        )
    return name


@dataclass
class DualStateFinetuneConfig:
    electrode_data_path: str = "source_data/split"
    graph_cache_dir: Optional[str] = None
    chgnet_model_name: ChgnetModelName = "0.2.0"
    chgnet_checkpoint: Optional[str] = None

    fusion_mode: FusionMode = "concat"
    export_embedding_mode: ExportEmbeddingMode = "concat"

    split_seed: int = 42
    inner_val_ratio: float = 0.1
    inner_val_seed_offset: int = 10001

    freeze_level: str = "L1"
    head_hidden: int = 64
    batch_size: int = 16
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    lr_scheduler: LRSchedulerType = "reduce_on_plateau"
    lr_scheduler_factor: float = 0.4
    lr_scheduler_patience: int = 3
    lr_scheduler_min_lr: float = 1e-8
    lr_scheduler_cooldown: int = 0
    weight_decay: float = 1e-8
    max_epochs: int = 500
    early_stop_train_val_mae_rel_gap: float = 0.3
    patience: int = 6
    grad_clip: float = 1.0
    device: str = "auto"

    preload_graphs: bool = True
    preload_device: PreloadDevice = "cpu"
    val_preload_device: PreloadDevice = "cpu"
    num_workers: int = 0
    pin_memory: bool = True
    use_amp: bool = False
    eval_train_each_epoch: bool = False

    output_dir: str = "chgnet_finetune_dual/results/voltage_repr"
    finetune_checkpoint: Optional[str] = None
    export_descriptor_name: str = "chgnet_ft_v_dual"
    export_device: str = "cpu"
    structure_column_prefix: str = "chg_struct_emb"
    graph_cache_overwrite: bool = False

    max_elements: int = 6
    descriptor_root: str = "descriptors"

    def resolve_paths(self, base_dir: Path) -> "DualStateFinetuneConfig":
        model_name = validate_chgnet_model_name(self.chgnet_model_name)
        cache_dir = self.graph_cache_dir or default_graph_cache_dir(model_name)
        updates = {"chgnet_model_name": model_name, "graph_cache_dir": cache_dir}
        for field in (
            "electrode_data_path",
            "graph_cache_dir",
            "output_dir",
            "descriptor_root",
        ):
            value = updates.get(field, getattr(self, field))
            if value and not Path(value).is_absolute():
                updates[field] = str(base_dir / value)
        if self.chgnet_checkpoint and not Path(self.chgnet_checkpoint).is_absolute():
            updates["chgnet_checkpoint"] = str(base_dir / self.chgnet_checkpoint)
        if self.finetune_checkpoint and not Path(self.finetune_checkpoint).is_absolute():
            updates["finetune_checkpoint"] = str(base_dir / self.finetune_checkpoint)
        return replace(self, **updates) if updates else self

    def resolved_finetune_checkpoint(self) -> Path:
        if self.finetune_checkpoint:
            return Path(self.finetune_checkpoint)
        return Path(self.output_dir) / "best_backbone.pt"

    def to_json_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def embedding_dim(mode: ExportEmbeddingMode | FusionMode) -> int:
        return 128 if mode == "concat" else 64
