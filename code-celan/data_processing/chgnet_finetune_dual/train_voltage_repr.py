"""Fine-tune CHGNet on charge/discharge graphs to predict average_voltage."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

DATA_PROCESSING_ROOT = Path(__file__).resolve().parents[1]
if str(DATA_PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_PROCESSING_ROOT))

from chgnet_finetune_dual.config import DualStateFinetuneConfig
from chgnet_finetune_dual.device_utils import resolve_preload_device, resolve_torch_device
from chgnet_finetune_dual.split import build_chgnet_finetune_split, load_split_assignment
from chgnet_finetune_dual.dataset import DualStateVoltageGraphDataset, collate_dual_voltage_graphs
from chgnet_finetune_dual.graph_cache import build_dual_graph_cache
from chgnet_finetune_dual.model import DualStateVoltageModel, configure_freeze, load_chgnet_backbone
from chgnet_finetune_dual.structures import load_voltage_by_battery_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _resolve_device(requested: str) -> torch.device:
    return resolve_torch_device(requested or "auto")


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if len(y_true) > 1 and ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _optimizer_learning_rates(optimizer: torch.optim.Optimizer) -> dict:
    groups = optimizer.param_groups
    return {
        "lr_backbone": float(groups[0]["lr"]),
        "lr_head": float(groups[1]["lr"]),
    }


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: DualStateFinetuneConfig,
):
    if config.lr_scheduler == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience,
            min_lr=config.lr_scheduler_min_lr,
            cooldown=config.lr_scheduler_cooldown,
        )
    if config.lr_scheduler == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=config.max_epochs,
            eta_min=config.lr_scheduler_min_lr,
        )
    raise ValueError(f"Unknown lr_scheduler: {config.lr_scheduler}")


@torch.no_grad()
def evaluate_model(model, loader, device, *, use_amp: bool = False) -> tuple[float, dict]:
    model.eval()
    preds_all = []
    targets_all = []
    for charge_graphs, discharge_graphs, voltages, _ in loader:
        voltages = voltages.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            preds, _ = model(charge_graphs, discharge_graphs)
        preds_all.append(preds.detach().float().cpu().numpy())
        targets_all.append(voltages.detach().cpu().numpy())
    y_pred = np.concatenate(preds_all)
    y_true = np.concatenate(targets_all)
    metrics = _regression_metrics(y_true, y_pred)
    return metrics["mae"], metrics


def _build_dataloader(dataset, config: DualStateFinetuneConfig, *, shuffle: bool) -> DataLoader:
    use_cuda = resolve_torch_device(config.device).type == "cuda"
    num_workers = config.num_workers if use_cuda else 0
    pin_memory = config.pin_memory and use_cuda
    loader_kwargs = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_dual_voltage_graphs,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def train_dual_voltage_repr(config: DualStateFinetuneConfig) -> Path:
    base_dir = DATA_PROCESSING_ROOT
    config = config.resolve_paths(base_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(config.device)
    use_amp = config.use_amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    logger.info(
        "Using device: %s | chgnet=%s | fusion_mode=%s | batch_size=%d | amp=%s | preload=%s | "
        "preload_device=%s | val_preload_device=%s | workers=%d | lr_scheduler=%s",
        device,
        config.chgnet_checkpoint or config.chgnet_model_name,
        config.fusion_mode,
        config.batch_size,
        use_amp,
        config.preload_graphs,
        config.preload_device,
        config.val_preload_device,
        config.num_workers if device.type == "cuda" else 0,
        config.lr_scheduler,
    )

    split_assignment = load_split_assignment(config.electrode_data_path)
    voltage_by_id = load_voltage_by_battery_id(config.electrode_data_path)

    ft_split = build_chgnet_finetune_split(
        projv2_train_ids=split_assignment.train_battery_ids,
        projv2_val_ids=split_assignment.val_battery_ids,
        projv2_test_ids=split_assignment.test_battery_ids,
        voltage_by_battery_id=voltage_by_id,
        inner_val_ratio=config.inner_val_ratio,
        split_seed=config.split_seed,
        inner_val_seed_offset=config.inner_val_seed_offset,
    )
    split_path = output_dir / "chgnet_inner_split.json"
    ft_split.save(split_path)

    cache_scope = (
        ft_split.chgnet_train_ids
        + ft_split.chgnet_inner_val_ids
        + ft_split.projv2_val_ids
        + ft_split.projv2_test_ids
    )
    build_dual_graph_cache(
        config.electrode_data_path,
        cache_scope,
        config.graph_cache_dir,
        chgnet_model_name=config.chgnet_model_name,
        overwrite=config.graph_cache_overwrite,
    )

    train_preload_device = resolve_preload_device(config.preload_device, device, log=False)
    val_preload_device = resolve_preload_device(config.val_preload_device, device, log=False)
    logger.info(
        "Graph preload targets: train=%s, inner_val=%s",
        train_preload_device,
        val_preload_device,
    )

    train_ds = DualStateVoltageGraphDataset(
        ft_split.chgnet_train_ids,
        voltage_by_id,
        config.graph_cache_dir,
        preload_graphs=config.preload_graphs,
        preload_device=train_preload_device,
    )
    val_ds = DualStateVoltageGraphDataset(
        ft_split.chgnet_inner_val_ids,
        voltage_by_id,
        config.graph_cache_dir,
        preload_graphs=config.preload_graphs,
        preload_device=val_preload_device,
    )
    if len(train_ds) < len(ft_split.chgnet_train_ids):
        logger.warning(
            "Train samples with graph cache: %d/%d",
            len(train_ds),
            len(ft_split.chgnet_train_ids),
        )
    if len(val_ds) < len(ft_split.chgnet_inner_val_ids):
        logger.warning(
            "Inner-val samples with graph cache: %d/%d",
            len(val_ds),
            len(ft_split.chgnet_inner_val_ids),
        )
    train_loader = _build_dataloader(train_ds, config, shuffle=True)
    val_loader = _build_dataloader(val_ds, config, shuffle=False)

    chgnet = load_chgnet_backbone(
        config.chgnet_model_name,
        config.chgnet_checkpoint,
        device=str(device),
    )
    model = DualStateVoltageModel(
        chgnet,
        fusion_mode=config.fusion_mode,
        head_hidden=config.head_hidden,
    ).to(device)
    trainable_modules = configure_freeze(model, config.freeze_level)
    logger.info("Freeze level %s; trainable modules: %s", config.freeze_level, trainable_modules)

    backbone_params = [p for p in model.chgnet.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": config.lr_backbone},
            {"params": head_params, "lr": config.lr_head},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = _build_lr_scheduler(optimizer, config)
    criterion = nn.MSELoss()
    if use_amp:
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    history = []
    best_val_mae = float("inf")
    best_epoch = -1
    patience_left = config.patience
    backbone_path = output_dir / "best_backbone.pt"

    for epoch in range(config.max_epochs):
        model.train()
        train_losses = []
        for charge_graphs, discharge_graphs, voltages, _ in train_loader:
            voltages = voltages.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                preds, _ = model(charge_graphs, discharge_graphs)
                loss = criterion(preds, voltages)
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                if config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        config.grad_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        config.grad_clip,
                    )
                optimizer.step()
            train_losses.append(float(loss.item()))

        val_mae, val_metrics = evaluate_model(model, val_loader, device, use_amp=use_amp)
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_mae)
        else:
            scheduler.step()
        current_lrs = _optimizer_learning_rates(optimizer)

        train_mae, train_metrics = evaluate_model(
            model, train_loader, device, use_amp=use_amp
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "train_mae": train_mae,
                "train_rmse": train_metrics["rmse"],
                "val_mae": val_mae,
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                **current_lrs,
            }
        )
        logger.info(
            "Epoch %d | train_mae=%.4f val_mae=%.4f val_r2=%.4f | "
            "lr_backbone=%.2e lr_head=%.2e",
            epoch,
            train_mae,
            val_mae,
            val_metrics["r2"],
            current_lrs["lr_backbone"],
            current_lrs["lr_head"],
        )

        if (
            config.early_stop_train_val_mae_rel_gap > 0
            and np.isfinite(train_mae)
            and val_mae > 0
            and (val_mae - train_mae) / val_mae >= config.early_stop_train_val_mae_rel_gap
        ):
            rel_gap = (val_mae - train_mae) / val_mae
            logger.info(
                "Early stopping at epoch %d (best epoch %d): train_mae=%.4f is %.1f%% "
                "lower than val_mae=%.4f (threshold %.0f%%).",
                epoch,
                best_epoch,
                train_mae,
                rel_gap * 100.0,
                val_mae,
                config.early_stop_train_val_mae_rel_gap * 100.0,
            )
            break

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_left = config.patience
            model.save_backbone(str(backbone_path))
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping at epoch %d (best epoch %d).", epoch, best_epoch)
                break

    metrics_path = output_dir / "metrics.json"
    config_path = output_dir / "config.json"
    metrics_path.write_text(
        json.dumps(
            {
                "lr_scheduler": config.lr_scheduler,
                "best_epoch": best_epoch,
                "best_inner_val_mae": best_val_mae,
                "final_learning_rates": _optimizer_learning_rates(optimizer),
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    config_path.write_text(json.dumps(config.to_json_dict(), indent=2), encoding="utf-8")
    logger.info("Saved best backbone to %s", backbone_path)
    return backbone_path


def main() -> None:
    train_dual_voltage_repr(DualStateFinetuneConfig())


if __name__ == "__main__":
    main()
