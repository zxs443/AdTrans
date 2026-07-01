"""Training, evaluation, and Optuna search for MLP baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.trial import Trial

BASELINE_ROOT = Path(__file__).resolve().parent
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

import adtrans_bridge  

from config import (
    DATASET_CONFIG,
    INPUT_CONFIG,
    MODEL_CONFIG,
    OPTIMIZER_CONFIG,
    PLOT_CONFIG,
    RESULTS_DIR,
    TRAINING_CONFIG,
    format_input_config_summary,
    input_config_tag,
)
from flat_data import create_flat_loaders, infer_input_dim, load_processed_data, unpack_flat_batch
from model import MLPBaseline
from adtrans_bridge import calculate_metrics
from plot_parity import visualize_baseline_predictions


def compute_l1_penalty(model: nn.Module) -> torch.Tensor:
    if not TRAINING_CONFIG["l1_regularization"]["enabled"]:
        device = next(model.parameters()).device
        return torch.zeros((), device=device)
    l1_lambda = TRAINING_CONFIG["l1_regularization"]["lambda"]
    return l1_lambda * sum(p.abs().sum() for p in model.parameters())


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    for batch in loader:
        features, targets, _, _ = unpack_flat_batch(batch, device)
        optimizer.zero_grad()
        preds = model(features)
        loss = criterion(preds, targets) + compute_l1_penalty(model)
        loss.backward()
        if TRAINING_CONFIG["gradient_clipping"]["enabled"]:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                TRAINING_CONFIG["gradient_clipping"]["clip_value"],
            )
        optimizer.step()
        batch_size = features.size(0)
        n_samples += batch_size
        total_loss += loss.item() * batch_size
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device) -> Tuple[float, np.ndarray, np.ndarray, list, list]:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    preds_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    formulas_all = []
    battery_ids_all = []

    for batch in loader:
        features, targets, formulas, battery_ids = unpack_flat_batch(batch, device)
        preds = model(features)
        loss = criterion(preds, targets)
        batch_size = features.size(0)
        n_samples += batch_size
        total_loss += loss.item() * batch_size
        preds_list.append(preds.detach().cpu().numpy())
        targets_list.append(targets.detach().cpu().numpy())
        formulas_all.extend(formulas)
        battery_ids_all.extend(battery_ids)

    avg_loss = total_loss / max(n_samples, 1)
    predictions = np.concatenate(preds_list) if preds_list else np.array([])
    targets = np.concatenate(targets_list) if targets_list else np.array([])
    return avg_loss, predictions, targets, formulas_all, battery_ids_all


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
) -> Tuple[nn.Module, List[float], List[float], List[float]]:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=TRAINING_CONFIG["scheduler"]["factor"],
        patience=TRAINING_CONFIG["scheduler"]["patience"],
        min_lr=TRAINING_CONFIG["scheduler"]["min_lr"],
    )

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    learning_rates: List[float] = []

    for epoch in range(TRAINING_CONFIG["max_epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _, _, _ = evaluate_loader(model, val_loader, criterion, device)

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != old_lr:
            print(f"Epoch {epoch:03d} | LR {old_lr:.2e} -> {new_lr:.2e}", flush=True)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        learning_rates.append(new_lr)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | Train MSE={train_loss:.6f} | "
                f"Val MSE={val_loss:.6f}",
                flush=True,
            )

        if val_loss < best_val_loss - TRAINING_CONFIG["min_delta"]:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= TRAINING_CONFIG["patience"]:
            print(f"Early stop at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best val v_mse: {best_val_loss:.6f}", flush=True)
    return model, train_losses, val_losses, learning_rates


def suggest_hidden_dims(trial: Trial, input_dim: int, n_layers: int) -> List[int]:
    min_ratio, max_ratio = OPTIMIZER_CONFIG["hidden_dim_ratio_range"]
    dims: List[int] = []
    current = input_dim
    for layer_idx in range(n_layers):
        low = max(8, int(current * min_ratio))
        high = max(low, int(current * max_ratio))
        hidden = trial.suggest_int(f"hidden_dim_layer_{layer_idx}", low, high)
        dims.append(hidden)
        current = hidden
    return dims


def objective(trial: Trial, processed_data, input_dim: int, device, seed: int) -> float:
    batch_size = trial.suggest_categorical("batch_size", OPTIMIZER_CONFIG["batch_sizes"])
    dropout_rate = trial.suggest_float("dropout_rate", *OPTIMIZER_CONFIG["dropout_range"])
    learning_rate = trial.suggest_float(
        "learning_rate", *OPTIMIZER_CONFIG["learning_rate_range"], log=True
    )
    weight_decay = trial.suggest_float(
        "weight_decay", *OPTIMIZER_CONFIG["weight_decay_range"], log=True
    )
    hidden_dims = suggest_hidden_dims(trial, input_dim, MODEL_CONFIG["hidden_layers"])

    loaders = create_flat_loaders(processed_data, batch_size, seed)
    model = MLPBaseline(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        activation_config=TRAINING_CONFIG["model_activation"],
    ).to(device)

    print(f"  trial {trial.number} | params={model.count_parameters():,}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()
    _, _, val_losses, _ = train_with_early_stopping(
        model,
        loaders["train_loader"],
        loaders["val_loader"],
        optimizer,
        criterion,
        device,
    )
    return float(min(val_losses)) if val_losses else float("inf")


def optimize_hyperparameters(
    processed_data, input_dim: int, device, seed: int, n_trials: Optional[int] = None
):
    if n_trials is None:
        n_trials = OPTIMIZER_CONFIG["n_trials"]

    print(f"\nOptuna: {n_trials} trials (objective = val voltage MSE)", flush=True)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    def _on_trial_finished(_study, trial):
        value = trial.value
        value_text = f"{value:.6f}" if value is not None else "failed"
        print(f"  trial {trial.number + 1}/{n_trials}: val_v_mse={value_text}", flush=True)

    study.optimize(
        lambda trial: objective(trial, processed_data, input_dim, device, seed),
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[_on_trial_finished],
    )
    return study.best_params, study.best_value


def denormalize_predictions(predictions: np.ndarray, targets_mean: float, targets_std: float) -> np.ndarray:
    return predictions * targets_std + targets_mean


def evaluate_and_collect(
    model,
    loaders,
    processed_data,
    device,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, tuple]]:
    """Return unscaled metrics and per-split (pred, target, formulas, battery_ids) in z-score space."""
    criterion = nn.MSELoss()
    targets_mean = float(processed_data["targets_mean"])
    targets_std = float(processed_data["targets_std"])

    metrics: Dict[str, Dict[str, float]] = {}
    collected: Dict[str, tuple] = {}
    split_loaders = {
        "Train": loaders["train_loader"],
        "Validation": loaders["val_loader"],
        "Test": loaders["test_loader"],
    }
    for split_name, loader in split_loaders.items():
        _, preds, targets, formulas, battery_ids = evaluate_loader(
            model, loader, criterion, device
        )
        preds_orig = denormalize_predictions(preds, targets_mean, targets_std)
        targets_orig = denormalize_predictions(targets, targets_mean, targets_std)
        metrics[split_name] = calculate_metrics(targets_orig, preds_orig)
        collected[split_name] = (preds, targets, formulas, battery_ids)
    return metrics, collected


def run_baseline(
    base_path: Path,
    device: torch.device,
    seed: int,
    n_trials: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    print("[1/3] Preprocess flat features...", flush=True)
    processed_data, split_mode = load_processed_data(base_path, seed)
    print(f"Split mode: {split_mode} | blocks: {format_input_config_summary()}", flush=True)

    input_dim = infer_input_dim(processed_data)
    print(f"Flat input dim: {input_dim}", flush=True)

    print("[2/3] Hyperparameter optimization (Optuna)...", flush=True)
    best_params, best_value = optimize_hyperparameters(
        processed_data, input_dim, device, seed, n_trials=n_trials
    )
    print(f"Optuna best val v_mse: {best_value:.6f}", flush=True)

    print("[3/3] Final training and evaluation...", flush=True)
    loaders = create_flat_loaders(processed_data, best_params["batch_size"], seed)
    hidden_dims = [
        best_params[f"hidden_dim_layer_{i}"] for i in range(MODEL_CONFIG["hidden_layers"])
    ]
    model = MLPBaseline(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=best_params["dropout_rate"],
        activation_config=TRAINING_CONFIG["model_activation"],
    ).to(device)
    print(f"Trainable parameters: {model.count_parameters():,}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
    )
    criterion = nn.MSELoss()
    model, train_losses, val_losses, learning_rates = train_with_early_stopping(
        model,
        loaders["train_loader"],
        loaders["val_loader"],
        optimizer,
        criterion,
        device,
    )

    encoder_desc, decoder_desc = DATASET_CONFIG["descriptor_pair"]
    combined_name = f"{encoder_desc}_{decoder_desc}_{input_config_tag()}"
    save_dir = RESULTS_DIR / combined_name
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics, collected = evaluate_and_collect(model, loaders, processed_data, device)

    if PLOT_CONFIG.get("parity_plots_enabled", True):
        train_pred, train_true, train_formulas, train_battery_ids = collected["Train"]
        val_pred, val_true, val_formulas, val_battery_ids = collected["Validation"]
        test_pred, test_true, test_formulas, test_battery_ids = collected["Test"]
        visualize_baseline_predictions(
            train_true,
            train_pred,
            val_true,
            val_pred,
            test_true,
            test_pred,
            train_formulas,
            val_formulas,
            test_formulas,
            train_battery_ids,
            val_battery_ids,
            test_battery_ids,
            metrics,
            base_dir=str(save_dir),
            encoder_type=encoder_desc,
            decoder_type=decoder_desc,
            run_tag=combined_name,
            targets_mean=float(processed_data["targets_mean"]),
            targets_std=float(processed_data["targets_std"]),
        )

    best_params_to_save = {
        **best_params,
        "hidden_dims": hidden_dims,
        "input_dim": input_dim,
        "input_config": dict(INPUT_CONFIG),
        "dataset_config": dict(DATASET_CONFIG),
        "best_value": float(best_value) if best_value is not None else None,
        "seed": seed,
    }
    with open(save_dir / "best_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(best_params_to_save, f, indent=2, ensure_ascii=False)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.get_config(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "learning_rates": learning_rates,
        },
        save_dir / "model.pth",
    )

    np.save(
        save_dir / "data_stats.npy",
        {
            "targets_mean": processed_data["targets_mean"],
            "targets_std": processed_data["targets_std"],
            "input_dim": input_dim,
            "input_config": dict(INPUT_CONFIG),
            "dataset_config": dict(DATASET_CONFIG),
            "seed": seed,
            "model_config": model.get_config(),
        },
    )

    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results: {save_dir}", flush=True)
    print("Metrics (voltage, unscaled):", flush=True)
    for split_name, m in metrics.items():
        print(
            f"  {split_name:12s} MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R²={m['R2']:.4f}",
            flush=True,
        )
    return metrics
