"""Training, evaluation, and Optuna search for the Random Forest baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import optuna
from optuna.trial import Trial
from sklearn.ensemble import RandomForestRegressor

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
    format_input_config_summary,
    input_config_tag,
)
from flat_data import create_flat_arrays, load_processed_data
from adtrans_bridge import calculate_metrics
from plot_parity import visualize_baseline_predictions


def suggest_rf_params(trial: Trial) -> dict:
    """Sample one Random Forest configuration from the search space in OPTIMIZER_CONFIG."""
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators", *OPTIMIZER_CONFIG["n_estimators_range"], step=100
        ),
        "max_depth": trial.suggest_int(
            "max_depth", *OPTIMIZER_CONFIG["max_depth_range"], log=True
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split", *OPTIMIZER_CONFIG["min_samples_split_range"]
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", *OPTIMIZER_CONFIG["min_samples_leaf_range"]
        ),
        "max_features": trial.suggest_float(
            "max_features", *OPTIMIZER_CONFIG["max_features_range"], log=True
        ),
        "max_samples": trial.suggest_float(
            "max_samples", *OPTIMIZER_CONFIG["max_samples_range"]
        ),
    }


def build_model(params: dict, seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        max_samples=params["max_samples"] if MODEL_CONFIG["bootstrap"] else None,
        bootstrap=MODEL_CONFIG["bootstrap"],
        criterion=MODEL_CONFIG["criterion"],
        n_jobs=MODEL_CONFIG["n_jobs"],
        random_state=seed,
    )


def objective(trial: Trial, splits: dict, seed: int) -> float:
    params = suggest_rf_params(trial)
    model = build_model(params, seed)
    model.fit(splits["train"]["X"], splits["train"]["y"])
    val_pred = model.predict(splits["val"]["X"])
    val_mse = float(np.mean((val_pred - splits["val"]["y"]) ** 2))
    return val_mse


def optimize_hyperparameters(
    splits: dict, seed: int, n_trials: Optional[int] = None
) -> Tuple[dict, float]:
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
        lambda trial: objective(trial, splits, seed),
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[_on_trial_finished],
    )
    return study.best_params, study.best_value


def denormalize_predictions(predictions: np.ndarray, targets_mean: float, targets_std: float) -> np.ndarray:
    return predictions * targets_std + targets_mean


def count_forest_nodes(model: RandomForestRegressor) -> int:
    return int(sum(tree.tree_.node_count for tree in model.estimators_))


def evaluate_and_collect(
    model: RandomForestRegressor,
    splits: dict,
    processed_data,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, tuple]]:
    """Return unscaled metrics and per-split (pred, target, formulas, battery_ids) in z-score space."""
    targets_mean = float(processed_data["targets_mean"])
    targets_std = float(processed_data["targets_std"])

    metrics: Dict[str, Dict[str, float]] = {}
    collected: Dict[str, tuple] = {}
    for split_name, split_key in (
        ("Train", "train"),
        ("Validation", "val"),
        ("Test", "test"),
    ):
        split = splits[split_key]
        preds = model.predict(split["X"])
        targets = split["y"]
        preds_orig = denormalize_predictions(preds, targets_mean, targets_std)
        targets_orig = denormalize_predictions(targets, targets_mean, targets_std)
        metrics[split_name] = calculate_metrics(targets_orig, preds_orig)
        collected[split_name] = (preds, targets, split["formulas"], split["battery_ids"])
    return metrics, collected


def run_baseline(
    base_path: Path,
    seed: int,
    n_trials: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    print("[1/3] Preprocess flat features...", flush=True)
    processed_data, split_mode = load_processed_data(base_path, seed)
    print(f"Split mode: {split_mode} | blocks: {format_input_config_summary()}", flush=True)

    splits = create_flat_arrays(processed_data)
    input_dim = splits["input_dim"]
    print(f"Flat input dim: {input_dim}", flush=True)
    print(
        "Split sizes: train=%d val=%d test=%d"
        % (len(splits["train"]["y"]), len(splits["val"]["y"]), len(splits["test"]["y"])),
        flush=True,
    )

    print("[2/3] Hyperparameter optimization (Optuna)...", flush=True)
    best_params, best_value = optimize_hyperparameters(splits, seed, n_trials=n_trials)
    print(f"Optuna best val v_mse: {best_value:.6f}", flush=True)
    print(f"Best params: {best_params}", flush=True)

    print("[3/3] Final training and evaluation...", flush=True)
    model = build_model(best_params, seed)
    model.fit(splits["train"]["X"], splits["train"]["y"])
    print(
        f"Fitted forest: {len(model.estimators_)} trees, "
        f"{count_forest_nodes(model):,} total nodes",
        flush=True,
    )

    encoder_desc, decoder_desc = DATASET_CONFIG["descriptor_pair"]
    combined_name = f"{encoder_desc}_{decoder_desc}_{input_config_tag()}"
    save_dir = RESULTS_DIR / combined_name
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics, collected = evaluate_and_collect(model, splits, processed_data)

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
        "input_dim": input_dim,
        "input_config": dict(INPUT_CONFIG),
        "dataset_config": dict(DATASET_CONFIG),
        "model_config": dict(MODEL_CONFIG),
        "best_value": float(best_value) if best_value is not None else None,
        "seed": seed,
    }
    with open(save_dir / "best_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(best_params_to_save, f, indent=2, ensure_ascii=False)

    if PLOT_CONFIG.get("save_model_enabled", True):
        model_path = save_dir / "model.joblib"
        joblib.dump(model, model_path, compress=3)
        print(f"Saved model: {model_path}", flush=True)

    np.save(
        save_dir / "data_stats.npy",
        {
            "targets_mean": processed_data["targets_mean"],
            "targets_std": processed_data["targets_std"],
            "input_dim": input_dim,
            "input_config": dict(INPUT_CONFIG),
            "dataset_config": dict(DATASET_CONFIG),
            "seed": seed,
            "best_params": dict(best_params),
            "n_trees": len(model.estimators_),
            "n_forest_nodes": count_forest_nodes(model),
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
