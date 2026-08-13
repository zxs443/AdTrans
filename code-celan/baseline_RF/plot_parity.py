"""Parity plots for Random Forest baseline."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from adtrans_bridge import VISUALIZATION_CONFIG, plot_single_parity, uses_structure_input
from config import INPUT_CONFIG, input_config_tag


def _format_feature_list(features: List[str]) -> str:
    if not features:
        return ""
    if len(features) == 1:
        return features[0]
    if len(features) == 2:
        return f"{features[0]} and {features[1]}"
    return ", ".join(features[:-1]) + f" and {features[-1]}"


def build_baseline_parity_model_info(encoder_type: str, decoder_type: str) -> str:
    """Top-left parity caption for the flattened Random Forest baseline."""
    features = ["formula"]
    if uses_structure_input() and INPUT_CONFIG.get("use_structure"):
        features.append("structure")
    for name in VISUALIZATION_CONFIG.get("electrode_features") or []:
        if name and name not in features:
            features.append(name)

    return (
        f"Model: Random Forest baseline\n"
        f"Features: {_format_feature_list(features)}\n"
        f"Encoder encoding: {encoder_type}\n"
        f"Decoder encoding: {decoder_type}"
    )


def visualize_baseline_predictions(
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
    metrics: Dict[str, Dict[str, float]],
    *,
    base_dir: str,
    encoder_type: str,
    decoder_type: str,
    run_tag: Optional[str] = None,
    targets_mean: float = 0.0,
    targets_std: float = 1.0,
):
    """Save train/val/test parity PNGs + combined predictions/metrics CSV (Adtrans styling)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_tag = run_tag or f"{encoder_type}_{decoder_type}_{input_config_tag()}"
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    def _unscale(arr):
        return arr * targets_std + targets_mean

    train_true_u = _unscale(train_true)
    train_pred_u = _unscale(train_pred)
    val_true_u = _unscale(val_true)
    val_pred_u = _unscale(val_pred)
    test_true_u = _unscale(test_true)
    test_pred_u = _unscale(test_pred)

    predictions_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "battery_id": train_battery_ids,
                    "True": train_true_u,
                    "Predicted": train_pred_u,
                    "Formula": train_formulas,
                    "Data Set": "Train",
                }
            ),
            pd.DataFrame(
                {
                    "battery_id": val_battery_ids,
                    "True": val_true_u,
                    "Predicted": val_pred_u,
                    "Formula": val_formulas,
                    "Data Set": "Validation",
                }
            ),
            pd.DataFrame(
                {
                    "battery_id": test_battery_ids,
                    "True": test_true_u,
                    "Predicted": test_pred_u,
                    "Formula": test_formulas,
                    "Data Set": "Test",
                }
            ),
        ]
    )

    model_info = build_baseline_parity_model_info(encoder_type, decoder_type)
    for subset_name, true_u, pred_u, battery_ids in (
        ("Train", train_true_u, train_pred_u, train_battery_ids),
        ("Validation", val_true_u, val_pred_u, val_battery_ids),
        ("Test", test_true_u, test_pred_u, test_battery_ids),
    ):
        save_path = os.path.join(
            plots_dir,
            f"parity_{subset_name.lower()}_{file_tag}_{timestamp}.png",
        )
        plot_single_parity(
            true_u,
            pred_u,
            subset_name=subset_name,
            model_info=model_info,
            save_path=save_path,
            battery_ids=battery_ids,
        )
        print(f"Saved parity plot: {save_path}", flush=True)

    predictions_df.to_csv(
        os.path.join(base_dir, f"predictions_{file_tag}_{timestamp}.csv"),
        index=False,
    )
    pd.DataFrame(metrics).T.to_csv(
        os.path.join(base_dir, f"metrics_{file_tag}_{timestamp}.csv"),
    )
    return metrics, timestamp
