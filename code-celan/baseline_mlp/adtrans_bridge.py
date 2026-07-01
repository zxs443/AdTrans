"""Import AdTrans training utilities."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

BASELINE_ROOT = Path(__file__).resolve().parent
ADTRANS_ROOT = BASELINE_ROOT.parent / "AdTrans"


def ensure_adtrans_importable() -> Path:
    adtrans_root = str(ADTRANS_ROOT.resolve())
    if adtrans_root not in sys.path:
        sys.path.insert(0, adtrans_root)
    return ADTRANS_ROOT


def _load(name: str) -> Any:
    ensure_adtrans_importable()
    return importlib.import_module(name)


ensure_adtrans_importable()

_dataset = _load("data.dataset")
_structure_placement = _load("data.structure_placement")
_utils_config = _load("utils.config")
_utils_metrics = _load("utils.metrics")
_plot_predictions = _load("utils.plot_predictions")
_visualization_config = _load("utils.visualization_config")
_preprocessing = _load("data.preprocessing")

MaterialsDataset = _dataset.MaterialsDataset
prepare_descriptor_datasets = _preprocessing.prepare_descriptor_datasets
uses_structure_input = _structure_placement.uses_structure_input
encoder_structure_offset = _structure_placement.encoder_structure_offset
structure_per_token_dim = _structure_placement.structure_per_token_dim
N_ELEMENTS = _preprocessing.N_ELEMENTS
ADTRANS_DATASET_CONFIG = _utils_config.DATASET_CONFIG
VISUALIZATION_CONFIG = _visualization_config.VISUALIZATION_CONFIG
calculate_metrics = _utils_metrics.calculate_metrics
visualize_predictions = _plot_predictions.visualize_predictions
plot_single_parity = _plot_predictions.plot_single_parity


def apply_baseline_dataset_overrides() -> None:
    """Sync baseline DATASET_CONFIG into AdTrans preprocessing config."""
    from config import DATASET_CONFIG as baseline_dataset

    name = baseline_dataset.get("structure_descriptor_name")
    if name is not None:
        ADTRANS_DATASET_CONFIG["structure_descriptor_name"] = name
        _utils_config.DATASET_CONFIG["structure_descriptor_name"] = name


__all__ = [
    "ADTRANS_ROOT",
    "MaterialsDataset",
    "N_ELEMENTS",
    "prepare_descriptor_datasets",
    "uses_structure_input",
    "encoder_structure_offset",
    "structure_per_token_dim",
    "ADTRANS_DATASET_CONFIG",
    "VISUALIZATION_CONFIG",
    "calculate_metrics",
    "visualize_predictions",
    "plot_single_parity",
    "ensure_adtrans_importable",
    "apply_baseline_dataset_overrides",
]
