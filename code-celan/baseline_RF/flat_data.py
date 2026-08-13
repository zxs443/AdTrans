"""Flatten encoder/decoder/structure blocks into numpy arrays for sklearn models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from config import DATASET_CONFIG, INPUT_CONFIG, validate_dataset_config, validate_input_config
from adtrans_bridge import (
    MaterialsDataset,
    N_ELEMENTS,
    apply_baseline_dataset_overrides,
    prepare_descriptor_datasets,
    uses_structure_input,
)


class FlatFeatureDataset:
    """Concatenate selected standardized encoder/decoder/structure blocks into one vector."""

    def __init__(self, materials_dataset: MaterialsDataset):
        self.materials_dataset = materials_dataset
        self.structure_available = uses_structure_input()
        validate_input_config(self.structure_available)
        self.use_encoder = INPUT_CONFIG["use_encoder"]
        self.use_decoder = INPUT_CONFIG["use_decoder"]
        self.use_structure = INPUT_CONFIG["use_structure"] and self.structure_available
        self.input_dim = self._infer_input_dim()

    def _infer_input_dim(self) -> int:
        sample = self.materials_dataset[0]
        return self._flatten_sample(sample).numel()

    def _flatten_sample(self, sample: tuple) -> torch.Tensor:
        parts = []
        if self.use_encoder:
            parts.append(sample[1].reshape(-1))
        if self.use_decoder:
            parts.append(sample[2].reshape(-1))
        if self.use_structure:
            parts.append(sample[7].reshape(-1))
        if not parts:
            raise ValueError("No input features selected; check INPUT_CONFIG.")
        return torch.cat(parts, dim=0)

    def __len__(self) -> int:
        return len(self.materials_dataset)

    def __getitem__(self, idx: int):
        sample = self.materials_dataset[idx]
        features = self._flatten_sample(sample)
        target = sample[3]
        formula = sample[5]
        battery_id = sample[6]
        return features, target, formula, battery_id


def dataset_to_arrays(flat_ds: FlatFeatureDataset) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Materialize one split into (X, y, formulas, battery_ids); targets stay in z-score space."""
    features: List[torch.Tensor] = []
    targets: List[float] = []
    formulas: List[str] = []
    battery_ids: List[str] = []
    for idx in range(len(flat_ds)):
        feat, target, formula, battery_id = flat_ds[idx]
        features.append(feat)
        targets.append(float(target))
        formulas.append(formula)
        battery_ids.append(battery_id)
    X = torch.stack(features, dim=0).numpy().astype(np.float64)
    y = np.asarray(targets, dtype=np.float64)
    return X, y, formulas, battery_ids


def create_flat_arrays(processed_data) -> Dict[str, dict]:
    """Return {'train'|'val'|'test': {'X','y','formulas','battery_ids'}} plus the input dim."""
    splits: Dict[str, dict] = {}
    input_dim = None
    for split_name, dataset_key in (
        ("train", "train_dataset"),
        ("val", "val_dataset"),
        ("test", "test_dataset"),
    ):
        flat_ds = FlatFeatureDataset(processed_data[dataset_key])
        if input_dim is None:
            input_dim = flat_ds.input_dim
        X, y, formulas, battery_ids = dataset_to_arrays(flat_ds)
        splits[split_name] = {
            "X": X,
            "y": y,
            "formulas": formulas,
            "battery_ids": battery_ids,
        }
    splits["input_dim"] = input_dim
    return splits


def load_processed_data(base_path: Path, seed: int):
    validate_dataset_config(Path(base_path))
    apply_baseline_dataset_overrides()
    encoder_desc, decoder_desc = DATASET_CONFIG["descriptor_pair"]
    processed_data, _, split_mode = prepare_descriptor_datasets(
        str(base_path),
        encoder_desc,
        decoder_desc,
        N_ELEMENTS,
        seed=seed,
    )
    return processed_data, split_mode
