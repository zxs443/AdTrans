"""Flatten encoder/decoder/structure blocks into one vector."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from config import DATASET_CONFIG, INPUT_CONFIG, validate_dataset_config, validate_input_config
from adtrans_bridge import (
    MaterialsDataset,
    N_ELEMENTS,
    apply_baseline_dataset_overrides,
    prepare_descriptor_datasets,
    uses_structure_input,
)


class FlatFeatureDataset(Dataset):
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


def collate_flat_batch(batch):
    features, targets, formulas, battery_ids = zip(*batch)
    return (
        torch.stack(features, dim=0),
        torch.stack(targets, dim=0),
        formulas,
        battery_ids,
    )


def create_flat_datasets(processed_data) -> Tuple[FlatFeatureDataset, FlatFeatureDataset, FlatFeatureDataset, int]:
    train_ds = FlatFeatureDataset(processed_data["train_dataset"])
    val_ds = FlatFeatureDataset(processed_data["val_dataset"])
    test_ds = FlatFeatureDataset(processed_data["test_dataset"])
    input_dim = train_ds.input_dim
    return train_ds, val_ds, test_ds, input_dim


def infer_input_dim(processed_data) -> int:
    return FlatFeatureDataset(processed_data["train_dataset"]).input_dim


def create_flat_loaders(
    processed_data,
    batch_size: int,
    seed: int,
) -> dict:
    train_ds, val_ds, test_ds, input_dim = create_flat_datasets(processed_data)
    generator = torch.Generator().manual_seed(seed)
    return {
        "train_loader": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate_flat_batch,
        ),
        "val_loader": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_flat_batch,
        ),
        "test_loader": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_flat_batch,
        ),
        "input_dim": input_dim,
    }


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


def unpack_flat_batch(batch, device=None):
    features, targets, formulas, battery_ids = batch
    if device is not None:
        features = features.to(device)
        targets = targets.to(device)
    return features, targets, formulas, battery_ids
