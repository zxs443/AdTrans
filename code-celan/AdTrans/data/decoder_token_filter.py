"""Filter preprocessed decoder tensors to a subset of property tokens."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from data.dataset import MaterialsDataset
from utils.token_ranking import group_decoder_columns


def columns_for_keep_tokens(
    decoder_feature_columns: Sequence[str],
    keep_token_names: Sequence[str],
    decoder_token_dim: int,
) -> Tuple[List[str], List[int]]:
    """Return filtered column list and token indices (into full token list)."""
    groups = group_decoder_columns(decoder_feature_columns, decoder_token_dim)
    name_to_idx = {g['name']: i for i, g in enumerate(groups)}
    missing = [n for n in keep_token_names if n not in name_to_idx]
    if missing:
        raise ValueError(f"keep_token_names not found in decoder columns: {missing}")

    keep_indices = [name_to_idx[n] for n in keep_token_names]
    filtered_cols = []
    for idx in keep_indices:
        filtered_cols.extend(groups[idx]['columns'])
    return filtered_cols, keep_indices


def _subset_decoder_array(arr: np.ndarray, keep_indices: List[int]) -> np.ndarray:
    """arr: [N, K, D] -> [N, K', D]"""
    return arr[:, keep_indices, :]


def _subset_decoder_stats(stat: np.ndarray, keep_indices: List[int]) -> np.ndarray:
    """stat: [K, D] -> [K', D]"""
    return stat[keep_indices, :]


def _rebuild_dataset_with_subset(dataset: MaterialsDataset, keep_indices: List[int]) -> MaterialsDataset:
    dec = _subset_decoder_array(dataset.decoder_data.numpy(), keep_indices)
    kwargs = dict(
        encoder_data=dataset.encoder_data.numpy(),
        decoder_data=dec,
        targets=dataset.targets.numpy(),
        src_key_padding_mask=dataset.src_key_padding_mask.numpy(),
        formulas=dataset.formulas,
        battery_ids=dataset.battery_ids,
    )
    if dataset.structure_data is not None:
        kwargs['structure_data'] = dataset.structure_data.numpy()
    return MaterialsDataset(**kwargs)


def apply_keep_tokens_to_processed(
    processed_data: dict,
    decoder_feature_columns: Sequence[str],
    keep_token_names: Sequence[str],
) -> tuple[dict, List[str]]:
    """Subset splits and decoder stats to kept tokens (order = keep_token_names)."""
    decoder_token_dim = processed_data['decoder_token_dim']
    filtered_cols, keep_indices = columns_for_keep_tokens(
        decoder_feature_columns, keep_token_names, decoder_token_dim
    )

    out = dict(processed_data)
    for split in ('train', 'val', 'test'):
        key = f'{split}_dataset'
        out[key] = _rebuild_dataset_with_subset(processed_data[key], keep_indices)

    out['decoder_mean'] = _subset_decoder_stats(processed_data['decoder_mean'], keep_indices)
    out['decoder_std'] = _subset_decoder_stats(processed_data['decoder_std'], keep_indices)
    out['decoder_tokens'] = len(keep_indices)

    for k in ('train_loader', 'val_loader', 'test_loader'):
        out.pop(k, None)

    return out, filtered_cols
