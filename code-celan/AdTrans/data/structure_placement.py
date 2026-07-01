"""CHGNet structure-token placement helpers (encoder / decoder / all / none)."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import torch

from utils.config import DATASET_CONFIG

VALID_STRUCTURE_PLACEMENTS = ('none', 'encoder', 'decoder', 'all')


def get_structure_placement():
    placement = DATASET_CONFIG.get('structure_placement', 'none')
    if placement not in VALID_STRUCTURE_PLACEMENTS:
        raise ValueError(
            f"Invalid structure_placement={placement!r}. "
            f"Expected one of {VALID_STRUCTURE_PLACEMENTS}."
        )
    return placement


def _resolve_placement(placement=None):
    return get_structure_placement() if placement is None else placement


def structure_dual_token_split():
    return bool(DATASET_CONFIG.get('structure_dual_token_split', False))


def structure_input_dim():
    return int(DATASET_CONFIG.get('structure_descriptor_dim', 64))


def structure_per_token_dim():
    dim = structure_input_dim()
    if structure_dual_token_split():
        if dim % 2 != 0:
            raise ValueError(
                f"structure_dual_token_split requires an even structure_descriptor_dim, got {dim}."
            )
        return dim // 2
    return dim


def uses_structure_input(placement=None):
    return _resolve_placement(placement) != 'none'


def structure_in_encoder(placement=None):
    return _resolve_placement(placement) in ('encoder', 'all')


def structure_in_decoder(placement=None):
    return _resolve_placement(placement) in ('decoder', 'all')


def n_structure_tokens(placement=None):
    if not uses_structure_input(placement):
        return 0
    return 2 if structure_dual_token_split() else 1


def structure_discharge_token_index():
    return 1 if structure_dual_token_split() else 0


def structure_memory_from_encoder_hidden(memory: torch.Tensor, placement=None):
    if not structure_in_encoder(placement):
        return None
    n_tokens = n_structure_tokens(placement)
    if memory.size(1) < n_tokens:
        return None
    return memory[:, structure_discharge_token_index(), :]


def encoder_seq_len(n_elements, placement=None):
    offset = n_structure_tokens(placement) if structure_in_encoder(placement) else 0
    return n_elements + offset


def encoder_structure_offset(placement=None):
    return n_structure_tokens(placement) if structure_in_encoder(placement) else 0


def get_structure_token_name():
    return 'structure'


def structure_encoder_token_slice(placement=None):
    """Slice over structure token positions at the start of the encoder sequence."""
    if not structure_in_encoder(placement):
        return None
    return slice(0, n_structure_tokens(placement))


def sum_structure_encoder_attention(element_attn_per_sample, placement=None):
    """Sum attention mass over all structure tokens (XAI aggregation)."""
    token_slice = structure_encoder_token_slice(placement)
    if token_slice is None:
        return 0.0
    return float(element_attn_per_sample[token_slice].sum())


def get_encoder_attention_structure_labels(placement=None):
    """Labels for structure positions in encoder attention plots."""
    if not structure_in_encoder(placement):
        return []
    if structure_dual_token_split():
        return ['structure(charge)', 'structure(discharge)']
    return [get_structure_token_name()]


def get_encoder_attention_labels(formula, n_elements, battery_id, placement=None):
    """Encoder-axis labels: structure token(s) followed by element slots."""
    from utils.element_slot_order import encoder_slot_element_symbols

    elements = encoder_slot_element_symbols(
        formula, str(battery_id), max_slots=n_elements
    )
    labels = list(get_encoder_attention_structure_labels(placement))
    labels.extend(elements)
    return labels


def append_decoder_structure_token_names(token_names, placement=None):
    """Append structure token name(s) at decoder tail (XAI / naming)."""
    if not structure_in_decoder(placement):
        return token_names
    name = get_structure_token_name()
    return token_names + [name] * n_structure_tokens(placement)


def accumulate_structure_xai_importance(
    delta: float,
    element_importance_sum: Dict[str, float],
    property_importance_sum: Dict[str, float],
    *,
    token_name: Optional[str] = None,
) -> None:
    """Assign structure IG/ablation delta without double-counting when placement='all'."""
    name = token_name or get_structure_token_name()
    enc = structure_in_encoder()
    dec = structure_in_decoder()
    if enc and dec:
        half = delta * 0.5
        element_importance_sum[name] = element_importance_sum.get(name, 0.0) + half
        property_importance_sum[name] = property_importance_sum.get(name, 0.0) + half
    elif enc:
        element_importance_sum[name] = element_importance_sum.get(name, 0.0) + delta
    elif dec:
        property_importance_sum[name] = property_importance_sum.get(name, 0.0) + delta


def normalize_element_importance_excluding_structure(
    element_importance_df: pd.DataFrame,
    *,
    structure_token_name: Optional[str] = None,
) -> pd.DataFrame:
    """Renormalize element importances to sum 1; structure token excluded from denominator."""
    if element_importance_df.empty:
        return element_importance_df
    name = structure_token_name or get_structure_token_name()
    out = element_importance_df.copy()
    elem_mask = out['Element'] != name
    total = out.loc[elem_mask, 'Importance'].sum()
    if total > 0:
        out.loc[elem_mask, 'Importance'] = out.loc[elem_mask, 'Importance'] / total
    else:
        out.loc[elem_mask, 'Importance'] = 0.0
    return out.sort_values(by='Importance', ascending=False).reset_index(drop=True)
