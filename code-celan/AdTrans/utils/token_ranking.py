"""Select decoder tokens to keep/drop from a single XAI ranking CSV."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from utils.pruning_config import PRUNING_STAGE_CONFIG

# Magpie-style statistic suffixes (longest first to avoid partial matches).
_DECODER_STAT_SUFFIXES = (
    'weighted_median',
    'max_min_ratio',
    'max_mean_ratio',
    'sorted_ratio',
    'mean_diff',
    'max_diff',
    'max_ratio',
    'entropy',
    'range',
    'mean',
    'max',
    'min',
    'std',
    'q25',
    'q75',
    'iqr',
)

XAI_RANKING_METHODS = {
    'ig': 'ig_decoder_token_importance_{descriptor}.csv',
    'fa': 'fa_decoder_token_importance_{descriptor}.csv',
    'self_attention': 'decoder_token_self_attention_importance_{descriptor}.csv',
}


def token_name_from_decoder_column(column: str) -> str:

    for suffix in _DECODER_STAT_SUFFIXES:
        tail = f'_{suffix}'
        if column.endswith(tail) and len(column) > len(tail):
            return column[: -len(tail)]
    match = re.match(r'(.+)_\d+$', column)
    if match:
        return match.group(1)
    return column


def _token_name_from_column(column: str) -> str:
    return token_name_from_decoder_column(column)


# Standard Magpie property tokens (property_base/magpie CSV layout).
MAGPIE_PROPERTY_TOKEN_NAMES = frozenset({
    'Number', 'MendeleevNumber', 'AtomicWeight', 'MeltingT', 'Column', 'Row',
    'CovalentRadius', 'Electronegativity', 'NsValence', 'NpValence', 'NdValence',
    'NfValence', 'NValence', 'NsUnfilled', 'NpUnfilled', 'NdUnfilled', 'NfUnfilled',
    'NUnfilled', 'GSvolume_pa', 'GSbandgap', 'GSmagmom', 'SpaceGroupNumber',
})


def extra_decoder_features_for_caption(
    property_token_names: Sequence[str],
    decoder_descriptor_type: str,
) -> List[str]:
    """Decoder tokens for parity-plot Features line (skip standard Magpie when decoder is magpie)."""
    names = list(property_token_names)
    if decoder_descriptor_type == 'magpie':
        return [n for n in names if n not in MAGPIE_PROPERTY_TOKEN_NAMES]
    return names


def group_decoder_columns(decoder_feature_columns: Sequence[str], decoder_token_dim: int) -> List[dict]:
    """Map each property token to its CSV column names (fixed order)."""
    cols = list(decoder_feature_columns)
    if len(cols) % decoder_token_dim != 0:
        raise ValueError(
            f"Decoder columns {len(cols)} not divisible by decoder_token_dim={decoder_token_dim}."
        )
    groups = []
    for start in range(0, len(cols), decoder_token_dim):
        chunk = cols[start : start + decoder_token_dim]
        groups.append({'name': _token_name_from_column(chunk[0]), 'columns': chunk})
    return groups


def all_property_token_names(decoder_feature_columns: Sequence[str], decoder_token_dim: int) -> List[str]:
    return [g['name'] for g in group_decoder_columns(decoder_feature_columns, decoder_token_dim)]


def _read_ranking_csv(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"XAI ranking CSV not found: {path}")
    df = pd.read_csv(path)
    if 'Property' not in df.columns or 'Importance' not in df.columns:
        raise ValueError(f"Invalid ranking CSV (need Property, Importance): {path}")
    return df


def get_ranking_method() -> str:
    method = str(PRUNING_STAGE_CONFIG.get('ranking_method', 'ig')).lower()
    if method not in XAI_RANKING_METHODS:
        valid = ', '.join(sorted(XAI_RANKING_METHODS))
        raise ValueError(f"ranking_method must be one of [{valid}], got {method!r}.")
    return method


def load_xai_ranking(
    results_dir: Path | str,
    descriptor_name: str,
    method: str | None = None,
) -> pd.DataFrame:
    """Load one XAI decoder-token ranking CSV from a results folder."""
    results_dir = Path(results_dir)
    method = method or get_ranking_method()
    pattern = XAI_RANKING_METHODS[method]
    fname = pattern.format(descriptor=descriptor_name)
    return _read_ranking_csv(results_dir / fname)


def _exclude_from_ranking() -> set[str]:
    return set(PRUNING_STAGE_CONFIG.get('exclude_from_ranking', ['structure']))


def _protected_tokens_in_pool(pool: Sequence[str]) -> set[str]:
    protected = set(PRUNING_STAGE_CONFIG.get('protected_tokens', []))
    return protected & set(pool)


def _filter_rankable_tokens(
    df: pd.DataFrame,
    allowed_tokens: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    exclude = _exclude_from_ranking()
    out = df[~df['Property'].isin(exclude)].copy()
    if allowed_tokens is not None:
        allowed = set(allowed_tokens)
        out = out[out['Property'].isin(allowed)]
    return out.reset_index(drop=True)


def prepare_token_ranking(ranking_df: pd.DataFrame, allowed_tokens: List[str]) -> pd.DataFrame:
    """Sort tokens by Importance (descending) within the rankable pool."""
    filtered = _filter_rankable_tokens(ranking_df, allowed_tokens=allowed_tokens)
    ranked = filtered.sort_values('Importance', ascending=False).reset_index(drop=True)
    ranked['rank'] = range(1, len(ranked) + 1)
    return ranked


def select_keep_tokens(
    ranking_df: pd.DataFrame,
    n_keep: int,
    full_token_order: List[str],
    protected_tokens: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Keep ``n_keep`` tokens: all protected (if any) plus top-ranked non-protected tokens.
    Output preserves original CSV column order.
    """
    if n_keep > len(full_token_order):
        raise ValueError(f"n_keep={n_keep} exceeds available tokens={len(full_token_order)}.")
    if n_keep < 1:
        raise ValueError("n_keep must be >= 1.")

    protected = set(protected_tokens or [])
    unknown = protected - set(full_token_order)
    if unknown:
        raise ValueError(f"protected_tokens not in token pool: {sorted(unknown)}")

    protected_in_order = [name for name in full_token_order if name in protected]
    n_protected = len(protected_in_order)
    if n_keep < n_protected:
        raise ValueError(
            f"n_keep={n_keep} is smaller than protected token count={n_protected}."
        )

    n_rankable_keep = n_keep - n_protected
    rankable_order = [name for name in full_token_order if name not in protected]
    if n_rankable_keep > 0:
        ranked_keep = set(
            ranking_df.sort_values('Importance', ascending=False)
            .head(n_rankable_keep)['Property']
        )
        rankable_kept = {name for name in rankable_order if name in ranked_keep}
    else:
        rankable_kept = set()

    return [name for name in full_token_order if name in protected or name in rankable_kept]


def compute_keep_tokens_for_round(
    prev_results_dir: Path | str,
    descriptor_name: str,
    full_token_order: List[str],
    target_n_tokens: int,
    ranking_out_dir: Optional[Path | str] = None,
    token_pool: Optional[Sequence[str]] = None,
) -> tuple[List[str], pd.DataFrame]:
    """Rank using previous round XAI CSV; return keep list + sorted ranking table."""
    prev_results_dir = Path(prev_results_dir)
    if ranking_out_dir is not None:
        ranking_out_dir = Path(ranking_out_dir)
    pool = list(token_pool) if token_pool is not None else list(full_token_order)
    if not pool:
        raise ValueError("token_pool is empty; nothing to rank for pruning.")

    protected_in_pool = _protected_tokens_in_pool(pool)
    rankable_pool = [t for t in pool if t not in protected_in_pool]
    if target_n_tokens < len(protected_in_pool):
        raise ValueError(
            f"target_n_tokens={target_n_tokens} < protected token count={len(protected_in_pool)}."
        )

    method = get_ranking_method()
    ranking_df = load_xai_ranking(prev_results_dir, descriptor_name, method)
    allowed = set(rankable_pool)
    allowed &= set(_filter_rankable_tokens(ranking_df)['Property'].tolist())
    if not allowed:
        allowed = set(rankable_pool)
    allowed_list = [t for t in rankable_pool if t in allowed]

    ranked = prepare_token_ranking(ranking_df, allowed_list)
    keep = select_keep_tokens(
        ranked,
        target_n_tokens,
        pool,
        protected_tokens=sorted(protected_in_pool),
    )

    if ranking_out_dir is not None:
        ranking_out_dir.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(ranking_out_dir / f'token_ranking_{method}.csv', index=False)
        payload = {
            'ranking_method': method,
            'target_n_tokens': target_n_tokens,
            'token_pool_size': len(pool),
            'protected_tokens': sorted(protected_in_pool),
            'n_pruned': len(pool) - len(keep),
            'keep_tokens': keep,
            'drop_tokens': [t for t in pool if t not in set(keep)],
            'prev_results_dir': str(prev_results_dir),
        }
        with open(ranking_out_dir / 'token_keep.json', 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    return keep, ranked
