"""Summarize metrics across decoder-token pruning rounds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd


def append_round_summary(
    summary_path: Path | str,
    round_label: str,
    n_tokens: int,
    metrics: dict,
    keep_tokens: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    summary_path = Path(summary_path)
    row = {
        'round': round_label,
        'n_property_tokens': n_tokens,
        'save_dir': save_dir or '',
        'keep_tokens': ','.join(keep_tokens) if keep_tokens else '',
    }
    for split in ('Train', 'Validation', 'Test'):
        if split in metrics:
            row[f'{split}_MAE'] = metrics[split].get('MAE')
            row[f'{split}_RMSE'] = metrics[split].get('RMSE')
            row[f'{split}_R2'] = metrics[split].get('R2')

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.is_file():
        df = pd.read_csv(summary_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(summary_path, index=False)
    return df


def load_best_hyperparameters(path: Path | str) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def hpo_center_from_best_params(best_params: dict) -> dict:
    """Extract Optuna center fields from a saved best_hyperparameters.json."""
    center = {
        'learning_rate': best_params.get('learning_rate'),
        'dropout_rate': best_params.get('dropout_rate'),
        'batch_size': best_params.get('batch_size'),
        'weight_decay': best_params.get('weight_decay'),
        'output_dims': best_params.get('output_dims'),
    }
    return center
