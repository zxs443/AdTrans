"""Stage-2 decoder token pruning pipeline (invoked from main.py)."""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from data.decoder_token_filter import apply_keep_tokens_to_processed
from data.preprocessing import (
    prepare_descriptor_datasets,
    create_data_loaders,
    get_decoder_token_names,
    N_ELEMENTS,
)
from data.structure_placement import structure_per_token_dim
from models.transformer import VoltagePredictor
from training.trainer import ModelTrainer
from training.evaluator import ModelEvaluator
from training.optimizer import create_optimizer, optimize_hyperparameters
from utils.analysis import summarize_deviant_samples
from utils.config import TRAINING_CONFIG, MODEL_CONFIG, DATASET_CONFIG, OPTIMIZER_CONFIG
from utils.pruning_config import PRUNING_STAGE_CONFIG
from utils.pruning_report import (
    append_round_summary,
    hpo_center_from_best_params,
    load_best_hyperparameters,
)
from utils.token_ranking import (
    all_property_token_names,
    compute_keep_tokens_for_round,
)


def _round_label(stage_idx: int, n_tokens: int, n_prune: int | None = None) -> str:
    if stage_idx == 0:
        return f'stage_00_k{n_tokens}'
    if n_prune is not None:
        return f'round_{stage_idx:02d}_k{n_tokens}_p{n_prune}'
    return f'round_{stage_idx:02d}_k{n_tokens}'


def _build_pruning_schedule(n_full_tokens: int) -> list[dict]:
    """Stage list from prune_per_round drop counts (not absolute keep counts)."""
    prune_per_round = list(PRUNING_STAGE_CONFIG.get('prune_per_round') or [])
    min_tokens = max(1, int(PRUNING_STAGE_CONFIG.get('min_tokens', 1)))

    schedule = [{'stage_idx': 0, 'n_tokens': n_full_tokens, 'n_prune': 0}]
    current_n = n_full_tokens

    for round_idx, n_prune in enumerate(prune_per_round, start=1):
        n_prune = int(n_prune)
        if n_prune <= 0:
            print(f'[WARN] prune_per_round[{round_idx - 1}]={n_prune} <= 0; skipping.')
            continue

        next_n = max(min_tokens, current_n - n_prune)
        if next_n >= current_n:
            print(
                f'[WARN] Cannot prune {n_prune} from {current_n} tokens '
                f'(min_tokens={min_tokens}); stopping schedule expansion.'
            )
            break

        actual_prune = current_n - next_n
        schedule.append(
            {
                'stage_idx': len(schedule),
                'n_tokens': next_n,
                'n_prune': actual_prune,
            }
        )
        current_n = next_n
        if current_n <= min_tokens:
            break

    return schedule


def _resolve_pruning_trials(stage_idx: int) -> int:
    """Stage 0 -> n_trials; pruning rounds -> n_trials_pruning_round (from config.py)."""
    if stage_idx == 0:
        return int(OPTIMIZER_CONFIG['n_trials'])
    cfg = OPTIMIZER_CONFIG.get('n_trials_pruning_round', OPTIMIZER_CONFIG['n_trials'])
    prune_idx = stage_idx - 1
    if isinstance(cfg, (list, tuple)):
        if prune_idx < len(cfg):
            return int(cfg[prune_idx])
        return int(cfg[-1])
    return int(cfg)


def _xai_ranking_splits(is_final_stage: bool) -> str:
    """Ranking CSVs for pruning should not leak val/test; default is train-only."""
    _ = is_final_stage
    splits = str(PRUNING_STAGE_CONFIG.get('xai_ranking_splits', 'train')).lower()
    if splits not in ('train', 'all'):
        raise ValueError(f"xai_ranking_splits must be 'train' or 'all', got {splits!r}.")
    return splits


def _train_schedule_round(
    *,
    round_label: str,
    target_n_tokens: int,
    keep_token_names: list[str],
    save_dir: str,
    hpo_center: dict | None,
    base_path: str,
    n_elements: int,
    encoder_desc: str,
    decoder_desc: str,
    device: torch.device,
    seed: int,
    n_trials: int,
    is_final_stage: bool,
) -> dict:
    combined_descriptor_name = f'{encoder_desc}_{decoder_desc}'
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n{"=" * 72}')
    print(f'{round_label} | target_tokens={target_n_tokens} | save_dir={save_dir}')
    print(f'keep ({len(keep_token_names)}): {keep_token_names}')
    print(f'XAI ranking splits: {_xai_ranking_splits(is_final_stage)}')
    if not round_label.startswith('stage_'):
        print(f'Pruning ranking method: {PRUNING_STAGE_CONFIG.get("ranking_method")}')
    print(f'{"=" * 72}\n')

    processed_data, decoder_feature_columns, _split_mode = prepare_descriptor_datasets(
        base_path,
        encoder_desc,
        decoder_desc,
        n_elements,
        seed=seed,
    )
    full_token_order = all_property_token_names(
        decoder_feature_columns, processed_data['decoder_token_dim']
    )
    if len(keep_token_names) != len(full_token_order):
        processed_data, decoder_feature_columns = apply_keep_tokens_to_processed(
            processed_data,
            decoder_feature_columns,
            keep_token_names,
        )

    hpo_checkpoint_path = os.path.join(save_dir, 'hpo_best_trial_checkpoint.pt')
    center_cfg = PRUNING_STAGE_CONFIG.get('hpo_center', {})
    optuna_viz = PRUNING_STAGE_CONFIG.get('optuna_visualization', False)

    print(f'[HPO] {n_trials} trials (center={"yes" if hpo_center else "no"})')
    best_params, best_value, saved_hpo_checkpoint = optimize_hyperparameters(
        processed_data,
        n_elements,
        device,
        n_trials=n_trials,
        seed=seed,
        decoder_feature_columns=decoder_feature_columns,
        checkpoint_path=hpo_checkpoint_path,
        hpo_center=hpo_center,
        hpo_center_config=center_cfg,
        optuna_visualization=optuna_viz,
    )

    final_data_loaders = create_data_loaders(processed_data, best_params['batch_size'], seed)
    final_processed_data = {**processed_data, **final_data_loaders}

    decoder_token_names = get_decoder_token_names(
        decoder_feature_columns, processed_data['decoder_token_dim']
    )

    model = VoltagePredictor(
        encoder_input_dim=DATASET_CONFIG['encoder_token_dim'],
        d_model=MODEL_CONFIG['d_model'],
        encoder_nhead=MODEL_CONFIG['encoder_nhead'],
        decoder_nhead=MODEL_CONFIG['decoder_nhead'],
        pooling_nhead=MODEL_CONFIG['pooling_nhead'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout_rate=best_params['dropout_rate'],
        decoder_input_dim=DATASET_CONFIG['decoder_token_dim'],
        output_hidden_layers=MODEL_CONFIG['output_hidden_layers'],
        output_dims=best_params['output_dims'],
        decoder_token_names=decoder_token_names,
        structure_placement=DATASET_CONFIG.get('structure_placement', 'none'),
        structure_descriptor_dim=structure_per_token_dim(),
    ).to(device)

    optimizer = create_optimizer(
        model,
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'],
    )
    criterion = torch.nn.MSELoss()
    trainer = ModelTrainer(model, optimizer, criterion, device)

    if saved_hpo_checkpoint and os.path.isfile(saved_hpo_checkpoint):
        checkpoint = torch.load(saved_hpo_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        trained_model = model
        train_losses_history = []
        val_losses_history = []
        learning_rates_history = []
        print(
            f'[Train] Loaded HPO checkpoint trial {checkpoint.get("trial_number")}, '
            f'val_v_mse={checkpoint.get("best_val_loss", float("nan")):.4f}',
            flush=True,
        )
    else:
        print('[Train] Early stopping...')
        trained_model, train_losses_history, val_losses_history, learning_rates_history = (
            trainer.train_with_early_stopping(
                final_processed_data['train_loader'],
                final_processed_data['val_loader'],
            )
        )

    best_params_to_save = {
        **best_params,
        'best_value': float(best_value) if best_value is not None else None,
        'seed': seed,
        'n_property_tokens': target_n_tokens,
        'keep_tokens': keep_token_names,
        'xai_ranking_splits': _xai_ranking_splits(is_final_stage),
        'ranking_method': PRUNING_STAGE_CONFIG.get('ranking_method'),
    }
    with open(os.path.join(save_dir, 'best_hyperparameters.json'), 'w', encoding='utf-8') as f:
        json.dump(best_params_to_save, f, indent=2, ensure_ascii=False)

    model_config = trained_model.get_config()
    trained_model.save_model(os.path.join(save_dir, 'model.pth'), config=model_config)

    stats = {
        'encoder_mean': final_processed_data['encoder_mean'],
        'encoder_std': final_processed_data['encoder_std'],
        'decoder_mean': final_processed_data['decoder_mean'],
        'decoder_std': final_processed_data['decoder_std'],
        'targets_mean': final_processed_data['targets_mean'],
        'targets_std': final_processed_data['targets_std'],
        'formulas': final_processed_data['formulas'],
        'battery_ids': final_processed_data['battery_ids'],
        'features_per_element': final_processed_data['features_per_element'],
        'decoder_tokens': final_processed_data['decoder_tokens'],
        'decoder_token_dim': final_processed_data['decoder_token_dim'],
        'n_elements': N_ELEMENTS,
        'structure_dim': final_processed_data.get('structure_dim', 0),
        'structure_mean': final_processed_data.get('structure_mean'),
        'structure_std': final_processed_data.get('structure_std'),
        'structure_placement': final_processed_data.get('structure_placement'),
        'encoder_seq_len': final_processed_data.get('encoder_seq_len'),
        'train_indices': final_processed_data['train_indices'],
        'val_indices': final_processed_data['val_indices'],
        'test_indices': final_processed_data['test_indices'],
        'model_config': model_config,
        'seed': seed,
        'keep_tokens': keep_token_names,
    }
    np.save(os.path.join(save_dir, 'data_stats.npy'), stats)

    print('[Eval + XAI]...')
    evaluator = ModelEvaluator(trained_model, criterion, device)
    metrics, timestamp = evaluator.evaluate_and_log_results(
        final_processed_data['train_loader'],
        final_processed_data['val_loader'],
        final_processed_data['test_loader'],
        final_processed_data['formulas'][final_processed_data['train_indices']],
        final_processed_data['formulas'][final_processed_data['val_indices']],
        final_processed_data['formulas'][final_processed_data['test_indices']],
        final_processed_data['formulas'],
        n_elements,
        save_dir,
        combined_descriptor_name,
        decoder_token_names,
        final_processed_data['encoder_mean'],
        final_processed_data['decoder_mean'],
        final_processed_data['targets_mean'],
        final_processed_data['targets_std'],
        train_losses_history=train_losses_history,
        val_losses_history=val_losses_history,
        learning_rates_history=learning_rates_history,
        structure_mean=final_processed_data.get('structure_mean'),
        xai_ranking_splits=_xai_ranking_splits(is_final_stage),
    )

    predictions_file = os.path.join(
        save_dir, f'predictions_{combined_descriptor_name}_{timestamp}.csv'
    )
    summarize_deviant_samples(
        predictions_file,
        os.path.join(save_dir, 'data_stats.npy'),
        os.path.join(save_dir, f'deviant_samples_summary_{combined_descriptor_name}_{timestamp}.csv'),
    )

    return metrics


def run_pruning_pipeline(device: torch.device, seed: int) -> None:
    """Run stage0 + progressive token-pruning rounds."""
    encoder_desc, decoder_desc = DATASET_CONFIG['descriptor_pair']
    combined_descriptor_name = f'{encoder_desc}_{decoder_desc}'
    output_root = os.path.abspath(os.path.join('results', 'pruning', combined_descriptor_name))
    os.makedirs(output_root, exist_ok=True)
    n_elements = N_ELEMENTS
    base_path = '.'

    processed_full, full_decoder_columns, _ = prepare_descriptor_datasets(
        base_path, encoder_desc, decoder_desc, n_elements, seed=seed
    )
    full_token_order = all_property_token_names(
        full_decoder_columns, processed_full['decoder_token_dim']
    )
    schedule = _build_pruning_schedule(len(full_token_order))
    if len(schedule) < 1:
        raise ValueError('pruning schedule is empty.')

    summary_path = os.path.join(output_root, 'pruning_summary.csv')
    prev_results_dir: str | None = None
    hpo_center: dict | None = None
    keep_tokens = list(full_token_order)

    schedule_desc = [
        f"k{s['n_tokens']}" + (f"(p{s['n_prune']})" if s['n_prune'] else '')
        for s in schedule
    ]
    print(
        f'Pruning pipeline | device={device} | seed={seed} | '
        f'full_tokens={len(full_token_order)} | ranking_method={PRUNING_STAGE_CONFIG.get("ranking_method")} | '
        f'prune_per_round={PRUNING_STAGE_CONFIG.get("prune_per_round")} | '
        f'schedule={schedule_desc} | output={output_root}'
    )

    for stage in schedule:
        stage_idx = stage['stage_idx']
        target_n_tokens = stage['n_tokens']
        n_prune = stage['n_prune']
        is_final = stage is schedule[-1]
        label = _round_label(stage_idx, target_n_tokens, n_prune if n_prune else None)
        save_dir = os.path.join(output_root, label)

        if stage_idx == 0:
            keep_tokens = list(full_token_order)
            hpo_center = None
        else:
            if prev_results_dir is None:
                raise RuntimeError('prev_results_dir missing for pruning round.')
            ranking_dir = os.path.join(save_dir, 'ranking')
            keep_tokens, _ranked = compute_keep_tokens_for_round(
                prev_results_dir=os.path.abspath(prev_results_dir),
                descriptor_name=combined_descriptor_name,
                full_token_order=full_token_order,
                target_n_tokens=target_n_tokens,
                ranking_out_dir=ranking_dir,
                token_pool=keep_tokens,
            )
            if len(keep_tokens) != target_n_tokens:
                raise ValueError(
                    f'{label}: expected {target_n_tokens} tokens, got {len(keep_tokens)}.'
                )

        n_trials = _resolve_pruning_trials(stage_idx)
        metrics = _train_schedule_round(
            round_label=label,
            target_n_tokens=target_n_tokens,
            keep_token_names=keep_tokens,
            save_dir=save_dir,
            hpo_center=hpo_center,
            base_path=base_path,
            n_elements=n_elements,
            encoder_desc=encoder_desc,
            decoder_desc=decoder_desc,
            device=device,
            seed=seed,
            n_trials=n_trials,
            is_final_stage=is_final,
        )

        append_round_summary(
            summary_path,
            label,
            target_n_tokens,
            metrics,
            keep_tokens=keep_tokens,
            save_dir=save_dir,
        )

        prev_results_dir = save_dir
        hpo_center = hpo_center_from_best_params(
            load_best_hyperparameters(os.path.join(save_dir, 'best_hyperparameters.json'))
        )

        prune_msg = f' | pruned={n_prune}' if n_prune else ''
        print(
            f'\n{label} done | tokens={target_n_tokens}{prune_msg} | '
            f'Test MAE={metrics["Test"]["MAE"]:.4f}'
        )

    print(f'\nPruning pipeline finished. Summary: {summary_path}')
