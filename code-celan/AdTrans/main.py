"""Train and evaluate the voltage-prediction Transformer: single run or token-pruning pipeline."""

import os
import sys

from utils.stdio_utils import configure_realtime_stdio

configure_realtime_stdio()

import json
import torch
import numpy as np
from data.structure_placement import structure_per_token_dim
from models.transformer import VoltagePredictor
from data.preprocessing import (
    prepare_descriptor_datasets,
    create_data_loaders,
    get_decoder_token_names,
    N_ELEMENTS,
)
from training.trainer import ModelTrainer
from training.evaluator import ModelEvaluator
from training.optimizer import create_optimizer, optimize_hyperparameters
from training.pruning_pipeline import run_pruning_pipeline
from utils.config import TRAINING_CONFIG, MODEL_CONFIG, DATASET_CONFIG, OPTIMIZER_CONFIG
from utils.pruning_config import PRUNING_STAGE_CONFIG
from utils.analysis import summarize_deviant_samples


def set_seed(seed):
    """Fix random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_and_evaluate_model(n_elements, encoder_descriptor_type, decoder_descriptor_type, base_path=".", device=None, seed=None):
    """Load descriptors, run HPO, train, evaluate, and save artifacts (single-stage baseline)."""
    if device is None:
        device = torch.device(TRAINING_CONFIG['device'])

    print(f"\n[1/4] Preprocess: encoder={encoder_descriptor_type}, decoder={decoder_descriptor_type}, seed={seed}")

    combined_descriptor_name = f"{encoder_descriptor_type}_{decoder_descriptor_type}"
    save_dir = f"results/{combined_descriptor_name}"
    hpo_checkpoint_path = os.path.join(save_dir, "hpo_best_trial_checkpoint.pt")

    processed_data, decoder_feature_columns, split_mode = prepare_descriptor_datasets(
        base_path,
        encoder_descriptor_type,
        decoder_descriptor_type,
        n_elements,
        seed=seed,
    )
    print("[2/4] Hyperparameter optimization (Optuna)...")
    best_params, best_value, saved_hpo_checkpoint = optimize_hyperparameters(
        processed_data, n_elements, device,
        n_trials=OPTIMIZER_CONFIG['n_trials'], seed=seed,
        decoder_feature_columns=decoder_feature_columns,
        checkpoint_path=hpo_checkpoint_path,
    )

    final_data_loaders = create_data_loaders(processed_data, best_params['batch_size'], seed)

    final_processed_data = {
        **processed_data,
        **final_data_loaders
    }

    full_decoder_token_names = get_decoder_token_names(decoder_feature_columns, processed_data['decoder_token_dim'])

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
        decoder_token_names=full_decoder_token_names,
        structure_placement=DATASET_CONFIG.get('structure_placement', 'none'),
        structure_descriptor_dim=structure_per_token_dim(),
    ).to(device)

    optimizer = create_optimizer(
        model,
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    criterion = torch.nn.MSELoss()
    trainer = ModelTrainer(model, optimizer, criterion, device)

    os.makedirs(save_dir, exist_ok=True)

    if saved_hpo_checkpoint and os.path.isfile(saved_hpo_checkpoint):
        checkpoint = torch.load(saved_hpo_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        trained_model = model
        train_losses_history = []
        val_losses_history = []
        learning_rates_history = []
        print(
            f"[3/4] Loaded HPO best trial checkpoint "
            f"(trial {checkpoint.get('trial_number')}, "
            f"val_v_mse={checkpoint.get('best_val_loss', float('nan')):.4f}), "
            f"skipping final retraining.",
            flush=True,
        )
    else:
        print("[3/4] Training with early stopping...")
        trained_model, train_losses_history, val_losses_history, learning_rates_history = trainer.train_with_early_stopping(
            final_processed_data['train_loader'],
            final_processed_data['val_loader']
        )

    try:
        best_params_to_save = {**best_params, 'best_value': float(best_value) if best_value is not None else None, 'seed': seed}
        with open(os.path.join(save_dir, "best_hyperparameters.json"), "w", encoding="utf-8") as f:
            json.dump(best_params_to_save, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] Could not save best_hyperparameters.json: {e}")

    model_config_to_save = trained_model.get_config()
    trained_model.save_model(os.path.join(save_dir, "model.pth"), config=model_config_to_save)
    model_config = trained_model.get_config()

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
        'seed': seed
    }
    np.save(os.path.join(save_dir, "data_stats.npy"), stats)

    print("[4/4] Evaluation, plots, and XAI...")
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
        full_decoder_token_names,
        final_processed_data['encoder_mean'],
        final_processed_data['decoder_mean'],
        final_processed_data['targets_mean'],
        final_processed_data['targets_std'],
        train_losses_history=train_losses_history,
        val_losses_history=val_losses_history,
        learning_rates_history=learning_rates_history,
        structure_mean=final_processed_data.get('structure_mean'),
    )

    predictions_file_path = os.path.join(save_dir, f'predictions_{combined_descriptor_name}_{timestamp}.csv')
    data_stats_file_path = os.path.join(save_dir, 'data_stats.npy')
    output_summary_file = os.path.join(save_dir, f'deviant_samples_summary_{combined_descriptor_name}_{timestamp}.csv')

    summarize_deviant_samples(predictions_file_path, data_stats_file_path, output_summary_file)

    return metrics


def main():
    seed = int(PRUNING_STAGE_CONFIG.get('seed', DATASET_CONFIG['seed']))
    set_seed(seed)

    device_str = PRUNING_STAGE_CONFIG.get('device') or TRAINING_CONFIG['device']
    device = torch.device(device_str)

    try:
        if PRUNING_STAGE_CONFIG.get('enabled', False):
            run_pruning_pipeline(device, seed)
            return

        print(f"AdTrans_struct | device={device} | seed={seed} | descriptor_pair={DATASET_CONFIG['descriptor_pair']}")
        n_elements = N_ELEMENTS
        encoder_desc, decoder_desc = DATASET_CONFIG['descriptor_pair']

        metrics = train_and_evaluate_model(n_elements, encoder_desc, decoder_desc, device=device, seed=seed)
        print("\nDone. Metrics (voltage, unscaled):")
        for split in ("Train", "Validation", "Test"):
            m = metrics[split]
            print(f"  {split:12s} MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R²={m['R2']:.4f}")
    except FileNotFoundError as e:
        print(f"[ERR] Missing file: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERR] Data/config mismatch: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERR] {e}")
        raise


if __name__ == "__main__":
    main()
