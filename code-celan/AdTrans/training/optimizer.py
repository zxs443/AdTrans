import os 
import torch
import torch.nn as nn
import optuna
from optuna.trial import Trial
from data.structure_placement import structure_per_token_dim
from models.transformer import VoltagePredictor
from utils.config import OPTIMIZER_CONFIG, TRAINING_CONFIG, DATASET_CONFIG, MODEL_CONFIG
import optuna.visualization as optuna_vis 
from datetime import datetime 
from data.preprocessing import preprocess_data_once, create_data_loaders, get_decoder_token_names
from data.dataset import unpack_batch, forward_model, voltage_from_model_output
from training.loss_utils import (
    average_breakdown_dict,
    compute_training_loss,
    default_loss_accum,
    format_train_epoch_summary,
    format_val_epoch_summary,
)

_HPO_BEST_CHECKPOINT = {
    'value': float('inf'),
    'state_dict': None,
    'trial_number': None,
}

def create_optimizer(model, learning_rate, weight_decay):
    """Build AdamW optimizer."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

def create_scheduler(optimizer, config):
    """ReduceLROnPlateau from TRAINING_CONFIG."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['scheduler']['factor'],
        patience=config['scheduler']['patience'],
        min_lr=config['scheduler']['min_lr']
    )



def _batch_size_choices(hpo_center=None, hpo_center_config=None):
    batch_choices = list(OPTIMIZER_CONFIG['batch_sizes'])
    if hpo_center is None:
        return batch_choices
    cfg = hpo_center_config or {}
    center_batch = hpo_center.get('batch_size')
    if cfg.get('prefer_center_batch', True) and center_batch in batch_choices:
        return [center_batch] + [b for b in batch_choices if b != center_batch]
    return batch_choices


def _suggest_centered_float(
    trial: Trial,
    name: str,
    global_low: float,
    global_high: float,
    center: float,
    delta: float,
    log: bool = False,
):
    low = max(global_low, center - delta)
    high = min(global_high, center + delta)
    if low >= high:
        low, high = global_low, global_high
    if log:
        low = max(low, 1e-12)
        return trial.suggest_float(name, low, high, log=True)
    return trial.suggest_float(name, low, high)


def objective(
    trial: Trial,
    processed_data,
    n_elements,
    device,
    decoder_feature_columns=None,
    hpo_center=None,
    hpo_center_config=None,
):
    """Optuna objective: validation voltage MSE."""
    d_model = MODEL_CONFIG['d_model']
    encoder_nhead = MODEL_CONFIG['encoder_nhead']
    decoder_nhead = MODEL_CONFIG['decoder_nhead']
    pooling_nhead = MODEL_CONFIG['pooling_nhead']
    num_layers = MODEL_CONFIG['num_layers']
    output_hidden_layers = MODEL_CONFIG['output_hidden_layers']
    
    batch_size = trial.suggest_categorical(
        'batch_size',
        _batch_size_choices(hpo_center, hpo_center_config),
    )
    
    data_loaders = create_data_loaders(processed_data, batch_size, OPTIMIZER_CONFIG['seed'])
    train_loader = data_loaders['train_loader']
    val_loader = data_loaders['val_loader']
    encoder_input_dim = processed_data['features_per_element']
    decoder_input_dim = processed_data['decoder_token_dim']

    output_dims = []
    current_dim_for_output_layer = d_model 

    for i in range(output_hidden_layers):
        min_dim_ratio = OPTIMIZER_CONFIG['output_dim_ratio_range'][0]
        max_dim_ratio = OPTIMIZER_CONFIG['output_dim_ratio_range'][1]

        # Constrain output_dim_layer_i < previous width
        low = max(1, int(current_dim_for_output_layer * min_dim_ratio))
        high = min(current_dim_for_output_layer - 1, int(current_dim_for_output_layer * max_dim_ratio))
        
        if high < low:
            low = 1
            high = 1
            print(f"[WARN] Could not find a strictly decreasing dimension range for output layer {i}, forcing to 1.")
        
        next_dim = trial.suggest_int(f'output_dim_layer_{i}', low, high)
        output_dims.append(next_dim)
        current_dim_for_output_layer = next_dim 
    
    decoder_token_names = None
    if decoder_feature_columns is not None:
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
        dropout_rate=(
            _suggest_centered_float(
                trial,
                'dropout_rate',
                OPTIMIZER_CONFIG['dropout_range'][0],
                OPTIMIZER_CONFIG['dropout_range'][1],
                hpo_center['dropout_rate'],
                (hpo_center_config or {}).get('dropout_delta', 0.05),
            )
            if hpo_center and 'dropout_rate' in hpo_center
            else trial.suggest_float('dropout_rate', *OPTIMIZER_CONFIG['dropout_range'])
        ),
        decoder_input_dim=DATASET_CONFIG['decoder_token_dim'], 
        output_hidden_layers=MODEL_CONFIG['output_hidden_layers'],
        output_dims=output_dims,
        decoder_token_names=decoder_token_names,
        n_property_tokens=processed_data.get('decoder_tokens'),
        structure_placement=DATASET_CONFIG.get('structure_placement', 'none'),
        structure_descriptor_dim=structure_per_token_dim(),
    ).to(device)
    
    n_params = model.count_parameters()
    print(f"Trainable parameters: {n_params:,}")
    
    if hpo_center and 'learning_rate' in hpo_center:
        lr_ratio = (hpo_center_config or {}).get('lr_ratio', 3.0)
        center_lr = float(hpo_center['learning_rate'])
        lr_low = max(OPTIMIZER_CONFIG['learning_rate_range'][0], center_lr / lr_ratio)
        lr_high = min(OPTIMIZER_CONFIG['learning_rate_range'][1], center_lr * lr_ratio)
        if lr_low >= lr_high:
            lr_low, lr_high = OPTIMIZER_CONFIG['learning_rate_range']
        learning_rate = trial.suggest_float('learning_rate', lr_low, lr_high, log=True)
    else:
        learning_rate = trial.suggest_float(
            'learning_rate', *OPTIMIZER_CONFIG['learning_rate_range'], log=True
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=trial.suggest_float('weight_decay', *OPTIMIZER_CONFIG['weight_decay_range'], log=True)
    )
    
    scheduler = create_scheduler(optimizer, TRAINING_CONFIG)
    
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(TRAINING_CONFIG['max_epochs']):
        train_accum = default_loss_accum()
        val_accum = default_loss_accum()
        model.train()
        train_loss = 0.0
        train_n = 0
        if len(train_loader) > 0: 
            for batch in train_loader:
                (
                    _,
                    enc_in,
                    dec_in,
                    target,
                    src_mask,
                    _,
                    _,
                    structure_in,
                ) = unpack_batch(batch, device)
                optimizer.zero_grad()
                model_out = forward_model(
                    model, enc_in, dec_in, src_mask, structure_in
                )
                output = voltage_from_model_output(model_out)
                breakdown = compute_training_loss(
                    output,
                    target,
                    model,
                    criterion,
                )
                breakdown.total.backward()
                
                if TRAINING_CONFIG['gradient_clipping']['enabled']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['gradient_clipping']['clip_value'])
                    
                optimizer.step()
                batch_size = enc_in.size(0)
                train_n += batch_size
                train_loss += breakdown.total.item() * batch_size
                for key in train_accum:
                    train_accum[key] += breakdown.detached_dict()[key] * batch_size
            
            train_loss /= len(train_loader.dataset)
            train_accum = average_breakdown_dict(train_accum, train_n)
        else:
            print("[WARN] Training loader is empty.")
            train_loss = float('nan') 
        model.eval()
        val_loss = 0.0
        val_n = 0
        if len(val_loader) > 0: 
            with torch.no_grad():
                for batch in val_loader:
                    (
                        _,
                        enc_in,
                        dec_in,
                        target,
                        src_mask,
                        _,
                        _,
                        structure_in,
                    ) = unpack_batch(batch, device)
                    model_out = forward_model(
                        model, enc_in, dec_in, src_mask, structure_in
                    )
                    output = voltage_from_model_output(model_out)
                    breakdown = compute_training_loss(
                        output,
                        target,
                        model,
                        criterion,
                    )
                    val_loss += breakdown.voltage_mse.item() * enc_in.size(0)
                    batch_size = enc_in.size(0)
                    val_n += batch_size
                    for key in val_accum:
                        val_accum[key] += breakdown.detached_dict()[key] * batch_size

            
            val_loss /= len(val_loader.dataset)
            val_accum = average_breakdown_dict(val_accum, val_n)
        else:
            print("[WARN] Validation loader is empty, skipping validation phase.")
            val_loss = float('nan')
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"Epoch {epoch:03d} | LR {old_lr:.2e} -> {new_lr:.2e}")
        
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"{format_train_epoch_summary(train_accum, train_loss)} | "
                f"{format_val_epoch_summary(val_loss, val_accum)}",
                flush=True,
            )
        
        if val_loss < best_val_loss - TRAINING_CONFIG['min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= TRAINING_CONFIG['patience']:
            print(f"Early stop at epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    global _HPO_BEST_CHECKPOINT
    if best_model_state is not None and best_val_loss < _HPO_BEST_CHECKPOINT['value']:
        _HPO_BEST_CHECKPOINT['value'] = best_val_loss
        _HPO_BEST_CHECKPOINT['state_dict'] = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        _HPO_BEST_CHECKPOINT['trial_number'] = trial.number

    print(f"Best val v_mse: {best_val_loss:.4f}")
    return best_val_loss

def _build_enqueue_trial_params(hpo_center, output_hidden_layers):
    if hpo_center is None:
        return None
    params = {
        'batch_size': hpo_center.get('batch_size'),
        'dropout_rate': hpo_center.get('dropout_rate'),
        'learning_rate': hpo_center.get('learning_rate'),
        'weight_decay': hpo_center.get('weight_decay'),
    }
    output_dims = hpo_center.get('output_dims')
    if output_dims and len(output_dims) == output_hidden_layers:
        for i, dim in enumerate(output_dims):
            params[f'output_dim_layer_{i}'] = dim
    if any(v is None for k, v in params.items() if not k.startswith('output_dim_layer_')):
        return None
    return params


def optimize_hyperparameters(
    processed_data,
    n_elements,
    device,
    n_trials=None,
    seed=None,
    decoder_feature_columns=None,
    checkpoint_path=None,
    hpo_center=None,
    hpo_center_config=None,
    optuna_visualization=None,
):
    """Run Optuna search; objective is validation voltage MSE."""
    global _HPO_BEST_CHECKPOINT
    _HPO_BEST_CHECKPOINT = {
        'value': float('inf'),
        'state_dict': None,
        'trial_number': None,
    }

    if n_trials is None:
        n_trials = OPTIMIZER_CONFIG['n_trials']
    if seed is None:
        seed = OPTIMIZER_CONFIG['seed']
        
    print(f"\nOptuna: {n_trials} trials (objective = val voltage MSE)", flush=True)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=seed)
    )

    if hpo_center_config is None:
        hpo_center_config = {}
    if hpo_center and hpo_center_config.get('enqueue_center_trial', True):
        enqueue_params = _build_enqueue_trial_params(hpo_center, MODEL_CONFIG['output_hidden_layers'])
        if enqueue_params is not None:
            study.enqueue_trial(enqueue_params)
            print(
                f"Enqueued HPO center trial: batch={enqueue_params['batch_size']}, "
                f"lr={enqueue_params['learning_rate']:.2e}, "
                f"dropout={enqueue_params['dropout_rate']:.4f}",
                flush=True,
            )

    def _on_trial_finished(study_obj, trial):
        value = trial.value
        value_text = f"{value:.6f}" if value is not None else "failed"
        print(f"  trial {trial.number + 1}/{n_trials}: val_v_mse={value_text}", flush=True)

    try:
        study.optimize(
            lambda trial: objective(
                trial,
                processed_data,
                n_elements,
                device,
                decoder_feature_columns=decoder_feature_columns,
                hpo_center=hpo_center,
                hpo_center_config=hpo_center_config,
            ),
            n_trials=n_trials,
            show_progress_bar=False,
            callbacks=[_on_trial_finished],
        )
    except optuna.exceptions.TrialPruned:
        print("[WARN] Optuna study pruned early.")
    except Exception as e:
        print(f"[ERR] Optuna failed: {e}")
        raise
    
    best_params = study.best_params
   
    final_params = {
        'encoder_input_dim': DATASET_CONFIG['encoder_token_dim'],  
        'd_model': MODEL_CONFIG['d_model'],
        'encoder_nhead': MODEL_CONFIG['encoder_nhead'],
        'decoder_nhead': MODEL_CONFIG['decoder_nhead'],
        'pooling_nhead': MODEL_CONFIG['pooling_nhead'],
        'num_layers': MODEL_CONFIG['num_layers'],
        'dropout_rate': best_params['dropout_rate'],
        'decoder_input_dim': DATASET_CONFIG['decoder_token_dim'], 
        'model_activation': TRAINING_CONFIG['model_activation']['type'], 
        'learning_rate': best_params['learning_rate'],
        'weight_decay': best_params['weight_decay'],
        'batch_size': best_params['batch_size'],
        'output_hidden_layers': MODEL_CONFIG['output_hidden_layers'],
        'output_dims': [best_params[f'output_dim_layer_{i}'] for i in range(MODEL_CONFIG['output_hidden_layers'])],
        'structure_placement': DATASET_CONFIG.get('structure_placement', 'none'),
        'structure_descriptor_dim': structure_per_token_dim(),
        'structure_input_dim': DATASET_CONFIG.get('structure_descriptor_dim', 64),
        'structure_dual_token_split': DATASET_CONFIG.get('structure_dual_token_split', False),
    }
    
    print("\nOptuna best params:")
    for param, value in best_params.items():
        if not param.startswith('output_dim_layer_'):
            print(f"  {param}: {value}")
    print(f"  output_dims: {final_params['output_dims']}")
    print(f"Best val v_mse: {study.best_value:.4f} ({len(study.trials)} trials)")
    
    viz_enabled = (
        OPTIMIZER_CONFIG['visualization']['enabled']
        if optuna_visualization is None
        else bool(optuna_visualization)
    )
    if viz_enabled:
        plots_dir = "results/optuna"
        os.makedirs(plots_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if 'history' in OPTIMIZER_CONFIG['visualization']['plots_to_generate']:
            try:
                fig_history = optuna_vis.plot_optimization_history(study)
                fig_history.write_image(os.path.join(plots_dir, f"optimization_history_{timestamp}.png"))
                print(f"Saved Optuna history plot: {os.path.join(plots_dir, f'optimization_history_{timestamp}.png')}")
            except Exception as e:
                print(f"[WARN] Optuna history plot failed: {e}")

        if 'parallel_coordinate' in OPTIMIZER_CONFIG['visualization']['plots_to_generate']:
            try:
                fig_parallel = optuna_vis.plot_parallel_coordinate(study)
                fig_parallel.write_image(os.path.join(plots_dir, f"parallel_coordinate_{timestamp}.png"))
                print(f"Saved Optuna parallel plot: {os.path.join(plots_dir, f'parallel_coordinate_{timestamp}.png')}")
            except Exception as e:
                print(f"[WARN] Optuna parallel plot failed: {e}")

        if 'slice' in OPTIMIZER_CONFIG['visualization']['plots_to_generate']:

            all_params = [p for p in study.best_params.keys() if not p.startswith('output_dim_layer_')]
            
            for param_name in all_params:
                try:
                    fig_slice = optuna_vis.plot_slice(study, params=[param_name])
                    filename = f"slice_plot_{param_name}_{timestamp}.png"
                    fig_slice.write_image(os.path.join(plots_dir, filename))
                    print(f"Saved Optuna slice ({param_name}): {os.path.join(plots_dir, filename)}")
                except Exception as e:
                    print(f"[WARN] Optuna slice plot ({param_name}) failed: {e}")

        if 'param_importances' in OPTIMIZER_CONFIG['visualization']['plots_to_generate']:
            try:
                fig_importance = optuna_vis.plot_param_importances(study)
                fig_importance.write_image(os.path.join(plots_dir, f"param_importances_{timestamp}.png"))
                print(f"Saved Optuna importance plot: {os.path.join(plots_dir, f'param_importances_{timestamp}.png')}")
            except Exception as e:
                print(f"[WARN] Optuna importance plot failed: {e}")

    saved_checkpoint_path = None
    if _HPO_BEST_CHECKPOINT['state_dict'] is not None:
        if checkpoint_path is None:
            checkpoint_path = os.path.join("results", "hpo_best_trial_checkpoint.pt")
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        torch.save(
            {
                'state_dict': _HPO_BEST_CHECKPOINT['state_dict'],
                'best_val_loss': _HPO_BEST_CHECKPOINT['value'],
                'trial_number': _HPO_BEST_CHECKPOINT['trial_number'],
            },
            checkpoint_path,
        )
        saved_checkpoint_path = checkpoint_path
        print(
            f"HPO best trial checkpoint saved: trial {_HPO_BEST_CHECKPOINT['trial_number']}, "
            f"val_v_mse={_HPO_BEST_CHECKPOINT['value']:.4f} -> {checkpoint_path}",
            flush=True,
        )

    return final_params, study.best_value, saved_checkpoint_path
