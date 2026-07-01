import os 
import torch
import torch.nn as nn
import numpy as np
from utils.config import TRAINING_CONFIG
from training.evaluator import ModelEvaluator
from data.dataset import unpack_batch, forward_model, voltage_from_model_output
from training.loss_utils import (
    average_breakdown_dict,
    compute_training_loss,
    default_loss_accum,
    format_train_epoch_summary,
    format_val_epoch_summary,
)

class ModelTrainer:
    """Training loop with early stopping on validation voltage MSE."""
    
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=TRAINING_CONFIG['scheduler']['factor'],
            patience=TRAINING_CONFIG['scheduler']['patience'],
            min_lr=TRAINING_CONFIG['scheduler']['min_lr']
        )
        self.evaluator = ModelEvaluator(model, criterion, device)

    def _loss_from_batch(self, batch):
        (
            _,
            enc_in,
            dec_in,
            target,
            src_mask,
            _,
            _,
            structure_in,
        ) = unpack_batch(batch, self.device)
        model_out = forward_model(
            self.model, enc_in, dec_in, src_mask, structure_in
        )
        output = voltage_from_model_output(model_out)
        return compute_training_loss(
            output,
            target,
            self.model,
            self.criterion,
        )
        
    def train_epoch(self, train_loader):

        self.model.train()
        total_loss = 0
        accum = default_loss_accum()
        n_samples = 0
        for batch in train_loader:
            self.optimizer.zero_grad()
            breakdown = self._loss_from_batch(batch)
            breakdown.total.backward()
            
            if TRAINING_CONFIG['gradient_clipping']['enabled']:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), TRAINING_CONFIG['gradient_clipping']['clip_value']
                )
                
            self.optimizer.step()
            batch_size = batch[1].size(0)
            n_samples += batch_size
            total_loss += breakdown.total.item() * batch_size
            for key in accum:
                accum[key] += breakdown.detached_dict()[key] * batch_size

        epoch_breakdown = average_breakdown_dict(accum, n_samples)
        self._last_train_breakdown = epoch_breakdown
        return total_loss / max(n_samples, 1)
    
    def train_with_early_stopping(self, train_loader, val_loader, config=None):

        if config is None:
            config = TRAINING_CONFIG
            
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        train_losses_history = []
        val_losses_history = []
        learning_rates_history = []

        for epoch in range(1, config['max_epochs'] + 1):
            train_loss = self.train_epoch(train_loader)
            train_losses_history.append(train_loss)
            
            val_loss, val_breakdown, _, _, _, _, _, _ = self.evaluator.evaluate(
                val_loader, return_loss_breakdown=True
            )
            val_losses_history.append(val_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            learning_rates_history.append(current_lr)
            old_lr = current_lr
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"Epoch {epoch:03d} | Learning rate updated: {old_lr:.2e} -> {new_lr:.2e}")
            
            if val_loss < best_val_loss - config['min_delta']:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            train_bd = getattr(self, "_last_train_breakdown", {})
            print(
                f"Epoch {epoch:03d} | "
                f"{format_train_epoch_summary(train_bd, train_loss)} | "
                f"{format_val_epoch_summary(val_loss, val_breakdown)}",
                flush=True,
            )
            
            if patience_counter >= config['patience']:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.model, train_losses_history, val_losses_history, learning_rates_history 
    
    def predict(self, encoder_input, decoder_input, src_key_padding_mask, structure_input=None):
        self.model.eval()
        if isinstance(encoder_input, np.ndarray):
            encoder_input = torch.tensor(encoder_input, dtype=torch.float32)
        if isinstance(decoder_input, np.ndarray):
            decoder_input = torch.tensor(decoder_input, dtype=torch.float32)
        if isinstance(src_key_padding_mask, np.ndarray):
            src_key_padding_mask = torch.tensor(src_key_padding_mask, dtype=torch.bool)
        if structure_input is not None and isinstance(structure_input, np.ndarray):
            structure_input = torch.tensor(structure_input, dtype=torch.float32)

        with torch.no_grad():
            encoder_input = encoder_input.to(self.device)
            decoder_input = decoder_input.to(self.device)
            src_key_padding_mask = src_key_padding_mask.to(self.device)
            if structure_input is not None:
                structure_input = structure_input.to(self.device)
            output = voltage_from_model_output(forward_model(
                self.model,
                encoder_input,
                decoder_input,
                src_key_padding_mask,
                structure_input,
            ))
            return output.cpu().numpy()
