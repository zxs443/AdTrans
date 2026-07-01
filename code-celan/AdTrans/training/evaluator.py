import os 
import torch
import torch.nn as nn
import numpy as np
import pandas as pd 
from utils.metrics import calculate_metrics
from utils.plot_predictions import visualize_predictions
from utils.plot_attention import visualize_attention_weights
from utils.plot_xai import plot_periodic_table_importance, plot_decoder_token_importance, plot_integrated_gradients_importance, plot_feature_ablation_importance
from utils.plot_training import plot_training_curves, plot_learning_rate
from utils.config import TRAINING_CONFIG, XAI_CONFIG, DATASET_CONFIG
from data.preprocessing import N_ELEMENTS
from utils.visualization_config import VISUALIZATION_CONFIG
import collections
from xai.integrated_gradients import IntegratedGradientsExplainer 
from xai.feature_ablation import FeatureAblationExplainer
from utils.element_slot_order import encoder_slot_element_symbols
from data.dataset import unpack_batch, forward_model, voltage_from_model_output
from training.loss_utils import average_breakdown_dict, compute_training_loss, default_loss_accum
from data.structure_placement import (
    encoder_seq_len as get_encoder_seq_len,
    encoder_structure_offset,
    structure_in_decoder,
    structure_in_encoder,
    get_structure_token_name,
    uses_structure_input,
    n_structure_tokens,
    sum_structure_encoder_attention,
    accumulate_structure_xai_importance,
    normalize_element_importance_excluding_structure,
)


def _pad_attention_tensor(attn, target_rows, target_cols):
    """Pad attention weights [num_heads, seq_q, seq_k] for stacking across samples."""
    if attn is None:
        return None
    if attn.dim() != 3:
        raise ValueError(f"Expected 3D attention tensor, got shape {tuple(attn.shape)}")

    num_heads, seq_q, seq_k = attn.shape
    if seq_q > target_rows or seq_k > target_cols:
        attn = attn[:, :target_rows, :target_cols]
        seq_q, seq_k = attn.shape[1], attn.shape[2]
    if seq_q == target_rows and seq_k == target_cols:
        return attn

    padded = attn.new_zeros(num_heads, target_rows, target_cols)
    padded[:, :seq_q, :seq_k] = attn
    return padded


def _infer_decoder_attention_seq_len(dec_in, dec_self_attn_list):
    if dec_self_attn_list and len(dec_self_attn_list) > 0 and len(dec_self_attn_list[0]) > 0:
        return int(dec_self_attn_list[0][0].shape[-1])
    seq_len = dec_in.size(1)
    if structure_in_decoder():
        seq_len += n_structure_tokens()
    return seq_len


def _encoder_slot_symbols(formula: str, battery_id: str) -> list[str]:
    if not battery_id:
        return []
    try:
        return encoder_slot_element_symbols(
            formula,
            str(battery_id),
            max_slots=N_ELEMENTS,
        )
    except Exception:
        return []


def _stack_padded_attentions(attn_list, target_rows, target_cols):
    if not attn_list:
        return None
    padded = [_pad_attention_tensor(attn, target_rows, target_cols) for attn in attn_list]
    return torch.stack(padded, dim=0)

class ModelEvaluator:
    
    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device

    def evaluate(self, dataloader, return_loss_breakdown=False):
        self.model.eval()
        total_loss = 0
        loss_accum = default_loss_accum()
        n_samples = 0
        predictions = []
        targets = []

        num_layers = self.model.num_layers
        n_elements = N_ELEMENTS
        enc_seq_len = get_encoder_seq_len(n_elements)
        decoder_seq_len = None
        
        all_results_with_indices = [] 
        all_attention_weights_with_indices = []  

        with torch.no_grad():
            for batch in dataloader:
                (
                    original_idx,
                    enc_in,
                    dec_in,
                    target,
                    src_mask,
                    formulas_batch,
                    battery_ids_batch,
                    structure_in,
                ) = unpack_batch(batch, self.device)
                model_out = forward_model(
                    self.model, enc_in, dec_in, src_mask, structure_in
                )
                output = voltage_from_model_output(model_out)
                enc_self_attn_list = model_out[1]
                dec_self_attn_list = model_out[2]
                dec_cross_attn_list = model_out[3]
                if decoder_seq_len is None:
                    decoder_seq_len = _infer_decoder_attention_seq_len(dec_in, dec_self_attn_list)
                
                voltage_loss = self.criterion(output, target)
                total_loss += voltage_loss.item() * enc_in.size(0)
                if return_loss_breakdown:
                    breakdown = compute_training_loss(
                        output,
                        target,
                        self.model,
                        self.criterion,
                    )
                    batch_size = enc_in.size(0)
                    n_samples += batch_size
                    for key in loss_accum:
                        loss_accum[key] += breakdown.detached_dict()[key] * batch_size
                else:
                    batch_size = enc_in.size(0)
                
                for i in range(batch_size):
                    battery_id = battery_ids_batch[i]
                    if isinstance(battery_id, torch.Tensor):
                        battery_id = battery_id.item() if battery_id.numel() == 1 else str(battery_id)
                    all_results_with_indices.append((
                        original_idx[i].item(), 
                        output[i].item(),      
                        target[i].item(),     
                        formulas_batch[i],
                        str(battery_id),
                    ))
                    
                    attn_weights = {
                        'original_idx': original_idx[i].item(),
                        'enc_self_attn': [enc_self_attn_list[layer_idx][0][i] if len(enc_self_attn_list[layer_idx]) > 0 else None for layer_idx in range(num_layers)],
                        'dec_self_attn': [dec_self_attn_list[layer_idx][0][i] if len(dec_self_attn_list[layer_idx]) > 0 else None for layer_idx in range(num_layers)],
                        'dec_cross_attn': [dec_cross_attn_list[layer_idx][0][i] if len(dec_cross_attn_list[layer_idx]) > 0 else None for layer_idx in range(num_layers)],
                    }
                    all_attention_weights_with_indices.append(attn_weights)

        all_results_with_indices.sort(key=lambda x: x[0])
        all_attention_weights_with_indices.sort(key=lambda x: x['original_idx'])
        
        predictions = np.array([x[1] for x in all_results_with_indices])
        targets = np.array([x[2] for x in all_results_with_indices])
        all_formulas_in_dataloader = [x[3] for x in all_results_with_indices]
        all_battery_ids_in_dataloader = [x[4] for x in all_results_with_indices]

        if decoder_seq_len is None:
            decoder_seq_len = n_elements

        aggregated_encoder_self_attns = []
        for layer_idx in range(num_layers):
            layer_attns = [attn['enc_self_attn'][layer_idx] for attn in all_attention_weights_with_indices if attn['enc_self_attn'][layer_idx] is not None]
            stacked = _stack_padded_attentions(layer_attns, enc_seq_len, enc_seq_len)
            aggregated_encoder_self_attns.append(stacked)
        
        aggregated_decoder_self_attns = []
        for layer_idx in range(num_layers):
            layer_attns = [attn['dec_self_attn'][layer_idx] for attn in all_attention_weights_with_indices if attn['dec_self_attn'][layer_idx] is not None]
            stacked = _stack_padded_attentions(layer_attns, decoder_seq_len, decoder_seq_len)
            aggregated_decoder_self_attns.append(stacked)
        
        aggregated_decoder_cross_attns = []
        for layer_idx in range(num_layers):
            layer_attns = [attn['dec_cross_attn'][layer_idx] for attn in all_attention_weights_with_indices if attn['dec_cross_attn'][layer_idx] is not None]
            stacked = _stack_padded_attentions(layer_attns, decoder_seq_len, enc_seq_len)
            aggregated_decoder_cross_attns.append(stacked)

        avg_voltage_mse = total_loss / len(dataloader.dataset)
        val_breakdown = average_breakdown_dict(loss_accum, n_samples) if return_loss_breakdown else None
        base_return = (
            avg_voltage_mse,
            predictions,
            targets,
            aggregated_encoder_self_attns,
            aggregated_decoder_self_attns,
            aggregated_decoder_cross_attns,
            all_formulas_in_dataloader,
            all_battery_ids_in_dataloader,
        )
        if return_loss_breakdown:
            return (avg_voltage_mse, val_breakdown, *base_return[1:])
        return base_return

    def _calculate_global_ig_importance(
        self,
        all_encoder_attributions,
        all_decoder_attributions,
        all_formulas,
        decoder_token_names,
        n_elements,
        all_structure_attributions=None,
        all_battery_ids=None,
    ):
        """Global IG importance over elements and decoder tokens."""
        element_importance_sum = collections.defaultdict(float)
        decoder_token_importance_sum = collections.defaultdict(float)
        structure_token_name = get_structure_token_name()
        element_offset = encoder_structure_offset()

        total_samples = all_encoder_attributions.shape[0]


        for sample_idx in range(total_samples):
            sample_formula = all_formulas[sample_idx]
            sample_battery_id = (
                all_battery_ids[sample_idx] if all_battery_ids is not None else None
            )
            elements_in_sample = _encoder_slot_symbols(sample_formula, sample_battery_id)
            if not elements_in_sample:
                print(f"Warning: No encoder slot symbols for '{sample_formula}'. Skipping element IG.")
                continue

            element_attributions_per_sample = torch.sum(torch.abs(all_encoder_attributions[sample_idx]), dim=-1).cpu().numpy()

            if all_structure_attributions is not None:
                structure_importance = torch.sum(torch.abs(all_structure_attributions[sample_idx])).item()
                accumulate_structure_xai_importance(
                    structure_importance,
                    element_importance_sum,
                    decoder_token_importance_sum,
                    token_name=structure_token_name,
                )

            for i, element_symbol in enumerate(elements_in_sample):
                tensor_idx = element_offset + i
                if tensor_idx < all_encoder_attributions.shape[1]:
                    element_importance_sum[element_symbol] += element_attributions_per_sample[tensor_idx]
        
        element_importance_df = normalize_element_importance_excluding_structure(
            pd.DataFrame(element_importance_sum.items(), columns=['Element', 'Importance'])
        )

        num_property_tokens = all_decoder_attributions.shape[1]
        for sample_idx in range(total_samples):
            decoder_token_attributions_per_sample = torch.sum(torch.abs(all_decoder_attributions[sample_idx]), dim=-1).cpu().numpy()

            for i, token_name in enumerate(decoder_token_names):
                if token_name == structure_token_name:
                    continue
                elif i < num_property_tokens:
                    decoder_token_importance_sum[token_name] += decoder_token_attributions_per_sample[i]
                else:
                    print(f"Warning: Decoder token name '{token_name}' (index {i}) exceeds model decoder sequence length. Will be ignored.")

        decoder_token_importance_df = pd.DataFrame(decoder_token_importance_sum.items(), columns=['Property', 'Importance'])
        if not decoder_token_importance_df.empty:
            total_sum = decoder_token_importance_df['Importance'].sum()
            if total_sum > 0:
                decoder_token_importance_df['Importance'] = decoder_token_importance_df['Importance'] / total_sum
            else:
                decoder_token_importance_df['Importance'] = 0.0
            decoder_token_importance_df = decoder_token_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

        return element_importance_df, decoder_token_importance_df

    def _calculate_attention_importance_per_head(
        self,
        enc_self_attn_weights_aggregated_per_head,
        cross_attn_weights_aggregated_per_head,
        self_attn_weights_aggregated_per_head,
        all_formulas_for_attention,
        decoder_token_names,
        n_elements,
        all_battery_ids_for_attention=None,
    ):
        """Per-head attention importance (all samples)."""
        element_importance_sum = collections.defaultdict(float)
        decoder_token_importance_sum = collections.defaultdict(float)

        if enc_self_attn_weights_aggregated_per_head is not None:
            total_samples = enc_self_attn_weights_aggregated_per_head.shape[0]
        elif cross_attn_weights_aggregated_per_head is not None:
            total_samples = cross_attn_weights_aggregated_per_head.shape[0]
        elif self_attn_weights_aggregated_per_head is not None:
            total_samples = self_attn_weights_aggregated_per_head.shape[0]
        else:
            total_samples = 0

        if enc_self_attn_weights_aggregated_per_head is not None:
            encoder_seq_len = enc_self_attn_weights_aggregated_per_head.shape[1]
            element_offset = encoder_structure_offset()
            structure_token_name = get_structure_token_name()
            for sample_idx in range(total_samples):
                sample_formula = all_formulas_for_attention[sample_idx]
                sample_battery_id = (
                    all_battery_ids_for_attention[sample_idx]
                    if all_battery_ids_for_attention is not None
                    else None
                )
                elements_in_sample = _encoder_slot_symbols(sample_formula, sample_battery_id)
                if not elements_in_sample:
                    print(f"Warning: No encoder slot symbols for '{sample_formula}'. Skipping per-head importance.")
                    continue

                enc_self_attn_weights = enc_self_attn_weights_aggregated_per_head[sample_idx, :, :].cpu().numpy()

                element_attn_per_sample = np.sum(enc_self_attn_weights, axis=1)

                if structure_in_encoder() and encoder_seq_len > 0:
                    element_importance_sum[structure_token_name] += sum_structure_encoder_attention(
                        element_attn_per_sample
                    )

                for i, element_symbol in enumerate(elements_in_sample):
                    tensor_idx = element_offset + i
                    if tensor_idx < encoder_seq_len:
                        element_importance_sum[element_symbol] += element_attn_per_sample[tensor_idx]
        
        element_importance_df = normalize_element_importance_excluding_structure(
            pd.DataFrame(element_importance_sum.items(), columns=['Element', 'Importance'])
        )
        if element_importance_df.empty:
            element_importance_df = pd.DataFrame(columns=['Element', 'Importance'])

        if self_attn_weights_aggregated_per_head is not None:
            decoder_seq_len = self_attn_weights_aggregated_per_head.shape[2] 
            for sample_idx in range(total_samples):
                self_attn_weights = self_attn_weights_aggregated_per_head[sample_idx, :, :].cpu().numpy()

                decoder_token_attn_per_sample = np.sum(self_attn_weights, axis=0)

                for i, token_name in enumerate(decoder_token_names):
                    if i < decoder_seq_len: 
                        decoder_token_importance_sum[token_name] += decoder_token_attn_per_sample[i]
                    else:
                        print(f"Warning: Decoder token name '{token_name}' (index {i}) exceeds model decoder sequence length {decoder_seq_len}. Will be ignored.")

        decoder_token_importance_df = pd.DataFrame(decoder_token_importance_sum.items(), columns=['Property', 'Importance'])
        if not decoder_token_importance_df.empty:
            total_importance_sum = decoder_token_importance_df['Importance'].sum()
            if total_importance_sum > 0:
                decoder_token_importance_df['Importance'] = decoder_token_importance_df['Importance'] / total_importance_sum
            else:
                decoder_token_importance_df['Importance'] = 0.0
            decoder_token_importance_df = decoder_token_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        else:
            decoder_token_importance_df = pd.DataFrame(columns=['Property', 'Importance'])

        return element_importance_df, decoder_token_importance_df

    def _calculate_overall_attention_importance(
        self,
        aggregated_encoder_self_attns,
        aggregated_decoder_cross_attns,
        aggregated_decoder_self_attns,
        all_formulas_for_attention,
        decoder_token_names,
        n_elements,
        all_battery_ids_for_attention=None,
    ):
        """Global attention importance (all layers/heads)."""
        if not aggregated_encoder_self_attns:
            print("Warning: No encoder self-attention weights available for element importance calculation.")
            element_importance_df = pd.DataFrame()
        else:
            num_layers_enc = len(aggregated_encoder_self_attns)
            total_samples_enc = aggregated_encoder_self_attns[0].shape[0]
            num_heads_enc = aggregated_encoder_self_attns[0].shape[1]
            encoder_seq_len = aggregated_encoder_self_attns[0].shape[2]

            element_importance_sum = collections.defaultdict(float)
            element_offset = encoder_structure_offset()
            structure_token_name = get_structure_token_name()
            
            for sample_idx in range(total_samples_enc):
                sample_formula = all_formulas_for_attention[sample_idx]
                sample_battery_id = (
                    all_battery_ids_for_attention[sample_idx]
                    if all_battery_ids_for_attention is not None
                    else None
                )
                elements_in_sample = _encoder_slot_symbols(sample_formula, sample_battery_id)
                if not elements_in_sample:
                    print(f"Warning: No encoder slot symbols for '{sample_formula}'. Skipping encoder self-attn importance.")
                    continue

                for layer_idx in range(num_layers_enc):
                    for head_idx in range(num_heads_enc):
                        
                        enc_self_attn_weights = aggregated_encoder_self_attns[layer_idx][sample_idx, head_idx, :, :].cpu().numpy()

                        element_attn_per_sample_head_layer = np.sum(enc_self_attn_weights, axis=1)

                        if structure_in_encoder() and encoder_seq_len > 0:
                            element_importance_sum[structure_token_name] += sum_structure_encoder_attention(
                                element_attn_per_sample_head_layer
                            )

                        for i, element_symbol in enumerate(elements_in_sample):
                            tensor_idx = element_offset + i
                            if tensor_idx < encoder_seq_len:
                                element_importance_sum[element_symbol] += element_attn_per_sample_head_layer[tensor_idx]
            
            element_importance_df = normalize_element_importance_excluding_structure(
                pd.DataFrame(element_importance_sum.items(), columns=['Element', 'Importance'])
            )
        
        # cross-attn [B,H,dec,enc]; sum over dec (axis 0)
        if not aggregated_decoder_cross_attns:
            element_cross_importance_df = pd.DataFrame()
        else:
            num_layers_cross = len(aggregated_decoder_cross_attns)
            total_samples_cross = aggregated_decoder_cross_attns[0].shape[0]
            num_heads_cross = aggregated_decoder_cross_attns[0].shape[1]
            decoder_seq_len_cross = aggregated_decoder_cross_attns[0].shape[2]
            encoder_seq_len_cross = aggregated_decoder_cross_attns[0].shape[3]

            element_cross_importance_sum = collections.defaultdict(float)
            element_offset = encoder_structure_offset()
            structure_token_name = get_structure_token_name()
            
            for sample_idx in range(total_samples_cross):
                sample_formula = all_formulas_for_attention[sample_idx]
                sample_battery_id = (
                    all_battery_ids_for_attention[sample_idx]
                    if all_battery_ids_for_attention is not None
                    else None
                )
                elements_in_sample = _encoder_slot_symbols(sample_formula, sample_battery_id)
                if not elements_in_sample:
                    print(f"Warning: No encoder slot symbols for '{sample_formula}'. Skipping cross-attn importance.")
                    continue

                for layer_idx in range(num_layers_cross):
                    for head_idx in range(num_heads_cross):
                        cross_attn_weights = aggregated_decoder_cross_attns[layer_idx][sample_idx, head_idx, :, :].cpu().numpy()
                        
                        element_attn_per_sample_head_layer = np.sum(cross_attn_weights, axis=0)
                        
                        if structure_in_encoder() and encoder_seq_len_cross > 0:
                            element_cross_importance_sum[structure_token_name] += sum_structure_encoder_attention(
                                element_attn_per_sample_head_layer
                            )

                        for i, element_symbol in enumerate(elements_in_sample):
                            tensor_idx = element_offset + i
                            if tensor_idx < encoder_seq_len_cross:
                                element_cross_importance_sum[element_symbol] += element_attn_per_sample_head_layer[tensor_idx]
            
            element_cross_importance_df = normalize_element_importance_excluding_structure(
                pd.DataFrame(element_cross_importance_sum.items(), columns=['Element', 'Importance'])
            )
            if element_cross_importance_df.empty:
                element_cross_importance_df = pd.DataFrame(columns=['Element', 'Importance'])

        if not aggregated_decoder_self_attns:
            print("Warning: No decoder self-attention weights available for property importance calculation.")
            decoder_token_importance_df = pd.DataFrame()
        else:
            num_layers_self = len(aggregated_decoder_self_attns)
            total_samples_self = aggregated_decoder_self_attns[0].shape[0]
            num_heads_self = aggregated_decoder_self_attns[0].shape[1]
            decoder_seq_len_self = aggregated_decoder_self_attns[0].shape[2]

            decoder_token_importance_sum = collections.defaultdict(float)

            for sample_idx in range(total_samples_self):
                for layer_idx in range(num_layers_self):
                    for head_idx in range(num_heads_self):
                        self_attn_weights = aggregated_decoder_self_attns[layer_idx][sample_idx, head_idx, :, :].cpu().numpy()

                        decoder_token_attn_per_sample_head_layer = np.sum(self_attn_weights, axis=0)

                        for i, token_name in enumerate(decoder_token_names):
                            if i < decoder_seq_len_self: 
                                decoder_token_importance_sum[token_name] += decoder_token_attn_per_sample_head_layer[i]
                            else:
                                print(f"Warning: Decoder token name '{token_name}' (index {i}) exceeds model decoder sequence length {decoder_seq_len_self}. Will be ignored.")

            decoder_token_importance_df = pd.DataFrame(decoder_token_importance_sum.items(), columns=['Property', 'Importance'])
            if not decoder_token_importance_df.empty:
                total_importance_sum = decoder_token_importance_df['Importance'].sum()
                if total_importance_sum > 0:
                    decoder_token_importance_df['Importance'] = decoder_token_importance_df['Importance'] / total_importance_sum
                else:
                    decoder_token_importance_df['Importance'] = 0.0
                decoder_token_importance_df = decoder_token_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

        return element_importance_df, decoder_token_importance_df, element_cross_importance_df

    @staticmethod
    def _slice_aggregated_attentions(attn_layers, n_samples):
        """Keep only the first ``n_samples`` rows from each layer's attention stack."""
        if not attn_layers or n_samples is None:
            return attn_layers
        sliced = []
        for attn in attn_layers:
            if attn is None:
                sliced.append(None)
            else:
                sliced.append(attn[:n_samples])
        return sliced
    
    def evaluate_and_log_results(self, train_loader, val_loader, test_loader, 
                                 train_formulas, val_formulas, test_formulas, 
                                 all_raw_formulas, 
                                 n_elements, save_dir, combined_descriptor_name, decoder_token_names,
                                 encoder_mean, decoder_mean, targets_mean, targets_std,
                                 train_losses_history=None, val_losses_history=None, learning_rates_history=None,
                                 structure_mean=None,
                                 xai_ranking_splits='all'):
        """Eval, metrics, plots, XAI. xai_ranking_splits: 'all' | 'train' (pruning default)."""
        print("\nEvaluating model...")
        _, train_pred, train_true, train_enc_self_attn, train_dec_self_attn, train_dec_cross_attn, train_formulas_for_attn, train_battery_ids_for_attn = self.evaluate(train_loader)
        _, val_pred, val_true, val_enc_self_attn, val_dec_self_attn, val_dec_cross_attn, val_formulas_for_attn, val_battery_ids_for_attn = self.evaluate(val_loader)
        _, test_pred, test_true, test_enc_self_attn, test_dec_self_attn, test_dec_cross_attn, test_formulas_for_attn, test_battery_ids_for_attn = self.evaluate(test_loader)
        
        train_true_unscaled = train_true * targets_std + targets_mean
        train_pred_unscaled = train_pred * targets_std + targets_mean
        val_true_unscaled = val_true * targets_std + targets_mean
        val_pred_unscaled = val_pred * targets_std + targets_mean
        test_true_unscaled = test_true * targets_std + targets_mean
        test_pred_unscaled = test_pred * targets_std + targets_mean

        metrics = {
            'Train': calculate_metrics(train_true_unscaled, train_pred_unscaled),
            'Validation': calculate_metrics(val_true_unscaled, val_pred_unscaled),
            'Test': calculate_metrics(test_true_unscaled, test_pred_unscaled)
        }
        
        if VISUALIZATION_CONFIG['plot_training_curves_enabled'] and train_losses_history is not None and val_losses_history is not None:
            plot_training_curves(train_losses_history, val_losses_history, save_dir, combined_descriptor_name)
        
        if VISUALIZATION_CONFIG['plot_learning_rate_enabled'] and learning_rates_history is not None:
            plot_learning_rate(learning_rates_history, save_dir, combined_descriptor_name)

        metrics, timestamp = visualize_predictions( 
            train_true, train_pred, val_true, val_pred, test_true, test_pred,
            train_formulas_for_attn, val_formulas_for_attn, test_formulas_for_attn,
            train_battery_ids_for_attn, val_battery_ids_for_attn, test_battery_ids_for_attn,
            n_elements, metrics,
            base_dir=save_dir, descriptor_type=combined_descriptor_name,
            decoder_token_names=decoder_token_names,
            targets_mean=targets_mean, targets_std=targets_std
        )
        
        num_layers_model = self.model.num_layers
        
        all_enc_self_attn_aggregated = []
        all_dec_self_attn_aggregated = []
        all_dec_cross_attn_aggregated = []

        for layer_idx in range(num_layers_model):
            current_enc_self_attns = []
            if train_enc_self_attn and layer_idx < len(train_enc_self_attn) and train_enc_self_attn[layer_idx] is not None:
                current_enc_self_attns.append(train_enc_self_attn[layer_idx])
            if val_enc_self_attn and layer_idx < len(val_enc_self_attn) and val_enc_self_attn[layer_idx] is not None:
                current_enc_self_attns.append(val_enc_self_attn[layer_idx])
            if test_enc_self_attn and layer_idx < len(test_enc_self_attn) and test_enc_self_attn[layer_idx] is not None:
                current_enc_self_attns.append(test_enc_self_attn[layer_idx])
            if current_enc_self_attns:
                all_enc_self_attn_aggregated.append(torch.cat(current_enc_self_attns, dim=0))
            else:
                all_enc_self_attn_aggregated.append(None) 

            current_dec_self_attns = []
            if train_dec_self_attn and layer_idx < len(train_dec_self_attn) and train_dec_self_attn[layer_idx] is not None:
                current_dec_self_attns.append(train_dec_self_attn[layer_idx])
            if val_dec_self_attn and layer_idx < len(val_dec_self_attn) and val_dec_self_attn[layer_idx] is not None:
                current_dec_self_attns.append(val_dec_self_attn[layer_idx])
            if test_dec_self_attn and layer_idx < len(test_dec_self_attn) and test_dec_self_attn[layer_idx] is not None:
                current_dec_self_attns.append(test_dec_self_attn[layer_idx])
            if current_dec_self_attns:
                all_dec_self_attn_aggregated.append(torch.cat(current_dec_self_attns, dim=0))
            else:
                all_dec_self_attn_aggregated.append(None)

            current_dec_cross_attns = []
            if train_dec_cross_attn and layer_idx < len(train_dec_cross_attn) and train_dec_cross_attn[layer_idx] is not None:
                current_dec_cross_attns.append(train_dec_cross_attn[layer_idx])
            if val_dec_cross_attn and layer_idx < len(val_dec_cross_attn) and val_dec_cross_attn[layer_idx] is not None:
                current_dec_cross_attns.append(val_dec_cross_attn[layer_idx])
            if test_dec_cross_attn and layer_idx < len(test_dec_cross_attn) and test_dec_cross_attn[layer_idx] is not None:
                current_dec_cross_attns.append(test_dec_cross_attn[layer_idx])
            if current_dec_cross_attns:
                all_dec_cross_attn_aggregated.append(torch.cat(current_dec_cross_attns, dim=0))
            else:
                all_dec_cross_attn_aggregated.append(None)

        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        all_formulas_for_attention = train_formulas_for_attn + val_formulas_for_attn + test_formulas_for_attn
        all_battery_ids_for_attention = (
            train_battery_ids_for_attn + val_battery_ids_for_attn + test_battery_ids_for_attn
        )

        total_formulas = len(all_formulas_for_attention)
        if all_enc_self_attn_aggregated and all_enc_self_attn_aggregated[0] is not None:
            total_attn = all_enc_self_attn_aggregated[0].shape[0]
            if total_formulas != total_attn:
                print(f"Error: Formula list size ({total_formulas}) doesn't match attention weight array size ({total_attn})")
                print(f"Train: {len(train_formulas_for_attn)}, Val: {len(val_formulas_for_attn)}, Test: {len(test_formulas_for_attn)}")
                if train_enc_self_attn and train_enc_self_attn[0] is not None:
                    print(f"Train attn: {train_enc_self_attn[0].shape[0]}")
                if val_enc_self_attn and val_enc_self_attn[0] is not None:
                    print(f"Val attn: {val_enc_self_attn[0].shape[0]}")
                if test_enc_self_attn and test_enc_self_attn[0] is not None:
                    print(f"Test attn: {test_enc_self_attn[0].shape[0]}")
                raise ValueError(f"Size mismatch: formulas={total_formulas}, attention={total_attn}")

        print("\nVisualizing attention weights...")
        visualize_attention_weights(
            all_enc_self_attn_aggregated, 
            all_dec_self_attn_aggregated, 
            all_dec_cross_attn_aggregated,
            all_formulas_for_attention, 
            n_elements, self.model.d_model, decoder_token_names, plots_dir, combined_descriptor_name,
            all_dataset_battery_ids=all_battery_ids_for_attention,
        )
        
        xai_train_only = str(xai_ranking_splits).lower() == 'train'
        n_train_for_xai = len(train_formulas_for_attn) if xai_train_only else None
        if xai_train_only:
            print("XAI token ranking: train split only (no val/test).")

        if VISUALIZATION_CONFIG.get('calculate_global_attention_importance', False):
            print("\nCalculating global attention importance...")

            if xai_train_only:
                all_formulas_for_global_importance = list(train_formulas_for_attn)
                all_battery_ids_for_global_importance = list(train_battery_ids_for_attn)
            else:
                all_formulas_for_global_importance = train_formulas_for_attn + val_formulas_for_attn + test_formulas_for_attn
                all_battery_ids_for_global_importance = all_battery_ids_for_attention

            filtered_aggregated_enc_self_attns = [
                attn for attn in self._slice_aggregated_attentions(
                    all_enc_self_attn_aggregated, n_train_for_xai
                )
                if attn is not None
            ]
            
            filtered_aggregated_cross_attns = [
                attn for attn in self._slice_aggregated_attentions(
                    all_dec_cross_attn_aggregated, n_train_for_xai
                )
                if attn is not None
            ]

            filtered_aggregated_self_attns = [
                attn for attn in self._slice_aggregated_attentions(
                    all_dec_self_attn_aggregated, n_train_for_xai
                )
                if attn is not None
            ]

            element_importance_df, decoder_token_importance_df, element_cross_importance_df = self._calculate_overall_attention_importance(
                filtered_aggregated_enc_self_attns,
                filtered_aggregated_cross_attns,
                filtered_aggregated_self_attns,
                all_formulas_for_global_importance,
                decoder_token_names,
                n_elements,
                all_battery_ids_for_attention=all_battery_ids_for_global_importance,
            )

            if not element_importance_df.empty:
                element_importance_path = os.path.join(save_dir, f'element_self_attention_importance_{combined_descriptor_name}.csv')
                element_importance_df.to_csv(element_importance_path, index=False)
                print(f"Global Element Self Attention Importance saved to: {element_importance_path}")
                plot_periodic_table_importance(element_importance_df, save_dir, f'{combined_descriptor_name}_self_attention')

            if not decoder_token_importance_df.empty:
                decoder_token_importance_path = os.path.join(save_dir, f'decoder_token_self_attention_importance_{combined_descriptor_name}.csv')
                decoder_token_importance_df.to_csv(decoder_token_importance_path, index=False)
                print(f"Global Decoder Property Self Attention Importance saved to: {decoder_token_importance_path}")
                plot_decoder_token_importance(decoder_token_importance_df, save_dir, f'{combined_descriptor_name}_self_attention')
            
            if not element_cross_importance_df.empty:
                element_cross_importance_path = os.path.join(save_dir, f'element_cross_attention_importance_{combined_descriptor_name}.csv')
                element_cross_importance_df.to_csv(element_cross_importance_path, index=False)
                print(f"Global Element Cross Attention Importance saved to: {element_cross_importance_path}")
                plot_periodic_table_importance(element_cross_importance_df, save_dir, f'{combined_descriptor_name}_cross_attention')

        if (
            VISUALIZATION_CONFIG.get('calculate_per_head_attention_importance', False)
            and VISUALIZATION_CONFIG.get('save_per_layer_per_head_attention_heatmaps', True)
        ):
            print("\nCalculating importance per layer and per attention head...")
            all_formulas_for_per_head_importance = train_formulas_for_attn + val_formulas_for_attn + test_formulas_for_attn
            all_battery_ids_for_per_head_importance = all_battery_ids_for_attention

            for layer_idx in range(num_layers_model):
                if all_enc_self_attn_aggregated[layer_idx] is not None:
                    num_heads_current_layer = all_enc_self_attn_aggregated[layer_idx].shape[1]
                    for head_idx in range(num_heads_current_layer):
                        enc_self_attn_per_head = all_enc_self_attn_aggregated[layer_idx][:, head_idx, :, :]
                        
                        cross_attn_per_head = None
                        if all_dec_cross_attn_aggregated[layer_idx] is not None:
                            cross_attn_per_head = all_dec_cross_attn_aggregated[layer_idx][:, head_idx, :, :]
                        
                        self_attn_per_head = None
                        if all_dec_self_attn_aggregated[layer_idx] is not None:
                            self_attn_per_head = all_dec_self_attn_aggregated[layer_idx][:, head_idx, :, :]

                        element_importance_df_per_head, decoder_token_importance_df_per_head = self._calculate_attention_importance_per_head(
                            enc_self_attn_per_head,
                            cross_attn_per_head,
                            self_attn_per_head,
                            all_formulas_for_per_head_importance,
                            decoder_token_names,
                            n_elements,
                            all_battery_ids_for_attention=all_battery_ids_for_per_head_importance,
                        )

                        if not element_importance_df_per_head.empty:
                            element_importance_path_per_head = os.path.join(save_dir, f'element_importance_{combined_descriptor_name}_L{layer_idx+1}_H{head_idx+1}.csv')
                            element_importance_df_per_head.to_csv(element_importance_path_per_head, index=False)
                            print(f"Element importance (L{layer_idx+1}, H{head_idx+1}) saved to: {element_importance_path_per_head}")
                            plot_periodic_table_importance(element_importance_df_per_head, save_dir, f'{combined_descriptor_name}_L{layer_idx+1}_H{head_idx+1}_self_attention')

                        if not decoder_token_importance_df_per_head.empty:
                            decoder_token_importance_path_per_head = os.path.join(save_dir, f'decoder_token_importance_{combined_descriptor_name}_L{layer_idx+1}_H{head_idx+1}.csv')
                            decoder_token_importance_df_per_head.to_csv(decoder_token_importance_path_per_head, index=False)
                            print(f"Decoder property importance (L{layer_idx+1}, H{head_idx+1}) saved to: {decoder_token_importance_path_per_head}")
                            plot_decoder_token_importance(decoder_token_importance_df_per_head, save_dir, f'{combined_descriptor_name}_L{layer_idx+1}_H{head_idx+1}_self_attention')
                else:
                    print(f"Warning: No decoder cross-attention weights for layer {layer_idx+1}, skipping per-head importance calculation for this layer.")

        if XAI_CONFIG['integrated_gradients']['enabled']:
            print("\nCalculating Integrated Gradients attributions...")
            ig_explainer = IntegratedGradientsExplainer(
                self.model, 
                self.device, 
                baseline_type=XAI_CONFIG['integrated_gradients']['baseline_type'],
                encoder_mean=encoder_mean, 
                decoder_mean=decoder_mean,
                structure_mean=structure_mean,
            )

            all_ig_encoder_attributions = []
            all_ig_decoder_attributions = []
            all_ig_structure_attributions = []

            if xai_train_only:
                ig_loader_specs = [(train_loader, train_formulas)]
                ig_formulas_for_importance = list(train_formulas_for_attn)
                ig_battery_ids_for_importance = list(train_battery_ids_for_attn)
            else:
                ig_loader_specs = [
                    (train_loader, train_formulas),
                    (val_loader, val_formulas),
                    (test_loader, test_formulas),
                ]
                ig_formulas_for_importance = (
                    train_formulas_for_attn + val_formulas_for_attn + test_formulas_for_attn
                )
                ig_battery_ids_for_importance = all_battery_ids_for_attention

            for loader, _formulas_list in ig_loader_specs:
                for batch in loader:
                    unpacked = unpack_batch(batch, self.device)
                    _, enc_in, dec_in, target, src_mask, _, _, structure_in = unpacked
                    encoder_attrs, decoder_attrs, structure_attrs = ig_explainer.attribute(
                        enc_in, dec_in, src_key_padding_mask=src_mask,
                        structure_input=structure_in,
                        steps=XAI_CONFIG['integrated_gradients']['steps']
                    )
                    all_ig_encoder_attributions.append(encoder_attrs.cpu())
                    all_ig_decoder_attributions.append(decoder_attrs.cpu())
                    if structure_attrs is not None:
                        all_ig_structure_attributions.append(structure_attrs.cpu())
            
            aggregated_ig_encoder_attributions = torch.cat(all_ig_encoder_attributions, dim=0)
            aggregated_ig_decoder_attributions = torch.cat(all_ig_decoder_attributions, dim=0)
            aggregated_ig_structure_attributions = (
                torch.cat(all_ig_structure_attributions, dim=0) if all_ig_structure_attributions else None
            )

            enc_attr_sum = aggregated_ig_encoder_attributions.abs().sum().item()
            dec_attr_sum = aggregated_ig_decoder_attributions.abs().sum().item()
            struct_attr_sum = (
                aggregated_ig_structure_attributions.abs().sum().item()
                if aggregated_ig_structure_attributions is not None else 0.0
            )
            if enc_attr_sum == 0.0 and dec_attr_sum == 0.0 and struct_attr_sum == 0.0:
                print("Warning: All Integrated Gradients attributions are zero. Check model gradients and IG baseline settings.")
            elif (
                not torch.isfinite(aggregated_ig_encoder_attributions).all()
                or not torch.isfinite(aggregated_ig_decoder_attributions).all()
                or (
                    aggregated_ig_structure_attributions is not None
                    and not torch.isfinite(aggregated_ig_structure_attributions).all()
                )
            ):
                print("Warning: Non-finite values detected in Integrated Gradients attributions.")

            ig_element_importance_df, ig_decoder_token_importance_df = self._calculate_global_ig_importance(
                aggregated_ig_encoder_attributions,
                aggregated_ig_decoder_attributions,
                ig_formulas_for_importance,
                decoder_token_names,
                n_elements,
                all_structure_attributions=aggregated_ig_structure_attributions,
                all_battery_ids=ig_battery_ids_for_importance,
            )

            if not ig_element_importance_df.empty:
                ig_element_importance_path = os.path.join(save_dir, f'ig_element_importance_{combined_descriptor_name}.csv')
                ig_element_importance_df.to_csv(ig_element_importance_path, index=False)
                print(f"IG element importance saved to: {ig_element_importance_path}")
                plot_integrated_gradients_importance(
                    ig_element_importance_df, 
                    ig_decoder_token_importance_df, 
                    save_dir, 
                    combined_descriptor_name, 
                    plot_type='periodic_table'
                )

            if not ig_decoder_token_importance_df.empty:
                ig_decoder_token_importance_path = os.path.join(save_dir, f'ig_decoder_token_importance_{combined_descriptor_name}.csv')
                ig_decoder_token_importance_df.to_csv(ig_decoder_token_importance_path, index=False)
                print(f"IG decoder property importance saved to: {ig_decoder_token_importance_path}")
                plot_integrated_gradients_importance(
                    ig_element_importance_df, 
                    ig_decoder_token_importance_df, 
                    save_dir, 
                    combined_descriptor_name, 
                    plot_type='histogram',
                    top_k=XAI_CONFIG['integrated_gradients']['visualize_top_k_properties']
                )

        if XAI_CONFIG['feature_ablation']['enabled']:
            print("\nCalculating Feature Ablation attributions...")
            fa_explainer = FeatureAblationExplainer(
                self.model, 
                self.device, 
                ablation_value_type=XAI_CONFIG['feature_ablation']['ablation_value']
            )

            fa_loaders = [train_loader] if xai_train_only else None
            fa_formulas_ctx = (
                list(train_formulas_for_attn)
                if xai_train_only
                else train_formulas_for_attn + val_formulas_for_attn + test_formulas_for_attn
            )
            fa_element_importance_df, fa_decoder_token_importance_df = fa_explainer.explain(
                train_loader, val_loader, test_loader, 
                fa_formulas_ctx,
                n_elements, 
                decoder_token_names,
                ablate_elements=XAI_CONFIG['feature_ablation']['ablate_elements'],
                ablate_properties=XAI_CONFIG['feature_ablation']['ablate_properties'],
                data_loaders=fa_loaders,
            )

            if not fa_element_importance_df.empty:
                fa_element_importance_path = os.path.join(save_dir, f'fa_element_importance_{combined_descriptor_name}.csv')
                fa_element_importance_df.to_csv(fa_element_importance_path, index=False)
                print(f"Feature Ablation element importance saved to: {fa_element_importance_path}")
                plot_feature_ablation_importance(
                    fa_element_importance_df, 
                    fa_decoder_token_importance_df, 
                    save_dir, 
                    combined_descriptor_name, 
                    plot_type='periodic_table'
                )

            if not fa_decoder_token_importance_df.empty:
                fa_decoder_token_importance_path = os.path.join(save_dir, f'fa_decoder_token_importance_{combined_descriptor_name}.csv')
                fa_decoder_token_importance_df.to_csv(fa_decoder_token_importance_path, index=False)
                print(f"Feature Ablation decoder property importance saved to: {fa_decoder_token_importance_path}")
                plot_feature_ablation_importance(
                    fa_element_importance_df, 
                    fa_decoder_token_importance_df, 
                    save_dir, 
                    combined_descriptor_name, 
                    plot_type='histogram',
                    top_k=XAI_CONFIG['feature_ablation']['visualize_top_k_properties']
                )

        print(f"\nEvaluation Metrics:")
        print("Training Set:")
        print(f"MAE: {metrics['Train']['MAE']:.4f}")
        print(f"RMSE: {metrics['Train']['RMSE']:.4f}")
        print(f"R²: {metrics['Train']['R2']:.4f}")
        print("\nValidation Set:")
        print(f"MAE: {metrics['Validation']['MAE']:.4f}")
        print(f"RMSE: {metrics['Validation']['RMSE']:.4f}")
        print(f"R²: {metrics['Validation']['R2']:.4f}")
        print("\nTest Set:")
        print(f"MAE: {metrics['Test']['MAE']:.4f}")
        print(f"RMSE: {metrics['Test']['RMSE']:.4f}")
        print(f"R²: {metrics['Test']['R2']:.4f}")
        
        return metrics, timestamp
