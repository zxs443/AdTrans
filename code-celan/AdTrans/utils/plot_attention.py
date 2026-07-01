import os
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
import seaborn as sns
from datetime import datetime
import numpy as np
import torch
from utils.visualization_config import VISUALIZATION_CONFIG
from data.structure_placement import get_encoder_attention_labels
from utils.formula_utils import _count_unique_elements_in_formula
import random
from collections import defaultdict


plt.rcParams['font.family'] = VISUALIZATION_CONFIG['font_family']


def _sample_plot_label(sample_battery_id) -> str:
    """Human-readable sample label for plot titles."""
    return str(sample_battery_id)


def _dedupe_indices_preserve_order(indices: list[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)
    return ordered


def _indices_for_formula_targets(all_dataset_formulas, formula_targets: list) -> list[int]:
    """Resolve each configured formula to all matching dataset indices."""
    matched_indices: list[int] = []
    for f_target in formula_targets:
        indices = [i for i, formula in enumerate(all_dataset_formulas) if formula == f_target]
        if indices:
            if len(indices) > 1:
                print(
                    f"Formula '{f_target}' matched {len(indices)} samples; "
                    "plotting all matching entries in the same subfolder."
                )
            matched_indices.extend(indices)
        else:
            print(f"Warning: Specified formula '{f_target}' not found in the dataset, skipping visualization.")
    return _dedupe_indices_preserve_order(matched_indices)


def _sanitize_path_component(text: str, max_len: int = 120) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", str(text)).strip().strip(".")
    return cleaned[:max_len] or "sample"


def _max_attention_sample_index(
    all_encoder_self_attn_aggregated,
    all_decoder_self_attn_aggregated,
    all_decoder_cross_attn_aggregated,
) -> int | None:
    """Max global sample index shared by concatenated train+val+test attention tensors."""
    for layers in (
        all_encoder_self_attn_aggregated,
        all_decoder_self_attn_aggregated,
        all_decoder_cross_attn_aggregated,
    ):
        if layers and layers[0] is not None:
            return int(layers[0].shape[0]) - 1
    return None


def _attention_formula_targets(visualization_config) -> list | None:
    targets = visualization_config.get("attention_samples_to_plot")
    if isinstance(targets, list) and targets and not isinstance(targets[0], int):
        return targets
    return None


def _sample_plot_directories(
    sample_indices: list[int],
    all_dataset_formulas,
    plots_dir: str,
    formula_targets: list | None,
) -> dict[int, str]:
    """Map each sample index to its output directory (formula subfolder when configured)."""
    if not formula_targets:
        return {idx: plots_dir for idx in sample_indices}

    directories: dict[int, str] = {}
    indices_set = set(sample_indices)
    for f_target in formula_targets:
        subdir = os.path.join(plots_dir, _sanitize_path_component(f_target))
        os.makedirs(subdir, exist_ok=True)
        for idx in sample_indices:
            if idx in indices_set and all_dataset_formulas[idx] == f_target:
                directories[idx] = subdir

    for idx in sample_indices:
        if idx not in directories:
            fallback = os.path.join(
                plots_dir,
                _sanitize_path_component(all_dataset_formulas[idx]),
            )
            os.makedirs(fallback, exist_ok=True)
            directories[idx] = fallback
    return directories

def _create_custom_colormap():
    colors = VISUALIZATION_CONFIG.get('xai_colormap_colors', ['#F8CECC','#ebcccf','#e0cbd2','#d2cad5','#c6c8d9','#b1c6de','#91c2e6','#86C1E9'])
    colors_reversed = list(reversed(colors))
    cmap_name = 'xai_custom_cmap_reversed'
    return mcolors.LinearSegmentedColormap.from_list(cmap_name, colors_reversed)


def _encoder_axis_annotation_settings(n_keys: int) -> tuple[bool, str, int]:
    """Annotate heatmap cells when the encoder/key axis is small (same rule as encoder self-attention)."""
    if n_keys <= 10:
        return True, ".3f", 10
    return False, "", 8


def _plot_single_heatmap(weights, title, xlabel, ylabel, filename, annot=True, fmt=".3f", figsize=(8, 7), fontsize_labels=10, xtick_rotation=90):
    plt.figure(figsize=figsize)
    n_rows, n_cols = weights.shape
    is_cross_attn = xlabel is not ylabel
    is_decoder_self_attn = (xlabel == ylabel and len(xlabel) > 10)

    if is_cross_attn and annot:
        annot_fontsize = max(10, min(18, int(120 / max(n_rows, 1))))
        cbar_fontsize = max(12, int(fontsize_labels * 1.2))
    elif is_decoder_self_attn:
        annot_fontsize = max(10, int(fontsize_labels * 1.0))
        cbar_fontsize = max(12, int(fontsize_labels * 1.3))
    else:
        annot_fontsize = max(10, int(fontsize_labels * 1.0))
        cbar_fontsize = max(14, int(fontsize_labels * 1.3))

    ax = sns.heatmap(weights, cmap=_create_custom_colormap(), annot=annot, fmt=fmt,
                     annot_kws={'size': annot_fontsize, 'weight': 'bold', 'color': 'black'},
                     cbar_kws={'label': 'Attention Weight'})
    plt.title(title, fontsize=VISUALIZATION_CONFIG['font_size_title'], fontweight='bold')

    ax.set_xticks(np.arange(len(xlabel)) + 0.5)
    ax.set_xticklabels(
        xlabel,
        rotation=xtick_rotation,
        ha='right',
        fontsize=fontsize_labels * 0.8,
        fontweight='bold',
    )

    ax.set_yticks(np.arange(len(ylabel)) + 0.5)
    ax.set_yticklabels(
        ylabel,
        rotation=0,
        ha='right',
        fontsize=fontsize_labels * 0.8,
        fontweight='bold',
    )

    ax.set_xlabel("Key", fontsize=fontsize_labels, fontweight='bold')
    ax.set_ylabel("Query", fontsize=fontsize_labels, fontweight='bold')

    fig = plt.gcf()
    for ax_candidate in fig.axes:
        if ax_candidate != ax:
            ax_candidate.tick_params(labelsize=cbar_fontsize)
            label = ax_candidate.get_ylabel()
            if label:
                ax_candidate.set_ylabel(label, fontsize=cbar_fontsize)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(os.path.join(filename), format='png', bbox_inches='tight', dpi=VISUALIZATION_CONFIG['dpi'])
    plt.close()

def visualize_attention_weights(
    all_encoder_self_attn_aggregated, 
    all_decoder_self_attn_aggregated, 
    all_decoder_cross_attn_aggregated,
    all_dataset_formulas, 
    n_elements, 
    d_model, 
    decoder_token_names, 
    plots_dir, 
    descriptor_type,
    all_dataset_battery_ids=None,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(plots_dir, exist_ok=True)

    def _get_attention_sample_indices(all_dataset_formulas, visualization_config):
        samples_to_plot_global_indices = []
        total_samples_global = len(all_dataset_formulas)

        if visualization_config['attention_samples_to_plot'] is not None:
            if isinstance(visualization_config['attention_samples_to_plot'], list) and \
               len(visualization_config['attention_samples_to_plot']) > 0:
                if isinstance(visualization_config['attention_samples_to_plot'][0], int):
                    samples_to_plot_global_indices = [idx for idx in visualization_config['attention_samples_to_plot'] if idx < total_samples_global]
                else:
                    matched_indices = _indices_for_formula_targets(
                        all_dataset_formulas,
                        visualization_config['attention_samples_to_plot'],
                    )
                    samples_to_plot_global_indices = [
                        idx for idx in matched_indices if 0 <= idx < total_samples_global
                    ]
            else:
                print("Warning: Invalid attention_samples_to_plot configuration, using random sampling or default behavior.")
        else:
            if visualization_config['random_sample_attention']:
                samples_by_num_elements = defaultdict(list)
                for i, formula in enumerate(all_dataset_formulas):
                    num_unique_elements = _count_unique_elements_in_formula(formula)
                    samples_by_num_elements[num_unique_elements].append(i)
                
                for num_elements_key, indices_list in samples_by_num_elements.items():
                    num_samples_to_select = min(visualization_config['num_random_samples_per_element'], len(indices_list))
                    samples_to_plot_global_indices.extend(random.sample(indices_list, num_samples_to_select))
                samples_to_plot_global_indices.sort()

                if not samples_to_plot_global_indices:
                    print("Warning: No samples available for attention visualization after random sampling.")
            else:
                print("No attention visualization samples specified, and random sampling is not enabled. Skipping attention visualization.")
        return samples_to_plot_global_indices

    samples_to_plot_global_indices = _get_attention_sample_indices(all_dataset_formulas, VISUALIZATION_CONFIG)
    if not samples_to_plot_global_indices:
        return
    
    # Drop out-of-range sample indices
    max_valid_idx = len(all_dataset_formulas) - 1
    attn_max_idx = _max_attention_sample_index(
        all_encoder_self_attn_aggregated,
        all_decoder_self_attn_aggregated,
        all_decoder_cross_attn_aggregated,
    )
    if attn_max_idx is not None:
        max_valid_idx = min(max_valid_idx, attn_max_idx)
    
    valid_indices = [idx for idx in samples_to_plot_global_indices if 0 <= idx <= max_valid_idx]
    invalid_indices = [idx for idx in samples_to_plot_global_indices if idx < 0 or idx > max_valid_idx]
    
    if invalid_indices:
        print(f"Warning: Found {len(invalid_indices)} invalid indices (max valid: {max_valid_idx}). Skipping them.")
    
    if not valid_indices:
        print(f"Error: No valid indices found after filtering. Cannot proceed with visualization.")
        return
    
    samples_to_plot_global_indices = valid_indices
    formula_targets = _attention_formula_targets(VISUALIZATION_CONFIG)
    sample_plot_directories = _sample_plot_directories(
        samples_to_plot_global_indices,
        all_dataset_formulas,
        plots_dir,
        formula_targets,
    )

    save_per_head_heatmaps = VISUALIZATION_CONFIG.get('save_per_layer_per_head_attention_heatmaps', True)

    if save_per_head_heatmaps and all_encoder_self_attn_aggregated:
        num_encoder_layers = len(all_encoder_self_attn_aggregated)
        
        for layer_idx in range(num_encoder_layers):
            current_layer_attn_weights = all_encoder_self_attn_aggregated[layer_idx]
            if current_layer_attn_weights is None: continue 
            num_heads = current_layer_attn_weights.shape[1]

            for global_sample_idx in samples_to_plot_global_indices:
                sample_plots_dir = sample_plot_directories[global_sample_idx]
                if global_sample_idx >= current_layer_attn_weights.shape[0]:
                    print(f"Warning: Index {global_sample_idx} out of bounds for encoder layer {layer_idx+1} (max: {current_layer_attn_weights.shape[0]-1}). Skipping.")
                    continue
                if global_sample_idx >= len(all_dataset_formulas):
                    print(f"Warning: Index {global_sample_idx} out of bounds for formula list (max: {len(all_dataset_formulas)-1}). Skipping.")
                    continue
                    
                sample_formula = all_dataset_formulas[global_sample_idx]
                if (
                    all_dataset_battery_ids is None
                    or global_sample_idx >= len(all_dataset_battery_ids)
                ):
                    continue
                sample_battery_id = all_dataset_battery_ids[global_sample_idx]
                encoder_labels = get_encoder_attention_labels(
                    sample_formula, n_elements, sample_battery_id
                )
                actual_seq_len = len(encoder_labels)

                if actual_seq_len == 0:
                    continue
                
                for head_idx in range(num_heads):
                    attn_weights_sample = current_layer_attn_weights[global_sample_idx, head_idx, :actual_seq_len, :actual_seq_len].cpu().numpy()
                    
                    title = f"Enc Self-Attn L{layer_idx+1}H{head_idx+1} - {_sample_plot_label(sample_battery_id)}"
                    filename = os.path.join(sample_plots_dir, f"encoder_self_attn_L{layer_idx+1}_H{head_idx+1}_S{global_sample_idx}_{timestamp}.png")
                    
                    figsize_enc = (max(4, actual_seq_len * 0.8), max(4, actual_seq_len * 0.8))
                    annot_enc, fmt_enc, fontsize_labels_enc = _encoder_axis_annotation_settings(actual_seq_len)

                    _plot_single_heatmap(attn_weights_sample, title, encoder_labels, encoder_labels, filename, 
                                        annot=annot_enc, fmt=fmt_enc, figsize=figsize_enc, fontsize_labels=fontsize_labels_enc)
    
    if save_per_head_heatmaps and all_decoder_self_attn_aggregated:
        num_decoder_layers = len(all_decoder_self_attn_aggregated)
        
        for layer_idx in range(num_decoder_layers):
            current_layer_attn_weights = all_decoder_self_attn_aggregated[layer_idx]
            if current_layer_attn_weights is None: continue
            num_heads = current_layer_attn_weights.shape[1]

            for global_sample_idx in samples_to_plot_global_indices:
                sample_plots_dir = sample_plot_directories[global_sample_idx]
                if global_sample_idx >= current_layer_attn_weights.shape[0]:
                    print(f"Warning: Index {global_sample_idx} out of bounds for decoder self-attention layer {layer_idx+1} (max: {current_layer_attn_weights.shape[0]-1}). Skipping.")
                    continue
                if global_sample_idx >= len(all_dataset_formulas):
                    print(f"Warning: Index {global_sample_idx} out of bounds for formula list (max: {len(all_dataset_formulas)-1}). Skipping.")
                    continue
                    
                sample_formula = all_dataset_formulas[global_sample_idx]
                if (
                    all_dataset_battery_ids is None
                    or global_sample_idx >= len(all_dataset_battery_ids)
                ):
                    continue
                sample_battery_id = all_dataset_battery_ids[global_sample_idx]

                for head_idx in range(num_heads):
                    aggregated_attn_weights = current_layer_attn_weights[global_sample_idx, head_idx, :, :].cpu().numpy()
                    
                    title = f"Dec Self-Attn L{layer_idx+1}H{head_idx+1} - {_sample_plot_label(sample_battery_id)}"
                    filename = os.path.join(sample_plots_dir, f"decoder_self_attn_L{layer_idx+1}_H{head_idx+1}_S{global_sample_idx}_{timestamp}.png")
                    
                    current_decoder_tokens = aggregated_attn_weights.shape[0]
                    figsize_dec = (max(8, current_decoder_tokens * 0.8), max(8, current_decoder_tokens * 0.8))
                    annot_dec = True if current_decoder_tokens <= 25 else False
                    fmt_dec = ".3f" if current_decoder_tokens <= 25 else ""
                    fontsize_labels_dec = 10 if current_decoder_tokens <= 25 else 8

                    _plot_single_heatmap(aggregated_attn_weights, title, decoder_token_names, decoder_token_names, filename,
                                        annot=annot_dec, fmt=fmt_dec, figsize=figsize_dec, fontsize_labels=fontsize_labels_dec)

    if save_per_head_heatmaps and all_decoder_cross_attn_aggregated:
        num_decoder_layers = len(all_decoder_cross_attn_aggregated)
        
        for layer_idx in range(num_decoder_layers):
            current_layer_attn_weights = all_decoder_cross_attn_aggregated[layer_idx]
            if current_layer_attn_weights is None: continue 
            num_heads = current_layer_attn_weights.shape[1]

            for global_sample_idx in samples_to_plot_global_indices:
                sample_plots_dir = sample_plot_directories[global_sample_idx]
                if global_sample_idx >= current_layer_attn_weights.shape[0]:
                    print(f"Warning: Index {global_sample_idx} out of bounds for decoder cross-attention layer {layer_idx+1} (max: {current_layer_attn_weights.shape[0]-1}). Skipping.")
                    continue
                if global_sample_idx >= len(all_dataset_formulas):
                    print(f"Warning: Index {global_sample_idx} out of bounds for formula list (max: {len(all_dataset_formulas)-1}). Skipping.")
                    continue
                    
                sample_formula = all_dataset_formulas[global_sample_idx]
                if (
                    all_dataset_battery_ids is None
                    or global_sample_idx >= len(all_dataset_battery_ids)
                ):
                    continue
                sample_battery_id = all_dataset_battery_ids[global_sample_idx]
                encoder_labels = get_encoder_attention_labels(
                    sample_formula, n_elements, sample_battery_id
                )
                encoder_seq_len_plot = len(encoder_labels)

                if encoder_seq_len_plot == 0:
                    continue
                
                for head_idx in range(num_heads):
                    attn_weights_sample = current_layer_attn_weights[global_sample_idx, head_idx, :, :encoder_seq_len_plot].cpu().numpy()
                    aggregated_attn_weights = attn_weights_sample
                    
                    title = f"Dec Cross-Attn L{layer_idx+1}H{head_idx+1} - {_sample_plot_label(sample_battery_id)}"
                    filename = os.path.join(sample_plots_dir, f"decoder_cross_attn_L{layer_idx+1}_H{head_idx+1}_S{global_sample_idx}_{timestamp}.png")
                    
                    current_decoder_tokens = aggregated_attn_weights.shape[0]
                    current_encoder_seq_len = aggregated_attn_weights.shape[1]
                    
                    figsize_cross = (max(8, current_encoder_seq_len * 0.8), max(8, current_decoder_tokens * 0.8))
                    annot_cross, fmt_cross, fontsize_labels_cross = _encoder_axis_annotation_settings(
                        current_encoder_seq_len
                    )

                    _plot_single_heatmap(aggregated_attn_weights, title, encoder_labels, decoder_token_names, filename,
                                        annot=annot_cross, fmt=fmt_cross, figsize=figsize_cross, fontsize_labels=fontsize_labels_cross)

    for global_sample_idx in samples_to_plot_global_indices:
        sample_plots_dir = sample_plot_directories[global_sample_idx]
        # Validate index before accessing
        if global_sample_idx >= len(all_dataset_formulas):
            print(f"Warning: Index {global_sample_idx} out of bounds for formula list (max: {len(all_dataset_formulas)-1}). Skipping.")
            continue
        
        max_attn_idx = _max_attention_sample_index(
            all_encoder_self_attn_aggregated,
            all_decoder_self_attn_aggregated,
            all_decoder_cross_attn_aggregated,
        )
        if max_attn_idx is not None and global_sample_idx > max_attn_idx:
            print(f"Warning: Index {global_sample_idx} out of bounds for attention weight arrays (max: {max_attn_idx}). Skipping.")
            continue
        
        sample_formula = all_dataset_formulas[global_sample_idx]
        if (
            all_dataset_battery_ids is None
            or global_sample_idx >= len(all_dataset_battery_ids)
        ):
            continue
        sample_battery_id = all_dataset_battery_ids[global_sample_idx]

        sample_enc_self_attn_list = []
        for layer_idx in range(len(all_encoder_self_attn_aggregated)):
            if (all_encoder_self_attn_aggregated[layer_idx] is not None and 
                global_sample_idx < all_encoder_self_attn_aggregated[layer_idx].shape[0]):
                sample_enc_self_attn_list.append(all_encoder_self_attn_aggregated[layer_idx][global_sample_idx])
            else:
                sample_enc_self_attn_list.append(None)
        
        sample_dec_self_attn_list = []
        for layer_idx in range(len(all_decoder_self_attn_aggregated)):
            if (all_decoder_self_attn_aggregated[layer_idx] is not None and 
                global_sample_idx < all_decoder_self_attn_aggregated[layer_idx].shape[0]):
                sample_dec_self_attn_list.append(all_decoder_self_attn_aggregated[layer_idx][global_sample_idx])
            else:
                sample_dec_self_attn_list.append(None)
        
        sample_dec_cross_attn_list = []
        for layer_idx in range(len(all_decoder_cross_attn_aggregated)):
            if (all_decoder_cross_attn_aggregated[layer_idx] is not None and 
                global_sample_idx < all_decoder_cross_attn_aggregated[layer_idx].shape[0]):
                sample_dec_cross_attn_list.append(all_decoder_cross_attn_aggregated[layer_idx][global_sample_idx])
            else:
                sample_dec_cross_attn_list.append(None)
        
        _visualize_aggregated_attention_for_sample(
            sample_idx=global_sample_idx,
            sample_formula=sample_formula,
            sample_battery_id=sample_battery_id,
            encoder_self_attn_list=sample_enc_self_attn_list,
            decoder_self_attn_list=sample_dec_self_attn_list,
            decoder_cross_attn_list=sample_dec_cross_attn_list,
            n_elements=n_elements,
            decoder_token_names=decoder_token_names,
            plots_dir=sample_plots_dir,
            descriptor_type=descriptor_type,
            timestamp=timestamp,
            plot_single_heatmap_func=_plot_single_heatmap
        )

def _visualize_aggregated_attention_for_sample(
    sample_idx: int,
    sample_formula: str,
    sample_battery_id,
    encoder_self_attn_list: list, 
    decoder_self_attn_list: list, 
    decoder_cross_attn_list: list, 
    n_elements: int,
    decoder_token_names: list,
    plots_dir: str,
    descriptor_type: str,
    timestamp: str,
    plot_single_heatmap_func: callable 
):
    sample_label = _sample_plot_label(sample_battery_id)
    print(f"Plotting aggregated attention heatmap for sample {sample_label} (index: {sample_idx})...")

    encoder_labels = get_encoder_attention_labels(
        sample_formula, n_elements, sample_battery_id
    )
    encoder_seq_len_plot = len(encoder_labels)

    if encoder_self_attn_list and any(attn is not None for attn in encoder_self_attn_list):
        valid_attns = [attn for attn in encoder_self_attn_list if attn is not None]
        if valid_attns:
            stacked_attns = torch.stack(valid_attns, dim=0)
            aggregated_enc_self_attn = stacked_attns.mean(dim=(0, 1))[:encoder_seq_len_plot, :encoder_seq_len_plot].cpu().numpy()
            
            if aggregated_enc_self_attn.shape[0] > 0 and aggregated_enc_self_attn.shape[1] > 0:
                title = f"Aggregated Enc Self-Attn - {sample_label}"
                filename = os.path.join(plots_dir, f"aggregated_encoder_self_attn_S{sample_idx}_{descriptor_type}_{timestamp}.png")
                figsize_enc = (max(4, encoder_seq_len_plot * 0.8), max(4, encoder_seq_len_plot * 0.8))
                annot_enc, fmt_enc, fontsize_labels_enc = _encoder_axis_annotation_settings(encoder_seq_len_plot)
                plot_single_heatmap_func(aggregated_enc_self_attn, title, encoder_labels, encoder_labels, filename, 
                                        annot=annot_enc, fmt=fmt_enc, figsize=figsize_enc, fontsize_labels=fontsize_labels_enc)
            else:
                print(f"Warning: Aggregated encoder self-attention weights for sample {sample_label} have zero dimensions, skipping plot.")
        else:
            print(f"Warning: No valid encoder self-attention weights available for aggregation for sample {sample_label}.")

    if decoder_self_attn_list and any(attn is not None for attn in decoder_self_attn_list):
        valid_attns = [attn for attn in decoder_self_attn_list if attn is not None]
        if valid_attns:
            stacked_attns = torch.stack(valid_attns, dim=0)
            aggregated_dec_self_attn = stacked_attns.mean(dim=(0, 1)).cpu().numpy()
            
            if aggregated_dec_self_attn.shape[0] > 0 and aggregated_dec_self_attn.shape[1] > 0:
                title = f"Aggregated Dec Self-Attn - {sample_label}"
                filename = os.path.join(plots_dir, f"aggregated_decoder_self_attn_S{sample_idx}_{descriptor_type}_{timestamp}.png")
                current_decoder_tokens = aggregated_dec_self_attn.shape[0]
                figsize_dec = (max(8, current_decoder_tokens * 0.8), max(8, current_decoder_tokens * 0.8))
                annot_dec = True if current_decoder_tokens <= 25 else False
                fmt_dec = ".3f" if current_decoder_tokens <= 25 else ""
                fontsize_labels_dec = 10 if current_decoder_tokens <= 25 else 8
                plot_single_heatmap_func(aggregated_dec_self_attn, title, decoder_token_names, decoder_token_names, filename,
                                        annot=annot_dec, fmt=fmt_dec, figsize=figsize_dec, fontsize_labels=fontsize_labels_dec)
            else:
                print(f"Warning: Aggregated decoder self-attention weights for sample {sample_label} have zero dimensions, skipping plot.")
        else:
            print(f"Warning: No valid decoder self-attention weights available for aggregation for sample {sample_label}.")

    if decoder_cross_attn_list and any(attn is not None for attn in decoder_cross_attn_list):
        valid_attns = [attn for attn in decoder_cross_attn_list if attn is not None]
        if valid_attns:
            stacked_attns = torch.stack(valid_attns, dim=0)
            aggregated_dec_cross_attn = stacked_attns.mean(dim=(0, 1))[:, :encoder_seq_len_plot].cpu().numpy()
            
            if aggregated_dec_cross_attn.shape[0] > 0 and aggregated_dec_cross_attn.shape[1] > 0:
                title = f"Aggregated Dec Cross-Attn - {sample_label}"
                filename = os.path.join(plots_dir, f"aggregated_decoder_cross_attn_S{sample_idx}_{descriptor_type}_{timestamp}.png")
                current_decoder_tokens = aggregated_dec_cross_attn.shape[0]
                current_encoder_seq_len = aggregated_dec_cross_attn.shape[1]
                figsize_cross = (max(8, current_encoder_seq_len * 0.8), max(8, current_decoder_tokens * 0.8))
                annot_cross, fmt_cross, fontsize_labels_cross = _encoder_axis_annotation_settings(
                    current_encoder_seq_len
                )
                plot_single_heatmap_func(aggregated_dec_cross_attn, title, encoder_labels, decoder_token_names, filename,
                                        annot=annot_cross, fmt=fmt_cross, figsize=figsize_cross, fontsize_labels=fontsize_labels_cross)
            else:
                print(f"Warning: Aggregated decoder cross-attention weights for sample {sample_label} have zero dimensions, skipping plot.")
        else:
            print(f"Warning: No valid decoder cross-attention weights available for aggregation for sample {sample_label}.")
