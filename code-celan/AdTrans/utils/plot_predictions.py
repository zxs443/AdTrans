import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from matplotlib.lines import Line2D
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.visualization_config import VISUALIZATION_CONFIG


plt.rcParams['font.family'] = VISUALIZATION_CONFIG['font_family']


def _create_joint_grid(plot_df, hue=None, palette=None):
    kwargs = {'data': plot_df, 'x': 'True', 'y': 'Predicted'}
    if hue is not None:
        kwargs['hue'] = hue
        if palette is not None:
            kwargs['palette'] = palette
    try:
        return sns.JointGrid(**kwargs, height=8)
    except TypeError:
        plain_kwargs = {k: v for k, v in kwargs.items() if k not in ('hue', 'palette')}
        try:
            return sns.JointGrid(**plain_kwargs, height=8)
        except TypeError:
            return sns.JointGrid(x='True', y='Predicted', data=plot_df, size=8)


def _plot_joint_scatter(grid, *, color, cfg, plot_df=None, ion_palette=None):
    edge_w = cfg.get('scatter_edge_width', 0.35)
    if plot_df is not None and ion_palette is not None and 'Ion' in plot_df.columns:
        ax = grid.ax_joint
        for ion in _ordered_ions(ion_palette):
            subset = plot_df[plot_df['Ion'] == ion]
            ax.scatter(
                subset['True'],
                subset['Predicted'],
                c=ion_palette.get(ion, '#a8a8a8'),
                alpha=cfg['scatter_alpha'],
                edgecolors='black',
                linewidths=edge_w,
                s=cfg['scatter_size'],
                label=ion,
                zorder=2,
            )
        return

    if hasattr(grid, 'plot_joint'):
        if hasattr(sns, 'scatterplot'):
            grid.plot_joint(
                sns.scatterplot,
                color=color,
                alpha=cfg['scatter_alpha'],
                edgecolor='black',
                linewidth=edge_w,
                s=cfg['scatter_size'],
            )
        else:
            grid.plot_joint(
                plt.scatter,
                color=color,
                alpha=cfg['scatter_alpha'],
                edgecolors='black',
                linewidths=edge_w,
                s=cfg['scatter_size'],
            )
        return

    grid.plot(
        plt.scatter,
        color=color,
        alpha=cfg['scatter_alpha'],
        edgecolors='black',
        linewidths=edge_w,
        s=cfg['scatter_size'],
    )


def _draw_hatched_marginal_hist(ax, values, *, bins, facecolor, cfg, orientation='vertical'):
    counts, bin_edges = np.histogram(np.asarray(values, dtype=float), bins=bins)
    bar_size = (bin_edges[1] - bin_edges[0]) * 0.9
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_kwargs = dict(
        facecolor=facecolor,
        edgecolor=cfg.get('parity_marginal_edgecolor', 'black'),
        hatch=cfg.get('parity_marginal_hatch', '//'),
        alpha=cfg['hist_alpha'],
        linewidth=cfg.get('parity_marginal_edge_width', 0.5),
    )
    if orientation == 'vertical':
        ax.bar(centers, counts, width=bar_size, align='center', **bar_kwargs)
    else:
        ax.barh(centers, counts, height=bar_size, align='center', **bar_kwargs)


def _plot_joint_marginals(grid, *, color, cfg, plot_df, axis_lo, axis_hi):
    bins = np.linspace(axis_lo, axis_hi, 24)
    ax_mx = getattr(grid, 'ax_marg_x', None)
    ax_my = getattr(grid, 'ax_marg_y', None)
    if ax_mx is not None:
        _draw_hatched_marginal_hist(
            ax_mx,
            plot_df['True'],
            bins=bins,
            facecolor=color,
            cfg=cfg,
            orientation='vertical',
        )
    if ax_my is not None:
        _draw_hatched_marginal_hist(
            ax_my,
            plot_df['Predicted'],
            bins=bins,
            facecolor=color,
            cfg=cfg,
            orientation='horizontal',
        )


def ion_type_from_battery_id(battery_id) -> str:
    """Extract working-ion symbol from battery_id suffix (visualization only, not a model feature)."""
    text = str(battery_id).strip()
    if '_' in text:
        return text.rsplit('_', 1)[-1]
    return text


def _ordered_ions(ion_palette: dict) -> List[str]:
    preferred = VISUALIZATION_CONFIG.get('ion_type_order') or []
    ordered = [ion for ion in preferred if ion in ion_palette]
    remaining = sorted(set(ion_palette.keys()) - set(ordered))
    return ordered + remaining


def _build_ion_palette(ion_types) -> dict:
    configured = VISUALIZATION_CONFIG.get('ion_type_palette') or {}
    preferred_order = VISUALIZATION_CONFIG.get('ion_type_order') or []
    fallback_colors = [configured[ion] for ion in preferred_order if ion in configured]
    if not fallback_colors:
        fallback_colors = list(configured.values())

    unique_ions = sorted({str(ion) for ion in ion_types})
    palette = {}
    next_color_idx = 0
    for ion in unique_ions:
        if ion in configured:
            palette[ion] = configured[ion]
        elif fallback_colors:
            palette[ion] = fallback_colors[next_color_idx % len(fallback_colors)]
            next_color_idx += 1
        else:
            tab10 = sns.color_palette('tab10', n_colors=max(len(unique_ions), 1))
            palette[ion] = tab10[next_color_idx % len(tab10)]
            next_color_idx += 1
    return palette


def _add_ion_legend(ax, ion_palette, cfg):
    edge_w = cfg.get('scatter_edge_width', 0.35)
    handles = [
        Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            markerfacecolor=ion_palette[ion],
            markeredgecolor='black',
            markeredgewidth=edge_w,
            markersize=8,
            label=ion,
        )
        for ion in _ordered_ions(ion_palette)
    ]
    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 0.005),
        loc='lower right',
        fontsize=cfg['font_size_legend'],
        handletextpad=0.05,
        borderpad=0.2,
        framealpha=float(cfg.get('legend_framealpha', 0.82)),
    )


def build_parity_model_info(descriptor_type: Optional[str]) -> Optional[str]:
    """Build top-left parity-plot caption from encoder/decoder descriptor pair."""
    if not descriptor_type or '_' not in descriptor_type:
        return None

    encoder_type, decoder_type = descriptor_type.split('_', 1)
    return (
        f'Encoder encoding: {encoder_type}\n'
        f'Decoder encoding: {decoder_type}'
    )


def _align_marginals_to_joint(grid) -> None:
    """Align marginal axes to joint plot bbox."""
    ax_j = grid.ax_joint
    fig = ax_j.figure
    fig.canvas.draw()
    pos_j = ax_j.get_position()
    top_j = pos_j.y0 + pos_j.height
    right_j = pos_j.x0 + pos_j.width

    ax_mx = getattr(grid, 'ax_marg_x', None)
    ax_my = getattr(grid, 'ax_marg_y', None)
    gap_h = max(0.0, ax_my.get_position().x0 - right_j) if ax_my is not None else 0.0

    if ax_mx is not None:
        pmx = ax_mx.get_position()
        ax_mx.set_position([pos_j.x0, top_j + gap_h, pos_j.width, pmx.height])
    if ax_my is not None:
        pmy = ax_my.get_position()
        ax_my.set_position([pmy.x0, pos_j.y0, pmy.width, pos_j.height])


def _metrics_from_arrays(true_vals: np.ndarray, pred_vals: np.ndarray) -> dict:
    return {
        'R2': float(r2_score(true_vals, pred_vals)),
        'MAE': float(mean_absolute_error(true_vals, pred_vals)),
        'RMSE': float(np.sqrt(mean_squared_error(true_vals, pred_vals))),
    }


def plot_single_parity(
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    *,
    subset_name: str,
    model_info: Optional[str],
    save_path: str,
    battery_ids=None,
) -> None:
    """JointGrid parity plot for a single dataset split (Train / Validation / Test)."""
    cfg = VISUALIZATION_CONFIG
    subset_metrics = _metrics_from_arrays(true_vals, pred_vals)

    plot_df = pd.DataFrame({
        'True': true_vals,
        'Predicted': pred_vals,
        'Data Set': subset_name,
    })
    ion_palette = None
    if battery_ids is not None:
        plot_df['Ion'] = [ion_type_from_battery_id(bid) for bid in battery_ids]
        ion_palette = _build_ion_palette(plot_df['Ion'])

    color = cfg['palette'].get(subset_name, '#a8a8a8')
    lo = float(cfg.get('parity_axis_lo', -1.5))
    hi = float(cfg.get('parity_axis_hi', 7.0))
    ticks = list(cfg.get('parity_axis_ticks', (0, 2, 4, 6)))

    plt.figure(figsize=cfg['figsize'], dpi=cfg['dpi'])
    g = _create_joint_grid(plot_df)
    use_ion_hue = ion_palette is not None
    if use_ion_hue:
        _plot_joint_scatter(
            g,
            color=color,
            cfg=cfg,
            plot_df=plot_df,
            ion_palette=ion_palette,
        )
    else:
        _plot_joint_scatter(g, color=color, cfg=cfg)

    marginal_color = cfg.get('parity_marginal_color', '#ffffff')
    _plot_joint_marginals(
        g,
        color=marginal_color,
        cfg=cfg,
        plot_df=plot_df,
        axis_lo=lo,
        axis_hi=hi,
    )

    ax = g.ax_joint
    line_range = np.array([lo, hi])
    ax.plot(
        line_range,
        line_range,
        '--',
        color='gray',
        linewidth=2,
        label='Perfect Prediction (y=x)',
        alpha=0.9,
        zorder=0,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect('equal', adjustable='box')

    if hasattr(g, 'ax_marg_x') and g.ax_marg_x is not None:
        g.ax_marg_x.set_xlim(lo, hi)
        g.ax_marg_x.set_xticks(ticks)
    if hasattr(g, 'ax_marg_y') and g.ax_marg_y is not None:
        g.ax_marg_y.set_ylim(lo, hi)
        g.ax_marg_y.set_yticks(ticks)

    metrics_block = (
        f'{subset_name} R²={subset_metrics["R2"]:.3f}\n'
        f'MAE={subset_metrics["MAE"]:.3f}  RMSE={subset_metrics["RMSE"]:.3f}'
    )
    info_text = f'{model_info}\n{metrics_block}' if model_info else metrics_block

    ax.text(
        0.03,
        0.995,
        info_text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=cfg['font_size_legend'],
        linespacing=1.5,
    )

    ax.set_xlabel('True average voltage (V)', fontsize=cfg.get('parity_font_size_label', cfg['font_size_label']))
    ax.set_ylabel('Predicted average voltage (V)', fontsize=cfg.get('parity_font_size_label', cfg['font_size_label']))

    tick_fontsize = cfg.get('parity_font_size_tick', cfg.get('font_size_tick', cfg['font_size_label']))
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    if hasattr(g, 'ax_marg_x') and g.ax_marg_x is not None:
        g.ax_marg_x.tick_params(axis='x', which='major', labelsize=tick_fontsize)
    if hasattr(g, 'ax_marg_y') and g.ax_marg_y is not None:
        g.ax_marg_y.tick_params(axis='y', which='major', labelsize=tick_fontsize)

    if use_ion_hue and ion_palette is not None:
        _add_ion_legend(ax, ion_palette, cfg)

    plt.subplots_adjust(right=0.85, top=0.9)
    _align_marginals_to_joint(g)

    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=cfg['dpi'])
    plt.close()


def visualize_predictions(
    train_true,
    train_pred,
    val_true,
    val_pred,
    test_true,
    test_pred,
    train_formulas,
    val_formulas,
    test_formulas,
    train_battery_ids,
    val_battery_ids,
    test_battery_ids,
    n_elements,
    metrics,
    base_dir='checkpoints',
    descriptor_type=None,
    decoder_token_names=None,
    targets_mean=0.0,
    targets_std=1.0,
):
    """Parity plots per split + predictions/metrics CSV."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    train_true_unscaled = train_true * targets_std + targets_mean
    train_pred_unscaled = train_pred * targets_std + targets_mean
    val_true_unscaled = val_true * targets_std + targets_mean
    val_pred_unscaled = val_pred * targets_std + targets_mean
    test_true_unscaled = test_true * targets_std + targets_mean
    test_pred_unscaled = test_pred * targets_std + targets_mean

    train_predictions_df_full = pd.DataFrame({
        'battery_id': train_battery_ids,
        'True': train_true_unscaled,
        'Predicted': train_pred_unscaled,
        'Formula': train_formulas,
        'Data Set': 'Train',
    })
    val_predictions_df_full = pd.DataFrame({
        'battery_id': val_battery_ids,
        'True': val_true_unscaled,
        'Predicted': val_pred_unscaled,
        'Formula': val_formulas,
        'Data Set': 'Validation',
    })
    test_predictions_df_full = pd.DataFrame({
        'battery_id': test_battery_ids,
        'True': test_true_unscaled,
        'Predicted': test_pred_unscaled,
        'Formula': test_formulas,
        'Data Set': 'Test',
    })

    predictions_df_to_save = pd.concat([
        train_predictions_df_full,
        val_predictions_df_full,
        test_predictions_df_full,
    ])

    model_info = build_parity_model_info(descriptor_type)
    split_data = [
        ('Train', train_true_unscaled, train_pred_unscaled, train_battery_ids),
        ('Validation', val_true_unscaled, val_pred_unscaled, val_battery_ids),
        ('Test', test_true_unscaled, test_pred_unscaled, test_battery_ids),
    ]
    for subset_name, true_u, pred_u, battery_ids in split_data:
        save_path = os.path.join(
            plots_dir,
            f'parity_{subset_name.lower()}_{descriptor_type}_{timestamp}.png',
        )
        plot_single_parity(
            true_u,
            pred_u,
            subset_name=subset_name,
            model_info=model_info,
            save_path=save_path,
            battery_ids=battery_ids,
        )
        print(f'Parity plot saved to: {save_path}')

    predictions_df_to_save.to_csv(
        os.path.join(base_dir, f'predictions_{descriptor_type}_{timestamp}.csv'),
        index=False,
    )

    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(os.path.join(base_dir, f'metrics_{descriptor_type}_{timestamp}.csv'))

    return metrics, timestamp
