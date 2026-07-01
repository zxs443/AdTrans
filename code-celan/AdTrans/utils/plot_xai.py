import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from datetime import datetime
import numpy as np
from data.structure_placement import normalize_element_importance_excluding_structure
from utils.visualization_config import VISUALIZATION_CONFIG

plt.rcParams['font.family'] = VISUALIZATION_CONFIG['font_family']


def _savefig_path(path: str) -> str:
    abs_path = os.path.abspath(path)
    if os.name == 'nt' and len(abs_path) >= 260 and not abs_path.startswith('\\\\?\\'):
        if abs_path.startswith('\\\\'):
            return '\\\\?\\UNC\\' + abs_path[2:]
        return '\\\\?\\' + abs_path
    return abs_path


PERIODIC_TABLE_LAYOUT = [
    ["H", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "He"],
    ["Li", "Be", "", "", "", "", "", "", "", "", "", "", "B", "C", "N", "O", "F", "Ne"],
    ["Na", "Mg", "", "", "", "", "", "", "", "", "", "", "Al", "Si", "P", "S", "Cl", "Ar"],
    ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
     "Ga", "Ge", "As", "Se", "Br", "Kr"],
    ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
     "In", "Sn", "Sb", "Te", "I", "Xe"],
    ["Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
     "Tl", "Pb", "Bi", "Po", "At", "Rn"],
    ["Fr", "Ra", "Ac", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
     "Nh", "Fl", "Mc", "Lv", "Ts", "Og"],
]
LANTHANIDES = ["Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
ACTINIDES = ["Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]


PERIODIC_TABLE_IMPORTANCE_FLOOR = 1e-4


def _all_periodic_table_elements() -> set[str]:
    elems: set[str] = set()
    for row in PERIODIC_TABLE_LAYOUT:
        elems.update(elem for elem in row if elem)
    elems.update(LANTHANIDES)
    elems.update(ACTINIDES)
    return elems


def _plotted_element_values(importance_dict: dict) -> np.ndarray:
    drawable = _all_periodic_table_elements()
    values = [
        float(importance_dict[elem])
        for elem in drawable
        if elem in importance_dict and np.isfinite(importance_dict[elem]) and importance_dict[elem] > 0
    ]
    return np.asarray(values, dtype=float)


def _periodic_table_color_norm(values: np.ndarray):
    floor = PERIODIC_TABLE_IMPORTANCE_FLOOR
    positive = values[values > 0]
    if len(positive) == 0:
        positive = np.array([floor, 1.0])
    vmin = floor
    vmax = max(float(positive.max()), vmin * 1.01)
    span_ratio = vmax / vmin
    if span_ratio >= 20:
        return mcolors.LogNorm(vmin=vmin, vmax=vmax), True
    return mcolors.Normalize(vmin=vmin, vmax=vmax), False


def _set_periodic_table_colorbar_ticks(cbar, vmin: float, vmax: float, use_log: bool) -> None:
    floor = PERIODIC_TABLE_IMPORTANCE_FLOOR
    vmin = max(vmin, floor)
    if use_log:
        exp_min = max(int(np.floor(np.log10(vmin))), int(np.floor(np.log10(floor))))
        exp_max = int(np.ceil(np.log10(vmax)))
        ticks = [10.0 ** exp for exp in range(exp_min, exp_max + 1)]
        ticks = [t for t in ticks if floor <= t <= vmax]
        if floor not in ticks:
            ticks.insert(0, floor)

    else:
        ticks = [t for t in np.linspace(vmin, vmax, num=5) if t >= floor]
        if floor not in ticks:
            ticks.insert(0, floor)
        if not ticks or abs(ticks[-1] - vmax) / max(vmax, 1e-12) > 0.05:
            ticks.append(vmax)
    cbar.set_ticks(ticks)
    cbar.ax.set_xlim(vmin, vmax)


def _format_importance_value(val):
    if val < PERIODIC_TABLE_IMPORTANCE_FLOOR:
        return "<0.0001"
    return f"{val:.4f}"


def _color_value_for_periodic_table(val: float) -> float:
    if val <= 0:
        return PERIODIC_TABLE_IMPORTANCE_FLOOR
    return max(float(val), PERIODIC_TABLE_IMPORTANCE_FLOOR)


def _build_histogram_colors(n):
    color_hi = VISUALIZATION_CONFIG.get('xai_histogram_color_hi', '#e9b539')
    color_lo = VISUALIZATION_CONFIG.get('xai_histogram_color_lo', '#d0d3d4')
    rgb_hi = np.array(mcolors.to_rgb(color_hi))
    rgb_lo = np.array(mcolors.to_rgb(color_lo))
    if n <= 0:
        return []
    return [
        mcolors.to_hex((1 - t) * rgb_hi + t * rgb_lo)
        for t in np.linspace(0, 1, n)
    ]


def _apply_histogram_style(ax):
    axis_lw = VISUALIZATION_CONFIG.get('xai_histogram_axis_linewidth', 2.4)
    label_fontsize = VISUALIZATION_CONFIG.get('xai_histogram_label_fontsize', 28)
    tick_fontsize = VISUALIZATION_CONFIG.get('xai_histogram_tick_fontsize', 28)

    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(axis_lw)
    ax.spines["bottom"].set_linewidth(axis_lw)
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.set_xlabel("Importance", fontweight="bold", labelpad=10, fontsize=label_fontsize)


def _periodic_table_inset_colorbar_ax(ax):
    """Place colorbar in the s-block / p-block gap (data coordinates)."""
    x0, y0, width, height = VISUALIZATION_CONFIG.get(
        'xai_periodic_table_colorbar_data_rect',
        [2.5, 8.20, 9.0, 0.30],
    )
    xlim_lo, xlim_hi = ax.get_xlim()
    ylim_lo, ylim_hi = ax.get_ylim()
    rel_x = (x0 - xlim_lo) / (xlim_hi - xlim_lo)
    rel_y = (y0 - ylim_lo) / (ylim_hi - ylim_lo)
    rel_w = width / (xlim_hi - xlim_lo)
    rel_h = height / (ylim_hi - ylim_lo)
    return ax.inset_axes([rel_x, rel_y, rel_w, rel_h])


def _plot_periodic_table_base(element_importance_df: pd.DataFrame, save_path: str, title: str, cbar_label: str):
    """Periodic table heatmap (element importance)."""
    element_importance_df = normalize_element_importance_excluding_structure(element_importance_df)
    importance_dict = dict(zip(element_importance_df["Element"], element_importance_df["Importance"]))

    plot_values = _plotted_element_values(importance_dict)
    norm, use_log = _periodic_table_color_norm(plot_values)
    vmin = float(norm.vmin)
    vmax = float(norm.vmax)

    cmap_name = VISUALIZATION_CONFIG.get('xai_periodic_table_cmap', 'viridis')
    cmap = plt.get_cmap(cmap_name)

    cell_size = VISUALIZATION_CONFIG.get('xai_periodic_table_cell_size', 1.0)
    gap = VISUALIZATION_CONFIG.get('xai_periodic_table_cell_gap', 0.08)
    box_size = cell_size - gap
    missing_color = VISUALIZATION_CONFIG.get('xai_periodic_table_missing_color', '#eaeaea')
    cell_alpha = VISUALIZATION_CONFIG.get('xai_periodic_table_cell_alpha', 0.7)
    fontsize_symbol = VISUALIZATION_CONFIG.get('xai_periodic_table_fontsize_symbol', 22)
    fontsize_value = VISUALIZATION_CONFIG.get('xai_periodic_table_fontsize_value', 14)
    title_fontsize = VISUALIZATION_CONFIG.get('xai_periodic_table_title_fontsize', 18)

    fig, ax = plt.subplots(
        figsize=(
            VISUALIZATION_CONFIG['periodic_table_figsize_width'],
            VISUALIZATION_CONFIG['periodic_table_figsize_height'],
        )
    )

    def draw_cell(x, y, elem):
        val = importance_dict.get(elem, np.nan)
        if np.isnan(val):
            color = missing_color
        else:
            color = cmap(norm(_color_value_for_periodic_table(val)))

        rect = FancyBboxPatch(
            (x + gap / 2, y + gap / 2),
            box_size,
            box_size,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=0,
            facecolor=color,
            alpha=cell_alpha,
        )
        ax.add_patch(rect)

        ax.text(
            x + 0.5, y + 0.68,
            elem,
            ha='center', va='center',
            fontsize=fontsize_symbol, fontweight='bold',
        )

        if not np.isnan(val):
            ax.text(
                x + 0.5, y + 0.30,
                _format_importance_value(val),
                ha='center', va='center',
                fontsize=fontsize_value, fontweight='bold',
            )

    for i, row in enumerate(PERIODIC_TABLE_LAYOUT):
        if all(elem == "" for elem in row):
            continue
        for j, elem in enumerate(row):
            if elem != "":
                y = len(PERIODIC_TABLE_LAYOUT) - i + 2
                draw_cell(j, y, elem)

    for j, elem in enumerate(LANTHANIDES):
        draw_cell(j + 2, 2, elem)

    for j, elem in enumerate(ACTINIDES):
        draw_cell(j + 2, 1, elem)

    ax.set_xlim(0, 18)
    ax.set_ylim(1, 10)
    ax.axis('off')

    cbar_data = VISUALIZATION_CONFIG.get(
        'xai_periodic_table_colorbar_data_rect',
        [2.5, 8.20, 9.0, 0.30],
    )
    title_gap = VISUALIZATION_CONFIG.get('xai_periodic_table_title_gap_above_cbar', 0.05)
    title_x = cbar_data[0] + cbar_data[2] / 2.0
    title_y = cbar_data[1] + cbar_data[3] + title_gap
    ax.text(
        title_x,
        title_y,
        title,
        ha='center',
        va='bottom',
        fontsize=title_fontsize,
        fontweight='bold',
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = _periodic_table_inset_colorbar_ax(ax)
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
    _set_periodic_table_colorbar_ticks(cbar, vmin, vmax, use_log)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=VISUALIZATION_CONFIG.get('xai_periodic_table_colorbar_ticksize', 16))
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')

    plt.savefig(
        _savefig_path(save_path),
        dpi=VISUALIZATION_CONFIG['dpi'],
        bbox_inches='tight',
        pad_inches=0.08,
        format='png',
    )
    plt.close()
    print(f"Element importance periodic table saved to: {save_path}")


def _build_periodic_table_title(descriptor_type: str, method: str = 'attention') -> str:
    if method == 'integrated_gradients':
        return "Element integrated gradients scores"
    if method == 'feature_ablation':
        return "Element feature ablation scores"
    if 'cross_attention' in descriptor_type:
        return "Element cross-attention scores"
    if 'self_attention' in descriptor_type:
        return "Element self-attention scores"
    return "Element attention scores"


def _resolve_histogram_top_k(token_importance_df: pd.DataFrame, top_k: int | None) -> int:
    if top_k is None:
        top_k = VISUALIZATION_CONFIG.get('xai_histogram_top_k_default', 15)
    if token_importance_df is None or token_importance_df.empty:
        return int(top_k)
    return min(int(top_k), len(token_importance_df))


def _apply_top_k_to_title(title: str, top_k: int) -> str:
    return re.sub(r'Top-\d+', f'Top-{top_k}', title, count=1)


def _build_histogram_title(descriptor_type: str, top_k: int, method: str = 'attention') -> str:
    title_fontsize = VISUALIZATION_CONFIG.get('xai_histogram_title_fontsize', 28)
    if method == 'integrated_gradients':
        title = f"Top-{top_k} important features based on integrated gradients"
    elif method == 'feature_ablation':
        title = f"Top-{top_k} important features based on feature ablation"
    elif 'cross_attention' in descriptor_type:
        title = f"Top-{top_k} important features based on cross attention"
    elif 'self_attention' in descriptor_type:
        title = f"Top-{top_k} important features based on self attention"
    else:
        title = f"Top-{top_k} important features based on pooling attention"
    return title, title_fontsize


def plot_periodic_table_importance(element_importance_df: pd.DataFrame, save_dir: str, descriptor_type: str):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    save_path = os.path.join(plots_dir, f'element_importance_periodic_table_{descriptor_type}_{timestamp}.png')
    title = _build_periodic_table_title(descriptor_type)
    _plot_periodic_table_base(element_importance_df, save_path, title, cbar_label='')


def _plot_histogram_base(
    token_importance_df: pd.DataFrame,
    save_path: str,
    title: str,
    xlabel: str,
    top_k: int = None,
):

    display_k = _resolve_histogram_top_k(token_importance_df, top_k)
    title = _apply_top_k_to_title(title, display_k)

    df = token_importance_df.sort_values("Importance", ascending=False).head(display_k).reset_index(drop=True)
    if df.empty:
        print(f"Warning: No data to plot histogram at {save_path}")
        return

    title_fontsize = VISUALIZATION_CONFIG.get('xai_histogram_title_fontsize', 28)
    fig, ax = plt.subplots(
        figsize=(
            VISUALIZATION_CONFIG['decoder_token_hist_figsize_width'],
            VISUALIZATION_CONFIG['decoder_token_hist_figsize_height'],
        )
    )

    colors = _build_histogram_colors(len(df))
    bars = ax.barh(
        np.arange(len(df)),
        df["Importance"],
        color=colors,
        edgecolor="white",
        linewidth=1,
    )
    ax.invert_yaxis()

    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["Property"], fontweight="bold")
    _apply_histogram_style(ax)
    ax.set_title(title, fontweight="bold", pad=1, fontsize=title_fontsize)

    plt.tight_layout()
    plt.savefig(
        _savefig_path(save_path),
        dpi=VISUALIZATION_CONFIG['dpi'],
        bbox_inches="tight",
        facecolor="white",
        format='png',
    )
    plt.close()
    print(f"Decoder property importance histogram saved to: {save_path}")


def plot_decoder_token_importance(
    token_importance_df: pd.DataFrame,
    save_dir: str,
    descriptor_type: str,
    top_k: int = None,
):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    if top_k is None:
        top_k = VISUALIZATION_CONFIG.get('xai_histogram_top_k_default', 15)

    save_path = os.path.join(plots_dir, f'decoder_token_importance_histogram_{descriptor_type}_{timestamp}.png')
    title, _ = _build_histogram_title(descriptor_type, top_k)
    _plot_histogram_base(token_importance_df, save_path, title, xlabel='Importance', top_k=top_k)


def plot_integrated_gradients_importance(
    element_importance_df: pd.DataFrame,
    token_importance_df: pd.DataFrame,
    save_dir: str,
    descriptor_type: str,
    plot_type: str,
    top_k: int = None,
):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    if plot_type == 'periodic_table':
        save_path = os.path.join(plots_dir, f'ig_element_importance_periodic_table_{descriptor_type}_{timestamp}.png')
        title = _build_periodic_table_title(descriptor_type, method='integrated_gradients')
        _plot_periodic_table_base(element_importance_df, save_path, title, cbar_label='')

    elif plot_type == 'histogram':
        save_path = os.path.join(plots_dir, f'ig_decoder_token_importance_histogram_{descriptor_type}_{timestamp}.png')
        if top_k is None:
            top_k = VISUALIZATION_CONFIG.get('xai_histogram_top_k_default', 15)
        title, _ = _build_histogram_title(descriptor_type, top_k, method='integrated_gradients')
        _plot_histogram_base(token_importance_df, save_path, title, 'Importance', top_k)
    else:
        print(f"Warning: Unsupported plot_type '{plot_type}'.")


def plot_feature_ablation_importance(
    element_importance_df: pd.DataFrame,
    token_importance_df: pd.DataFrame,
    save_dir: str,
    descriptor_type: str,
    plot_type: str,
    top_k: int = None,
):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    if plot_type == 'periodic_table':
        save_path = os.path.join(plots_dir, f'fa_element_importance_periodic_table_{descriptor_type}_{timestamp}.png')
        title = _build_periodic_table_title(descriptor_type, method='feature_ablation')
        _plot_periodic_table_base(element_importance_df, save_path, title, cbar_label='')

    elif plot_type == 'histogram':
        save_path = os.path.join(plots_dir, f'fa_decoder_token_importance_histogram_{descriptor_type}_{timestamp}.png')
        if top_k is None:
            top_k = VISUALIZATION_CONFIG.get('xai_histogram_top_k_default', 15)
        title, _ = _build_histogram_title(descriptor_type, top_k, method='feature_ablation')
        _plot_histogram_base(token_importance_df, save_path, title, 'Importance', top_k)
    else:
        print(f"Warning: Unsupported plot_type '{plot_type}'.")
