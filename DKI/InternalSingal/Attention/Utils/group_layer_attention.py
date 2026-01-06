import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Utils import sample_layer_attention
from Utils.sample_layer_attention import compute_line_attention_for_sample
# 5.1-5.2 Plot heatmaps

def plot_layer_line_heatmap(attn_mat, title, out_path,
                            cmap="Blues", dpi=300):
    """
    attn_mat: [num_layers, num_lines]
    y: layer, x: line index
    """
    attn_mat = np.asarray(attn_mat)
    L, N = attn_mat.shape

    # Simple adaptive: width/font/tick density
    fig_w = min(max(8, N * 0.35), 20)
    fig_h = min(max(5, L * 0.35), 12)
    step = 1 if N <= 30 else (2 if N <= 60 else (5 if N <= 120 else 10))
    # step = 1
    xticks = np.arange(0, N, step)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(attn_mat, aspect="auto", interpolation="nearest", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=20)            # Right side tick label font size
    # cbar.set_label("Attention Score", fontsize=20) 
    ax.set_yticks(np.arange(L))
    ax.set_yticklabels([f"L{i}" for i in range(L)], fontsize=20)

    ax.set_xticks(xticks)
    ax.set_xticklabels([str(i+1) for i in xticks], ha="center", fontsize=20)
# , rotation=60
    ax.set_xlabel("Value Positions (1=earliest, 32=latest)", fontsize=20)
    ax.set_ylabel("Layer", fontsize=22)
    ax.set_title(title, fontsize=22)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)



def analyze_group_attention(samples, model, tokenizer, group_name: str, out_dir: Path, max_samples: int = 100):
    out_dir.mkdir(parents=True, exist_ok=True)

    num_layers_ref = None
    used = 0

    print(f"\n[Group {group_name}] analyzing up to {max_samples} samples...")
    probs_latest_list = []
    probs_earliest_list = []
    probs_index_list = []
    for i, sample_record in enumerate(samples[:max_samples]):
        try:
            res = compute_line_attention_for_sample(sample_record, model, tokenizer)
        except Exception as e:
            print(f"  [skip sample {sample_record['raw_sample']['index']}] reason: {e}")
            continue

        # attn_e = res["attn_earliest"]   # [L, T]
        attn_l = res["attn_latest"]     # [L, T]
        num_layers, num_lines = attn_l.shape

        if num_layers_ref is None:
            num_layers_ref = num_layers
        elif num_layers != num_layers_ref:
            print(f"  [skip] num_layers mismatch: {num_layers} vs {num_layers_ref}")
            continue

        # ---------- Normalize row-level attention for latest token ----------
        # For each layer: sum_over_lines(attn_l[L, :]), soft-normalize
        sum_latest = attn_l.sum(axis=1, keepdims=True) + 1e-9
        probs_latest = attn_l / sum_latest               # [L, T]

        # ---------- Normalize row-level attention for earliest token ----------
        # sum_earliest = attn_e.sum(axis=1, keepdims=True) + 1e-9
        # probs_earliest = attn_e / sum_earliest           # [L, T]
        probs_latest_list.append(probs_latest)       
        # probs_earliest_list.append(probs_earliest)   

        used += 1
        if (i + 1) % 10 == 0:
            print(f"  processed {i+1} samples, used={used}")
        
        # Predicted row
        probs_index_list.append(res['idx'])

    if used == 0:
        print(f"[Group {group_name}] no valid samples for attention analysis.")
        return

    # Group-averaged full-row attention distribution
    probs_latest_mean = np.stack(probs_latest_list, axis=0).mean(axis=0)     # [L, T]
    # probs_earliest_mean = np.stack(probs_earliest_list, axis=0).mean(axis=0) # [L, T]

    # Plot group-level full-row heatmap
    plot_layer_line_heatmap(
        probs_latest_mean,
        title=f"Layer-Wise Attention Scores - {group_name}",
        out_path=out_dir / f"{group_name}_latest_all_lines_heatmap.png",
    )
    np.save(f"DKI/InternalSingal/Attention/attn_plots_llama_{group_name}.npy", probs_latest_mean)
    # plot_layer_line_heatmap(
    #     probs_earliest_mean,
    #     title=f"Earliest token attention over all lines - {group_name}",
    #     out_path=out_dir / f"{group_name}_earliest_all_lines_heatmap.png",
    # )
    return probs_latest_list, probs_index_list


# Does the maximum attention value mean the model will definitely output the corresponding value? Compare 50 correct samples and 50 incorrect samples

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
def extend_with_oof(attn_layer: np.ndarray) -> np.ndarray:
    """
    attn_layer: (N, 32)
    Add a column of OOF (value 0) at the end, returns (N, 33)
    """
    num_samples, num_pos = attn_layer.shape
    extended = np.zeros((num_samples, num_pos + 1), dtype=attn_layer.dtype)
    extended[:, :num_pos] = attn_layer
    return extended


def plot_attention_three_big_figs(
    attn_group,         # (N, L_total, 32)
    pred_idx_group,     # (N,) 0..31, OOF is -1
    layers_to_plot=(7, 12, 21),  # 0-based: layers 8, 13, 22
    group_name="correct",
    cmap="magma",
    target_idx=31,
    save_dir="DKI/InternalSingal/Attention/attn_plots_llama",
    dpi=300,
    cell_inch=0.28,
    y_tick_step=5
):
    # Color scheme
    single_palettes = {
        "monoPeach": ["#fff7f2", "#ffd9c9", "#ffb08a", "#ff7844", "#b3471f"],
    }
    cmaps = {name: LinearSegmentedColormap.from_list(name, cols) for name, cols in single_palettes.items()}
    cmap = cmaps["monoPeach"]
    os.makedirs(save_dir, exist_ok=True)

    num_samples, L_total, num_pos = attn_group.shape
    assert num_pos == 32, "Last dimension should be attention for 32 values"

    # Legend definition (only shown in first plot)
    legend_handles = [
        Line2D([0], [0], marker="*", linestyle="None", markerfacecolor="none", 
               markeredgecolor="b", markeredgewidth=1.0, markersize=12, label="predicted"),
        Line2D([0], [0], marker="^", linestyle="None", color="r",
               markeredgewidth=1.0, markersize=10, label="argmax attn"),
        Line2D([0], [0], marker="o", linestyle="None", color="g",
               markeredgewidth=1.0, markersize=10, label="pred==argmax attn"),
    ]

    for layer_idx, layer_id in enumerate(layers_to_plot):
        attn_layer = attn_group[:, layer_id, :]  # (N, 32)
        attn_ext = extend_with_oof(attn_layer)    # (N, 33)
        attn_ext_transposed = attn_ext.T           # (33, N)
        nrows, ncols = attn_ext_transposed.shape

        # Calculate figure size
        fig_w = ncols * cell_inch + 2.2
        fig_h = nrows * cell_inch + 2.5
        
        # Adaptive top margin: need more space when sample count is low
        is_first_layer = (layer_idx == 0)
        needs_legend = is_first_layer
        # When sample count < Y-axis count, need more top space
        # Need more space when legend is present: legend(0.98) + title(0.92) + spacing
        top_space_needed = 0.20 if needs_legend else 0.10
        if ncols < nrows:
            top_space_needed += 0.05
        
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(111)
        
        # Set margins to ensure title and legend don't overlap
        # top parameter controls top position of subplot area, need space for legend and title
        fig.subplots_adjust(
            top=0.88 if needs_legend else 0.95,  # Subplot top at 0.88 with legend, 0.95 without legend
            bottom=0.1,
            left=0.1,
            right=0.88
        )

        # Draw heatmap
        extent = (-0.5, ncols - 0.5, nrows - 0.5, -0.5)
        im = ax.imshow(attn_ext_transposed, cmap=cmap, interpolation="nearest",
                      origin="upper", aspect="equal", extent=extent)

        # Grid lines
        ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
        ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.3)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.axhline(num_pos - 0.5, color="gray", linewidth=1.2)

        # Draw markers
        cell_pt = cell_inch * 72.0
        marker_sizes = {
            'star': (0.70 * cell_pt) ** 2,
            'tri': (0.62 * cell_pt) ** 2,
            'cir': (0.58 * cell_pt) ** 2
        }
        
        matches = 0
        for s in range(num_samples):
            pred = int(pred_idx_group[s])
            argmax_row = int(np.argmax(attn_layer[s]))
            
            if argmax_row == pred:
                matches += 1
                ax.scatter(s, pred, marker="o", s=marker_sizes['cir'], 
                          color="g", zorder=7, clip_on=True)
            else:
                row_pred = (nrows - 1) if pred == -1 else pred
                if 0 <= row_pred < nrows:
                    ax.scatter(s, row_pred, marker="*", s=marker_sizes['star'],
                              facecolors="none", edgecolors="b", linewidths=2.0,
                              zorder=5, clip_on=True)
                ax.scatter(s, argmax_row, marker="^", s=marker_sizes['tri'],
                          color="r", zorder=6, clip_on=True)

        # Set axis labels
        ax.set_xticks(range(0, ncols, y_tick_step))
        ax.set_xticklabels([f"S{i+1}" for i in range(0, ncols, y_tick_step)], fontsize=20)
        ax.set_xlabel(f"Sample Index (S1,..., S{num_samples})", fontsize=25, labelpad=10)
        
        y_labels = [f"{i+1}" for i in range(num_pos)] + ["OOF"]
        ax.set_yticks(range(nrows))
        ax.set_yticklabels(y_labels, fontsize=15)
        ax.set_ylabel("Value Occurrence Position (1–32, OOF)", fontsize=22, labelpad=10)

        # Title and legend
        match_rate = matches / num_samples
        title = f"Heatmap - Layer {layer_id}: Samples={num_samples}, Match Rate={match_rate*100:.1f}%"
        
        if needs_legend:
            # Legend at top, title below legend, subplot below title
            # Legend position: 0.98 (top of figure)
            # Title position: 0.92 (below legend with sufficient spacing)
            fig.legend(handles=legend_handles, bbox_to_anchor=(0.5, 0.97),
                      loc="upper center", ncol=3, frameon=True, fontsize=25)
            # fig.suptitle(title, fontsize=20, y=0.92)  # Title below legend, no overlap
            fig.suptitle(title, fontsize=25, y=0.88)
        else:
            fig.suptitle(title, fontsize=25, y=0.90)  # Without legend, title can be higher

        # Colorbar
        cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=20)

        # Save
        out_path = os.path.join(save_dir, f"predict_groundtruth_{group_name}_L{layer_id+1:02d}.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

def group_line(attn_group, pred_idx_group, group_name="correct",
               save_dir="DKI/InternalSingal/Attention/attn_plots_llama",
               dpi=300):
    """
    Plot consistency score line chart for each layer
    
    Parameters:
    - attn_group: np.ndarray (N, L_total, 32), N samples, L_total layers, 32 value positions
    - pred_idx_group: np.ndarray (N,), predicted id for each sample (0-31, or -1 for OOF)
    - group_name: str, group name
    - save_dir: str, save directory
    - dpi: int, image resolution
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_samples, L_total, num_pos = attn_group.shape
    assert num_pos == 32, "Last dimension should be attention for 32 values"
    
    # Calculate consistency score for each layer
    consistency_scores = []
    layer_labels = []
    
    for layer_id in range(L_total):
        attn_layer = attn_group[:, layer_id, :]  # (N, 32)
        
        # For each sample, calculate argmax position
        argmax_positions = np.argmax(attn_layer, axis=1)  # (N,)
        
        # Calculate consistency: whether argmax position matches predicted id
        # Note: -1 in pred_idx_group indicates OOF, needs special handling
        matches = 0
        valid_samples = 0
        
        for s in range(num_samples):
            pred = pred_idx_group[s]
            argmax_pos = argmax_positions[s]
            
            # If pred is -1 (OOF), skip or handle specially
            if pred == -1:
                # OOF case: can skip, or consider match when argmax is at position 32
                # Here choose to skip OOF samples
                continue
            
            valid_samples += 1
            if argmax_pos == pred:
                matches += 1
        
        # Calculate consistency ratio
        if valid_samples > 0:
            consistency = matches / valid_samples
        else:
            consistency = 0.0
        
        consistency_scores.append(consistency)
        layer_labels.append(f"L{layer_id}")
    print(f"{group_name}:{consistency_scores}")
    # Plot line chart
    plt.rcParams.update({
        "font.size": 20,
        "figure.figsize": (14, 6),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    
    # X-axis: layer numbers (L0-L27)
    x_positions = np.arange(L_total)
    
    # Plot line chart
    ax.plot(x_positions, consistency_scores, marker='o', markersize=6, 
            linewidth=2, color='#ff7844', label='Consistency Score')
    
    # Set X-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(layer_labels, fontsize=18, rotation=45, ha='right')
    ax.set_xlabel(f"Layer (L0-L{L_total-1})", fontsize=22, labelpad=10)
    
    # Set Y-axis
    ax.set_ylabel("Consistency Score", fontsize=22, labelpad=10)
    ax.set_ylim(0, 1.09)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f"{i:.1f}" for i in np.arange(0, 1.1, 0.1)], fontsize=18)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Title
    ax.set_title(f"Layer-wise Consistency Score - {group_name}", 
                 fontsize=22, pad=15)
                    #  f"(Argmax attention position == Predicted position)"
    # Add value annotations (optional, can show if layer count is not too many)
    if L_total <= 40:
        for i, score in enumerate(consistency_scores):
            ax.text(i, score + 0.02, f"{score:.2f}", 
                   ha='center', va='bottom', fontsize=12, color='gray')
    
    # Save
    out_path = os.path.join(save_dir, f"consistency_score_{group_name}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved consistency score plot to {out_path}")
    print(f"Consistency scores: {[f'{s:.3f}' for s in consistency_scores]}")