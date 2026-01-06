from pathlib import Path
from Utils import sample_multi_head_attention
from Utils.sample_multi_head_attention import compute_line_attention_for_sample_multihead
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

def plot_head_line_heatmap(data, title, out_path, dpi=300):
    """
    data: np.ndarray [H, T], rows are heads, columns are line indices
    """
    import matplotlib.pyplot as plt
    
    # Configurable parameters (refer to mainhead.py)
    FONT_BASE = 20
    TITLE_SIZE = 20
    LABEL_SIZE = 20
    TICK_SIZE = 20
    CBar_SIZE = 20
    XTICK_MAX = 32
    YTICK_MAX = 40
    WIDTH_PER_COL = 0.33
    HEIGHT_PER_ROW = 0.33
    FIG_W_MIN, FIG_W_MAX = 10, 22
    FIG_H_MIN, FIG_H_MAX = 6, 22
    
    # Set global font size
    mpl.rcParams.update({
        "font.size": FONT_BASE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
    })
    
    single_palettes = {
        "monoBlue":   ["#f7fbff", "#deebf7", "#9ecae1", "#3182bd", "#08519c"],
        "monoGreen":  ["#f7fff6", "#e6f7ea", "#9fe6b8", "#39b26a", "#0a6b3a"],
        "monoPurple": ["#fbf7ff", "#efe3ff", "#d6b8ff", "#9a55ff", "#5a00d1"],
        "monoTeal":   ["#f3fffb", "#d9fff4", "#95efde", "#2bbfa9", "#006a59"],
        "monoPeach":  ["#fff7f2", "#ffd9c9", "#ffb08a", "#ff7844", "#b3471f"],
        "monoGray":   ["#ffffff", "#f0f0f0", "#bdbdbd", "#6b6b6b", "#111111"],
    }
    cmaps = {name: LinearSegmentedColormap.from_list(name, cols) for name, cols in single_palettes.items()}
    cmap = cmaps["monoPurple"]
    
    data = np.asarray(data)
    H, T = data.shape
    
    # Adaptive figure size (refer to mainhead.py)
    fig_w = min(max(FIG_W_MIN, T * WIDTH_PER_COL), FIG_W_MAX)
    fig_h = min(max(FIG_H_MIN, H * HEIGHT_PER_ROW), FIG_H_MAX)
    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap)
    
    # Title and axes
    ax.set_title(title)
    ax.set_xlabel("Value Positions (1=earliest, 32=latest)")
    ax.set_ylabel("Head Index")
    
    # Intelligent tick label thinning (refer to mainhead.py)
    x_idx = np.arange(T)
    if T > XTICK_MAX:
        step = int(np.ceil(T / XTICK_MAX))
        x_show = x_idx[::step]
        x_labels = [str(i) for i in x_show]
    else:
        x_show = x_idx
        x_labels = [str(i + 1) for i in x_idx]
    ax.set_xticks(x_show)
    ax.set_xticklabels(x_labels, rotation=90)
    
    y_idx = np.arange(H)
    y_labels = [f"H{i}" for i in range(H)]
    if H > YTICK_MAX:
        step = int(np.ceil(H / YTICK_MAX))
        y_show = y_idx[::step]
        y_labels_show = [y_labels[i] for i in y_show]
    else:
        y_show = y_idx
        y_labels_show = y_labels
    ax.set_yticks(y_show)
    ax.set_yticklabels(y_labels_show)
    
    # Colorbar (refer to mainhead.py)
    cbar = plt.colorbar(im, ax=ax, shrink=1.0)
    # cbar.set_label("Attention proportion", fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=CBar_SIZE)
    
    # Borders & details
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(axis='both', which='major', length=4)
    
    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def analyze_group_attention_multihead(
    samples,
    model,
    tokenizer,
    layers_to_show,
    group_name: str,
    out_dir: Path,
    max_samples: int ,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    num_layers_ref = None
    num_heads_ref = None
    used = 0

    print(f"\n[Group {group_name}] analyzing up to {max_samples} samples (multi-head)...")
    probs_latest_list = []
    probs_earliest_list = []
    probs_index_list = []

    for i, sample_record in enumerate(samples[:max_samples]):
        try:
            res = compute_line_attention_for_sample_multihead(sample_record, model, tokenizer)
        except Exception as e:
            print(f"  [skip sample {sample_record['raw_sample']['index']}] reason: {e}")
            continue

        attn_e = res["attn_earliest"]   # [L, H, T]
        attn_l = res["attn_latest"]     # [L, H, T]
        num_layers, num_heads, num_lines = attn_l.shape

        if num_layers_ref is None:
            num_layers_ref = num_layers
            num_heads_ref = num_heads
        else:
            if num_layers != num_layers_ref or num_heads != num_heads_ref:
                print(f"  [skip] shape mismatch: layers {num_layers} vs {num_layers_ref}, "
                      f"heads {num_heads} vs {num_heads_ref}")
                continue

        # Here attn_e / attn_l have already been normalized per-(layer,head) in the function,
        # can directly collect as probability distributions.
        probs_latest_list.append(attn_l)      # [L, H, T]
        probs_earliest_list.append(attn_e)    # [L, H, T]
        probs_index_list.append(res["idx"])

        used += 1
        if (i + 1) % 10 == 0:
            print(f"  processed {i+1} samples, used={used}")

    if used == 0:
        print(f"[Group {group_name}] no valid samples for multi-head attention analysis.")
        return None, None, None

    # Group average: get [L, H, T]
    probs_latest_mean   = np.stack(probs_latest_list, axis=0).mean(axis=0)
    probs_earliest_mean = np.stack(probs_earliest_list, axis=0).mean(axis=0)


    for layer_idx in layers_to_show:
        if layer_idx < num_layers_ref:
            latest_layer_heat = probs_latest_mean[layer_idx]     # [H, T]
            earliest_layer_heat = probs_earliest_mean[layer_idx] # [H, T]

            plot_head_line_heatmap(
                latest_layer_heat,
                title=f"Latest token attention (layer {layer_idx})-{group_name}",
                out_path=out_dir / f"{group_name}_latest_layer{layer_idx}_head_line_heatmap.png",
            )
            # plot_head_line_heatmap(
            #     earliest_layer_heat,
            #     title=f"Earliest token attention (layer {layer_idx}) - head × line - {group_name}",
            #     out_path=out_dir / f"{group_name}_earliest_layer{layer_idx}_head_line_heatmap.png",
            # )
    latest_head_mean = probs_latest_mean.mean(axis=0)     # [H, T]
    earliest_head_mean = probs_earliest_mean.mean(axis=0) # [H, T]
    np.save(f"DKI/InternalSingal/Attention/MutliHeadAttentionVIZ/latest_head_mean_{group_name}.npy", latest_head_mean)
    plot_head_line_heatmap(
        latest_head_mean,
        title=f"Head-Wise Attention Scores -{group_name}",
        out_path=out_dir / f"{group_name}_latest_all_layers_head_line_heatmap.png",
    )
    # plot_head_line_heatmap(
    #     earliest_head_mean,
    #     title=f"Earliest token attention (all layers avg)",
    #     out_path=out_dir / f"{group_name}_earliest_all_layers_head_line_heatmap.png",
    # )
    # return probs_latest_list[:50], probs_earliest_list[:50], probs_index_list[:50]
