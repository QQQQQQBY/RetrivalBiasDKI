from Utils.utils import *
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import json
SYSTEM_MSG = (
    "Only output a single JSON object and nothing else. No code, no prose, no markdown.\n"
    "If you violate this, replace your entire reply with exactly: "
    '{"rationale":"FORMAT_ERROR","final":[]}'
)

def render_chat(tokenizer, messages):
    # Generate chat text for input (including template)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def sample_hidden_state(sample_record, model, tokenizer):
    """
    {
       "lines": [value_str_0, value_str_1, ...],   # In line order
       "hidden_values":   np.ndarray [L, T, D],    # Value hidden for each layer and line (averaged over value tokens in that line)
       "hidden_earliest": np.ndarray [L, D] or None, # Hidden state at earliest slot for each layer
       "hidden_latest":   np.ndarray [L, D] or None, # Hidden state at latest slot for each layer
       "idx": int,                                  # Model predicted latest line number (0-based, -1 means oof)
    }
    """
    prompt = sample_record["raw_sample"]["present_prompt"]
    model_output = sample_record["model_output"]
    parsed = sample_record["parsed"]  # (cue, earliest_pred, latest_pred)
    if parsed is None:
        raise ValueError("Sample has no parsed JSON output.")
    _, earliest_pred, latest_pred = parsed
    messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": model_output}
        ]
    chat_text = render_chat(tokenizer, messages)

    enc = tokenizer(
        chat_text,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    input_ids = enc["input_ids"][0].to(model.device)
    offset_mapping = enc["offset_mapping"][0].tolist()


    # 1) Token indices for record area (value)
    value_spans = extract_value_spans_in_prompt(chat_text)
    num_lines = len(value_spans)
    value_tokens_per_line = []
    for (v_start, v_end, v_str) in value_spans:
        token_indices = char_span_to_token_indices(offset_mapping, v_start, v_end)
        value_tokens_per_line.append(token_indices)
    
    start_str = len(render_chat(tokenizer, [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt}
        ]))
    # 2) Token indices for earliest/latest predicted values in output
    def find_value_tokens_in_output(value_str):

        pos = chat_text.find(value_str, start_str)
        if pos == -1:
            return []
        span_start = pos
        span_end = pos + len(value_str)
        token_indices = char_span_to_token_indices(offset_mapping, span_start, span_end)
        if not token_indices:
            return None
        q_idx = min(token_indices)
        return q_idx
    
    q_idx_earliest = find_value_tokens_in_output(earliest_pred) if earliest_pred else None
    q_idx_latest   = find_value_tokens_in_output(latest_pred)   if latest_pred   else None
    if q_idx_earliest is None and q_idx_latest is None:
        raise ValueError("Cannot locate earliest/latest value tokens in output.")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = outputs.hidden_states
    hidden_per_layer = hidden_states[1:] 
    num_layers = len(hidden_per_layer)
    seq_len = hidden_per_layer[0].shape[1]
    d_model = hidden_per_layer[0].shape[2]
    # Value hidden for each layer and line (averaged over value tokens in that line)
    hidden_values = np.zeros((num_layers, num_lines, d_model), dtype=np.float32)
    hidden_earliest = None
    hidden_latest   = None
    if q_idx_earliest is not None:
        hidden_earliest = np.zeros((num_layers, d_model), dtype=np.float32)
    if q_idx_latest is not None:
        hidden_latest   = np.zeros((num_layers, d_model), dtype=np.float32)
    for layer_idx, layer_hidden in enumerate(hidden_per_layer):
        # layer_hidden: [1, seq, D]
        layer_hidden = layer_hidden[0].to(torch.float32)  # [seq, D]
        layer_np = layer_hidden.cpu().numpy()              # [seq, D]
        for line_idx, token_indices in enumerate(value_tokens_per_line):
            if not token_indices:
                continue
            # [num_tokens, D]
            token_vecs = layer_np[token_indices, :]
            hidden_values[layer_idx, line_idx, :] = token_vecs.mean(axis=0)
        if q_idx_earliest is not None:
            hidden_earliest[layer_idx, :] = layer_np[q_idx_earliest, :]
        if q_idx_latest is not None:
            hidden_latest[layer_idx, :] = layer_np[q_idx_latest, :]
    idx, is_oof = get_predicted_latest_index(sample_record)
    if is_oof:
        idx = -1

    return {
        "lines": [v_str for (_, _, v_str) in value_spans],
        "hidden_values":   hidden_values,   # [L, T, D]
        "hidden_earliest": hidden_earliest, # [L, D] or None
        "hidden_latest":   hidden_latest,   # [L, D] or None
        "idx": idx,
    }

def compute_per_layer_argmax_indices(hidden_values, hidden_q):
    """
    For each layer, compute similarity between answer hidden and each line's value hidden,
    and return the index of the line with maximum similarity for each layer and the corresponding maximum similarity.

    Parameters:
    - hidden_values: np.ndarray, shape [L, T, D]
    - hidden_q:      np.ndarray, shape [L, D], hidden state of a query (latest or earliest)

    Returns:
    - best_idx_per_layer: np.ndarray, shape [L], argmax line number for each layer (0 ~ T-1)
    """
    if hidden_q is None:
        return None, None
    L, T, D = hidden_values.shape
    best_idx_per_layer = np.full(L, -1, dtype=int)
    sims_per_layer = np.zeros((L, T), dtype=np.float32)
    eps = 1e-9
    for l in range(L):
        q = hidden_q[l]          # [D]
        vals = hidden_values[l]  # [T, D]
        q_norm = q / (np.linalg.norm(q) + eps)
        vals_norm = vals / (np.linalg.norm(vals, axis=1, keepdims=True) + eps)
        sims = vals_norm @ q_norm  # [T]
        idx = int(sims.argmax())
        best_idx_per_layer[l] = idx
        sims_per_layer[l, :] = sims
    return best_idx_per_layer, sims_per_layer



def analyze_hidden_state(samples, model, tokenizer, group_name, out_dir, title_prefix):
    
    latest_idxs_list = []
    used = 0
    group_best_idx_per_latest = []
    group_latest_sim_per_layer = []
    group_best_idx_per_layer_earliest = []
    group_earliest_sim_per_layer = []
    predict_sim_list = []
    # Hidden states
    for i, sample_record in enumerate(samples):
        try:
            res = sample_hidden_state(sample_record, model, tokenizer)
        except Exception as e:
            print(f"  [skip sample {sample_record['raw_sample']['index']}] reason: {e}")
            continue

        hidden_values   = res["hidden_values"]    # [L, T, D]
        hidden_latest   = res["hidden_latest"]    # [L, D] or None
        hidden_earliest = res["hidden_earliest"]
        
        best_idx_per_layer_latest, latest_sim_per_layer = compute_per_layer_argmax_indices(
            hidden_values, hidden_latest
        )
        best_idx_per_layer_earliest, earliest_sim_per_layer = compute_per_layer_argmax_indices(
            hidden_values, hidden_earliest
        )

        group_best_idx_per_latest.append(best_idx_per_layer_latest)
        group_latest_sim_per_layer.append(latest_sim_per_layer)
        group_best_idx_per_layer_earliest.append(best_idx_per_layer_earliest)
        group_earliest_sim_per_layer.append(earliest_sim_per_layer)
        # Get similarity scores for all layers at predicted position for current sample [L]
        # Only process valid prediction positions (idx >= 0)
        if res['idx'] >= 0:
            predict_sim_list.append(latest_sim_per_layer[:, res['idx']])
        latest_idxs_list.append(res['idx'])
        used += 1
        if (i + 1) % 10 == 0:
            print(f"  processed {i+1} samples, used={used}")
    
    # Compute similarity scores at corresponding answer positions for predicted values, mean and std across multiple samples and layers
    # predict_sim_list: similarity for all layers at predicted position for each sample (only includes valid prediction positions)
    if predict_sim_list:
        predict_sim_array = np.array(predict_sim_list)  # [N_valid, L]
        latest_idxs = np.array(latest_idxs_list)
        
        # Get indices of all valid prediction positions and corresponding prediction positions
        valid_mask = latest_idxs >= 0
        valid_predict_idxs = latest_idxs[valid_mask]  # Prediction positions for valid samples
        
        # Compute statistics for similarity scores corresponding to each position (1, 2, 3, ...)
        position_stats = {}
        unique_positions = np.unique(valid_predict_idxs)  # Only process valid positions
        
        for pos in unique_positions:
            # Find all valid samples with prediction position pos
            mask = (valid_predict_idxs == pos)
            if mask.sum() > 0:
                # Get similarity for all layers at this position for these samples [num_samples, L]
                sims_at_pos = predict_sim_array[mask, :]  # [num_samples, L]
                
                # Compute mean and std (across samples and layers)
                mean_sim = float(np.mean(sims_at_pos))
                std_sim = float(np.std(sims_at_pos))
                
                # Positions start from 0, but output starts from 1
                position_stats[str(int(pos) + 1)] = {
                    "mean": mean_sim,
                    "std": std_sim,
                    "num_samples": int(mask.sum())
                }
        
        # Save as JSON file
        json_path = out_dir / f"{group_name}_predicted_position_similarity_stats.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(position_stats, f, indent=2, ensure_ascii=False)
        
        print(f"  Saved predicted position similarity statistics to: {json_path}")
        print(f"  Predicted position similarity statistics (position: mean±std, num_samples):")
        for pos_key in sorted(position_stats.keys(), key=lambda x: int(x)):
            stats = position_stats[pos_key]
            print(f"    {pos_key}: {stats['mean']:.4f}±{stats['std']:.4f} (n={stats['num_samples']})")
    
    # print(f"{title_prefix}: {group_name}: {predict_sim_mean}")
    # Plot line charts for the above lists, first average multiple values
    best_idx_per_layers_latest = np.stack(group_best_idx_per_latest, axis=0)      # [N, L],(50, 28)
    latest_sim_per_layers      = np.stack(group_latest_sim_per_layer, axis=0)      # [N, L, T], (50, 28, 32)
    best_idx_per_layers_earliest = np.stack(group_best_idx_per_layer_earliest, axis=0)  # [N, L]
    earliest_sim_per_layers      = np.stack(group_earliest_sim_per_layer, axis=0)       # [N, L, T]
    
    latest_idxs = np.array(latest_idxs_list)
    
    # Compute mean: average over N samples
    latest_sim_mean = latest_sim_per_layers.mean(axis=0)      # [L, T]
    earliest_sim_mean = earliest_sim_per_layers.mean(axis=0)  # [L, T]
    print(f"Saving {group_name} as npy file")
    np.save(f"DKI/InternalSingal/HiddenState/Embeddings/latest_sim_mean_{group_name}.npy", latest_sim_mean)
    # Plot line charts
    plot_multi_layer_line_charts(
        latest_sim_mean,
        title_prefix=title_prefix,
        group_name=group_name,
        out_dir=out_dir
    )
    
    # plot_multi_layer_line_charts(
    #     earliest_sim_mean,
    #     title_prefix="Earliest token similarity",
    #     group_name=group_name,
    #     out_dir=out_dir
    # )
    
    # Plot line chart after averaging across layers
    latest_sim_mean_all_layers = latest_sim_mean.mean(axis=0)      # [T], average across layers
    latest_sim_var_all_layers = latest_sim_mean.var(axis=0)
    earliest_sim_mean_all_layers = earliest_sim_mean.mean(axis=0)  # [T], average across layers
    print("Mean similarity scores at each candidate value after averaging across layers")
    print(f"{title_prefix}: {group_name}: {latest_sim_mean_all_layers}")
    print("Variance of similarity scores at each candidate value after averaging across layers")
    print(f"{title_prefix}: {group_name}: {latest_sim_var_all_layers}")

    plot_all_layers_averaged_line_chart(
        latest_sim_mean_all_layers,
        title_prefix=title_prefix,
        group_name=group_name,
        out_dir=out_dir
    )
    
    # plot_all_layers_averaged_line_chart(
    #     earliest_sim_mean_all_layers,
    #     title_prefix="Earliest token similarity",
    #     group_name=group_name,
    #     out_dir=out_dir
    # )
    
    # Compute and plot consistency score line chart
    # Consistency score: proportion of layers where the line number corresponding to maximum similarity matches the model's final predicted output value
    consistency_scores_latest = compute_consistency_scores(
        best_idx_per_layers_latest, latest_idxs
    )  # [L]
    
    print("Model consistency scores across layers:")
    print(f"{title_prefix}: {group_name}: {consistency_scores_latest}")
    # consistency_scores_earliest = compute_consistency_scores(
    #     best_idx_per_layers_earliest, latest_idxs  # Note: earliest also uses latest_idxs as ground truth
    # )  # [L]
    
    plot_consistency_scores(
        consistency_scores_latest,
        title_prefix=title_prefix,
        group_name=group_name,
        out_dir=out_dir
    )
    
    # plot_consistency_scores(
    #     consistency_scores_earliest,
    #     title_prefix="Earliest token consistency",
    #     group_name=group_name,
    #     out_dir=out_dir
    # )


def plot_multi_layer_line_charts(sim_per_layers, title_prefix, group_name, out_dir, dpi=300, layers_to_show=None):
    """
    Plot multi-layer line charts, all layers displayed on one large figure
    
    Parameters:
    - sim_per_layers: np.ndarray [L, T], average similarity for each layer and line
    - title_prefix: str, title prefix
    - group_name: str, group name
    - out_dir: Path, output directory
    - dpi: int, image resolution
    - layers_to_show: list, list of layer numbers to display, None means display all layers
    """
    # Configurable parameters (subplot font slightly smaller)
    FONT_BASE = 12
    TITLE_SIZE = 14
    LABEL_SIZE = 12
    TICK_SIZE = 10
    
    sim_per_layers = np.asarray(sim_per_layers)
    L, T = sim_per_layers.shape
    
    if layers_to_show is None:
        layers_to_show = list(range(L))
    
    # Filter valid layer numbers
    valid_layers = [idx for idx in layers_to_show if idx < L]
    num_layers = len(valid_layers)
    
    if num_layers == 0:
        print(f"  No valid layers to plot for {title_prefix}")
        return
    
    # Compute subplot layout: 28 layers can use 4x7 or 7x4, here use 4 rows 7 columns
    nrows = 4
    ncols = 7
    if num_layers <= 28:
        # If number of layers is less than 28, adjust layout
        nrows = int(np.ceil(num_layers / ncols))
    # nrows = 4
    # ncols = 8
    # if num_layers <= 32:
    #     # If number of layers is less than 32, adjust layout
    #     nrows = int(np.ceil(num_layers / ncols))
    
    # Create large figure
    fig_w = min(max(20, T * 0.15 * ncols), 30)
    fig_h = min(max(12, 1.5 * nrows), 20)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), constrained_layout=True)
    
    # If only one row, ensure axes is 2D array
    if nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    x_values = np.arange(T)  # X-axis: line numbers (0 to T-1)
    
    # Plot one subplot for each layer
    for plot_idx, layer_idx in enumerate(valid_layers):
        row = plot_idx // ncols
        col = plot_idx % ncols
        ax = axes[row, col]
        
        # Plot line chart
        y_values = sim_per_layers[layer_idx, :]  # [T]
        ax.plot(x_values, y_values, marker='o', markersize=2, linewidth=1.5)
        
        # Set title (subplot title)
        ax.set_title(f"Layer {layer_idx}", fontsize=TITLE_SIZE)
        
        # Set labels (only show on edge subplots)
        if row == nrows - 1:
            ax.set_xlabel("Value Positions", fontsize=LABEL_SIZE)
        if col == 0:
            ax.set_ylabel("Similarity Score", fontsize=LABEL_SIZE)
        
        # Set ticks
        step = max(1, T // 8)  # Ticks are sparser in subplots
        xticks = np.arange(0, T, step)
        if len(xticks) == 0:
            xticks = np.arange(0, T)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(i) for i in xticks], rotation=45, ha="right", fontsize=TICK_SIZE)
        ax.tick_params(axis='y', labelsize=TICK_SIZE)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Borders
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.tick_params(axis='both', which='major', length=2)
    
    # Hide extra subplots
    for plot_idx in range(num_layers, nrows * ncols):
        row = plot_idx // ncols
        col = plot_idx % ncols
        axes[row, col].axis('off')
    
    # Set large figure title
    fig.suptitle(f"{title_prefix} across layers - {group_name}", fontsize=16, y=1.05)
    
    # Save
    out_path = out_dir / f"{group_name}_{title_prefix.lower().replace(' ', '_')}_all_layers_linechart.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
    print(f"  Saved multi-layer line chart ({num_layers} layers) for {title_prefix}")


def plot_all_layers_averaged_line_chart(sim_values, title_prefix, group_name, out_dir, dpi=300):
    """
    Plot similarity line chart after averaging across layers for 32 values
    
    Parameters:
    - sim_values: np.ndarray [T], similarity for each line after averaging across layers
    - title_prefix: str, title prefix
    - group_name: str, group name
    - out_dir: Path, output directory
    - dpi: int, image resolution
    """
    # Configurable parameters (reference mainhead.py)
    FONT_BASE = 20
    TITLE_SIZE = 20
    LABEL_SIZE = 20
    TICK_SIZE = 20
    
    # Set global font size
    mpl.rcParams.update({
        "font.size": FONT_BASE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
    })
    
    sim_values = np.asarray(sim_values)
    T = sim_values.shape[0]
    
    # Adaptive figure size
    fig_w = min(max(10, T * 0.3), 20)
    fig_h = min(max(6, 1 * 0.8), 10)
    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    
    x_values = np.arange(T)  # X-axis: line numbers (0 to T-1, total 32 values)
    
    # Plot line chart
    ax.plot(x_values, sim_values, marker='o', markersize=4, linewidth=2)
    
    # Set title and labels
    ax.set_title(f"{title_prefix} (all layers averaged) - {group_name}")
    ax.set_xlabel("Value Positions")
    ax.set_ylabel("Similarity Score")
    
    # Set ticks
    step = 1 if T <= 30 else (2 if T <= 60 else (5 if T <= 120 else 10))
    xticks = np.arange(0, T, step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(i + 1) for i in xticks], rotation=60, ha="right")
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Borders
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(axis='both', which='major', length=4)
    
    # Save
    out_path = out_dir / f"{group_name}_{title_prefix.lower().replace(' ', '_')}_all_layers_avg_linechart.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
    print(f"  Saved all-layers-averaged line chart for {title_prefix}")


def compute_consistency_scores(best_idx_per_layers, predicted_idxs):
    """
    Compute consistency score: proportion of layers where the line number corresponding to maximum similarity matches the model's final predicted output value
    
    Parameters:
    - best_idx_per_layers: np.ndarray [N, L], argmax line number for each sample at each layer
    - predicted_idxs: np.ndarray [N], model's final predicted output value for each sample
    
    Returns:
    - consistency_scores: np.ndarray [L], consistency score for each layer (between 0 and 1)
    """
    best_idx_per_layers = np.asarray(best_idx_per_layers)
    predicted_idxs = np.asarray(predicted_idxs)
    
    N, L = best_idx_per_layers.shape
    
    consistency_scores = np.zeros(L, dtype=np.float32)
    
    for layer_idx in range(L):
        # For each layer, compute proportion where argmax matches predicted value
        matches = (best_idx_per_layers[:, layer_idx] == predicted_idxs)
        consistency_scores[layer_idx] = matches.sum() / N
    
    return consistency_scores


def plot_consistency_scores(consistency_scores, title_prefix, group_name, out_dir, dpi=300):
    """
    Plot consistency score line chart: X-axis is arrangement of L layers, Y-axis is consistency score
    Reference the plotting method of group_line
    
    Parameters:
    - consistency_scores: np.ndarray [L], consistency score for each layer
    - title_prefix: str, title prefix
    - group_name: str, group name
    - out_dir: Path, output directory
    - dpi: int, image resolution
    """
    consistency_scores = np.asarray(consistency_scores)
    L = consistency_scores.shape[0]
    
    # Generate layer labels
    layer_labels = [f"L{i}" for i in range(L)]
    
    # Plot line chart (reference the plotting method of group_line)
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
    x_positions = np.arange(L)
    
    # Plot line chart
    ax.plot(x_positions, consistency_scores, marker='o', markersize=6, 
            linewidth=2, color='#ff7844', label='Consistency Score')
    
    # Set X-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(layer_labels, fontsize=18, rotation=45, ha='right')
    ax.set_xlabel(f"Layer (L0-L{L-1})", fontsize=22, labelpad=10)
    
    # Set Y-axis
    ax.set_ylabel("Consistency Score", fontsize=22, labelpad=10)
    ax.set_ylim(0, 1.09)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f"{i:.1f}" for i in np.arange(0, 1.1, 0.1)], fontsize=18)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Title
    ax.set_title(f"{title_prefix} Similarity Score - {group_name}", 
                 fontsize=22, pad=15)
    
    # Add value annotations (can display if number of layers is not too many)
    if L <= 40:
        for i, score in enumerate(consistency_scores):
            ax.text(i, score + 0.02, f"{score:.2f}", 
                   ha='center', va='bottom', fontsize=12, color='gray')
    
    # Save
    out_path = out_dir / f"{group_name}_{title_prefix.lower().replace(' ', '_')}_consistency_scores.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
    # Output consistency scores to console
    print(f"  Saved consistency scores chart for {title_prefix}")
    print(f"  Consistency scores: {[f'{s:.3f}' for s in consistency_scores]}")

