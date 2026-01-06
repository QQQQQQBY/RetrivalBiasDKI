from Utils.utils import *
import numpy as np
import torch
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

def get_sample_logits(sample_record, model, tokenizer):
    """
    Get logits and softmax logits when model predicts earliest_pred and latest_pred at different value_tokens_per_line
    
    Returns:
    {
       "lines": [value_str_0, value_str_1, ...],   # In line order
       "logits_values_earliest":   np.ndarray [T],    # Logits for each row value when predicting earliest (multi-token average)
       "logits_values_latest":     np.ndarray [T],    # Logits for each row value when predicting latest (multi-token average)
       "probs_values_earliest":     np.ndarray [T],    # Model confidence for earliest_pred when predicting earliest (multi-token sum, all values set to same value)
       "probs_values_latest":       np.ndarray [T],    # Model confidence for latest_pred when predicting latest (multi-token sum, all values set to same value)
       "idx": int,                                  # Model predicted latest row number (0-based, -1 means oof)
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
    
    # Token index list for predicted earliest / latest values in output (may contain multiple tokens)
    def find_value_tokens_in_output(value_str):
        pos = chat_text.find(value_str, start_str)
        if pos == -1:
            pos = chat_text.find(value_str)
            if pos == -1:
                return None

        span_start = pos
        span_end = pos + len(value_str)
        token_indices = char_span_to_token_indices(offset_mapping, span_start, span_end)
        if not token_indices:
            return None
        # Filter out tokens in prompt, only keep tokens in output
        output_token_indices = []
        for idx in sorted(token_indices):
            if offset_mapping[idx][0] >= start_str:
                output_token_indices.append(idx)
        if not output_token_indices:
            return None
        return output_token_indices
    
    q_token_indices_earliest = find_value_tokens_in_output(earliest_pred) if earliest_pred else None
    q_token_indices_latest   = find_value_tokens_in_output(latest_pred)   if latest_pred   else None
    if q_token_indices_earliest is None and q_token_indices_latest is None:
        raise ValueError("Cannot locate earliest/latest value tokens in output.")
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            output_hidden_states=True,
            use_cache=False,
        )
    
    # Get logits: [batch_size, seq_len, vocab_size]
    logits = outputs.logits[0]  # [seq_len, vocab_size]
    vocab_size = logits.shape[1]
    
    # Initialize result arrays: each value returns a scalar value (logits average)
    logits_values_earliest = np.zeros(num_lines, dtype=np.float32)
    logits_values_latest = np.zeros(num_lines, dtype=np.float32)
    
    # Compute softmax (for subsequent probability calculation)
    # First convert logits to float32 to avoid BFloat16 type unable to convert to numpy
    logits_float32 = logits.float() if logits.dtype == torch.bfloat16 else logits
    logits_np = logits_float32.cpu().numpy()  # [seq_len, vocab_size]
    probs = torch.softmax(logits_float32, dim=-1).cpu().numpy()  # [seq_len, vocab_size]
    
    # Helper function: compute model confidence for predicted value (multi-token sum)
    def compute_pred_probs_sum(q_token_indices):
        if q_token_indices is None:
            return 0.0
        return sum(probs[q_idx-1, input_ids[q_idx].item()] for q_idx in q_token_indices)
    
    # Helper function: compute logits values
    def compute_logits_values(q_token_indices, logits_values):
        if q_token_indices is None:
            return
        # Average logits across multiple token positions
        logits_avg = np.mean([logits_np[q_idx-1, :] for q_idx in q_token_indices], axis=0)
        # Compute logits average for each value
        for line_idx, token_indices in enumerate(value_tokens_per_line):
            if token_indices:
                value_logits = [logits_avg[input_ids[token_idx].item()] for token_idx in token_indices]
                # value_logits = value_logits[0]
                if value_logits:
                    logits_values[line_idx] = np.mean(value_logits)
    
    # Compute confidence and logits values
    earliest_pred_probs_sum = compute_pred_probs_sum(q_token_indices_earliest)
    latest_pred_probs_sum = compute_pred_probs_sum(q_token_indices_latest)
    compute_logits_values(q_token_indices_earliest, logits_values_earliest)
    compute_logits_values(q_token_indices_latest, logits_values_latest)
    
    # Use token_indices corresponding to the value predicted by the model to get probs_values_latest
    # Use compute_pred_probs_sum logic to get model confidence for latest_pred (multi-token sum)
    probs_values_latest = latest_pred_probs_sum/len(q_token_indices_latest)  # float, model confidence for latest_pred
    
    # Compute softmax probability for each candidate value position (for plotting)
    probs_all_candidates = torch.softmax(torch.from_numpy(logits_values_latest), dim=0).numpy()  # [T]
    
    # Get predicted idx
    idx, is_oof = get_predicted_latest_index(sample_record)
    if is_oof:
        idx = -1
    
    # Get confidence score corresponding to predicted value (directly use latest_pred_probs_sum)
    predicted_confidence = None
    if latest_pred_probs_sum > 0:
        # Use model confidence for latest_pred (multi-token sum)
        predicted_confidence = float(latest_pred_probs_sum)/len(q_token_indices_latest)
    elif idx == -1:
        # OOF case: if latest_pred_probs_sum is 0, may need special handling
        predicted_confidence = None
    
    return {
        "lines": [v_str for (_, _, v_str) in value_spans],
        "logits_values_earliest": logits_values_earliest,   # [T]
        "logits_values_latest": logits_values_latest,       # [T]
        "probs_values_latest": probs_values_latest,        # float, model confidence for latest_pred (using compute_pred_probs_sum)
        "probs_all_candidates": probs_all_candidates,       # [T] softmax probability for all candidate values (for plotting)
        "predicted_confidence": predicted_confidence,       # float, confidence score corresponding to predicted value
        "idx": idx,
    }

def group_analyse_logits(samples, model, tokenizer, group_name, out_dir):
    all_samples_logits_latest = []
    all_samples_probs_latest = []  # Collect softmax probabilities for all candidate values (for plotting)
    latest_idxs_list = []
    confidence_by_predicted_value_latest = {}
    
    for sample_record in samples:
        try:
            res = get_sample_logits(sample_record, model, tokenizer)
        except Exception as e:
            print(f"  [skip sample {sample_record['raw_sample']['index']}] reason: {e}")
            continue
        
        all_samples_logits_latest.append(res['logits_values_latest'])
        all_samples_probs_latest.append(res['probs_all_candidates'])  # Use softmax probabilities for all candidate values for plotting
        predicted_idx = res['idx']
        predicted_confidence = res['predicted_confidence']
        latest_idxs_list.append(predicted_idx)
        
        # Collect confidence scores corresponding to predicted values (grouped by predicted candidate value index)
        # Including OOF case (idx=-1)
        if predicted_confidence is not None:
            confidence_by_predicted_value_latest.setdefault(predicted_idx, []).append(predicted_confidence)
    
    if len(all_samples_logits_latest) == 0:
        print(f"  No valid samples for {group_name}")
        return
    
    # Convert to arrays and compute statistics
    all_samples_logits_latest = np.stack(all_samples_logits_latest, axis=0)  # [N, T]
    all_samples_probs_latest = np.stack(all_samples_probs_latest, axis=0)  # [N, T]
    T = all_samples_logits_latest.shape[1]
    
    # Store logits values
    np.save(f"DKI/InternalSingal/Logits/logits_reports/logits_latest_{group_name}.npy", all_samples_logits_latest)
    
    # Save confidence scores to JSON file
    confidence_data = {
        "latest": {str(k): [float(x) for x in v] for k, v in confidence_by_predicted_value_latest.items()}
    }
    confidence_json_path = out_dir / f"{group_name}_confidence_scores.json"
    confidence_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(confidence_json_path, 'w', encoding='utf-8') as f:
        json.dump(confidence_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved confidence scores to {confidence_json_path}")
    
    # Compute statistics and plot
    all_samples_logits_latest_mean = all_samples_logits_latest.mean(axis=0)
    all_samples_logits_latest_std = all_samples_logits_latest.std(axis=0)
    
    # Plot
    plot_logits_line_chart(
        all_samples_logits_latest_mean,
        all_samples_logits_latest_std,
        "Latest token logits",
        group_name,
        out_dir
    )
    
    plot_probs_violin_by_position(
        all_samples_probs_latest,
        "Latest token probs",
        group_name,
        out_dir
    )
    
    print(f"  Processed {len(all_samples_logits_latest)} samples for {group_name}")


def plot_logits_line_chart(logits_mean, logits_std, title_prefix, group_name, out_dir, dpi=300):
    """
    Plot logits line chart with error band (standard deviation)
    
    Parameters:
    - logits_mean: np.ndarray [T], mean logits
    - logits_std: np.ndarray [T], logits standard deviation
    - title_prefix: str, title prefix
    - group_name: str, group name
    - out_dir: Path, output directory
    - dpi: int, image resolution
    """
    # Configurable parameters (refer to mainhead.py)
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
    
    logits_mean = np.asarray(logits_mean)
    logits_std = np.asarray(logits_std)
    T = logits_mean.shape[0]
    
    # Adaptive figure size
    fig_w = min(max(10, T * 0.3), 20)
    fig_h = min(max(6, 1 * 0.8), 10)
    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    
    x_values = np.arange(T)  # X-axis: row numbers (0 to T-1)
    
    # Plot line chart
    ax.plot(x_values, logits_mean, marker='o', markersize=4, linewidth=2, label='Mean')
    
    # Plot error band (standard deviation)
    ax.fill_between(
        x_values,
        logits_mean - logits_std,
        logits_mean + logits_std,
        alpha=0.3,
        label='±1 std'
    )
    
    # Set title and labels
    ax.set_title(f"{title_prefix} - {group_name}")
    ax.set_xlabel("Record line index (0=earliest, T-1=latest)")
    ax.set_ylabel("Logits")
    
    # Set ticks
    step = 1 if T <= 30 else (2 if T <= 60 else (5 if T <= 120 else 10))
    xticks = np.arange(0, T, step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(i) for i in xticks], rotation=60, ha="right")
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend()
    
    # Borders
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(axis='both', which='major', length=4)
    
    # Save
    out_path = out_dir / f"{group_name}_{title_prefix.lower().replace(' ', '_')}_linechart.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
    print(f"  Saved line chart for {title_prefix}")


def plot_probs_violin_by_position(probs_conf_array, title_prefix, group_name, out_dir, dpi=300, trim_quantile=1.0, remove_outliers=True, ndig=3):
    """
    Plot violin chart of confidence scores for each candidate value position
    
    Parameters:
    - probs_conf_array: np.ndarray [N, T], confidence scores for N samples at T candidate value positions
    - title_prefix: str, title prefix
    - group_name: str, group name
    - out_dir: Path, output directory
    - dpi: int, image resolution
    - trim_quantile: float, trim threshold
    - remove_outliers: bool, whether to remove outliers
    - ndig: int, number of decimal places
    """
    if probs_conf_array.size == 0:
        print(f"  No probs data for {title_prefix}")
        return
    
    N, T = probs_conf_array.shape
    
    # Collect confidence scores for each position
    probs_by_position = []
    for pos_idx in range(T):
        probs_at_pos = probs_conf_array[:, pos_idx]  # [N]
        
        # Compute trim threshold
        if trim_quantile < 1.0:
            thresh = float(np.quantile(probs_at_pos, trim_quantile))
            if remove_outliers:
                probs_at_pos = probs_at_pos[probs_at_pos <= thresh]
            else:
                probs_at_pos = np.minimum(probs_at_pos, thresh)
        
        probs_by_position.append(probs_at_pos.tolist())
    
    # Compute mean for each position
    mean_probs = []
    for pos_idx in range(T):
        if len(probs_by_position[pos_idx]) > 0:
            mean_probs.append(float(np.mean(probs_by_position[pos_idx])))
        else:
            mean_probs.append(0.0)
    
    # Set plotting parameters
    plt.rcParams.update({
        "font.size": 20,
        "figure.figsize": (16, 6),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Prepare data: one violin plot for each position
    positions = np.arange(1, T + 1)  # 1 to T
    
    # Prepare data: if a column is empty, place [0.0] as placeholder
    data_for_violin = []
    for bucket in probs_by_position:
        if bucket:
            data_for_violin.append(bucket)
        else:
            data_for_violin.append([0.0])
    
    # Draw violin plot
    parts = ax.violinplot(
        data_for_violin,
        positions=positions,
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        vert=True
    )
    
    # Beautify violin colors
    cmap = plt.cm.viridis
    for i, pc in enumerate(parts['bodies']):
        color = cmap(i / max(1, len(parts['bodies']) - 1))
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.9)
        pc.set_linewidth(0.6)
    
    # Draw mean points in violin (white points with black edges)
    for i, pos in enumerate(positions):
        mean_p = mean_probs[i]
        # If column is actually empty (no data after trim), show as gray small point and mark with '-'
        if mean_p > 0.003:
            ax.scatter(pos, mean_p, color='white', edgecolors='black', zorder=10, s=50)
            # Annotate mean value above point (regular decimal, not scientific notation)
            mean_label = f"{mean_p:.{ndig}f}"
            ax.text(pos, mean_p + 0.05, mean_label, ha='center', va='bottom', fontsize=10, color='black')
        else:
            ax.scatter(pos, 0.0, color='lightgray', edgecolors='black', zorder=9, s=30)
            ax.text(pos, max(max(mean_probs), 1e-6) * 0.01, "—", ha='center', va='bottom', fontsize=15, color='gray')
    
    # Axes and labels
    ax.set_xlabel("Value index (1 = earliest, 32 = latest)")
    ax.set_ylabel("Predicted probability (confidence)")
    title_trim = f" (trim q={trim_quantile}, removed={remove_outliers})" if trim_quantile < 1.0 else ""
    ax.set_title(f"{title_prefix} distribution (violin) - {group_name}" + title_trim)
    
    ax.set_xticks(positions)
    ax.set_xticklabels([str(int(p)) for p in positions])
    ax.set_xlim(0.3, T + 0.7)
    
    # Y-axis upper limit: based on trimmed mean and trimmed data maximum
    all_vals = [v for bucket in probs_by_position for v in bucket]
    if all_vals:
        observed_max = max(max(mean_probs), max(all_vals), 1e-6)
        ax.set_ylim(0.0, max(observed_max * 1.05, 1e-6))
    else:
        ax.set_ylim(0.0, 1.0)
    
    # Disable y-axis scientific notation
    from matplotlib.ticker import ScalarFormatter
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    fmt.set_useOffset(False)
    ax.yaxis.set_major_formatter(fmt)
    
    plt.tight_layout()
    out_path = out_dir / f"{group_name}_{title_prefix.lower().replace(' ', '_')}_violin.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    
    print(f"  Saved violin plot for {title_prefix} (n_samples={N}, n_positions={T})")



