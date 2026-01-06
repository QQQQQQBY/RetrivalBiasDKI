from Utils.utils import *
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

# 4 For this single sample only, no cross-sample averaging, visualize correct and incorrect samples separately
def compute_line_attention_for_sample(sample_record, model, tokenizer):
    """
    Returns:
    {
       "lines": [value_str_0, value_str_1, ...],   # In line order
       "attn_earliest": np.ndarray [num_layers, num_lines],
       "attn_latest":   np.ndarray [num_layers, num_lines],
    }
    Only for this single sample, no cross-sample averaging.
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
    # 2) Token indices for predicted earliest / latest values in output
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

    # q_idx_earliest = find_value_tokens_in_output(earliest_pred) if earliest_pred else []
    q_idx_latest = find_value_tokens_in_output(latest_pred) if latest_pred else []

    if not q_idx_latest:
        raise ValueError("Cannot locate earliest/latest value tokens in output.")

    # 3) Forward compute attention
    with torch.no_grad():
        outputs = model(input_ids=input_ids.unsqueeze(0), output_attentions=True, use_cache=False)

    attns = outputs.attentions  # tuple(num_layers, [1, n_heads, seq, seq])
    if attns is None:
        raise RuntimeError("Model did not return attentions; ensure config.output_attentions=True.")

    num_layers = len(attns)
    attn_earliest = np.zeros((num_layers, num_lines), dtype=np.float32)
    attn_latest = np.zeros((num_layers, num_lines), dtype=np.float32)

    for layer_idx, layer_attn in enumerate(attns):
        # layer_attn: [1, n_heads, seq, seq]
        layer_attn = layer_attn[0].to(torch.float32)      # [n_heads, seq, seq]
        attn_mean = layer_attn.mean(dim=0)                # Average over heads -> [seq, seq]
        attn_mean_np = attn_mean.cpu().numpy()

        # earliest: For each line, compute sum_q sum_k A[q,k] / |Q|
        # if q_idx_earliest is not None:
        #     for line_idx, token_indices in enumerate(value_tokens_per_line):
        #         if not token_indices:
        #             continue
        #         # sum_k A[q, k] over value tokens
        #         s = attn_mean_np[q_idx_earliest, token_indices].sum()
        #         attn_earliest[layer_idx, line_idx] = s

        # Same for latest
        if q_idx_latest is not None:
            for line_idx, token_indices in enumerate(value_tokens_per_line):
                if not token_indices:
                    continue
                s = attn_mean_np[q_idx_latest, token_indices].sum()
                attn_latest[layer_idx, line_idx] = s

    # sum_earliest = attn_earliest.sum(axis=1, keepdims=True) + 1e-9
    # probs_earliest = attn_earliest / sum_earliest
    sum_latest = attn_latest.sum(axis=1, keepdims=True) + 1e-9
    probs_latest = attn_latest / sum_latest 
    
    idx, is_oof = get_predicted_latest_index(sample_record)
    if is_oof:
        idx = -1
    # Can normalize to "proportion" later, here return raw sum first
    return {
        "lines": [v_str for (_, _, v_str) in value_spans],
        # "attn_earliest": probs_earliest,
        "attn_latest": probs_latest,
        "idx": idx
    }


# 4.1.1-4.2.2 Plot visualization
import numpy as np
import matplotlib.pyplot as plt

def plot_layer_line_heatmap_sample(attn_mat, lines, title, out_path,
                                   cmap="Blues", dpi=300, max_label_len=18):
    """
    attn_mat: [num_layers, num_lines]
    lines: List of value strings for each line
    """
    attn_mat = np.asarray(attn_mat)
    L, N = attn_mat.shape

    # Figure size & font (simple adaptive)
    fig_w = min(max(8, N * 0.35), 20)
    fig_h = min(max(5, L * 0.35), 12)

    # x-axis labels: truncate + if too many, show every few
    # def trunc(s):
    #     s = str(s)
    #     return s if len(s) <= max_label_len else s[:max_label_len] + "…"

    step = 1 if N <= 30 else (2 if N <= 60 else (5 if N <= 120 else 10))
    xticks = np.arange(0, N, step)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(attn_mat, aspect="auto", interpolation="nearest", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Attention proportion")
    cbar.ax.tick_params(labelsize=20)            # Right side tick label font size
    cbar.set_label("Attention proportion", fontsize=20) 

    ax.set_yticks(np.arange(L))
    ax.set_yticklabels([f"L{i}" for i in range(L)], fontsize=20)

    ax.set_xticks(xticks)
    ax.set_xticklabels([str(i) for i in xticks], rotation=60, ha="right", fontsize=20)

    ax.set_title(title, fontsize=20)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

