
import torch
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
def compute_line_attention_for_sample_multihead(sample_record, model, tokenizer):
    """
    Multi-head version: returns row-level attention for each layer, each head, for each row value (normalized by row)

    Returns:
    {
       "lines": [value_str_0, value_str_1, ...],   # In line order
       "attn_earliest": np.ndarray [L, H, T],      # Attention proportion for each layer, head, row
       "attn_latest":   np.ndarray [L, H, T],
       "idx": int,                                 # Model predicted latest row number (0-based, -1 means oof)
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


    q_idx_earliest = find_value_tokens_in_output(earliest_pred) if earliest_pred else None
    q_idx_latest   = find_value_tokens_in_output(latest_pred)   if latest_pred   else None

    if q_idx_earliest is None and q_idx_latest is None:
        raise ValueError("Cannot locate earliest/latest value tokens in output.")

    # 3) Forward compute attention (enable output_attentions)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            output_attentions=True,
            use_cache=False,
        )

    attns = outputs.attentions  # tuple(num_layers, [1, n_heads, seq, seq])
    if attns is None:
        raise RuntimeError("Model did not return attentions; ensure config.output_attentions=True.")

    num_layers = len(attns)
    n_heads = attns[0].shape[1]

    # [L, H, T]: row-level attention sum for each layer, head, row
    attn_earliest = np.zeros((num_layers, n_heads, num_lines), dtype=np.float32)
    attn_latest   = np.zeros((num_layers, n_heads, num_lines), dtype=np.float32)

    for layer_idx, layer_attn in enumerate(attns):
        # layer_attn: [1, n_heads, seq, seq]
        layer_attn = layer_attn[0].to(torch.float32)  # [H, seq, seq]

        # Calculate separately for each head
        for head_idx in range(n_heads):
            head_attn = layer_attn[head_idx]          # [seq, seq]
            head_attn_np = head_attn.cpu().numpy()

            # earliest: For each row, compute sum_k A[q, k] over value tokens
            if q_idx_earliest is not None:
                for line_idx, token_indices in enumerate(value_tokens_per_line):
                    if not token_indices:
                        continue
                    s = head_attn_np[q_idx_earliest, token_indices].sum() / len(token_indices)
                    attn_earliest[layer_idx, head_idx, line_idx] = s

            # Same for latest
            if q_idx_latest is not None:
                for line_idx, token_indices in enumerate(value_tokens_per_line):
                    if not token_indices:
                        continue
                    s = head_attn_np[q_idx_latest, token_indices].sum()/len(token_indices)
                    attn_latest[layer_idx, head_idx, line_idx] = s

    # 4) Normalize rows within each layer and head → probability distribution
    #    shape: [L, H, 1]
    sum_earliest = attn_earliest.sum(axis=2, keepdims=True) + 1e-9
    probs_earliest = attn_earliest / sum_earliest

    sum_latest = attn_latest.sum(axis=2, keepdims=True) + 1e-9
    probs_latest = attn_latest / sum_latest

    idx, is_oof = get_predicted_latest_index(sample_record)
    if is_oof:
        idx = -1

    return {
        "lines": [v_str for (_, _, v_str) in value_spans],
        "attn_earliest": probs_earliest,   # [L, H, T]
        "attn_latest":   probs_latest,     # [L, H, T]
        "idx": idx,
    }
