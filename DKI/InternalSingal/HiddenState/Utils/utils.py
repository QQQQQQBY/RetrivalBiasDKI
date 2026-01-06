
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "Qwen2.5-7B-Instruct"
# MODEL_PATH = "llama-3.1-8B-instruct"
def load_model_and_tokenizer():
    print(f"Loading model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,   # or torch.float16
        device_map="auto",
        load_in_4bit=True,            # Can enable if GPU memory is tight; for precise attention alignment, prefer full precision
        attn_implementation="eager"
    )
    model.eval()
    return model, tokenizer

def extract_value_spans_in_prompt(prompt: str):
    """
    In present_prompt text:
    - Find the records block between START: and END
    - For each line 'cue:value', extract value's character range [start_char, end_char)
    Returns: [(value_start, value_end, value_str), ...]
    """
    start_marker = "START:\n"
    end_marker = "\nEND"

    start_pos = prompt.index(start_marker) + len(start_marker)
    end_pos = prompt.index(end_marker)

    records_block = prompt[start_pos:end_pos]  # Excluding START/END
    lines = records_block.splitlines()

    spans = []
    cur = start_pos
    for line in lines:
        if not line.strip():
            cur += len(line) + 1
            continue
        # line is like "manually:slumming"
        colon_pos = line.index(":")
        value_str = line[colon_pos + 1:]
        line_start = cur
        line_end = cur + len(line)
        value_start = line_start + colon_pos + 1
        value_end = line_end  # Include entire value to end of line

        spans.append((value_start, value_end, value_str))
        cur += len(line) + 1  # +1 for '\n'
    return spans

def char_span_to_token_indices(offset_mapping, span_start, span_end):
    """
    offset_mapping: List[(start_char, end_char)] for full_text
    Returns: List of token indices that intersect with [span_start, span_end)
    """
    indices = []
    for i, (s, e) in enumerate(offset_mapping):
        # Special tokens may be (0,0), skip them
        if s == e == 0:
            continue
        # Include if there's overlap
        if not (e <= span_start or s >= span_end):
            indices.append(i)
    return indices

def get_predicted_latest_index(sample):
    """
    Input: single sample dict
    Output:
        idx: 0-based index of predicted latest in value sequence; returns None if OOF
        is_oof: bool, True means latest is not in any line (or is UNKNOWN / parsing failed)
    """
    # 1. Parse target cue (can get from entities)
    raw = sample.get("raw_sample", {})
    entities = raw.get("entities", [])
    if not entities:
        # No entity information, treat as OOF
        return None, True
    target_cue = entities[0]

    # 2. Parse model output JSON, get latest field
    model_output_str = sample.get("parsed", "")
    pred_latest = model_output_str[2]

    # Predicting UNKNOWN is also treated as OOF
    if pred_latest == "" or pred_latest.upper() == "UNKNOWN":
        return None, True

    # 3. Parse cue:value lines between START..END from prompt
    prompt = raw.get("present_prompt", "")

    start_marker = "START:\n"
    end_marker = "\nEND"

    try:
        start_idx = prompt.index(start_marker) + len(start_marker)
        end_idx = prompt.index(end_marker, start_idx)
    except ValueError:
        # START/END not found
        return None, True

    block = prompt[start_idx:end_idx].strip("\n")

    values = []  # Only keep values corresponding to target cue in order
    for line in block.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        cue, val = line.split(":", 1)
        cue = cue.strip()
        val = val.strip()
        if cue == target_cue:
            values.append(val)

    if not values:
        # No lines for this cue were parsed
        return None, True

    # 4. Find the position of predicted latest in values
    #    If there are duplicate values, take the "last position" as index
    indices = [i for i, v in enumerate(values) if v == pred_latest]

    if not indices:
        # Not found, treat as OOF
        return None, True

    last_idx = max(indices)  # 0-based index
    return last_idx, False