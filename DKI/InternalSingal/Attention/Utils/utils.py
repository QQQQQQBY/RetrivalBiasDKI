import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# MODEL_PATH = "Qwen2.5-7B-Instruct"
MODEL_PATH = "llama-3.1-8B-instruct"
MAX_SAMPLES_TO_SCAN = 2000
N_CORRECT = 36
N_WRONG = 164
# N_CORRECT = 8
# N_WRONG = 192
MAX_NEW_TOKENS = 64 
SYSTEM_MSG = (
    "Only output a single JSON object and nothing else. No code, no prose, no markdown.\n"
    "If you violate this, replace your entire reply with exactly: "
    '{"rationale":"FORMAT_ERROR","final":[]}'
)
# 2. Load model
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

# 3.1 Parse generated content
def parse_json_output(text: str):
    """
    Extract JSON object from model output and parse into (cue, earliest, latest)
    Supports multiple formats:
    1. Standard JSON format: {"cue":"...","earliest":"...","latest":"..."}
    2. Code block format: ```json\n{"cue":"...","earliest":"...","latest":"..."}\n```
    3. Plain text format: "cue":"..." or cue:...
    4. Supports extra text before/after, escape characters, etc.
    """
    if not text:
        return None
    
    candidates = []
    
    # Method 1: Prioritize extracting from ```json ... ``` code blocks
    for m in re.finditer(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.I):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and ("cue" in obj or "earliest" in obj or "latest" in obj):
                candidates.append(obj)
        except json.JSONDecodeError:
            pass
    
    # Method 2: Try extracting from any ``` ... ``` code blocks
    if not candidates:
        for m in re.finditer(r"```\s*(\{[\s\S]*?\})\s*```", text):
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict) and ("cue" in obj or "earliest" in obj or "latest" in obj):
                    candidates.append(obj)
            except json.JSONDecodeError:
                pass
    
    # Method 3: Use bracket matching to find all complete {...} JSON objects
    if not candidates:
        s = text
        start = s.find("{")
        while start != -1:
            stack = 0
            in_str = False
            esc = False
            for i in range(start, len(s)):
                ch = s[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        stack += 1
                    elif ch == "}":
                        stack -= 1
                        if stack == 0:
                            piece = s[start:i+1]
                            try:
                                obj = json.loads(piece)
                                if isinstance(obj, dict) and ("cue" in obj or "earliest" in obj or "latest" in obj):
                                    candidates.append(obj)
                            except json.JSONDecodeError:
                                pass
                            break
                if stack == 0 and i > start:
                    break
            start = s.find("{", start+1)
    
    # Method 4: If still not found, try simple regex matching (non-greedy)
    if not candidates:
        m = re.search(r"\{.*?\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and ("cue" in obj or "earliest" in obj or "latest" in obj):
                    candidates.append(obj)
            except json.JSONDecodeError:
                pass
    
    # Method 5: If JSON parsing all fails, try directly extracting key-value pairs with regex
    if not candidates:
        extracted = {}
        # Match "cue":"value" or 'cue':'value' format
        for key in ["cue", "earliest", "latest"]:
            pattern1 = rf'["\']?{re.escape(key)}["\']?\s*:\s*["\']([^"\'\\]*(?:\\.[^"\'\\]*)*)["\']'
            match1 = re.search(pattern1, text, re.IGNORECASE)
            if match1:
                value = match1.group(1)
                # Handle escape characters
                value = value.replace('\\"', '"').replace("\\'", "'").replace('\\n', '\n').replace('\\\\', '\\')
                extracted[key] = value
                continue
            
            # Match key:value format (without quotes)
            pattern2 = rf'["\']?{re.escape(key)}["\']?\s*:\s*([^\s,}}\]]+)'
            match2 = re.search(pattern2, text, re.IGNORECASE)
            if match2:
                value = match2.group(1).strip().rstrip(',}').strip('"\'')
                extracted[key] = value
        
        if extracted.get("cue") or extracted.get("earliest") or extracted.get("latest"):
            candidates.append(extracted)
    
    # Select the most suitable from candidates (prefer those containing all fields, otherwise select the last one)
    if candidates:
        # Prefer selecting those containing all three fields
        full_candidates = [c for c in candidates if all(k in c for k in ["cue", "earliest", "latest"])]
        if full_candidates:
            obj = full_candidates[-1]  # Take the last one (usually the final answer)
        else:
            obj = candidates[-1]  # Otherwise take the last candidate
        
        cue = obj.get("cue")
        earliest = obj.get("earliest")
        latest = obj.get("latest")
        return cue, earliest, latest
    
    return None

# 3.2 Generate once first to determine correct and incorrect samples
def call_model(model, tokenizer, messages, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Generate JSON output once using greedy decoding"""
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,               # Output string first, then tokenize uniformly
        add_generation_prompt=True    # Append assistant start marker to let model continue writing
    )
    inputs = tokenizer(chat_text, return_tensors="pt", return_offsets_mapping=True).to(model.device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None and tokenizer.eos_token_id is not None:
        pad_token_id = tokenizer.eos_token_id
    input_ids = inputs["input_ids"][0].to(model.device)
    with torch.no_grad():
        generated = model.generate(
            input_ids.unsqueeze(0),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=pad_token_id,
        )
    # Only take the newly generated part
    gen_ids = generated[:, inputs["input_ids"].shape[1]:]
    # full_ids = torch.cat([input_ids, gen_ids], dim=1)
    text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text
# , full_ids

# 3. Collect 100 correct samples + 100 incorrect samples
def collect_correct_and_wrong_samples(data, model, tokenizer):
    correct_samples = []
    wrong_samples = []
    scanned = 0

    for sample in tqdm(data):
        if scanned >= MAX_SAMPLES_TO_SCAN:
            break
        if len(correct_samples) >= N_CORRECT and len(wrong_samples) >= N_WRONG:
            break

        scanned += 1
        user_prompt = sample["present_prompt"]
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_prompt},
        ]
        # prompt = "Only output a single JSON object and nothing else. No code, no prose, no markdown.\n" + sample["present_prompt"]
        gt_earliest, gt_latest = extract_gt_words(sample["words"])

        model_out = call_model(model, tokenizer, messages=messages)
        parsed = parse_json_output(model_out)

        if parsed is None:
            continue
            # is_correct = False
            # out_earliest, out_latest = None, None
        elif None in parsed:
            continue    
        else:
            _, out_earliest, out_latest = parsed
            # is_correct = (out_earliest == gt_earliest and out_latest == gt_latest)
            is_correct = (out_latest == gt_latest)


        record = {
            "raw_sample": sample,
            "gt_earliest": gt_earliest,
            "gt_latest": gt_latest,
            "model_output": model_out,
            "parsed": parsed,
            # "full_ids": full_ids
        }

        if is_correct and len(correct_samples) < N_CORRECT:
            correct_samples.append(record)
        elif not is_correct and len(wrong_samples) < N_WRONG:
            wrong_samples.append(record)

        if scanned % 10 == 0:
            print(f"Scanned {scanned} samples: "
                  f"{len(correct_samples)} correct, {len(wrong_samples)} wrong")

    print(f"\nTotal scanned: {scanned}")
    print(f"Collected {len(correct_samples)} correct samples, "
          f"{len(wrong_samples)} wrong samples")
    return correct_samples, wrong_samples

# 4.1 Get character span for each value
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

# 4.2 Get offset_mapping corresponding to value character span
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



def extract_gt_words(words_list):
    """
    Example structure of words:
    [
      {"0": {"manually": "slumming"}},
      {"1": {"manually": "sanction"}}
    ]
    Convention: 0 -> earliest, 1 -> latest
    """
    earliest = None
    latest = None
    for item in words_list:
        idx_str, cue_map = list(item.items())[0]
        idx = int(idx_str)
        cue, value = list(cue_map.items())[0]
        if idx == 0:
            earliest = value
        elif idx == 1:
            latest = value
    return earliest, latest


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