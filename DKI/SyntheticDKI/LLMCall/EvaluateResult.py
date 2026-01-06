import argparse
import json
import math
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
# --------------------------
# Utilities
# --------------------------

def _is_target_shape(obj) -> bool:
    # Shape:
    # {
    #   "rationale": "....",
    #   "final": [ {"key":...,"earliest":...,"latest":...}, ... ]
    # }
    if not isinstance(obj, dict):
        return False
    if "rationale" not in obj or "final" not in obj:
        return False
    if not isinstance(obj["rationale"], str):
        return False
    if not isinstance(obj["final"], list):
        return False
    for it in obj["final"]:
        if not (isinstance(it, dict)
                and set(it.keys()) == {"key", "earliest", "latest"}
                and all(isinstance(it[k], str) or it[k] is None for k in ("key","earliest","latest"))):
            return False
    return True

import json, re

def _extract_first_matching_json(text: str):
    """
    Returns the first/last JSON object matching the target schema.
    Strategy:
      a) Objects in ```json ... ``` (priority)
      b) Objects in any ``` ... ```
      c) Find all { ... } pairs in the full text using bracket matching, then json.loads each
      d) Select objects matching _is_target_shape; if multiple, take the "last one"
    """
    candidates = []

    # a) ```json ... ```
    for m in re.finditer(r"```json\s*({[\s\S]*?})\s*```", text, flags=re.I):
        try:
            obj = json.loads(m.group(1))
            candidates.append(obj)
        except json.JSONDecodeError:
            pass

    # b) Other ``` ... ``` (many models use ``` without marking json)
    if not candidates:
        for m in re.finditer(r"```\s*({[\s\S]*?})\s*```", text):
            try:
                obj = json.loads(m.group(1))
                candidates.append(obj)
            except json.JSONDecodeError:
                pass

    # c) Bracket matching in full text, enumerate all { ... } fragments (avoid regex greedy capture)
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
                                candidates.append(obj)
                            except json.JSONDecodeError:
                                pass
                            break
            start = s.find("{", start+1)

    # d) Filter objects matching schema; if multiple, take the "last one" (usually after "Here is the output")
    good = [o for o in candidates if _is_target_shape(o)]
    if good:
        return good[-1]  # Usually the last one is the "final answer JSON"
    # Fallback: accept any object with "final" that is a list
    fallback = [o for o in candidates if isinstance(o, dict) and isinstance(o.get("final"), list)]
    if fallback:
        return fallback[-1]

    return None


def levenshtein(a: Optional[str], b: Optional[str]) -> Optional[int]:
    if a is None or b is None:
        return None
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost     # substitution
            )
            prev = cur
    return dp[lb]

def eq(a: Optional[str], b: Optional[str], strict_case: bool) -> bool:
    if a is None or b is None:
        return False
    if b in a:
        return True
    if strict_case:
        return a == b
    # Lenient: ignore leading/trailing spaces and case
    return str(a).strip().lower() == str(b).strip().lower()

# --------------------------
# Ground truth & prediction parsing
# --------------------------
def build_gt(sample: Dict) -> Tuple[List[str], Dict[str, Tuple[Optional[str], Optional[str]]]]:
    """
    Returns:
      entities_order: In the order given by sample['entities']
      gt_map: {entity: (gt_earliest, gt_latest)}
    """
    entities = sample.get("entities", [])
    words = sample.get("words", [])
    gt0, gt1 = {}, {}
    for it in words:
        if "0" in it:
            gt0 = it["0"]
        if "1" in it:
            gt1 = it["1"]
    gt_map = {}
    for e in entities:
        gt_map[e] = (gt0.get(e), gt1.get(e) if gt1 else None)
    return entities, gt_map

def extract_key_value_pairs(text: str, keys: List[str]) -> Dict[str, Optional[str]]:
    """
    Generic key-value pair extraction function that extracts values for specified keys from a string.
    Supports multiple formats:
    1. JSON format: {"key": "value"}
    2. Code block format: ```json\n{"key": "value"}\n```
    3. Plain text format: "key":"value" or key:value
    
    Args:
        text: Input text
        keys: List of keys to extract, e.g., ["cue", "earliest", "latest"]
    
    Returns:
        Dictionary containing extracted key-value pairs
    """
    result = {}
    if not text:
        return result
    
    # Method 1: Try to parse JSON (including JSON in code blocks)
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract from code blocks
        obj = _extract_first_matching_json(text)
        if obj is None:
            # Try to extract JSON from code blocks
            json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text, flags=re.I)
            if json_match:
                try:
                    obj = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
    
    # If JSON parsing succeeds, extract directly
    if isinstance(obj, dict):
        for key in keys:
            result[key] = obj.get(key)
        return result
    
    # Method 2: Use regex to directly extract key-value pairs from string
    # Match "key":"value" or 'key':'value' format
    for key in keys:
        # Match "key":"value" format (considering escape characters)
        pattern1 = rf'["\']?{re.escape(key)}["\']?\s*:\s*["\']([^"\'\\]*(?:\\.[^"\'\\]*)*)["\']'
        match1 = re.search(pattern1, text, re.IGNORECASE)
        if match1:
            value = match1.group(1)
            # Handle escape characters
            value = value.replace('\\"', '"').replace("\\'", "'").replace('\\n', '\n').replace('\\\\', '\\')
            result[key] = value
            continue
        
        # Match key:value format (without quotes)
        pattern2 = rf'["\']?{re.escape(key)}["\']?\s*:\s*([^\s,}}\]]+)'
        match2 = re.search(pattern2, text, re.IGNORECASE)
        if match2:
            value = match2.group(1).strip().rstrip(',}').strip('"\'')
            result[key] = value
    
    return result


def parse_pred(sample: Dict) -> Optional[List[Tuple[str, Optional[str], Optional[str]]]]:
    """
    Extracts JSON from sample['recall_response'], supports multiple formats:
    1. New format (single entity): {"cue":"pinwheel","earliest":"atrocity","latest":"encumber"}
    2. Old format (multiple entities): {"rationale":"...","final":[{"key":"...","earliest":"...","latest":"..."}]}
    3. Code block format: ```json\n{"cue":"...","earliest":"...","latest":"..."}\n```
    
    Automatically compatible: with/without code blocks, extra text before/after, code before JSON cases.
    If JSON parsing fails, uses regex to directly extract key-value pairs.
    """
    raw = sample.get("recall_response")
    if raw is None:
        return None

    obj = None
    # First try pure JSON
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = _extract_first_matching_json(raw)

    # If JSON object parsing succeeds
    if isinstance(obj, dict):
        # Format 1: New format - single object with "cue" field
        if "cue" in obj and ("earliest" in obj or "latest" in obj):
            cue = obj.get("cue")
            earliest = obj.get("earliest")
            latest = obj.get("latest")
            return [(cue, earliest, latest)]
        
        # Format 2: Old format - contains "final" array
        final = obj.get("final")
        if isinstance(final, list):
            out = []
            for item in final:
                if isinstance(item, dict):
                    # Support "key" or "cue" field
                    k = item.get("key") or item.get("cue")
                    pe = item.get("earliest")
                    pl = item.get("latest")
                    out.append((k, pe, pl))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    out.append((None, item[0], item[1]))
                else:
                    out.append((None, None, None))
            return out if out else None
    
    # If JSON parsing fails, use generic key-value pair extraction function
    extracted = extract_key_value_pairs(raw, ["cue", "earliest", "latest"])
    if extracted.get("cue") or extracted.get("earliest") or extracted.get("latest"):
        cue = extracted.get("cue")
        earliest = extracted.get("earliest")
        latest = extracted.get("latest")
        return [(cue, earliest, latest)]
    
    # If still not extracted, try extracting "key" field (compatible with old format)
    extracted_key = extract_key_value_pairs(raw, ["key", "earliest", "latest"])
    if extracted_key.get("key") or extracted_key.get("earliest") or extracted_key.get("latest"):
        key = extracted_key.get("key")
        earliest = extracted_key.get("earliest")
        latest = extracted_key.get("latest")
        return [(key, earliest, latest)]
    
    return None


# --------------------------
# Evaluation
# --------------------------
def evaluate_sample(sample: Dict, strict_case: bool = True) -> Dict:
    entities, gt_map = build_gt(sample)
    preds = parse_pred(sample)
    
    # Count valid JSON: if preds is not None, JSON was successfully parsed
    has_valid_json = preds is not None

    per_entity = []
    if preds is None:
        preds = [(None, None, None)] * len(entities)

    # Align predictions with entities order:
    # 1) If predictions contain key/cue, use key->(pe,pl) mapping to align;
    # 2) If predictions don't contain key or key is missing, align by position.
    key2pred = {}
    all_have_key = True
    for tpl in preds:
        k, pe, pl = tpl
        if k is None:
            all_have_key = False
            break
        key2pred[k] = (pe, pl)
    
    # If all predictions have key, use key mapping alignment; otherwise align by position
    use_positional = not all_have_key

    aligned = []
    if use_positional:
        # Align by position
        for i, e in enumerate(entities):
            if i < len(preds):
                aligned.append((e, preds[i][1], preds[i][2]))
            else:
                aligned.append((e, None, None))
    else:
        # Align by key mapping
        for e in entities:
            aligned.append((e, ) + key2pred.get(e, (None, None)))

    # Score each entity
    for (e, pe, pl) in aligned:
        gt_e, gt_l = gt_map[e]
        earliest_ok = eq(pe, gt_e, strict_case) if gt_e is not None else (pe in (None, "UNKNOWN"))
        latest_ok   = eq(pl, gt_l, strict_case) if gt_l is not None else None
        pair_ok     = (earliest_ok and latest_ok) if latest_ok is not None else None

        swap_err = None
        if gt_e is not None and gt_l is not None and pe is not None and pl is not None:
            swap_err = eq(pe, gt_l, strict_case) and eq(pl, gt_e, strict_case)

        per_entity.append({
            "entity": e,
            "gt_earliest": gt_e,
            "gt_latest": gt_l,
            "pred_earliest": pe,
            "pred_latest": pl,
            "earliest_correct": earliest_ok,
            "latest_correct": latest_ok,
            "pair_correct": pair_ok,
            "swap_error": swap_err,
            "pred_unknown_earliest": (pe == "UNKNOWN"),
            "pred_unknown_latest": (pl == "UNKNOWN") if pl is not None else None,
            "earliest_edit": levenshtein(pe, gt_e),
            "latest_edit": levenshtein(pl, gt_l),
        })

    # Aggregate
    def mean_bool(xs):
        xs = [x for x in xs if isinstance(x, bool)]
        return sum(1 for x in xs if x) / len(xs) if xs else None

    def mean_num(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return sum(xs) / len(xs) if xs else None

    earliest_acc = mean_bool([r["earliest_correct"] for r in per_entity])
    latest_acc   = mean_bool([r["latest_correct"]   for r in per_entity])
    pair_acc     = mean_bool([r["pair_correct"]     for r in per_entity])
    swap_rate    = mean_bool([r["swap_error"]       for r in per_entity])
    unk_e_rate   = mean_bool([r["pred_unknown_earliest"] for r in per_entity])
    unk_l_rate   = mean_bool([r["pred_unknown_latest"] for r in per_entity])
    avg_edit_e   = mean_num([r["earliest_edit"] for r in per_entity])
    avg_edit_l   = mean_num([r["latest_edit"]   for r in per_entity])

        # ---- NEW: collect error cases (insert this section) ----
    error_items = []
    for r in per_entity:
        errs = []
        if r.get("swap_error"):
            errs.append("swap")
        if r.get("earliest_correct") is False:
            errs.append("earliest_mismatch")
        if r.get("latest_correct") is False:
            errs.append("latest_mismatch")
        if r.get("pred_unknown_earliest"):
            errs.append("unknown_earliest")
        if r.get("pred_unknown_latest") is True:  # May be None
            errs.append("unknown_latest")

        if errs:
            error_items.append({
                "entity": r["entity"],
                "error_types": errs,
                "gt_earliest": r["gt_earliest"],
                "gt_latest": r["gt_latest"],
                "pred_earliest": r["pred_earliest"],
                "pred_latest": r["pred_latest"],
                "earliest_edit": r["earliest_edit"],
                "latest_edit": r["latest_edit"],
            })

    summary = {
        "n_entities": len(per_entity),
        "earliest_acc": earliest_acc,
        "latest_acc": latest_acc,
        "pair_acc": pair_acc,
        "swap_rate": swap_rate,
        "unknown_earliest_rate": unk_e_rate,
        "unknown_latest_rate": unk_l_rate,
        "avg_edit_earliest": avg_edit_e,
        "avg_edit_latest": avg_edit_l,
        "has_valid_json": has_valid_json,
        "has_errors": len(error_items) > 0,
    }
    return {"per_entity": per_entity, "summary": summary, "errors": error_items}

# --------------------------
# CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="Qwen2.5-7B-Instruct 32", help="JSON file containing samples (single sample or array of samples)")
    ap.add_argument("--file", default="DKI/SyntheticDKI/LLMCall/Results/UpdateTimes_32_gpt-5-nano.json", help="JSON file containing samples (single sample or array of samples)")
    ap.add_argument("--strict_case", action="store_true", help="Strict case sensitivity (default: case-insensitive)")
    args = ap.parse_args()
    print(args.type)
    print("")
    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data if isinstance(data, list) else [data]
    all_summaries = []
    error_indices = []
    latest_correct_indices = []
    valid_json_count = 0
    
    for index, sample in enumerate(samples):
        out = evaluate_sample(sample, strict_case=args.strict_case)
        all_summaries.append(out["summary"])
        
        # Count valid JSON
        if out["summary"].get("has_valid_json", False):
            valid_json_count += 1
        
        # Collect error sample indices
        if out["summary"].get("has_errors", False):
            error_indices.append(index)
        
        # Collect indices of samples with correct latest
        # If latest_acc is 1.0 (all entities' latest are correct), or all entities' latest_correct are True
        latest_acc = out["summary"].get("latest_acc")
        if latest_acc is not None and latest_acc == 1.0:
            latest_correct_indices.append(index)
        elif latest_acc is None:
            # If latest_acc is None, check latest_correct for all entities in per_entity
            per_entity = out.get("per_entity", [])
            if per_entity:
                all_latest_correct = all(
                    r.get("latest_correct") is True 
                    for r in per_entity 
                    if r.get("latest_correct") is not None
                )
                if all_latest_correct:
                    latest_correct_indices.append(index)

    # If multiple samples, output overall average
    if len(all_summaries) > 1:
        def avg_of(key):
            vals = [s[key] for s in all_summaries if s[key] is not None]
            return sum(vals)/len(vals) if vals else None
        
        overall = {k: avg_of(k) for k in all_summaries[0].keys()}
        overall["n_samples"] = len(all_summaries)
        
        # Output statistics in English format
        print("=" * 80)
        print("Evaluation Results Statistics")
        print("=" * 80)
        total_samples = overall.get('n_samples', 0)
        print(f"Total samples: {total_samples}")
        print(f"Valid JSON count: {valid_json_count} ({valid_json_count/total_samples*100:.2f}%)" if total_samples > 0 else f"Valid JSON count: {valid_json_count}")
        
        if overall.get('earliest_acc') is not None:
            print(f"Earliest accuracy: {overall.get('earliest_acc', 0):.4f}")
        else:
            print("Earliest accuracy: N/A")
            
        if overall.get('latest_acc') is not None:
            print(f"Latest accuracy: {overall.get('latest_acc', 0):.4f}")
        else:
            print("Latest accuracy: N/A")
            
        if overall.get('pair_acc') is not None:
            print(f"Pair accuracy: {overall.get('pair_acc', 0):.4f}")
        else:
            print("Pair accuracy: N/A")
            
        if overall.get('swap_rate') is not None:
            print(f"Swap error rate: {overall.get('swap_rate', 0):.4f}")
        else:
            print("Swap error rate: N/A")
            
        if overall.get('unknown_earliest_rate') is not None:
            print(f"Unknown Earliest rate: {overall.get('unknown_earliest_rate', 0):.4f}")
        else:
            print("Unknown Earliest rate: N/A")
            
        if overall.get('unknown_latest_rate') is not None:
            print(f"Unknown Latest rate: {overall.get('unknown_latest_rate', 0):.4f}")
        else:
            print("Unknown Latest rate: N/A")
            
        if overall.get('avg_edit_earliest') is not None:
            print(f"Average Earliest edit distance: {overall.get('avg_edit_earliest', 0):.2f}")
        else:
            print("Average Earliest edit distance: N/A")
            
        if overall.get('avg_edit_latest') is not None:
            print(f"Average Latest edit distance: {overall.get('avg_edit_latest', 0):.2f}")
        else:
            print("Average Latest edit distance: N/A")
            
        print("=" * 80)
        print(f"Sample indices with prediction errors: {error_indices}")
        print(f"Number of error samples: {len(error_indices)}")
        print(f"Sample indices with correct Latest: {latest_correct_indices}")
        print(f"Number of samples with correct Latest: {len(latest_correct_indices)}")
        print("=" * 80)
    else:
        # Single sample case
        summary = all_summaries[0]
        print("=" * 80)
        print("Evaluation Results Statistics")
        print("=" * 80)
        print(f"Valid JSON: {'Yes' if summary.get('has_valid_json', False) else 'No'}")
        if summary.get('earliest_acc') is not None:
            print(f"Earliest accuracy: {summary.get('earliest_acc', 0):.4f}")
        if summary.get('latest_acc') is not None:
            print(f"Latest accuracy: {summary.get('latest_acc', 0):.4f}")
        if summary.get('pair_acc') is not None:
            print(f"Pair accuracy: {summary.get('pair_acc', 0):.4f}")
        print("=" * 80)
        if error_indices:
            print(f"Sample indices with prediction errors: {error_indices}")
        else:
            print("Sample indices with prediction errors: []")
        if latest_correct_indices:
            print(f"Sample indices with correct Latest: {latest_correct_indices}")
        else:
            print("Sample indices with correct Latest: []")
        print("=" * 80)


if __name__ == "__main__":
    main()
