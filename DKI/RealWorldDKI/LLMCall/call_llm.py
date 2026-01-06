"""
Test the performance of prompt datasets on LLM models
"""

import argparse
import json
import os
import re
import requests
from typing import Dict, List, Any, Optional
from tqdm import tqdm


def call_llm_api(prompt: str, instruction: str, api_base: str, model: str, 
                 api_key: str, temperature: float = 0.0) -> Optional[str]:
    """
    Call LLM API
    
    Args:
        prompt: User input
        instruction: Instruction
        api_base: API base URL
        model: Model name
        api_key: API key
        temperature: Temperature parameter
    
    Returns:
        Model response text, returns None if failed
    """
    url = f"{api_base}/chat/completions"
    
    # Build messages
    messages = [
        {"role": "system", "content": (
      "Only output a single JSON object and nothing else. No code, no prose, no markdown.\n"
      "If you violate this, replace your entire reply with exactly: "
      '{"rationale":"FORMAT_ERROR","final":[]}'
    )},
        {"role": "user", "content": instruction+prompt}
    ]
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=500)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"API call failed: {e}")
        return None


def extract_json_from_response(response: str) -> Optional[Dict]:
    """
    Extract JSON object from response
    
    Args:
        response: Model response text
    
    Returns:
        Parsed JSON dictionary, returns None if failed
    """
    if not response:
        return None
    
    # Try direct parsing
    try:
        return json.loads(response)
    except:
        pass
    
    # Try extracting JSON code block (supports nested JSON)
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    # Try finding JSON object containing earliest and latest (supports nesting)
    # Use smarter method: find first { then match to corresponding }
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(response):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                # Found a complete JSON object
                json_str = response[start_idx:i+1]
                try:
                    parsed = json.loads(json_str)
                    # Check if contains earliest/latest or final fields
                    if ("earliest" in parsed and "latest" in parsed) or "final" in parsed:
                        return parsed
                except:
                    pass
    
    # Try finding any complete JSON object (starting from first {)
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(response):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                json_str = response[start_idx:i+1]
                try:
                    return json.loads(json_str)
                except:
                    pass
    
    return None


def normalize_name(name) -> str:
    """
    Normalize name for comparison
    If name contains ':' separator (e.g., "cue:value"), extract the part after ':'
    
    Args:
        name: Name (can be string, None, or other types)
    
    Returns:
        Normalized string (lowercase)
    """
    # Handle None and non-string types
    if name is None:
        return ""
    if not isinstance(name, str):
        name = str(name)
    if not name:
        return ""
    # Remove leading and trailing spaces
    name = name.strip()
    # If contains ':' separator, extract the part after it (value part)
    if ':' in name:
        parts = name.split(':', 1)  # Only split the first ':'
        if len(parts) == 2:
            name = parts[1]  # Take the value part
    # Convert to lowercase
    return name.lower()


def extract_earliest_latest(data: Dict) -> tuple:
    """
    Extract earliest and latest values from JSON dictionary
    Supports multiple formats:
    1. {"earliest": "...", "latest": "..."}
    2. {"final": [{"key": "...", "earliest": "...", "latest": "..."}]}
    3. Cases with only earliest or latest (extract separately)
    
    Args:
        data: JSON dictionary
    
    Returns:
        (earliest, latest) tuple, returns empty strings if not present
    """
    # Format 1: Directly contains earliest and latest
    if "earliest" in data or "latest" in data:
        earliest = data.get("earliest", "")
        latest = data.get("latest", "")
        # Ensure return type is string
        if earliest is None:
            earliest = ""
        if latest is None:
            latest = ""
        return str(earliest), str(latest)
    
    # Format 2: Contains final array
    if "final" in data and isinstance(data["final"], list) and len(data["final"]) > 0:
        # Take first element (usually only one)
        first_item = data["final"][0]
        if isinstance(first_item, dict):
            earliest = first_item.get("earliest", "")
            latest = first_item.get("latest", "")
            # Ensure return type is string
            if earliest is None:
                earliest = ""
            if latest is None:
                latest = ""
            return str(earliest), str(latest)
    
    return "", ""


def evaluate_response(predicted: Dict, ground_truth: Dict) -> Dict[str, bool]:
    """
    Evaluate prediction results
    
    Args:
        predicted: Predicted JSON dictionary
        ground_truth: Ground truth dictionary
    
    Returns:
        Evaluation results dictionary
    """
    results = {
        "earliest_correct": False,
        "latest_correct": False,
        "both_correct": False,
        "has_valid_json": predicted is not None
    }
    
    if predicted is None:
        return results
    
    # Extract earliest and latest from predicted
    pred_earliest, pred_latest = extract_earliest_latest(predicted)
    pred_earliest = normalize_name(pred_earliest)
    pred_latest = normalize_name(pred_latest)
    
    # Extract earliest and latest from ground_truth
    gt_earliest, gt_latest = extract_earliest_latest(ground_truth)
    gt_earliest = normalize_name(gt_earliest)
    gt_latest = normalize_name(gt_latest)
    
    results["earliest_correct"] = pred_earliest == gt_earliest
    results["latest_correct"] = pred_latest == gt_latest
    results["both_correct"] = results["earliest_correct"] and results["latest_correct"]
    
    return results


def process_dataset(dataset_path: str, api_base: str, model: str, api_key: str, 
                   temperature: float = 0.0, output_path: str = None) -> Dict[str, Any]:
    """
    Process a single dataset
    
    Args:
        dataset_path: Dataset file path
        api_base: API base URL
        model: Model name
        api_key: API key
        temperature: Temperature parameter
        output_path: Output file path
    
    Returns:
        Evaluation results statistics
    """
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    results = []
    stats = {
        "total": len(dataset),
        "earliest_correct": 0,
        "latest_correct": 0,
        "both_correct": 0,
        "valid_json": 0,
        "invalid_json": 0,
        "api_errors": 0
    }
    
    print(f"\nProcessing dataset: {dataset_path}")
    print(f"Total samples: {len(dataset)}")
    
    for item in tqdm(dataset, desc="Processing"):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        ground_truth_str = item.get("output", "")
        
        # Parse ground truth
        try:
            ground_truth = json.loads(ground_truth_str)
        except:
            # If parsing fails, try using fields from item
            # Prefer parsed result from output field, if fails use earliest and latest fields
            ground_truth = {
                "earliest": item.get("earliest", ""),
                "latest": item.get("latest", "")
            }
        
        # Call API
        response = call_llm_api(input_text, instruction, api_base, model, api_key, temperature)
        
        # Extract JSON
        predicted = extract_json_from_response(response) if response else None
        
        # Evaluate
        eval_results = evaluate_response(predicted, ground_truth)
        
        # Update statistics
        if predicted is not None:
            stats["valid_json"] += 1
        else:
            stats["invalid_json"] += 1
        
        if response is None:
            stats["api_errors"] += 1
        
        if eval_results["earliest_correct"]:
            stats["earliest_correct"] += 1
        if eval_results["latest_correct"]:
            stats["latest_correct"] += 1
        if eval_results["both_correct"]:
            stats["both_correct"] += 1
        
        # Save results
        result_item = {
            **item,
            "response": response,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "evaluation": eval_results
        }
        results.append(result_item)
    
    # Calculate accuracy
    if stats["total"] > 0:
        stats["earliest_accuracy"] = stats["earliest_correct"] / stats["total"]
        stats["latest_accuracy"] = stats["latest_correct"] / stats["total"]
        stats["both_accuracy"] = stats["both_correct"] / stats["total"]
        stats["valid_json_rate"] = stats["valid_json"] / stats["total"]
    
    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Test prompt dataset performance on LLM")
    parser.add_argument("--type", type=str, default="original")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--api_base", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output_path", type=str, default="DKI/RealWorldDKI/LLMCall/Results/realworld_32_llama_3.1_8b.json")
    parser.add_argument("--dataset_path", type=str, default="DKI/RealWorldDKI/Dataset/realworld_32.json")
    args = parser.parse_args()
    print(args.type)
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"API Base: {args.api_base}")
    print("=" * 80)
    all_stats = {}
    

    output_path = args.output_path
        
    stats = process_dataset(
        dataset_path=args.dataset_path,
        api_base=args.api_base,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        output_path=output_path
    )
        
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total samples: {stats['total']}")
    print(f"  Earliest accuracy: {stats.get('earliest_accuracy', 0):.4f} ({stats['earliest_correct']}/{stats['total']})")
    print(f"  Latest accuracy: {stats.get('latest_accuracy', 0):.4f} ({stats['latest_correct']}/{stats['total']})")
    print(f"  Both correct: {stats.get('both_accuracy', 0):.4f} ({stats['both_correct']}/{stats['total']})")
    print(f"  Valid JSON rate: {stats.get('valid_json_rate', 0):.4f} ({stats['valid_json']}/{stats['total']})")
    print(f"  API errors: {stats['api_errors']}")
    
    # Calculate overall statistics
    total_samples = sum(s["total"] for s in all_stats.values())
    total_earliest_correct = sum(s["earliest_correct"] for s in all_stats.values())
    total_latest_correct = sum(s["latest_correct"] for s in all_stats.values())
    total_both_correct = sum(s["both_correct"] for s in all_stats.values())
    total_valid_json = sum(s["valid_json"] for s in all_stats.values())
    
    print("\n" + "=" * 80)
    print("Overall Statistics:")
    print(f"  Total samples: {total_samples}")
    if total_samples > 0:
        print(f"  Earliest accuracy: {total_earliest_correct/total_samples:.4f} ({total_earliest_correct}/{total_samples})")
        print(f"  Latest accuracy: {total_latest_correct/total_samples:.4f} ({total_latest_correct}/{total_samples})")
        print(f"  Both correct: {total_both_correct/total_samples:.4f} ({total_both_correct}/{total_samples})")
        print(f"  Valid JSON rate: {total_valid_json/total_samples:.4f} ({total_valid_json}/{total_samples})")
    print("=" * 80)

    print("Test completed!")


if __name__ == "__main__":
    main()

