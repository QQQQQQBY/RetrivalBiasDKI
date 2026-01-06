import json
from openai import OpenAI
import os
import time
from collections import defaultdict

# ===== Local LLM Configuration =====

API_BASE = ""
API_KEY = ""  # Most local APIs don't verify key, just a placeholder

MODEL_NAME = ""

MODEL_NAME_new = ""
print(MODEL_NAME)
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
)

# ===== Utility function: Construct CUE_PHRASE based on category =====
def get_cue_phrase_for_category(category: str, entity: str, relation: str) -> str:
    """
    Construct CUE_PHRASE based on category, keeping rules consistent with story generation
    """
    if category == "athletes_byPayment":
        # athletes_byPayment: "football club of <player>"
        return f"{relation} of {entity}"
    
    elif category == "companies_byRevenue":
        # companies_byRevenue: "Chief Executive Officer of <company>"
        if entity.lower() in relation.lower():
            return relation
        else:
            return f"{relation} of {entity}"
    
    elif category == "organizations":
        # organizations: "general secretary of <organization>"
        if entity.lower() in relation.lower():
            return relation
        else:
            return f"{relation} of {entity}"
    
    elif category == "countries_byGDP":
        # countries_byGDP: relation itself already contains country name, e.g., "President of Italy"
        return relation
    
    else:
        # Default case
        if entity.lower() in relation.lower():
            return relation
        else:
            return f"{relation} of {entity}"


# ===== Construct evaluation prompts for different categories =====
def build_eval_prompt_for_category(category: str, cue_phrase: str, text: str) -> str:
    """
    Construct different evaluation prompts based on category
    """
    if category == "athletes_byPayment":
        # Football club evaluation: find "The football club of <player> was <club>"
        return f"""You are an information extraction system.

        You are given:
        1) a CUE string
        2) a long narrative TEXT.

        CUE: "{cue_phrase}"

        Within the TEXT, cue–value records for this CUE appear in sentences that contain
        the exact substring:

        "The {cue_phrase} was VALUE"

        or

        "the {cue_phrase} was VALUE"

        where VALUE is a club name exactly as it appears in the text.

        Definitions:
        - Earliest VALUE: the club name from the FIRST sentence in TEXT (from top to bottom)
        that contains "The {cue_phrase} was VALUE" or "the {cue_phrase} was VALUE".
        - Latest VALUE: the club name from the LAST sentence in TEXT that contains that substring.

        Your task:
        - Read the TEXT carefully.
        - Find all clubs that appear in such sentences for this CUE.
        - Among them, identify the earliest and latest according to the above definitions.
        - If no such club appears in the TEXT, use "UNKNOWN" for both.

        Output format (VERY IMPORTANT):
        - Output ONLY a single valid JSON object.
        - No extra text, no explanation, no markdown.
        - The JSON object must have exactly the following fields:

        {{
        "cue": "{cue_phrase}",
        "earliest": "<VERBATIM or UNKNOWN>",
        "latest": "<VERBATIM or UNKNOWN>"
        }}

        Where:
        - "cue" MUST be exactly "{cue_phrase}".
        - "earliest" and "latest" MUST be either:
        - a club name copied verbatim from the TEXT, or
        - the string "UNKNOWN" (all caps) if you cannot find any match.

        Now here is the TEXT:

        {text}
"""
    
    elif category == "companies_byRevenue":
        # CEO and organizational position evaluation: find various natural language expressions
        return f"""You are an information extraction system.

        You are given:
        - a CUE string
        - a long narrative TEXT.

        CUE: "{cue_phrase}"

        Within the TEXT, some sentences state that a person holds this role.
        Treat any sentence that clearly asserts that a person is the {cue_phrase} as a cue–value record.
        Typical patterns may look like:
        - "<PERSON> was the {cue_phrase}"
        - "the {cue_phrase} was <PERSON>"
        or other natural variants that unambiguously say that someone is the {cue_phrase}.

        Definitions:
        - Earliest VALUE: the PERSON name from the FIRST sentence in TEXT (from top to bottom)
        that asserts someone is the {cue_phrase}.
        - Latest VALUE: the PERSON name from the LAST sentence in TEXT that asserts someone
        is the {cue_phrase}.

        Your task:
        - Read the TEXT carefully.
        - Find all people who are described as the {cue_phrase}.
        - Among them, identify the earliest and latest according to the above definitions.
        - If no such person appears in the TEXT, use "UNKNOWN" for both.

        Output format (VERY IMPORTANT):
        - Output ONLY a single valid JSON object.
        - No extra text, no explanation, no markdown.
        - The JSON object must have exactly the following fields:

        {{
        "cue": "{cue_phrase}",
        "earliest": "<VERBATIM or UNKNOWN>",
        "latest": "<VERBATIM or UNKNOWN>"
        }}

        Where:
        - "cue" MUST be exactly "{cue_phrase}".
        - "earliest" and "latest" MUST be either:
        - a PERSON name copied verbatim from the TEXT, or
        - the string "UNKNOWN" (all caps) if you cannot find any match.

        Now here is the TEXT:

        {text}
"""
    elif category == "organizations":
        return f"""You are an information extraction system.

You are given:
1) a ROLE_PHRASE string
2) a long narrative TEXT.

Within the TEXT, records for this role appear in sentences that contain
the exact substring:

  "<VALUE> served as the {cue_phrase}"

where VALUE is a person name exactly as it appears in the text.

Definitions:
- Earliest VALUE: the VALUE from the first sentence in TEXT (from top to bottom)
  that contains a substring of the form "<VALUE> served as the {cue_phrase}".
- Latest VALUE: the VALUE from the last sentence in TEXT that contains such a substring.

Your task:
- Find the earliest and latest VALUE for this ROLE_PHRASE according to the definitions.
- If no such sentence exists, return "UNKNOWN" for both.

Output format:
- Output ONLY a single valid JSON object, with no extra text, no explanation, no markdown.
- The JSON object must have the following fields:

{{
  "role_phrase": "{cue_phrase}",
  "earliest": "<VERBATIM or UNKNOWN>",
  "latest": "<VERBATIM or UNKNOWN>"
}}

Where:
- "role_phrase" must be exactly the string "{cue_phrase}".
- "earliest" and "latest" must be either:
  - a VALUE copied verbatim from the TEXT, or
  - the string "UNKNOWN" if you cannot find any matching sentence.

Now here is the TEXT:

{text}
"""
    elif category == "countries_byGDP":
        # Country position evaluation: find "During/In <period>, the <cue> was <value>"
        return f"""You are an information extraction system.

You are given:
1) a CUE string
2) a long narrative TEXT.

Within the TEXT, cue–value records for this CUE appear in sentences that contain
the exact substring:

  "the {cue_phrase} was VALUE"

or

  "The {cue_phrase} was VALUE"

where VALUE is a person name exactly as it appears in the text.

Definitions:
- Earliest VALUE: the VALUE from the first sentence in TEXT (from top to bottom)
  that contains "the {cue_phrase} was VALUE" or "The {cue_phrase} was VALUE".
- Latest VALUE: the VALUE from the last sentence in TEXT that contains that pattern.

Your task:
- Find the earliest and latest VALUE for the given CUE according to these definitions.
- If no such sentence exists for the CUE, return "UNKNOWN" for both.

Output format:
- Output ONLY a single valid JSON object, with no extra text, no explanation, no markdown.
- The JSON object must have the following fields:

{{
  "cue": "{cue_phrase}",
  "earliest": "<VERBATIM or UNKNOWN>",
  "latest": "<VERBATIM or UNKNOWN>"
}}

Where:
- "cue" must be exactly the string "{cue_phrase}".
- "earliest" and "latest" must be either:
  - a VALUE copied verbatim from the TEXT, or
  - the string "UNKNOWN" if you cannot find any matching sentence.

Now here is the TEXT:

{text}
"""
    


# ===== Read merged data and ground truth data for each category =====
base_dir = "DKI/RealWorldDKI/LongTextDataset"
merged_stories_path = os.path.join(base_dir, "LongStory.json")

# Read merged story data
with open(merged_stories_path, "r", encoding="utf-8") as f:
    merged_stories = json.load(f)

print(f"Loaded {len(merged_stories)} merged stories.")

# Read ground truth data for each category
category_records = {}
categories = ["athletes_byPayment", "companies_byRevenue", "countries_byGDP", "organizations"]

for category in categories:
    records_path = os.path.join(base_dir, "category_outputs", category, "key_value_data.json")
    if os.path.exists(records_path):
        with open(records_path, "r", encoding="utf-8") as f:
            category_records[category] = json.load(f)
        print(f"Loaded {len(category_records[category])} records for {category}")
    else:
        print(f"Warning: {records_path} not found")
        category_records[category] = []


# ===== Build index mapping: from merged_stories index to category_records index =====
# Since merged_stories are merged in category order, we need to build a mapping relationship
category_index_map = {}
current_idx = 0

for category in categories:
    category_count = len(category_records[category])
    category_index_map[category] = {
        'start': current_idx,
        'end': current_idx + category_count,
        'records': category_records[category]
    }
    current_idx += category_count


# ===== Read existing evaluation results (if exists) =====
existing_results_path = os.path.join(base_dir, "long_story_results_gpt-5-nano.json")
existing_results = {}
existing_results_list = []

if os.path.exists(existing_results_path):
    print(f"\nLoading existing results from {existing_results_path}")
    with open(existing_results_path, "r", encoding="utf-8") as f:
        existing_results_list = json.load(f)
    
    # Create index mapping for easy lookup
    for result in existing_results_list:
        idx = result.get("index", -1)
        if idx >= 0:
            existing_results[idx] = result
    
    print(f"Loaded {len(existing_results_list)} existing results")
    null_count = sum(1 for r in existing_results_list if r.get("model_parsed") is None)
    print(f"Found {null_count} items with model_parsed=None that need to be re-evaluated")
else:
    print(f"\nNo existing results file found at {existing_results_path}, will evaluate all items")

# ===== Call model for earliest/latest testing =====
eval_results = []
# Record which items have been processed (re-evaluated or using existing results)
processed_indices = set()

for story_obj in merged_stories:
    idx = story_obj.get("index", -1)
    category = story_obj.get("category", "")
    original_idx = story_obj.get("original_index", -1)
    
    # Check if existing result exists and model_parsed is not null
    if idx in existing_results:
        existing_result = existing_results[idx]
        if existing_result.get("model_parsed") is not None:
            # Already have valid result, use it directly
            print(f"=== Sample {idx}: Using existing result (model_parsed is not null) ===")
            eval_results.append(existing_result)
            processed_indices.add(idx)
            continue
        else:
            # model_parsed is null, need to re-evaluate
            print(f"=== Re-evaluating sample {idx} (model_parsed is null) ===")
            # Get basic information from existing result
            entity = existing_result.get("entity")
            relation = existing_result.get("relation")
            cue_phrase = existing_result.get("cue_phrase")
            earliest_true = existing_result.get("earliest_true")
            latest_true = existing_result.get("latest_true")
    else:
        # No existing result, need full evaluation
        print(f"=== Evaluating sample {idx} (category: {category}, original_index: {original_idx}) ===")
        
        # Get corresponding ground truth data
        if category not in category_index_map:
            print(f"Warning: Unknown category {category}, skipping.")
            continue
        
        category_info = category_index_map[category]
        if original_idx < 0 or original_idx >= len(category_info['records']):
            print(f"Warning: Invalid original_index {original_idx} for category {category}, skipping.")
            continue
        
        records = category_info['records'][original_idx]
        if not records or len(records) == 0:
            print(f"Warning: Empty records for sample {idx}, skipping.")
            continue
        
        # Extract entity, relation, and ground truth values
        entity, relation, _ = records[0]  # e.g., "Cristiano Ronaldo", "football club", "Sporting CP"
        cue_phrase = get_cue_phrase_for_category(category, entity, relation)
        earliest_true = records[0][2]      # First value
        latest_true = records[-1][2]       # Last value
    
    text = story_obj.get("generated_text", "") or ""
    if not text.strip():
        print(f"Sample {idx} has empty generated_text, skipping.")
        # If re-evaluating from existing result, update original result; otherwise create new result
        if idx in existing_results:
            updated_result = existing_results[idx].copy()
            updated_result.update({
                "model_raw": "",
                "model_parsed": None,
                "earliest_pred": None,
                "latest_pred": None,
                "earliest_correct": None,
                "latest_correct": None,
                "note": "empty_generated_text",
            })
            eval_results.append(updated_result)
            processed_indices.add(idx)
        else:
            eval_results.append({
                "index": idx,
                "category": category,
                "original_index": original_idx,
                "entity": entity if 'entity' in locals() else None,
                "relation": relation if 'relation' in locals() else None,
                "cue_phrase": cue_phrase if 'cue_phrase' in locals() else None,
                "earliest_true": earliest_true if 'earliest_true' in locals() else None,
                "latest_true": latest_true if 'latest_true' in locals() else None,
                "model_raw": "",
                "model_parsed": None,
                "earliest_pred": None,
                "latest_pred": None,
                "earliest_correct": None,
                "latest_correct": None,
                "note": "empty_generated_text",
            })
            processed_indices.add(idx)
        continue
    
    prompt = build_eval_prompt_for_category(category, cue_phrase, text)
    
    try:
        # Loop calling model until getting non-empty response
        max_retries = 20  # Maximum retry count to avoid infinite loop
        retry_count = 0
        raw_content = ""
        response = None
        
        while retry_count < max_retries:
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,   # Extraction tasks recommend 0 for determinism
                    max_tokens=1024,
                )
                
                raw_content = response.choices[0].message.content or ""
                
                # If response is not empty, break loop
                if raw_content.strip():
                    break
                else:
                    retry_count += 1
                    print(f"  Warning: Empty response, retrying ({retry_count}/{max_retries})...")
                    time.sleep(1)  # Wait 1 second before retry
                    
            except Exception as retry_error:
                retry_count += 1
                print(f"  Error during API call, retrying ({retry_count}/{max_retries}): {retry_error}")
                time.sleep(2)  # Wait longer when error occurs
                if retry_count >= max_retries:
                    raise retry_error
        
        if not raw_content.strip():
            print(f"  Warning: Still empty after {max_retries} retries, using empty string")
        
        print("Raw model output:", raw_content[:200] + "..." if len(raw_content) > 200 else raw_content)
        
        # Try to parse JSON
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            # Try to extract JSON part
            import re
            json_match = re.search(r'\{[^{}]*\}', raw_content)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                except:
                    parsed = None
            else:
                parsed = None
        
        if parsed is not None:
            pred_earliest = parsed.get("earliest")
            pred_latest = parsed.get("latest")
            earliest_correct = (pred_earliest == earliest_true) if pred_earliest else False
            latest_correct = (pred_latest == latest_true) if pred_latest else False
        else:
            pred_earliest = None
            pred_latest = None
            earliest_correct = None
            latest_correct = None
        
        # If re-evaluating from existing result, update original result; otherwise create new result
        if idx in existing_results:
            # Update existing result
            updated_result = existing_results[idx].copy()
            updated_result.update({
                "model_raw": raw_content,
                "model_parsed": parsed,
                "earliest_pred": pred_earliest,
                "latest_pred": pred_latest,
                "earliest_correct": earliest_correct,
                "latest_correct": latest_correct,
            })
            eval_results.append(updated_result)
            processed_indices.add(idx)
        else:
            # Create new result
            eval_results.append({
                "index": idx,
                "category": category,
                "original_index": original_idx,
                "entity": entity,
                "relation": relation,
                "cue_phrase": cue_phrase,
                "earliest_true": earliest_true,
                "latest_true": latest_true,
                "model_raw": raw_content,
                "model_parsed": parsed,
                "earliest_pred": pred_earliest,
                "latest_pred": pred_latest,
                "earliest_correct": earliest_correct,
                "latest_correct": latest_correct,
            })
            processed_indices.add(idx)
        
    except Exception as e:
        print(f"Error when querying model on sample {idx}: {e}")
        
        # If re-evaluating from existing result, update original result; otherwise create new result
        if idx in existing_results:
            updated_result = existing_results[idx].copy()
            updated_result.update({
                "model_raw": "",
                "model_parsed": None,
                "earliest_pred": None,
                "latest_pred": None,
                "earliest_correct": None,
                "latest_correct": None,
                "error": str(e),
            })
            eval_results.append(updated_result)
            processed_indices.add(idx)
        else:
            eval_results.append({
                "index": idx,
                "category": category,
                "original_index": original_idx,
                "entity": entity if 'entity' in locals() else None,
                "relation": relation if 'relation' in locals() else None,
                "cue_phrase": cue_phrase if 'cue_phrase' in locals() else None,
                "earliest_true": earliest_true if 'earliest_true' in locals() else None,
                "latest_true": latest_true if 'latest_true' in locals() else None,
                "model_raw": "",
                "model_parsed": None,
                "earliest_pred": None,
                "latest_pred": None,
                "earliest_correct": None,
                "latest_correct": None,
                "error": str(e),
            })
            processed_indices.add(idx)
    
    time.sleep(0.2)  # Can sleep a bit to avoid calling too frequently

# ===== Ensure all items are included in results =====
# If there are existing results but unprocessed items, also add them to results
if existing_results:
    for idx, existing_result in existing_results.items():
        if idx not in processed_indices:
            print(f"=== Sample {idx}: Adding existing result (not processed) ===")
            eval_results.append(existing_result)
            processed_indices.add(idx)

# ===== Sort by index to ensure correct order =====
eval_results.sort(key=lambda x: x.get("index", -1))

# ===== Save evaluation results =====
out_path = os.path.join(base_dir, f"eval_merged_del_n_story_results_{MODEL_NAME_new}.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(eval_results, f, ensure_ascii=False, indent=2)

print(f"\nSaved eval results to {out_path}")

# ===== Statistics =====
print("\n" + "="*60)
print("Evaluation Summary")
print("="*60)

total = len(eval_results)
earliest_correct_count = sum(1 for r in eval_results if r.get("earliest_correct") == True)
latest_correct_count = sum(1 for r in eval_results if r.get("latest_correct") == True)
both_correct_count = sum(1 for r in eval_results if r.get("earliest_correct") == True and r.get("latest_correct") == True)

print(f"\nTotal samples: {total}")
print(f"Earliest correct: {earliest_correct_count}/{total} ({earliest_correct_count/total*100:.2f}%)")
print(f"Latest correct: {latest_correct_count}/{total} ({latest_correct_count/total*100:.2f}%)")
print(f"Both correct: {both_correct_count}/{total} ({both_correct_count/total*100:.2f}%)")

# Statistics by category
print("\n" + "-"*60)
print("Results by Category")
print("-"*60)

for category in categories:
    category_results = [r for r in eval_results if r.get("category") == category]
    if not category_results:
        continue
    
    cat_total = len(category_results)
    cat_earliest_correct = sum(1 for r in category_results if r.get("earliest_correct") == True)
    cat_latest_correct = sum(1 for r in category_results if r.get("latest_correct") == True)
    cat_both_correct = sum(1 for r in category_results if r.get("earliest_correct") == True and r.get("latest_correct") == True)
    
    print(f"\n{category}:")
    print(f"  Total: {cat_total}")
    print(f"  Earliest correct: {cat_earliest_correct}/{cat_total} ({cat_earliest_correct/cat_total*100:.2f}%)")
    print(f"  Latest correct: {cat_latest_correct}/{cat_total} ({cat_latest_correct/cat_total*100:.2f}%)")
    print(f"  Both correct: {cat_both_correct}/{cat_total} ({cat_both_correct/cat_total*100:.2f}%)")

print("\n" + "="*60)

