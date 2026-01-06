import json
from typing import List, Dict, Tuple
import random
import re

def parse_answer(answer_str: str) -> Tuple[str, str, str]:
    name_match = answer_str.split("|")[0].strip()
    name = name_match
    start_match = re.search(r'S:\s*\+(\d{4}-\d{2}-\d{2})T', answer_str)
    start_date = start_match.group(1) if start_match else None
    end_match = re.search(r'E:\s*\+(\d{4}-\d{2}-\d{2})T', answer_str)
    end_date = end_match.group(1) if end_match else None
    
    return name, start_date, end_date

def build_instruction_input_output(role: str, kv_list: List[str], 
                                   entities: List[str], earliest: str, latest: str, instruction: str, input_text:str) -> Tuple[str, str, str]:
    """
    instruction, input, output
    Returns:
        (instruction, input, output)
    """
    input_text = input_text.format(
        entities_json=json.dumps(entities, ensure_ascii=False),
        kv_list="\n".join(kv_list),
    )
    final = {
            "cue": entities[0],
            "latest": latest,
            "earliest": earliest
        }
    # output_dict = {
    #     "final": final_list
    # }
    output = json.dumps(final, ensure_ascii=False, indent=2)
    
    return instruction, input_text, output

def process_single_entry(entity_name: str, role: str, answers: List[str], 
                         default_end_date: str, instruction: str, input_text:str) -> Dict:
    if not answers:
        return None

    records = []
    original_records = [] 
    for answer_str in answers:
        name, start_date, end_date = parse_answer(answer_str)
        
        if name and start_date:
            has_end_date = end_date is not None
            if not end_date:
                end_date = default_end_date
            
            records.append((name, start_date, end_date))
            original_records.append((name, start_date, end_date, has_end_date))
    
    if not records:
        return None

    sorted_records = sorted(records, key=lambda x: x[1]) 

    kv_list = []
    kv_list_index = []
    index = 0
    for name, start_date, end_date in sorted_records:
        index = index + 1
        kv_record = f"{index}. {role}:{name}"
        kv_list_index.append(kv_record)
        kv_list.append(f"{role}:{name}")
    if kv_list:
        first_kv = kv_list[0]
        earliest = first_kv.split(":", 1)[1] if ":" in first_kv else ""        

        last_kv = kv_list[-1]
        latest = last_kv.split(":", 1)[1] if ":" in last_kv else ""
    else:
        earliest, latest = "", ""

    entities = [role]
    instruction, input_text, output = build_instruction_input_output(
        role, kv_list, entities, earliest, latest, instruction, input_text
    )
    
    result = {
        "entity": entity_name,
        "role": role,
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "earliest": earliest,
        "latest": latest
    }
    
    return result


def process_data(input_file: str, output_file: str, 
                         default_end_date: str, instruction: str, input_text:str):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = []
    data_types = ["countries_byGDP", "athletes_byPayment", "companies_byRevenue", "organizations"]
    
   
    for data_type in data_types:
        type_data = data.get(data_type, {})
        if data_type == "countries_byGDP":
            for country, roles in type_data.items():
                for role, role_data in roles.items(): 
                    answers = role_data.get("answers", [])
                    result = process_single_entry(country, role, answers, default_end_date, instruction, input_text)
                    if result:
                        result["country"] = country  
                        results.append(result)
        elif data_type == "athletes_byPayment":
            for athlete, athlete_data in type_data.items():
                questions = athlete_data.get("questions", {})
                generic_question = questions.get("generic", "")
                if "club" in generic_question.lower():
                    role = f"club of {athlete}"
                elif "team" in generic_question.lower():
                    role = f"team of {athlete}"
                else:
                    role = f"organization of {athlete}"
                
                answers = athlete_data.get("answers", [])
                result = process_single_entry(athlete, role, answers, default_end_date, instruction, input_text)
                if result:
                    results.append(result)

        elif data_type == "companies_byRevenue":

            for company, company_data in type_data.items():
                questions = company_data.get("questions", {})
                generic_question = questions.get("generic", "")
                if "chief executive officer" in generic_question.lower() or "ceo" in generic_question.lower():
                    role = f"Chief Executive Officer of {company}"
                elif "president" in generic_question.lower():
                    role = f"President of {company}"
                else:
                    role = f"executive of {company}"
                
                answers = company_data.get("answers", [])
                result = process_single_entry(company, role, answers, default_end_date, instruction, input_text)
                if result:
                    results.append(result)

        elif data_type == "organizations":

            for organization, roles in type_data.items():
                for role, role_data in roles.items():
                    answers = role_data.get("answers", [])
                    full_role = f"{role} of {organization}"
                    result = process_single_entry(organization, full_role, answers, default_end_date,instruction, input_text)
                    if result:
                        results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results