import argparse
from dataclasses import dataclass, asdict
import json 
import pandas as pd
import numpy as np
from math import floor, ceil
import random
from typing import List
from tqdm import tqdm


def gen_dict(seed, entity, round):
    df = pd.read_csv("DKI/SyntheticDKI/Words.csv")
    word_list = df["Word"].dropna().astype(str).str.strip().tolist()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(word_list), size=len(entity), replace=False)
    selected = [word_list[i] for i in idx]
    Results = []
    answer = {}
    for j in range(len(selected)):
        text = ""+entity[j]+":"+selected[j]
        Results.append(text)
        answer[entity[j]] = selected[j]
    return Results, answer 

def gen_word_list(R: int, seed: int, N: int):
    rounds = []
    rng = np.random.default_rng(seed)
    df = pd.read_csv("DKI/SyntheticDKI/Words.csv")
    word_list = df["Word"].dropna().astype(str).str.strip().tolist()
    entity_idx = rng.choice(len(word_list), size=N, replace=False)
    ENTITIES = [word_list[i] for i in entity_idx]
    answer_list = []
    for r in range(R):
        KV_list, answer = gen_dict(seed + r + 1, ENTITIES, r)
        rounds.extend(KV_list)  
        if r == 0:
            answer_list.append({r:answer})
        elif r == R - 1:
            answer_list.append({1:answer})
    return rounds, answer_list, ENTITIES  


def build_prompts(kv_list, tem_prompt, entities, R):
    index = 0
    # kv_record = []
    # for item in kv_list:
    #     index = index + 1
    #     kv_record.append(f"{index}. {item}")
    present = tem_prompt.format(cue_json = json.dumps(entities, ensure_ascii=False), records_text = "\n".join(kv_list))
    return {"present": present}


@dataclass
class GenerateData:
    index: int
    words: List[str]
    entities: List[str]
    present_prompt: str

def generate_data_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testsample", type=int, default=200)
    parser.add_argument("--R", type=int, default=32) 
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--output", type=str, default="DKI/SyntheticDKI/InterventionDataset/synthetic_FewShot_32.json")
    parser.add_argument("--f", type=int, default=2) # 0,1,2,3,4 random seed
    tem_prompt = (
    "You are given a long list of cue–value records and a single target cue (CUE).\n"
    "For this target cue, return its earliest and latest VALUE according to FIRST and LAST "
    "occurrence order in the provided record list (top to bottom).\n"
    "\n"
    "Below are two examples. Follow the same pattern: \n"
    "EXAMPLE 1 \n"
    "edgewise:artistic\n"
    "edgewise:tributes\n"
    "edgewise:overplay\n"
    "edgewise:cowardly\n"
    "edgewise:applause\n"
    "edgewise:slavered\n"
    "edgewise:coincide\n"
    "edgewise:teletype\n"
    "edgewise:sunburnt\n"
    "\n"
    "Correct output:\n"
    "{\n"
    "  \"cue\":\"edgewise\",\n"
    "  \"latest\":\"sunburnt\",\n"
    "  \"earliest\":\"artistic\"\n"
    "}\n"
    "\n"
    "EXAMPLE 2 \n"
    "tributes:coherent\n"
    "tributes:allergen\n"
    "tributes:shivered\n"
    "tributes:cowardly\n"
    "tributes:arranged\n"
    "tributes:emeritus\n"
    "tributes:teletype\n"
    "tributes:antennae\n"
    "\n"
    "Correct output:\n"
    "{\n"
    "  \"cue\":\"tributes\",\n"
    "  \"latest\":\"antennae\",\n"
    "  \"earliest\":\"coherent\"\n"
    "}\n"
    "\n"
    "Now solve the following instance in exactly the same way.\n"
    "\n"
    "CUE (JSON array): {cue_json}\n"
    "\n"
     "INPUT FORMAT\n"
    "- Each record is one line in the form: cue:value\n"
    "- Boundaries: lines strictly between the literal markers START: and END\n"
    "\n"
    "START:\n"
    "{records_text}\n"
    "END\n"
    "\n"
    "Output (valid JSON only):\n"
    '{{"cue":"<cue>","earliest":"<VERBATIM or UNKNOWN>", "latest":"<VERBATIM or UNKNOWN>"}},\n'
    "\n"
    "Rules:\n"
    "- Only the cue specified by CUE; ignore all other cues.\n"
    "- Earliest/Latest are defined strictly by first/last appearance order within START..END.\n"
    "- Keep VALUE VERBATIM; JSON-escape only as required.\n"
    "- Output exactly one JSON object and nothing else. No code, no prose, no markdown.\n"
    "\n"
)
    parser.add_argument("--tem_prompt", type=str, default=tem_prompt)
    parser.add_argument("--kv_input", type=str, default="")
    args = parser.parse_args()
    dataset_list = []
    for i in tqdm(range(args.testsample)):
        random.seed(i+args.f)
        num = random.randint(1, 1000)
        KV_list, answer, entities = gen_word_list(args.R, i + num, args.N)
        prompts = build_prompts(KV_list, args.tem_prompt, entities, 1)
        dataset_list.append(GenerateData(index = i, words = answer, entities= entities, present_prompt = prompts['present']))

    json_data = [asdict(d) for d in dataset_list]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    generate_data_main()