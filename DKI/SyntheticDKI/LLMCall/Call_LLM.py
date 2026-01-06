import argparse
from Utils import utils
import random
import json
from tqdm import tqdm
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="update times: 512")
    parser.add_argument("--model", type=str, default="gpt-5-nano")
    parser.add_argument("--api_base", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="DKI/SyntheticDKI/Dataset/UpdateTimes_32.json")
    parser.add_argument("--output", type=str, default="DKI/SyntheticDKI/LLMCall/Results/UpdateTimes_32_gpt-5-nano.json")
    args = parser.parse_args()
    random.seed(args.seed)
    with open(args.dataset, 'r', encoding='utf-8')as f:
        dataset = json.load(f)
    response_list = []
    print(f"========={args.model}, {args.type} =========")
    print()
    for data in tqdm(dataset):
        try:
            rec_ready = utils.call_llm_api_online(data["present_prompt"], None, args.api_base, args.model, args.temperature, args.api_key)
            data["recall_response"] = rec_ready
            response_list.append(data)
        except:
            continue
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(response_list, f, indent=2, ensure_ascii=False)
    print("Run Completed")
    print()
    
if __name__ == "__main__":
    main()