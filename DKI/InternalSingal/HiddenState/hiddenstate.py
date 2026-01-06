
import torch
import json
from pathlib import Path
from Utils.utils import *
from Utils import group_analyse_hidden_state
from Utils.group_analyse_hidden_state import analyze_hidden_state

SAVE_DIR = "DKI/InternalSingal/HiddenState"
MAX_NEW_TOKENS = 64  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    with open("DKI/InternalSingal/Attention/attn_plots_qwen25/correct_samples.json", 'r') as f:
        correct_samples =  json.load(f)
    with open("DKI/InternalSingal/Attention/attn_plots_qwen25/wrong_samples.json", 'r') as f:
        wrong_samples = json.load(f)

    # with open("DKI/InternalSingal/Attention/attn_plots_llama/correct_samples.json", 'r') as f:
    #     correct_samples =  json.load(f)
    # with open("DKI/InternalSingal/Attention/attn_plots_llama/wrong_samples.json", 'r') as f:
    #     wrong_samples = json.load(f)

    out_dir = Path(SAVE_DIR)

    model, tokenizer = load_model_and_tokenizer()
    if correct_samples:
        analyze_hidden_state(
            correct_samples,
            model,
            tokenizer,
            group_name="36 Correct Sample",
            out_dir=out_dir,
            title_prefix="Qwen2.5-7B"
        )
    
    if wrong_samples:
        analyze_hidden_state(
            wrong_samples,
            model,
            tokenizer,
            group_name="164 Wrong Samples",
            out_dir=out_dir,
            title_prefix="Qwen2.5-7B"
        )

if __name__ == "__main__":
    main()