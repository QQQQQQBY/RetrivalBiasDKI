
import torch
from pathlib import Path
import json
from Utils.utils import *
from Utils import group_analyse_logits
from Utils.group_analyse_logits import group_analyse_logits

SAVE_DIR = "DKI/InternalSingal/Logits/logits_reports"
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
        # Mean and variance of logits at different positions
        group_analyse_logits(
            correct_samples,
            model,
            tokenizer,
            group_name="36 Correct Samples",
            out_dir=out_dir,
        )
    if wrong_samples:
        # Mean and variance of logits at different positions
        group_analyse_logits(
            wrong_samples,
            model,
            tokenizer,
            group_name="164 Wrong Samples",
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()