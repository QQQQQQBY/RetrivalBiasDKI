import json
from pathlib import Path
import torch
from Utils.utils import *
import numpy as np
from Utils.group_multi_head_attention import *

# SAVE_DIR = "DKI/InternalSingal/Attention/MutliHeadAttentionVIZ"
SAVE_DIR = "DKI/InternalSingal/Attention/MultiHeadLLaMA"
N_CORRECT = 36
N_WRONG = 164
out_dir = Path(SAVE_DIR)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def main():
    print(f"Loading correct and wrong samples ...")
    with open("DKI/InternalSingal/Attention/attn_plots_qwen25/correct_samples.json", 'r') as f:
        correct_samples =  json.load(f)
    with open("DKI/InternalSingal/Attention/attn_plots_qwen25/wrong_samples.json", 'r') as f:
        wrong_samples = json.load(f)
    # with open("DKI/InternalSingal/Attention/attn_plots_llama/correct_samples.json", 'r') as f:
    #     correct_samples =  json.load(f)
    # with open("DKI/InternalSingal/Attention/attn_plots_llama/wrong_samples.json", 'r') as f:
    #     wrong_samples = json.load(f)
    model, tokenizer = load_model_and_tokenizer()
    layers_to_show = [8]
    if correct_samples:
        analyze_group_attention_multihead(
            correct_samples,
            model,
            tokenizer,
            layers_to_show,
            group_name="36 Correct Samples - Qwen",
            out_dir=out_dir,
            max_samples=N_CORRECT,
        )
    if wrong_samples:
        analyze_group_attention_multihead(
            wrong_samples,
            model,
            tokenizer,
            layers_to_show,
            group_name="164 Wrong Samples - Qwen",
            out_dir=out_dir,
            max_samples=N_WRONG,
        )

if __name__ == "__main__":
    main()