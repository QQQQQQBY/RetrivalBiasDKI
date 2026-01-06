
# Correct and incorrect samples of the model. When the model correctly predicts earliest and latest values, can its attention clearly point to the first and last positions? When the model incorrectly predicts latest values, how is its attention distribution?
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from Utils.utils import *
from Utils.sample_layer_attention import *
from Utils.group_layer_attention import *
# ========= Configuration Area =========
DATA_PATH = "DKI/SyntheticDKI/Dataset/synthetic_32.json"  # TODO: Replace with your data file path
# MODEL_PATH = "Qwen2.5-7B-Instruct"
MODEL_PATH = "llama-3.1-8B-instruct"
# SAVE_DIR = "DKI/InternalSingal/Attention/SaveAttentionWeight/attn_plots_qwen25"   # Attention plot save directory
SAVE_DIR = "DKI/InternalSingal/Attention/SaveAttentionWeight/attn_plots_llama" 
MAX_NEW_TOKENS = 64  # Length of generated JSON output, generally sufficient
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # 1. Read data
    data_path = Path(DATA_PATH)
    print(f"Loading data from {data_path} ...")
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Load model
    model, tokenizer = load_model_and_tokenizer()

    # 3. Collect 100 correct samples + 100 incorrect samples
    # correct_samples, wrong_samples = collect_correct_and_wrong_samples(
    #     data, model, tokenizer
    # )
    # with open("DKI/InternalSingal/Attention/SaveAttentionWeight/attn_plots_qwen25/correct_samples.json", 'w') as f:
    #     json.dump(correct_samples, f, indent=2, ensure_ascii=False)
    # with open("DKI/InternalSingal/Attention/SaveAttentionWeight/attn_plots_qwen25/wrong_samples.json", 'w') as f:
    #     json.dump(wrong_samples, f, indent=2, ensure_ascii=False)

    # with open("DKI/InternalSingal/Attention/SaveAttentionWeight/attn_plots_qwen25/correct_samples.json", 'r') as f:
    #     correct_samples =  json.load(f)
    # with open("DKI/InternalSingal/Attention/SaveAttentionWeight/attn_plots_qwen25/wrong_samples.json", 'r') as f:
    #     wrong_samples = json.load(f)
    with open("DKI/InternalSingal/Attention/SaveAttentionWeight/attn_plots_llama/correct_samples.json", 'r') as f:
        correct_samples =  json.load(f)
    with open("DKI/InternalSingal/Attention/SaveAttentionWeight/attn_plots_llama/wrong_samples.json", 'r') as f:
        wrong_samples = json.load(f)
    out_dir = Path(SAVE_DIR)


    # Group average
    # 5. Perform row-level attention statistics & plot comparison curves for 100 correct samples / 100 incorrect samples
    if correct_samples:
        # 5.1 Correct sample calculation + plotting
        probs_latest_list_correct, probs_index_list_correct = analyze_group_attention(
            correct_samples,
            model,
            tokenizer,
            group_name="8 Correct Samples - LLaMA",
            out_dir=out_dir,
            max_samples=N_CORRECT,
        )

    if wrong_samples:
        # 5.2 Incorrect sample calculation + plotting
        probs_latest_list_wrong, probs_index_list_wrong = analyze_group_attention(
            wrong_samples,
            model,
            tokenizer,
            group_name="192 Wrong Samples - LLaMA",
            out_dir=out_dir,
            max_samples=N_WRONG,
        )

    print(f"Done. Curves saved under: {out_dir.absolute()}")

    # layers_to_show = [8, 18]  # If your layer numbers start from 1
    layers_to_show = [5, 27]

    probs_latest_list_correct = np.array(probs_latest_list_correct)
    probs_latest_list_wrong = np.array(probs_latest_list_wrong)
    # n_samples_correct = probs_latest_correct.shape[0]
    # n_samples_error   = probs_latest_list_wrong.shape[0]
    # n_pos             = probs_latest_correct.shape[2]
    
    # ---------- Save data needed for plotting ----------
    print(f"Saving plotting data to {out_dir}...")
    np.save(out_dir / "probs_latest_list_correct.npy", probs_latest_list_correct)
    np.save(out_dir / "probs_latest_list_wrong.npy", probs_latest_list_wrong)
    np.save(out_dir / "probs_index_list_correct.npy", np.array(probs_index_list_correct))
    np.save(out_dir / "probs_index_list_wrong.npy", np.array(probs_index_list_wrong))
    np.save(out_dir / "layers_to_show.npy", np.array(layers_to_show))
    print(f"Data saved successfully!")
    
    # ---------- Plot correct samples ----------
    plot_attention_three_big_figs(
        attn_group=probs_latest_list_correct,
        pred_idx_group=probs_index_list_correct,
        layers_to_plot=layers_to_show,
        group_name="8 Correct Samples",
    )

    # ---------- Plot incorrect samples ----------
    plot_attention_three_big_figs(
        attn_group=probs_latest_list_wrong[:100],
        pred_idx_group=probs_index_list_wrong,
        layers_to_plot=layers_to_show,
        group_name="100 Wrong Samples",
    )

    group_line(attn_group=probs_latest_list_correct,
        pred_idx_group=probs_index_list_correct,group_name="8 Correct Samples", save_dir=out_dir)

    group_line(attn_group=probs_latest_list_wrong,
        pred_idx_group=probs_index_list_wrong,group_name="192 Wrong Samples", save_dir=out_dir)



if __name__ == "__main__":
    main()
