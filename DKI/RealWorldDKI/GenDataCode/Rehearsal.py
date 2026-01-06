import argparse
from dataclasses import dataclass, asdict
import json 
import pandas as pd
import numpy as np
from math import floor, ceil
import random
from typing import List, Dict
from utils import process_data


def generate_data_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="DKI/RealWorldDKI/temporal_interval_qa.json")
    parser.add_argument("--output_file", type=str, default="DKI/RealWorldDKI/Dataset/RealWorldDataset_Rehearsal.json")
    parser.add_argument("--default_end_date", type=str, default="2024-12-31")
    args = parser.parse_args()

    instruction = (
        "You are given a long list of cue–value records and a single target cue (CUE).\n"
        "For this target cue, return its earliest and latest VALUE according to FIRST and LAST "
        "occurrence order in the provided record list (top to bottom).\n"
        "\n"
        "Thinking rule:\n"
        "Rehearse each new cue:value pair three times when you read it. Do this internally and do not output the rehearsals into text.\n"
    )
    input_text = (
        "CUE (JSON array): {entities_json}\n"
        "\n"
        "INPUT FORMAT\n"
        "- Each record is one line in the form: cue:value\n"
        "- Boundaries: lines strictly between the literal markers START: and END\n"
        "\n"
        "START:\n"
        "{kv_list}\n"
        "END\n"
        "\n"
        "Output (valid JSON only):\n"
        '{{"cue":"<cue>","latest":"<VERBATIM or UNKNOWN>", "earliest":"<VERBATIM or UNKNOWN>"}},\n'
        "\n"
        "Rules:\n"
        "- Only the cue specified by CUE; ignore all other cues.\n"
        "- Earliest/Latest are defined strictly by first/last appearance order within START..END.\n"
        "- Keep VALUE VERBATIM; JSON-escape only as required.\n"
        "- Output exactly one JSON object and nothing else. No code, no prose, no markdown.\n"
        "\n"
    )
    args = parser.parse_args()

    process_data(args.input, args.output_file, args.default_end_date, instruction, input_text)


if __name__ == "__main__":
    generate_data_main()
