# DKI (Dynamic Knowledge Instances)

We introduce a Dynamic Knowledge Instance (DKI) evaluation framework, modeling multi-updates of the same fact as a cue paired with a sequence of updated values, and assess models via endpoint probing of the earliest (initial) and latest (current) states. Across diverse LLMs, we observe that retrieval bias intensifies as updates increase, earliest-state accuracy stays high while latest-state accuracy drops substantially. Diagnostic analyses of attention, hidden-state similarity, and output logits further reveal that these signals become flatter and weakly discriminative on errors, providing little stable basis for identifying the latest update. Finally, intervention strategies inspired by cognitive heuristics yield only modest gains and do not eliminate the bias.

<p align="center">
  <img src="DKI/DKIFig.png" 
       alt="DKI"
       width="80%" 
       style="border: 1px solid #eee; box-shadow: 2px 2px 5px rgba(0,0,0,0.1)">
</p>
</p>

<p align="center">
  <em>  Dynamic Knowledge Instance (DKI) evaluation framework  </em>
</p>

## 🦋 Project Structure

### 🧭 `SyntheticDKI/`
Synthetic dataset generation and evaluation framework for DKI tasks.

- **`GenDataCode/`**: Scripts for generating synthetic datasets with various intervention strategies
  - `GenSynData.py`: Base synthetic data generation
  - `Index.py`: Index-based intervention (adds monotonically increasing indices)
  - `Integration.py`: Integration-based intervention (emphasizes internal organization rules)
  - `Forgetting.py`: Forgetting-based intervention (encourages forgetting previous pairs)
  - `Rehearsal.py`: Rehearsal-based intervention
  - `Semantic.py`: Semantic-based intervention
  - `CoT.py`: Chain-of-Thought prompting
  - `FewShot.py`: Few-shot prompting
  - `2_shot.py`: 2-shot prompting variant

- **`LLMCall/`**: LLM API calling and result evaluation
  - `Call_LLM.py`: Main script for calling LLM APIs on synthetic datasets
  - `EvaluateResult.py`: Evaluation utilities for parsing and assessing model outputs
  - `Results/`: Directory for storing evaluation results
  - `Utils/`: Utility functions for JSON parsing and data processing

- **`Dataset/`**: Generated synthetic datasets (JSON format)
- **`InterventionDataset/`**: Datasets with different intervention strategies applied
- **`Words.csv`**: Word list used for synthetic data generation

### 🚀 `RealWorldDKI/`
Real-world dataset processing and evaluation for DKI tasks.

- **`GenDataCode/`**: Scripts for processing real-world temporal data
  - `GenData.py`: Base data generation from real-world sources
  - `Index.py`: Index-based intervention variant
  - `Integration.py`: Integration-based intervention variant
  - `Forgetting.py`: Forgetting-based intervention variant
  - `Rehearsal.py`: Rehearsal-based intervention variant
  - `Semantic.py`: Semantic-based intervention variant
  - `CoT.py`: Chain-of-Thought prompting variant
  - `FewShot.py`: Few-shot prompting variant
  - `utils.py`: Shared utilities for data processing

- **`LLMCall/`**: LLM API calling for real-world datasets
  - `call_llm.py`: Main script for calling LLM APIs on real-world datasets
  - `long_call_llm.py`: Specialized script for long narrative texts with category-specific prompts
  - `Results/`: Directory for storing evaluation results

- **`Dataset/`**: Processed real-world datasets
- **`InterventionDataset/`**: Real-world datasets with intervention strategies applied
- **`LongTextDataset/`**: Long narrative text datasets (e.g., stories about athletes, companies, countries)
- **`temporal_interval_qa.json`**: Source temporal interval question-answering data

### 🧠 `InternalSingal/`
Internal signal analysis tools for understanding model behavior on DKI tasks.

- **`Attention/`**: Attention weight analysis
  - `attention_value.py`: Main script for analyzing attention patterns in correct vs. incorrect predictions
  - `multi_head_attention.py`: Multi-head attention analysis
  - `Utils/`: Utility functions for attention extraction and visualization
    - `utils.py`: Model loading, JSON parsing, sample collection utilities
    - `sample_layer_attention.py`: Single-sample layer-wise attention analysis
    - `group_layer_attention.py`: Group-level layer-wise attention analysis and visualization
    - `group_multi_head_attention.py`: Multi-head attention analysis across sample groups
    - `sample_multi_head_attention.py`: Single-sample multi-head attention analysis
  - `SaveAttentionWeight/`: Directory for saved attention weight data
  - `MutliHeadAttentionVIZ/`: Directory for multi-head attention visualizations

- **`HiddenState/`**: Hidden state analysis
  - `hiddenstate.py`: Main script for analyzing hidden states
  - `Utils/`: Utility functions for hidden state extraction and analysis
    - `utils.py`: Model loading, value span extraction, token mapping utilities
    - `group_analyse_hidden_state.py`: Group-level hidden state analysis with similarity computation and visualization
  - `Embeddings/`: Directory for saved hidden state embeddings

- **`Logits/`**: Logits analysis
  - `logits.py`: Main script for analyzing logits at different value positions
  - `Utils/`: Utility functions for logits extraction and analysis
    - `utils.py`: Model loading, value span extraction, token mapping utilities
    - `group_analyse_logits.py`: Group-level logits analysis with mean/variance computation and visualization
  - `logits_reports/`: Directory for logits analysis reports and visualizations

## 💡 Key Features

1. **Synthetic Data Generation**: Generate controlled datasets with various intervention strategies to study model behavior
2. **Real-World Evaluation**: Test models on real-world temporal data from multiple domains
3. **Internal Signal Analysis**: Deep dive into model internals (attention, hidden states, logits) to understand prediction mechanisms
4. **Comprehensive Evaluation**: Parse and evaluate model outputs with detailed metrics and statistics

## 🔧 Usage

### 📈 Generating Synthetic Data
```bash
cd SyntheticDKI/GenDataCode
python GenSynData.py --testsample 200 --R 32 --output ../Dataset/synthetic_32.json
```

### 🧩 Calling LLM APIs
```bash
cd SyntheticDKI/LLMCall
python Call_LLM.py --type original --model <model_name> --api_base <api_url> --api_key <key>
```

### 📝 Analyzing Attention Patterns
```bash
cd InternalSingal/Attention
python attention_value.py
```

### ⏳ Analyzing Hidden States
```bash
cd InternalSingal/HiddenState
python hiddenstate.py
```

### 📊 Analyzing Logits
```bash
cd InternalSingal/Logits
python logits.py
```

## 🧬 Data Format

The datasets follow a consistent format:
- Each sample contains `index`, `words`, `entities`, and `present_prompt`
- Prompts include cue-value records between `START:` and `END` markers
- Model outputs are expected in JSON format: `{"cue": "...", "earliest": "...", "latest": "..."}`

## ⚡ Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- NumPy
- Matplotlib
- scikit-learn
- tqdm

## 🧾 Notes

- Model paths and API configurations should be updated in respective configuration sections
- GPU is recommended for internal signal analysis
- Some scripts use 4-bit quantization for memory efficiency; adjust based on your hardware

