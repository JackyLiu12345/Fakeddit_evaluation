# Fakeddit Multimodal LLM Evaluation

Evaluate and fine-tune multimodal LLMs on the [Fakeddit](https://github.com/entitize/fakeddit) fake-news detection dataset using Hugging Face models.

---

## Contents

- [Quick Start](#quick-start)
- [Evaluation — Zero-Shot Baseline](#evaluation--zero-shot-baseline)
- [Evaluation — ICL Prompting Strategies](#evaluation--icl-prompting-strategies)
- [LoRA SFT Training (Qwen2-VL)](#lora-sft-training-qwen2-vl)
- [Adapter Inference](#adapter-inference)
- [Model Comparison Report Template](#model-comparison-report-template)
- [Module Overview](#module-overview)
- [Hardware Requirements](#hardware-requirements)

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/JackyLiu12345/Fakeddit_evaluation.git
cd Fakeddit_evaluation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the default zero-shot evaluation
python evaluate.py
```

---

## Evaluation — Zero-Shot Baseline

The default evaluation runs zero-shot inference with LLaVA-Next on 10% of the test split for 2-way classification.

```bash
# Default settings: LLaVA-Next, test split, 2-way, 10% sample
python evaluate.py

# 3-way classification with Qwen2-VL
python evaluate.py --model Qwen/Qwen2-VL-7B-Instruct --task 3

# 6-way on validate split, larger sample
python evaluate.py --task 6 --split validate --sample-fraction 0.25

# Custom output file, run on CPU
python evaluate.py --output my_results.csv --device cpu

# Quick test with only 20 samples
python evaluate.py --max-samples 20
```

All flags and their defaults:

| Flag | Default | Description |
|------|---------|-------------|
| `--split` | `test` | Dataset split: `train`, `validate`, `test` |
| `--task` | `2` | Classification task: `2`, `3`, or `6` |
| `--model` | `llava-hf/llava-v1.6-mistral-7b-hf` | HuggingFace model ID |
| `--sample-fraction` | `0.1` | Fraction of multimodal rows to evaluate |
| `--output` | `results.csv` | Output CSV path |
| `--device` | `auto` | `cuda`, `cpu`, or `auto` |
| `--max-samples` | `None` | Cap on number of samples (useful for testing) |
| `--data-dir` | `None` | Path to local TSV files (bypasses Google Drive) |
| `--verbose` | `False` | Enable DEBUG logging |

---

## Evaluation — ICL Prompting Strategies

The evaluation script supports three prompting strategies via `--prompt-strategy`.

### Few-shot balanced

Selects `--num-demos-per-class` demonstrations uniformly at random per class:

```bash
python evaluate.py \
    --prompt-strategy few_shot_balanced \
    --num-demos-per-class 1
```

### Few-shot hard-negative

Preferentially selects examples with short/ambiguous titles or sensational language (e.g. "breaking", "shocking"):

```bash
python evaluate.py \
    --prompt-strategy few_shot_hard_negative \
    --num-demos-per-class 2
```

### With rationale demos (chain-of-thought style)

```bash
python evaluate.py \
    --prompt-strategy few_shot_balanced \
    --include-rationale-demos
```

### Self-consistency (majority voting)

Run N independent generations per sample and apply majority vote:

```bash
python evaluate.py \
    --prompt-strategy few_shot_hard_negative \
    --self-consistency-n 3
```

### Full ICL example

```bash
python evaluate.py \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --task 2 \
    --split test \
    --sample-fraction 0.05 \
    --prompt-strategy few_shot_balanced \
    --num-demos-per-class 2 \
    --include-rationale-demos \
    --self-consistency-n 3 \
    --output results_icl_balanced.csv \
    --max-samples 100
```

The output CSV includes extra metadata columns:

| Column | Description |
|--------|-------------|
| `prompt_strategy` | Strategy used (`zero_shot`, `few_shot_balanced`, `few_shot_hard_negative`) |
| `self_consistency_n` | Number of generations per sample |
| `num_demos_used` | Total demos injected into each prompt |

---

## LoRA SFT Training (Qwen2-VL)

Fine-tune a Qwen2-VL model with LoRA (or QLoRA for 4-bit quantisation) on Fakeddit.

### Quickstart

```bash
# Train with defaults (Qwen2-VL-2B, 10% of train split, 2-way, QLoRA auto)
python train_lora_qwen2vl.py

# 3-way classification, bf16, 5% data
python train_lora_qwen2vl.py --task 3 --sample-fraction 0.05 --bf16

# Disable QLoRA (full fp16 LoRA)
python train_lora_qwen2vl.py --no-use-qlora --fp16

# Larger model, custom output, cap training samples
python train_lora_qwen2vl.py \
    --model-name Qwen/Qwen2-VL-7B-Instruct \
    --output-dir outputs/qwen2vl-7b-lora \
    --max-train-samples 500
```

### Training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-name` | `Qwen/Qwen2-VL-2B-Instruct` | Base model to fine-tune |
| `--split` | `train` | Dataset split |
| `--sample-fraction` | `0.1` | Fraction of split to use |
| `--task` | `2` | Classification task |
| `--output-dir` | `outputs/qwen2vl-lora` | Adapter save path |
| `--per-device-train-batch-size` | `1` | Batch size per GPU |
| `--gradient-accumulation-steps` | `8` | Effective batch = batch × accumulation |
| `--num-train-epochs` | `1` | Training epochs |
| `--learning-rate` | `2e-4` | Peak LR for AdamW |
| `--max-train-samples` | `None` | Hard cap on training samples |
| `--bf16` | `False` | Use bfloat16 (Ampere GPU+) |
| `--fp16` | `False` | Use float16 |
| `--use-qlora / --no-use-qlora` | auto | 4-bit QLoRA (auto-detects CUDA + bitsandbytes) |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha scaling factor |
| `--lora-dropout` | `0.05` | LoRA dropout probability |
| `--data-dir` | `None` | Path to local TSV files (bypasses Google Drive) |

### Required extra packages

```bash
pip install peft bitsandbytes einops
```

`bitsandbytes` is only needed for `--use-qlora` / QLoRA mode.

---

## Adapter Inference

After training, run inference on a single example using the saved adapter:

```bash
python lora_infer_qwen2vl.py \
    --base-model Qwen/Qwen2-VL-2B-Instruct \
    --adapter-path outputs/qwen2vl-lora/adapter \
    --title "Breaking: Scientists make groundbreaking discovery" \
    --image-url "https://example.com/news_image.jpg"
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--base-model` | `Qwen/Qwen2-VL-2B-Instruct` | Base model HuggingFace ID |
| `--adapter-path` | *(required)* | Path to saved LoRA adapter directory |
| `--task` | `2` | Classification task |
| `--title` | *(required)* | News post title to classify |
| `--image-url` | *(required)* | Image URL |
| `--device` | `auto` | Compute device |
| `--max-new-tokens` | `64` | Max tokens to generate |

---

## Model Comparison Report Template

A ready-to-fill Markdown report template is included:

```
reports/model_comparison_template.md
```

It contains copy-paste tables for:

1. Experiment metadata (date, commit, hardware, CUDA, library versions)
2. Dataset setup (split, fraction, row counts before/after filtering)
3. Prompting strategy table
4. Model configuration table
5. Metrics summary (accuracy, macro-F1, per-class F1)
6. Error analysis categories
7. Latency/cost/throughput (sec/sample, samples/hour, GPU memory)
8. Qualitative examples (correct and incorrect)
9. Conclusions and next steps

**To use:**

```bash
# Copy the template to a new file and fill it in
cp reports/model_comparison_template.md reports/model_comparison_$(date +%Y-%m-%d).filled.md
```

Filled copies (`*.filled.md`) are excluded from Git by `.gitignore` by convention.

---

## Module Overview

| File | Description |
|------|-------------|
| `config.py` | Google Drive file IDs, label maps, default model and generation params |
| `data_loader.py` | Downloads Fakeddit TSV splits via `gdown`, filters to multimodal rows, samples |
| `prompts.py` | Zero-shot prompt templates for 2/3/6-way, formatted for LLaVA and Qwen2-VL |
| `icl.py` | Demo pool building, balanced/hard-negative selection, ICL prompt composition, majority vote |
| `utils.py` | `download_image()` with retry, `load_model_and_processor()`, `generate_response()` |
| `metrics.py` | Accuracy, per-class P/R/F1, macro-F1, confusion matrix |
| `evaluate.py` | Main evaluation CLI (zero-shot + ICL + self-consistency) |
| `train_lora_qwen2vl.py` | LoRA SFT training pipeline for Qwen2-VL |
| `lora_infer_qwen2vl.py` | Single-sample inference with a trained LoRA adapter |
| `reports/model_comparison_template.md` | Markdown report template for documenting experiment results |

---

## Troubleshooting: Google Drive Download Errors

If you see an error like:

```
gdown.exceptions.FileURLRetrievalError: Cannot retrieve the public link of the file.
You may need to change the permission to 'Anyone with the link', or have had many accesses.
```

This means Google Drive is blocking the automatic download (usually due to rate-limiting or permission changes). **Workaround — use local files:**

### Step 1: Download the TSV files manually

Open these links in your browser and save the files:

| Split | Google Drive link | Save as |
|-------|-------------------|---------|
| test | [Download](https://drive.google.com/uc?id=1GqQtt86gxdGMjbx7KxM4XQyWQGvpEX2M) | `multimodal_test_public.tsv` |
| validate | [Download](https://drive.google.com/uc?id=1yNEEzn3EjjhywIAb9Xli_mF5O65nQ9cx) | `multimodal_validate.tsv` |
| train | [Download](https://drive.google.com/uc?id=1iu-H12Rvmz_XW3lK9IH7bwMfOpKddhE8) | `multimodal_train.tsv` |

Or download them from the [official Fakeddit Google Drive folder](https://drive.google.com/drive/folders/1jU7qgDqU1je9Y0PMKJ_f31yXRo5uWGFm).

### Step 2: Place the files in a directory

```bash
mkdir -p data
# Move your downloaded files into the data/ folder
mv ~/Downloads/multimodal_test_public.tsv data/
```

### Step 3: Re-run with `--data-dir`

```bash
# Evaluation
python evaluate.py --data-dir ./data/

# Training
python train_lora_qwen2vl.py --data-dir ./data/
```

> **Tip:** The `--data-dir` flag works with all scripts and can be combined with any other flags (e.g. `--task`, `--split`, `--max-samples`).

---

## Hardware Requirements

| Model | Min VRAM | Recommended |
|-------|----------|-------------|
| LLaVA-Next 7B (float16) | ~16 GB | A100 40 GB |
| Qwen2-VL-2B (float16) | ~8 GB | A40 / A100 |
| Qwen2-VL-7B (float16) | ~20 GB | A100 40 GB |
| Qwen2-VL-2B (QLoRA 4-bit) | ~4 GB | Any modern GPU |
| Qwen2-VL-7B (QLoRA 4-bit) | ~8 GB | RTX 3090 / A100 |

CPU inference is supported but is significantly slower.

> **Tip:** For Google Colab, use `Qwen/Qwen2-VL-2B-Instruct` with `--use-qlora`
> on a T4/A100 runtime.
