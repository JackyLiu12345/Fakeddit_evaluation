# Fakeddit Multimodal LLM Evaluation

> **Evaluate open-source multimodal vision-language models on fake-news detection — no large file downloads, no OpenAI dependency.**

This repository provides a ready-to-run evaluation framework for testing [Hugging Face](https://huggingface.co) multimodal LLMs (e.g., LLaVA-Next, Qwen2-VL) on the [Fakeddit](https://github.com/entitize/Fakeddit) dataset.  The dataset TSV files are streamed directly from Google Drive into memory; only **10% of the dataset** is used by default for fast iteration.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Supported Models](#supported-models)
5. [Hardware Requirements](#hardware-requirements)
6. [CLI Reference](#cli-reference)
7. [Example Commands & Sample Output](#example-commands--sample-output)
8. [Streaming & 10 % Sampling](#streaming--10--sampling)
9. [Adding More Models](#adding-more-models)
10. [Citation](#citation)

---

## Project Overview

[Fakeddit](https://github.com/entitize/Fakeddit) is a large-scale multimodal fake-news detection dataset sourced from Reddit.  Each sample consists of a post **title** and an **image**, labelled under 2-way, 3-way, or 6-way classification schemes.

This project lets you:

- **Stream** the dataset TSV files from Google Drive — no manual download of multi-GB files.
- **Sample 10%** of the multimodal subset (rows with images) for rapid evaluation.
- Run **zero-shot multimodal inference** with any supported Hugging Face vision-language model.
- Compute **accuracy, per-class F1, macro-F1, and confusion matrix** automatically.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/JackyLiu12345/Fakeddit_evaluation.git
cd Fakeddit_evaluation

# 2. Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run evaluation (2-way classification, test split, 10% sample, LLaVA-Next)
python evaluate.py
```

Results are saved to `results.csv` and metrics are printed to stdout.

---

## Diagnostics Checklist Script

This repo includes an executable diagnostics script:

```bash
cd Fakeddit_evaluation
chmod +x diagnose.sh
./diagnose.sh
```

Optional arguments:

```bash
./diagnose.sh [split] [task] [sample_fraction]
# example
./diagnose.sh validate 3 0.01
```

To run a full smoke inference pass (heavier, loads model):

```bash
SMOKE=1 ./diagnose.sh test 2 0.01
```

The script checks environment/config, data loading and filtering, label distribution,
label parsing behavior, image download behavior, and (optionally) end-to-end
evaluation + metrics recomputation.

---

## Project Structure

```
Fakeddit_evaluation/
├── config.py          # Central configuration (model names, file IDs, label maps)
├── data_loader.py     # Stream TSV from Google Drive + 10% sampling
├── evaluate.py        # Main evaluation script (CLI entry-point)
├── metrics.py         # Accuracy, F1, confusion matrix — also standalone
├── prompts.py         # Zero-shot prompt templates for 2/3/6-way tasks
├── requirements.txt   # Python dependencies (HuggingFace ecosystem)
├── utils.py           # Image download, logging, HF model factory, inference
└── README.md
```

---

## Supported Models

| Model | HF Hub ID | Backend class |
|---|---|---|
| **LLaVA-Next** (default) | `llava-hf/llava-v1.6-mistral-7b-hf` | `LlavaNextForConditionalGeneration` |
| **Qwen2-VL** | `Qwen/Qwen2-VL-7B-Instruct` | `Qwen2VLForConditionalGeneration` |
| Any Vision2Seq model | `<any HF model ID>` | `AutoModelForVision2Seq` (fallback) |

The factory function `load_model_and_processor` in `utils.py` auto-detects the model type from its name.  See [Adding More Models](#adding-more-models) for how to extend support.

---

## Hardware Requirements

| Configuration | Recommendation |
|---|---|
| 7B models (LLaVA-Next, Qwen2-VL) | GPU with **≥ 16 GB VRAM** (e.g. A100, RTX 3090/4090) |
| CPU-only | Possible but very slow; use a smaller model or reduce `--sample-fraction` |
| Multi-GPU | Supported automatically via `device_map="auto"` and `accelerate` |

The code uses `torch.float16` on CUDA and `torch.float32` on CPU for memory efficiency.

---

## CLI Reference

```
python evaluate.py [OPTIONS]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--split` | str | `test` | Dataset split: `train`, `validate`, or `test` |
| `--task` | int | `2` | Classification task: `2`, `3`, or `6` |
| `--model` | str | `llava-hf/llava-v1.6-mistral-7b-hf` | HuggingFace Hub model ID |
| `--sample-fraction` | float | `0.1` | Fraction of multimodal rows to evaluate |
| `--output` | str | `results.csv` | Path for the per-sample results CSV |
| `--device` | str | `auto` | Compute device: `cuda`, `cpu`, or `auto` |
| `--verbose` | flag | off | Enable DEBUG-level logging |

**Standalone metrics computation:**

```bash
python metrics.py results.csv --task 2
```

---

## Example Commands & Sample Output

### 2-way classification with LLaVA-Next (default)

```bash
python evaluate.py --split test --task 2 --output results_2way.csv
```

```
============================================================
  Accuracy : 0.7142
  Macro-F1 : 0.6891
============================================================

Per-class report:
              precision    recall  f1-score   support
        real       0.76      0.74      0.75       312
        fake       0.65      0.68      0.66       209
    accuracy                           0.71       521
   macro avg       0.71      0.71      0.69       521
weighted avg       0.72      0.71      0.71       521

Confusion matrix (rows = true, cols = predicted):
            real  fake
real         231    81
fake          68   141
```

### 3-way classification with Qwen2-VL

```bash
python evaluate.py \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --split validate \
    --task 3 \
    --output results_3way.csv
```

### Evaluate on 25% of training data

```bash
python evaluate.py --split train --sample-fraction 0.25 --output results_train.csv
```

---

## Streaming & 10 % Sampling

The Fakeddit TSV files (especially `train`) can be hundreds of megabytes.  This project avoids storing them permanently:

1. `gdown` downloads the TSV into a **temporary file** that is deleted immediately after pandas parses it into memory.
2. Only rows where `hasImage == True` and `image_url` is non-empty are kept (multimodal subset).
3. **10% of those rows** are sampled (`df.sample(frac=0.1, random_state=42)`) — overridable via `--sample-fraction`.

This means the only large data on disk at any moment is the model weights.

---

## Adding More Models

1. Open `utils.py` and extend `load_model_and_processor`:

```python
elif "my_model_keyword" in model_name_lower:
    from transformers import MyModelClass, MyProcessorClass
    processor = MyProcessorClass.from_pretrained(model_name)
    model = MyModelClass.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
```

2. If the model needs a different prompt format, add a helper in `prompts.py` and update `_build_prompt` in `evaluate.py`.

---

## Citation

If you use Fakeddit in your research, please cite the original paper:

```bibtex
@article{nakamura2019fakeddit,
  title   = {Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection},
  author  = {Nakamura, Kai and Levy, Sharon and Wang, William Yang},
  journal = {arXiv preprint arXiv:1911.03854},
  year    = {2019}
}
```

Dataset repository: https://github.com/entitize/Fakeddit  
Paper: https://arxiv.org/abs/1911.03854
