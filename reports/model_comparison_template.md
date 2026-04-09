# Fakeddit Multimodal LLM Evaluation — Model Comparison Report

<!-- Copy this template and fill in the placeholders marked with [PLACEHOLDER]. -->
<!-- Save filled copies as reports/model_comparison_YYYY-MM-DD.filled.md        -->

---

## 1. Experiment Metadata

| Field               | Value                          |
|---------------------|--------------------------------|
| Date                | [YYYY-MM-DD]                   |
| Git commit          | [git rev-parse --short HEAD]   |
| Hardware            | [GPU model, e.g. NVIDIA A100]  |
| CUDA version        | [e.g. 12.1]                    |
| Python version      | [e.g. 3.11.5]                  |
| transformers ver.   | [e.g. 4.41.0]                  |
| torch version       | [e.g. 2.3.0]                   |
| peft version        | [e.g. 0.11.1]                  |
| Run by              | [username / CI pipeline]       |

---

## 2. Dataset Setup

| Field                          | Value                     |
|--------------------------------|---------------------------|
| Dataset                        | Fakeddit v2.0             |
| Split evaluated                | [test / validate / train] |
| Sample fraction                | [e.g. 10%]                |
| Rows before multimodal filter  | [PLACEHOLDER]             |
| Rows after multimodal filter   | [PLACEHOLDER]             |
| Rows evaluated                 | [PLACEHOLDER]             |
| Label column                   | [2_way_label / 3_way_label / 6_way_label] |
| Classification task            | [2-way / 3-way / 6-way]   |

---

## 3. Prompting Strategy Table

| Strategy ID | Strategy Name          | Demos per class | Rationale demos | Self-consistency N | Notes                    |
|-------------|------------------------|-----------------|-----------------|-------------------|--------------------------|
| S0          | zero_shot              | 0               | N/A             | 1                 | Baseline (default)       |
| S1          | few_shot_balanced      | [e.g. 1]        | No              | 1                 | Balanced class sampling  |
| S2          | few_shot_balanced      | [e.g. 2]        | Yes             | 1                 | With rationale demos     |
| S3          | few_shot_hard_negative | [e.g. 1]        | No              | 3                 | + Majority vote          |
| [Add rows]  | …                      | …               | …               | …                 | …                        |

---

## 4. Model Configuration Table

| Model ID | HuggingFace Model ID                         | Precision | QLoRA | LoRA r | LoRA α | Adapter path (if fine-tuned)     |
|----------|----------------------------------------------|-----------|-------|--------|--------|----------------------------------|
| M0       | llava-hf/llava-v1.6-mistral-7b-hf            | float16   | N/A   | N/A    | N/A    | —                                |
| M1       | Qwen/Qwen2-VL-2B-Instruct                    | float16   | N/A   | N/A    | N/A    | —                                |
| M2       | Qwen/Qwen2-VL-2B-Instruct (LoRA fine-tuned) | bfloat16  | Yes   | 16     | 32     | outputs/qwen2vl-lora/adapter     |
| M3       | Qwen/Qwen2-VL-7B-Instruct                    | bfloat16  | N/A   | N/A    | N/A    | —                                |
| [Add rows] | …                                           | …         | …     | …      | …      | …                                |

---

## 5. Metrics Summary Table

### 5.1 Overall Metrics

| Run ID | Model | Strategy | Accuracy | Macro-F1 | Notes                    |
|--------|-------|----------|----------|----------|--------------------------|
| R0     | M0    | S0       | [0.XXX]  | [0.XXX]  | Baseline                 |
| R1     | M1    | S0       | [0.XXX]  | [0.XXX]  | Qwen2-VL zero-shot       |
| R2     | M1    | S1       | [0.XXX]  | [0.XXX]  | Qwen2-VL few-shot bal.   |
| R3     | M1    | S3       | [0.XXX]  | [0.XXX]  | Qwen2-VL hard-neg + vote |
| R4     | M2    | S0       | [0.XXX]  | [0.XXX]  | Fine-tuned Qwen2-VL      |
| [Add]  | …     | …        | …        | …        | …                        |

### 5.2 Per-class F1 (best run: [Run ID])

| Class                  | Precision | Recall | F1     | Support |
|------------------------|-----------|--------|--------|---------|
| real                   | [0.XXX]   | [0.XXX]| [0.XXX]| [N]     |
| fake                   | [0.XXX]   | [0.XXX]| [0.XXX]| [N]     |
| **macro avg**          | [0.XXX]   | [0.XXX]| [0.XXX]| [N]     |

*(Extend rows for 3-way / 6-way tasks as needed.)*

---

## 6. Error Analysis

| Category                          | Count | % of total errors | Example IDs                     |
|-----------------------------------|-------|-------------------|---------------------------------|
| Image download failure            | [N]   | [XX%]             | [id1, id2, …]                   |
| Model output "unknown" (no parse) | [N]   | [XX%]             | [id1, id2, …]                   |
| Correct label in response but…    | [N]   | [XX%]             | [id1, id2, …]                   |
| Ambiguous / short title           | [N]   | [XX%]             | [id1, id2, …]                   |
| Sensational title (false positive)| [N]   | [XX%]             | [id1, id2, …]                   |
| Satire misclassified as real      | [N]   | [XX%]             | [id1, id2, …]                   |
| Other                             | [N]   | [XX%]             | [id1, id2, …]                   |

---

## 7. Latency / Cost / Throughput

| Run ID | Model | Strategy | Sec / sample | Samples / hour | Peak GPU mem (GB) | Notes          |
|--------|-------|----------|--------------|----------------|-------------------|----------------|
| R0     | M0    | S0       | [X.XX]       | [XXXX]         | [XX.X]            |                |
| R1     | M1    | S0       | [X.XX]       | [XXXX]         | [XX.X]            |                |
| R2     | M1    | S1       | [X.XX]       | [XXXX]         | [XX.X]            | +ICL overhead  |
| R4     | M2    | S0       | [X.XX]       | [XXXX]         | [XX.X]            | Fine-tuned     |
| [Add]  | …     | …        | …            | …              | …                 | …              |

> **Measurement note:** latency was measured as wall-clock time from image-URL
> to decoded label, averaged over [N] samples on [hardware].

---

## 8. Qualitative Examples

### 8.1 Correct Predictions

| Sample ID | Title (truncated)                        | True Label | Predicted | Strategy |
|-----------|------------------------------------------|------------|-----------|----------|
| [id]      | "[PLACEHOLDER title text]"               | real       | real      | S0       |
| [id]      | "[PLACEHOLDER title text]"               | fake       | fake      | S1       |
| [id]      | "[PLACEHOLDER title text]"               | true       | true      | S2       |

### 8.2 Incorrect Predictions (Interesting Failures)

| Sample ID | Title (truncated)                        | True Label | Predicted | Strategy | Hypothesis                        |
|-----------|------------------------------------------|------------|-----------|----------|-----------------------------------|
| [id]      | "[PLACEHOLDER title text]"               | fake       | real      | S0       | Image misleads model              |
| [id]      | "[PLACEHOLDER title text]"               | satire/parody | true  | S1       | Subtle satire not caught          |
| [id]      | "[PLACEHOLDER title text]"               | real       | fake      | S3       | Sensational wording triggers FP   |

---

## 9. Conclusions and Next Steps

### Key findings

- [PLACEHOLDER: Summarise which model + strategy achieves the best accuracy/F1.]
- [PLACEHOLDER: Note if ICL prompting improves over zero-shot, and by how much.]
- [PLACEHOLDER: Note if fine-tuned adapter outperforms frozen model.]
- [PLACEHOLDER: Comment on classes that are still hard to classify.]

### Recommended next steps

1. [ ] [PLACEHOLDER: e.g. Scale to 100% of test split for final numbers.]
2. [ ] [PLACEHOLDER: e.g. Try `num_demos_per_class=2` or `self_consistency_n=5`.]
3. [ ] [PLACEHOLDER: e.g. Fine-tune on more data (train split, 50% sample).]
4. [ ] [PLACEHOLDER: e.g. Add vision-language model X for comparison.]
5. [ ] [PLACEHOLDER: e.g. Investigate image-download failure rate and mitigate.]

---

*Generated from `reports/model_comparison_template.md`.  
Filled copies should be saved as `reports/model_comparison_YYYY-MM-DD.filled.md`.*
