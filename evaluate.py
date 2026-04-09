"""
evaluate.py — Main evaluation script for Fakeddit multimodal LLM evaluation.

Usage examples
--------------
# Evaluate with default settings (LLaVA-Next, test split, 2-way, 10% sample):
    python evaluate.py

# Evaluate Qwen2-VL on the validate split with 3-way classification:
    python evaluate.py --model Qwen/Qwen2-VL-7B-Instruct --split validate --task 3

# Use a larger sample fraction and a custom output file:
    python evaluate.py --sample-fraction 0.25 --output my_results.csv

# Run on CPU (slower but no GPU required):
    python evaluate.py --device cpu

# Use few-shot balanced ICL with 2 demos per class:
    python evaluate.py --prompt-strategy few_shot_balanced --num-demos-per-class 2

# Use hard-negative few-shot with self-consistency (3 generations, majority vote):
    python evaluate.py --prompt-strategy few_shot_hard_negative --self-consistency-n 3

# Quick test with only 20 samples:
    python evaluate.py --max-samples 20
"""

import argparse
import logging
import re
import sys
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

import config as cfg
from data_loader import load_split
from icl import (
    compose_icl_user_prompt,
    majority_vote,
    select_demos_balanced,
    select_demos_hard_negative,
)
from metrics import compute_metrics, print_metrics
from prompts import format_llava_prompt, format_qwen2vl_messages, get_prompts
from utils import (
    download_image,
    generate_response,
    load_model_and_processor,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a multimodal HuggingFace LLM on the Fakeddit dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Existing arguments (preserved for backward compatibility) ─────────────
    parser.add_argument(
        "--split",
        choices=["train", "validate", "test"],
        default="test",
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=[2, 3, 6],
        default=2,
        help="Classification task: 2-way, 3-way, or 6-way.",
    )
    parser.add_argument(
        "--model",
        default=cfg.DEFAULT_MODEL,
        help="HuggingFace Hub model ID.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=cfg.DEFAULT_SAMPLE_FRACTION,
        dest="sample_fraction",
        help="Fraction of the multimodal dataset to evaluate on (0 < f ≤ 1).",
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        help="Path to save the per-sample results CSV.",
    )
    parser.add_argument(
        "--device",
        default=cfg.DEFAULT_DEVICE,
        choices=["cuda", "cpu", "auto"],
        help="Compute device.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    # ── New ICL / prompting arguments ─────────────────────────────────────────
    parser.add_argument(
        "--prompt-strategy",
        choices=["zero_shot", "few_shot_balanced", "few_shot_hard_negative"],
        default="zero_shot",
        dest="prompt_strategy",
        help=(
            "Prompting strategy: zero_shot (default), few_shot_balanced, "
            "or few_shot_hard_negative."
        ),
    )
    parser.add_argument(
        "--num-demos-per-class",
        type=int,
        default=1,
        dest="num_demos_per_class",
        help="Number of in-context demonstrations to sample per class (ICL only).",
    )
    parser.add_argument(
        "--include-rationale-demos",
        action="store_true",
        dest="include_rationale",
        help=(
            "Include a brief rationale placeholder in each ICL demo "
            "(chain-of-thought style)."
        ),
    )
    parser.add_argument(
        "--self-consistency-n",
        type=int,
        default=1,
        dest="self_consistency_n",
        help=(
            "Number of independent generations per sample. "
            "If >1, applies majority voting over the N responses."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        dest="max_samples",
        help="Cap the number of samples evaluated (useful for quick testing).",
    )

    return parser.parse_args(argv)


# ── Label parsing ─────────────────────────────────────────────────────────────

def _parse_predicted_label(raw_text: str, n_way: int) -> str:
    """
    Extract the predicted label from the model's raw generated text.

    Parsing strategy (in order):
    1. Exact first-token match against known labels (case-insensitive).
    2. First-line match: check the first non-empty line for any known label.
    3. Longest-label substring match: find the longest label that appears
       anywhere in the response (to avoid partial matches like "true" inside
       "fake_with_true_text").
    4. Return ``"unknown"`` if all strategies fail.
    """
    label_map = cfg.LABEL_MAPS[n_way]
    labels_by_length = sorted(label_map.values(), key=len, reverse=True)

    raw_stripped = raw_text.strip()
    text_lower = raw_stripped.lower()

    # Strategy 1: Exact first-token match.
    first_token = re.split(r"[\s.,;:!?]+", text_lower)[0]
    if first_token:  # Guard against empty string when text starts with delimiters.
        for label in labels_by_length:
            if first_token == label.lower():
                return label

    # Strategy 2: First-line match — check whole first non-empty line.
    for line in raw_stripped.splitlines():
        line_lower = line.strip().lower()
        if not line_lower:
            continue
        for label in labels_by_length:
            if label.lower() in line_lower:
                return label
        break  # Only inspect the first non-empty line.

    # Strategy 3: Longest-label substring match anywhere in the full response.
    for label in labels_by_length:
        if label.lower() in text_lower:
            return label

    logger.debug("Could not parse label from response: %r", raw_text)
    return "unknown"


# ── Per-sample prompt builder ─────────────────────────────────────────────────

def _build_prompt(
    model_name: str,
    n_way: int,
    clean_title: str,
    demos: Optional[List[dict]] = None,
    include_rationale: bool = False,
):
    """
    Return the correctly formatted prompt for the given model type.

    For zero-shot, *demos* should be None or empty.
    For few-shot, *demos* is a list of {title, label} dicts.
    """
    model_lower = model_name.lower()

    if demos:
        # ICL path: compose a custom user prompt with demos embedded.
        user_prompt = compose_icl_user_prompt(
            clean_title, n_way, demos, include_rationale=include_rationale
        )
        system_prompt, _ = get_prompts(n_way, clean_title)  # reuse system prompt
    else:
        # Zero-shot path: use original prompt logic.
        system_prompt, user_prompt = get_prompts(n_way, clean_title)

    if "qwen" in model_lower:
        return format_qwen2vl_messages(system_prompt, user_prompt)
    else:
        # LLaVA and generic Vision2Seq models.
        return format_llava_prompt(user_prompt)


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(
    split: str,
    task: int,
    model_name: str,
    sample_fraction: float,
    output_path: str,
    device: str,
    prompt_strategy: str = "zero_shot",
    num_demos_per_class: int = 1,
    include_rationale: bool = False,
    self_consistency_n: int = 1,
    max_samples: Optional[int] = None,
) -> Optional[dict]:
    """
    Orchestrate data loading, model inference, and result saving.

    Parameters
    ----------
    split:        Dataset split name (``"train"``, ``"validate"``, ``"test"``).
    task:         Number of classes (2, 3, or 6).
    model_name:   HuggingFace Hub model ID.
    sample_fraction: Fraction of multimodal rows to evaluate.
    output_path:  Where to write the per-sample CSV.
    device:       Compute device string.
    prompt_strategy:
        One of ``"zero_shot"``, ``"few_shot_balanced"``,
        ``"few_shot_hard_negative"``.
    num_demos_per_class:
        Number of ICL demonstrations per class (only used for few-shot).
    include_rationale:
        Whether to include rationale placeholders in ICL demos.
    self_consistency_n:
        If > 1, generate N responses per sample and apply majority voting.
    max_samples:
        Cap on the number of samples to evaluate (None = no cap).

    Returns
    -------
    Metrics dict (see :func:`metrics.compute_metrics`) or *None* on failure.
    """
    label_col = cfg.LABEL_COLUMN[task]

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading '%s' split (%.0f%% sample) …", split, sample_fraction * 100)
    df = load_split(split, sample_fraction=sample_fraction)

    if label_col not in df.columns:
        logger.error("Label column '%s' not found in dataset.", label_col)
        return None

    # Apply max_samples cap.
    if max_samples is not None and max_samples < len(df):
        df = df.head(max_samples)
        logger.info("Capped evaluation to %d samples (--max-samples).", max_samples)

    logger.info("Evaluating %d samples for %d-way classification.", len(df), task)

    # ── 2. Build demo pool for ICL (few-shot strategies) ─────────────────────
    demos: List[dict] = []
    if prompt_strategy == "few_shot_balanced":
        demos = select_demos_balanced(
            df, task, label_col, num_demos_per_class=num_demos_per_class
        )
        logger.info("ICL (balanced): %d demos selected.", len(demos))
    elif prompt_strategy == "few_shot_hard_negative":
        demos = select_demos_hard_negative(
            df, task, label_col, num_demos_per_class=num_demos_per_class
        )
        logger.info("ICL (hard-negative): %d demos selected.", len(demos))

    num_demos_used = len(demos)

    # ── 3. Load model ─────────────────────────────────────────────────────────
    model, processor = load_model_and_processor(model_name, device=device)

    # ── 4. Prepare result columns ─────────────────────────────────────────────
    fieldnames = [
        "id",
        "true_label",
        "predicted_label",
        "raw_response",
        "prompt_strategy",
        "self_consistency_n",
        "num_demos_used",
    ]
    results_rows = []

    # ── 5. Evaluation loop ────────────────────────────────────────────────────
    label_map = cfg.LABEL_MAPS[task]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        sample_id = row.get("id", "")
        clean_title = str(row.get("clean_title", ""))
        image_url = str(row.get("image_url", ""))
        true_label_int = row.get(label_col)
        try:
            true_label = label_map[int(true_label_int)]
        except (KeyError, ValueError, TypeError):
            logger.warning(
                "Unexpected label value %r for sample %s — skipping.",
                true_label_int,
                sample_id,
            )
            continue

        # Download image with graceful failure.
        try:
            image = download_image(image_url)
        except Exception as exc:
            logger.warning("Skipping sample %s — image download failed: %s", sample_id, exc)
            results_rows.append({
                "id": sample_id,
                "true_label": true_label,
                "predicted_label": "unknown",
                "raw_response": f"IMAGE_DOWNLOAD_ERROR: {exc}",
                "prompt_strategy": prompt_strategy,
                "self_consistency_n": self_consistency_n,
                "num_demos_used": num_demos_used,
            })
            continue

        # Build prompt (zero-shot or ICL).
        prompt = _build_prompt(
            model_name,
            task,
            clean_title,
            demos=demos if demos else None,
            include_rationale=include_rationale,
        )

        # Run inference (potentially multiple times for self-consistency).
        raw_responses: List[str] = []
        try:
            for _ in range(self_consistency_n):
                raw_response = generate_response(
                    model,
                    processor,
                    image,
                    prompt,
                    device=device,
                    max_new_tokens=cfg.GENERATION_CONFIG["max_new_tokens"],
                )
                raw_responses.append(raw_response)
        except Exception as exc:
            logger.warning("Inference failed for sample %s: %s", sample_id, exc)
            raw_responses = [f"INFERENCE_ERROR: {exc}"]

        # Parse predictions and apply majority vote if self-consistency > 1.
        parsed_preds = [_parse_predicted_label(r, task) for r in raw_responses]
        if self_consistency_n > 1:
            predicted_label = majority_vote(parsed_preds)
        else:
            predicted_label = parsed_preds[0]

        results_rows.append({
            "id": sample_id,
            "true_label": true_label,
            "predicted_label": predicted_label,
            # Multiple responses (self-consistency > 1) are joined with " ||| ".
            # This separator is unlikely to appear in model outputs for this task.
            "raw_response": " ||| ".join(raw_responses),
            "prompt_strategy": prompt_strategy,
            "self_consistency_n": self_consistency_n,
            "num_demos_used": num_demos_used,
        })

    # ── 6. Save results ───────────────────────────────────────────────────────
    results_df = pd.DataFrame(results_rows, columns=fieldnames)
    results_df.to_csv(output_path, index=False)
    logger.info("Results saved to '%s'.", output_path)

    # ── 7. Compute & print metrics ────────────────────────────────────────────
    scored_df = results_df[results_df["predicted_label"] != "unknown"].copy()
    if scored_df.empty:
        logger.error("No valid predictions to score.")
        return None

    metrics = compute_metrics(scored_df, task)
    print_metrics(metrics)
    return metrics


# ── Entry-point ───────────────────────────────────────────────────────────────

def main(argv=None):
    args = _parse_args(argv)
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    if not (0 < args.sample_fraction <= 1.0):
        logger.error("--sample-fraction must be in (0, 1]. Got: %s", args.sample_fraction)
        sys.exit(1)

    run_evaluation(
        split=args.split,
        task=args.task,
        model_name=args.model,
        sample_fraction=args.sample_fraction,
        output_path=args.output,
        device=args.device,
        prompt_strategy=args.prompt_strategy,
        num_demos_per_class=args.num_demos_per_class,
        include_rationale=args.include_rationale,
        self_consistency_n=args.self_consistency_n,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
