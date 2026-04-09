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
"""

import argparse
import csv
import logging
import os
import sys
from typing import Optional

import pandas as pd
from tqdm import tqdm

import config as cfg
from data_loader import load_split
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
    return parser.parse_args(argv)


# ── Label parsing ─────────────────────────────────────────────────────────────

def _parse_predicted_label(raw_text: str, n_way: int) -> str:
    """
    Extract the predicted label from the model's raw generated text.

    Searches (case-insensitively) for any of the expected label strings inside
    *raw_text*.  Returns ``"unknown"`` if no label is found.
    """
    label_map = cfg.LABEL_MAPS[n_way]
    text_lower = raw_text.strip().lower()

    # Try exact / substring match for each label (longest first to avoid
    # partial matches, e.g. "true" matching inside "fake_with_true_text").
    for label in sorted(label_map.values(), key=len, reverse=True):
        if label.lower() in text_lower:
            return label

    logger.debug("Could not parse label from response: %r", raw_text)
    return "unknown"


# ── Per-sample prompt builder ─────────────────────────────────────────────────

def _build_prompt(model_name: str, n_way: int, clean_title: str):
    """Return the correctly formatted prompt for the given model type."""
    system_prompt, user_prompt = get_prompts(n_way, clean_title)
    model_lower = model_name.lower()

    if "qwen" in model_lower:
        return format_qwen2vl_messages(system_prompt, user_prompt)
    else:
        # LLaVA and generic Vision2Seq models use the instruction template.
        return format_llava_prompt(user_prompt)


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(
    split: str,
    task: int,
    model_name: str,
    sample_fraction: float,
    output_path: str,
    device: str,
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

    logger.info("Evaluating %d samples for %d-way classification.", len(df), task)

    # ── 2. Load model ─────────────────────────────────────────────────────────
    model, processor = load_model_and_processor(model_name, device=device)

    # ── 3. Prepare results file ───────────────────────────────────────────────
    fieldnames = ["id", "true_label", "predicted_label", "raw_response"]
    results_rows = []

    # ── 4. Evaluation loop ────────────────────────────────────────────────────
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
            })
            continue

        # Build prompt.
        prompt = _build_prompt(model_name, task, clean_title)

        # Run inference.
        try:
            raw_response = generate_response(
                model,
                processor,
                image,
                prompt,
                device=device,
                max_new_tokens=cfg.GENERATION_CONFIG["max_new_tokens"],
            )
        except Exception as exc:
            logger.warning("Inference failed for sample %s: %s", sample_id, exc)
            raw_response = f"INFERENCE_ERROR: {exc}"

        predicted_label = _parse_predicted_label(raw_response, task)

        results_rows.append({
            "id": sample_id,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "raw_response": raw_response,
        })

    # ── 5. Save results ───────────────────────────────────────────────────────
    results_df = pd.DataFrame(results_rows, columns=fieldnames)
    results_df.to_csv(output_path, index=False)
    logger.info("Results saved to '%s'.", output_path)

    # ── 6. Compute & print metrics ────────────────────────────────────────────
    # Exclude rows where we could not obtain a prediction.
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
    )


if __name__ == "__main__":
    main()
