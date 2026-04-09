"""
metrics.py — Evaluation metrics for Fakeddit classification results.

Computes accuracy, per-class precision/recall/F1, macro-F1, and prints a
confusion matrix.  Can be used as a module or run standalone:

    python metrics.py results.csv [--task 2]
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from config import LABEL_MAPS

logger = logging.getLogger(__name__)


def compute_metrics(
    results_df: pd.DataFrame,
    n_way: int,
) -> dict:
    """
    Compute classification metrics from a results DataFrame.

    Parameters
    ----------
    results_df:
        DataFrame with at least columns ``true_label`` and ``predicted_label``.
        Labels should be *string* label names (e.g. ``"real"``, ``"fake"``).
    n_way:
        Number of classes (2, 3, or 6).  Used to enumerate all expected labels.

    Returns
    -------
    dict with keys: accuracy, macro_f1, report (str), confusion_matrix (ndarray)
    """
    label_map = LABEL_MAPS[n_way]
    all_labels = list(label_map.values())

    y_true = results_df["true_label"].astype(str).tolist()
    y_pred = results_df["predicted_label"].astype(str).tolist()

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=all_labels,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    # Extract macro-F1 from the report dict for convenience.
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "report": report,
        "confusion_matrix": cm,
        "labels": all_labels,
    }


def print_metrics(metrics: dict) -> None:
    """Pretty-print the metrics dict returned by :func:`compute_metrics`."""
    print("\n" + "=" * 60)
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Macro-F1 : {metrics['macro_f1']:.4f}")
    print("=" * 60)
    print("\nPer-class report:")
    print(metrics["report"])

    labels = metrics["labels"]
    cm = metrics["confusion_matrix"]
    print("Confusion matrix (rows = true, cols = predicted):")
    # Header row
    col_w = max(len(lbl) for lbl in labels) + 2
    header = " " * col_w + "".join(f"{lbl:>{col_w}}" for lbl in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = f"{row_label:<{col_w}}" + "".join(f"{cm[i, j]:>{col_w}}" for j in range(len(labels)))
        print(row)
    print()


def evaluate_results_file(results_path: str, n_way: int) -> dict:
    """
    Load *results_path* CSV, compute metrics, and print them.

    Parameters
    ----------
    results_path:
        Path to the CSV produced by ``evaluate.py``.
    n_way:
        Classification task (2, 3, or 6).
    """
    df = pd.read_csv(results_path)
    if "true_label" not in df.columns or "predicted_label" not in df.columns:
        raise ValueError(
            "Results CSV must contain 'true_label' and 'predicted_label' columns."
        )

    metrics = compute_metrics(df, n_way)
    print_metrics(metrics)
    return metrics


# ── Standalone entry-point ────────────────────────────────────────────────────

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute metrics from a Fakeddit evaluation results CSV."
    )
    parser.add_argument("results_csv", help="Path to the results CSV file.")
    parser.add_argument(
        "--task",
        type=int,
        choices=[2, 3, 6],
        default=2,
        help="Classification task (2, 3, or 6-way).  Default: 2.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    evaluate_results_file(args.results_csv, args.task)
