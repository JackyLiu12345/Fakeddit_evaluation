"""
icl.py — In-Context Learning (ICL) helpers for Fakeddit multimodal evaluation.

Provides functions to:
- Build a demo pool from a dataset split.
- Select balanced or hard-negative demonstration examples.
- Compose ICL-augmented user prompts.
- Apply majority voting over multiple predictions.

All selection functions are deterministic given a fixed random_seed.
"""

import logging
import re
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd

from config import LABEL_COLUMN, LABEL_MAPS

logger = logging.getLogger(__name__)

# Sensational tokens used for hard-negative heuristic selection.
_SENSATIONAL_TOKENS = {
    "breaking",
    "shocking",
    "must see",
    "unbelievable",
    "you won't believe",
    "urgent",
    "exclusive",
    "bombshell",
    "exposed",
    "scandal",
    "alert",
    "warning",
    "incredible",
    "amazing",
    "outrageous",
    "jaw-dropping",
    "secret",
    "leaked",
    "disturbing",
    "viral",
}

# Short/ambiguous title heuristic: fewer than this many words.
_SHORT_TITLE_WORD_THRESHOLD = 6


def build_demo_pool(
    df: pd.DataFrame,
    task: int,
    label_col: str,
) -> pd.DataFrame:
    """
    Build a pool of candidate demonstrations from *df* for the given *task*.

    Keeps only rows that have:
    - A valid ``clean_title`` (non-null, non-empty).
    - A valid label value in the label map for *task*.

    Parameters
    ----------
    df:
        Full (sampled) DataFrame from :func:`data_loader.load_split`.
    task:
        Classification task (2, 3, or 6).
    label_col:
        Name of the label column (e.g. ``"2_way_label"``).

    Returns
    -------
    pandas.DataFrame with columns ``clean_title``, ``label_str``.
    """
    if task not in LABEL_MAPS:
        raise ValueError(f"task must be one of {list(LABEL_MAPS.keys())}, got {task}.")

    label_map = LABEL_MAPS[task]
    valid_int_labels = set(label_map.keys())

    # Require a valid label and non-empty title.
    pool = df.copy()

    if label_col not in pool.columns:
        logger.warning("Label column '%s' not found; demo pool is empty.", label_col)
        return pd.DataFrame(columns=["clean_title", "label_str"])

    pool = pool[pool[label_col].notna()].copy()
    pool = pool[pool["clean_title"].notna() & (pool["clean_title"].str.strip() != "")].copy()

    # Convert numeric labels to string label names.
    def _to_label_str(val):
        try:
            return label_map[int(val)]
        except (KeyError, ValueError, TypeError):
            return None

    pool["label_str"] = pool[label_col].apply(_to_label_str)
    pool = pool[pool["label_str"].notna()].copy()

    logger.debug("Demo pool size for task=%d: %d rows.", task, len(pool))
    return pool[["clean_title", "label_str"]].reset_index(drop=True)


def select_demos_balanced(
    df: pd.DataFrame,
    task: int,
    label_col: str,
    num_demos_per_class: int = 1,
    random_seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Select *num_demos_per_class* examples per class from the demo pool via
    uniform random sampling (stratified).

    Parameters
    ----------
    df:
        Source DataFrame (same one passed to the evaluator).
    task:
        Classification task (2, 3, or 6).
    label_col:
        Label column name.
    num_demos_per_class:
        Number of demonstrations to sample per class.
    random_seed:
        Random seed for reproducibility.

    Returns
    -------
    List of dicts with keys ``"title"`` and ``"label"``.
    """
    pool = build_demo_pool(df, task, label_col)
    if pool.empty:
        logger.warning("Demo pool is empty; returning no demos.")
        return []

    demos: List[Dict[str, str]] = []
    label_map = LABEL_MAPS[task]

    for label_str in label_map.values():
        class_rows = pool[pool["label_str"] == label_str]
        if class_rows.empty:
            logger.warning("No demos found for label '%s' (task=%d).", label_str, task)
            continue
        sampled = class_rows.sample(
            n=min(num_demos_per_class, len(class_rows)),
            random_state=random_seed,
        )
        for _, row in sampled.iterrows():
            demos.append({"title": str(row["clean_title"]), "label": label_str})

    logger.debug("Balanced demos selected: %d total.", len(demos))
    return demos


def _is_hard_negative(title: str) -> bool:
    """
    Return True if *title* is considered 'hard-negative' by heuristic:
    - Title word count is below the short-title threshold, OR
    - Title contains at least one sensational token.
    """
    title_lower = title.lower()
    word_count = len(title_lower.split())
    if word_count < _SHORT_TITLE_WORD_THRESHOLD:
        return True
    for token in _SENSATIONAL_TOKENS:
        if token in title_lower:
            return True
    return False


def select_demos_hard_negative(
    df: pd.DataFrame,
    task: int,
    label_col: str,
    num_demos_per_class: int = 1,
    random_seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Select *num_demos_per_class* hard-negative examples per class.

    Hard-negative heuristic: prefer examples where the title is short/ambiguous
    or contains sensational tokens (e.g. "breaking", "shocking").  Falls back
    to balanced random selection if too few hard-negative examples exist for a
    class.

    Parameters
    ----------
    df, task, label_col, num_demos_per_class, random_seed:
        Same semantics as :func:`select_demos_balanced`.

    Returns
    -------
    List of dicts with keys ``"title"`` and ``"label"``.
    """
    pool = build_demo_pool(df, task, label_col)
    if pool.empty:
        logger.warning("Demo pool is empty; returning no demos.")
        return []

    demos: List[Dict[str, str]] = []
    label_map = LABEL_MAPS[task]

    for label_str in label_map.values():
        class_rows = pool[pool["label_str"] == label_str].copy()
        if class_rows.empty:
            logger.warning("No demos found for label '%s' (task=%d).", label_str, task)
            continue

        # Filter to hard-negative candidates.
        hn_mask = class_rows["clean_title"].apply(_is_hard_negative)
        hn_rows = class_rows[hn_mask]

        if len(hn_rows) >= num_demos_per_class:
            sampled = hn_rows.sample(
                n=num_demos_per_class, random_state=random_seed
            )
        else:
            # Fall back to the full class pool if not enough HN examples.
            logger.debug(
                "Insufficient hard-negative examples for '%s' (%d < %d); "
                "falling back to all class examples.",
                label_str,
                len(hn_rows),
                num_demos_per_class,
            )
            sampled = class_rows.sample(
                n=min(num_demos_per_class, len(class_rows)),
                random_state=random_seed,
            )

        for _, row in sampled.iterrows():
            demos.append({"title": str(row["clean_title"]), "label": label_str})

    logger.debug("Hard-negative demos selected: %d total.", len(demos))
    return demos


def compose_icl_user_prompt(
    clean_title: str,
    task: int,
    demos: List[Dict[str, str]],
    include_rationale: bool = False,
) -> str:
    """
    Compose an ICL-augmented user prompt for a single evaluation sample.

    Prepends formatted demonstrations before the evaluation question.
    Each demo is text-only (no image); the final query still includes the
    image (handled by the caller via the processor).

    Parameters
    ----------
    clean_title:
        The post title to classify.
    task:
        Classification task (2, 3, or 6).
    demos:
        List of demo dicts, each with keys ``"title"`` and ``"label"``.
        Produced by :func:`select_demos_balanced` or
        :func:`select_demos_hard_negative`.
    include_rationale:
        If True, include a brief rationale placeholder in each demo entry.
        Useful for chain-of-thought style prompting.

    Returns
    -------
    str — The full user-visible prompt text (system prompt is kept separate).
    """
    if task not in LABEL_MAPS:
        raise ValueError(f"task must be one of {list(LABEL_MAPS.keys())}, got {task}.")

    label_map = LABEL_MAPS[task]
    label_names = ", ".join(label_map.values())

    label_list_str = "\n".join(f"- {lbl}" for lbl in label_map.values())

    # Build demonstration block.
    demo_lines: List[str] = []
    if demos:
        demo_lines.append("Here are some labeled examples to guide your classification:\n")
        for i, demo in enumerate(demos, start=1):
            demo_title = demo["title"]
            demo_label = demo["label"]
            if include_rationale:
                demo_lines.append(
                    f"Example {i}:\n"
                    f'  Title: "{demo_title}"\n'
                    f"  Rationale: [This is a {demo_label} post based on the title characteristics.]\n"
                    f"  Label: {demo_label}"
                )
            else:
                demo_lines.append(
                    f"Example {i}:\n"
                    f'  Title: "{demo_title}"\n'
                    f"  Label: {demo_label}"
                )
        demo_lines.append("")  # blank line before the query

    demo_block = "\n".join(demo_lines)

    # Build the evaluation query.
    query = (
        f"Now examine the image and the following news post title carefully.\n\n"
        f'Title: "{clean_title}"\n\n'
        f"Classify this news post into exactly one of these {task} categories:\n"
        f"{label_list_str}\n\n"
        f"Respond with ONLY the label name from this list: {label_names}."
    )

    if demo_block:
        return demo_block + query
    return query


def majority_vote(predictions: List[str]) -> str:
    """
    Return the most common prediction string from *predictions*.

    In case of a tie, the first-occurring majority label (alphabetically
    if still tied) is returned.  Returns ``"unknown"`` for an empty list.

    Parameters
    ----------
    predictions:
        List of predicted label strings (e.g. ``["real", "fake", "real"]``).

    Returns
    -------
    str — The majority-voted label.
    """
    if not predictions:
        return "unknown"
    counter = Counter(predictions)
    # most_common returns in insertion/count order; ties are broken
    # deterministically by Python's stable sort on the count.
    return counter.most_common(1)[0][0]
