"""
data_loader.py — Stream Fakeddit TSV files from Google Drive into memory.

No large files are written permanently to disk.  A temporary file is used
only while gdown transfers the data; it is deleted immediately afterwards.
"""

import io
import logging
import os
import tempfile

import gdown
import pandas as pd

from config import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAMPLE_FRACTION,
    GDRIVE_FILE_IDS,
)

logger = logging.getLogger(__name__)


def _download_tsv_to_dataframe(file_id: str) -> pd.DataFrame:
    """
    Download a TSV from Google Drive (identified by *file_id*) and return it
    as a :class:`pandas.DataFrame`.

    Strategy
    --------
    gdown does not support writing directly to a BytesIO buffer, so we use a
    temporary file.  The file is removed as soon as pandas has parsed it into
    memory, keeping disk usage negligible.
    """
    url = f"https://drive.google.com/uc?id={file_id}"

    # Create a temporary file that is automatically cleaned up.
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        logger.info("Downloading TSV from Google Drive (file_id=%s) …", file_id)
        gdown.download(url, tmp_path, quiet=False)

        logger.info("Parsing TSV into DataFrame …")
        df = pd.read_csv(tmp_path, sep="\t", low_memory=False)
    finally:
        # Always remove the temporary file, even if parsing fails.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.debug("Removed temporary file: %s", tmp_path)

    return df


def _filter_multimodal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows that have an associated image.

    A row is considered multimodal when:
    - ``hasImage`` is ``True`` (boolean) or ``1`` (integer).
    - ``image_url`` is non-empty / non-null.
    """
    has_image_col = df["hasImage"].astype(str).str.lower().isin({"true", "1"})
    has_url = df["image_url"].notna() & (df["image_url"].str.strip() != "")
    filtered = df[has_image_col & has_url].copy()
    logger.info(
        "Multimodal filter: %d → %d rows (kept %.1f%%)",
        len(df),
        len(filtered),
        100 * len(filtered) / max(len(df), 1),
    )
    return filtered


def load_split(
    split_name: str,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """
    Load a Fakeddit dataset split from Google Drive and return a sampled,
    multimodal-only :class:`pandas.DataFrame`.

    Parameters
    ----------
    split_name:
        One of ``"train"``, ``"validate"``, or ``"test"``.
    sample_fraction:
        Fraction of multimodal rows to keep (default: 0.10 → 10%).
    random_seed:
        Random seed for reproducible sampling.

    Returns
    -------
    pandas.DataFrame
        Sampled rows with at least the columns ``id``, ``clean_title``,
        ``image_url``, ``hasImage``, ``2_way_label``, ``3_way_label``,
        and ``6_way_label``.
    """
    if split_name not in GDRIVE_FILE_IDS:
        raise ValueError(
            f"Unknown split '{split_name}'. "
            f"Choose from: {list(GDRIVE_FILE_IDS.keys())}"
        )

    file_id = GDRIVE_FILE_IDS[split_name]
    df = _download_tsv_to_dataframe(file_id)

    # Keep only multimodal rows before sampling so the fraction refers to
    # the multimodal subset, not the entire dataset.
    df = _filter_multimodal(df)

    # Sample the requested fraction.
    n_before = len(df)
    df = df.sample(frac=sample_fraction, random_state=random_seed).reset_index(
        drop=True
    )
    logger.info(
        "Sampled %.0f%% of multimodal rows: %d → %d",
        sample_fraction * 100,
        n_before,
        len(df),
    )

    return df
