"""
data_loader.py — Load Fakeddit TSV files from Google Drive or a local directory.

Supports two loading modes:
1. **Google Drive (default):** Downloads TSV via ``gdown`` using file IDs from
   ``config.GDRIVE_FILE_IDS``.  A temporary file is used only during download
   and is deleted immediately afterwards.
2. **Local directory (``--data-dir``):** Reads pre-downloaded TSV files from a
   user-supplied directory, using filenames from ``config.LOCAL_TSV_FILENAMES``.
   This avoids Google Drive rate-limit / permission errors entirely.
"""

import logging
import os
import tempfile

import pandas as pd

from config import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAMPLE_FRACTION,
    GDRIVE_FILE_IDS,
    LOCAL_TSV_FILENAMES,
)

logger = logging.getLogger(__name__)

# ── Helpful manual-download instructions shown on gdown failure ───────────────
_MANUAL_DOWNLOAD_MSG = """\

╔══════════════════════════════════════════════════════════════════════╗
║  Google Drive download failed.                                     ║
║                                                                    ║
║  This usually means the file's sharing link has been rate-limited  ║
║  or its permissions are not set to "Anyone with the link".         ║
║                                                                    ║
║  WORKAROUND — download the TSV manually:                           ║
║                                                                    ║
║    1. Open this URL in a browser:                                  ║
║       https://drive.google.com/uc?id={file_id}                     ║
║    2. Save the file to a local folder, e.g.  ./data/               ║
║    3. Re-run with --data-dir pointing to that folder:              ║
║                                                                    ║
║       python evaluate.py --data-dir ./data/                        ║
║                                                                    ║
║  Expected filename: {filename}                                     ║
║  (or any .tsv file matching that split)                            ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def _download_tsv_to_dataframe(file_id: str, split_name: str) -> pd.DataFrame:
    """
    Download a TSV from Google Drive (identified by *file_id*) and return it
    as a :class:`pandas.DataFrame`.

    Uses ``gdown`` with ``fuzzy=True`` for better compatibility with Google
    Drive's confirmation pages and rate-limiting mechanisms.

    Parameters
    ----------
    file_id:
        Google Drive file identifier.
    split_name:
        Dataset split name (used only for error messages).

    Raises
    ------
    RuntimeError
        If the download or parsing fails.
    """
    import gdown

    url = f"https://drive.google.com/uc?id={file_id}"

    # Create a temporary file that is automatically cleaned up.
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        logger.info("Downloading TSV from Google Drive (file_id=%s) …", file_id)
        output = gdown.download(url, tmp_path, quiet=False, fuzzy=True)

        if output is None:
            # gdown returns None when the download fails.
            expected_fn = LOCAL_TSV_FILENAMES.get(split_name, f"{split_name}.tsv")
            raise RuntimeError(
                f"gdown returned None — download failed for split '{split_name}'."
                + _MANUAL_DOWNLOAD_MSG.format(
                    file_id=file_id, filename=expected_fn
                )
            )

        logger.info("Parsing TSV into DataFrame …")
        df = pd.read_csv(tmp_path, sep="\t", low_memory=False)
    except Exception as exc:
        # Re-raise with helpful instructions if not already a RuntimeError
        # with our message.
        if "WORKAROUND" not in str(exc):
            expected_fn = LOCAL_TSV_FILENAMES.get(split_name, f"{split_name}.tsv")
            raise RuntimeError(
                str(exc)
                + _MANUAL_DOWNLOAD_MSG.format(
                    file_id=file_id, filename=expected_fn
                )
            ) from exc
        raise
    finally:
        # Always remove the temporary file, even if parsing fails.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.debug("Removed temporary file: %s", tmp_path)

    return df


def _load_local_tsv(data_dir: str, split_name: str) -> pd.DataFrame:
    """
    Load a TSV file from a local directory.

    Tries ``config.LOCAL_TSV_FILENAMES[split_name]`` first, then falls back to
    any ``.tsv`` file whose name contains the *split_name* string.

    Parameters
    ----------
    data_dir:
        Path to the directory containing the TSV files.
    split_name:
        One of ``"train"``, ``"validate"``, ``"test"``.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    FileNotFoundError
        If no matching TSV file is found.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"--data-dir '{data_dir}' does not exist or is not a directory."
        )

    # Primary: try the canonical filename.
    canonical = LOCAL_TSV_FILENAMES.get(split_name)
    if canonical:
        canonical_path = os.path.join(data_dir, canonical)
        if os.path.isfile(canonical_path):
            logger.info("Loading local TSV: %s", canonical_path)
            return pd.read_csv(canonical_path, sep="\t", low_memory=False)

    # Fallback: look for any .tsv file containing the split name.
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".tsv") and split_name in fname.lower():
            fpath = os.path.join(data_dir, fname)
            logger.info("Loading local TSV (fallback match): %s", fpath)
            return pd.read_csv(fpath, sep="\t", low_memory=False)

    expected = canonical or f"*{split_name}*.tsv"
    raise FileNotFoundError(
        f"No TSV file found for split '{split_name}' in '{data_dir}'. "
        f"Expected filename: {expected}\n"
        f"Files present: {os.listdir(data_dir)}"
    )


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
    data_dir: str = None,
) -> pd.DataFrame:
    """
    Load a Fakeddit dataset split and return a sampled, multimodal-only
    :class:`pandas.DataFrame`.

    Data source priority:
    1. If *data_dir* is given, load from a local TSV file in that directory.
    2. Otherwise, download from Google Drive using ``gdown``.

    Parameters
    ----------
    split_name:
        One of ``"train"``, ``"validate"``, or ``"test"``.
    sample_fraction:
        Fraction of multimodal rows to keep (default: 0.10 → 10%).
    random_seed:
        Random seed for reproducible sampling.
    data_dir:
        Path to a local directory containing pre-downloaded Fakeddit TSV
        files.  When provided, Google Drive download is skipped entirely.

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

    if data_dir:
        # ── Local file path ───────────────────────────────────────────────
        df = _load_local_tsv(data_dir, split_name)
    else:
        # ── Google Drive download ─────────────────────────────────────────
        file_id = GDRIVE_FILE_IDS[split_name]
        df = _download_tsv_to_dataframe(file_id, split_name)

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
