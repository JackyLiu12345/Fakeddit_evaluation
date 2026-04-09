"""
config.py — Central configuration for Fakeddit multimodal LLM evaluation.
"""

# ── Google Drive file IDs for each dataset split (Fakeddit v2.0) ──────────────
# Users can verify these from the official Google Drive folder:
# https://drive.google.com/drive/folders/1jU7qgDqU1je9Y0PMKJ_f31yXRo5uWGFm
GDRIVE_FILE_IDS = {
    "train":    "1iu-H12Rvmz_XW3lK9IH7bwMfOpKddhE8",
    "validate": "1yNEEzn3EjjhywIAb9Xli_mF5O65nQ9cx",
    "test":     "1GqQtt86gxdGMjbx7KxM4XQyWQGvpEX2M",
}

# ── Default model ─────────────────────────────────────────────────────────────
DEFAULT_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"

# ── Sampling ──────────────────────────────────────────────────────────────────
DEFAULT_SAMPLE_FRACTION = 0.1   # Use 10% of the dataset by default
DEFAULT_RANDOM_SEED = 42

# ── Device ────────────────────────────────────────────────────────────────────
# "auto" → use CUDA if available, otherwise CPU
DEFAULT_DEVICE = "auto"

# ── Generation parameters ─────────────────────────────────────────────────────
GENERATION_CONFIG = {
    "max_new_tokens": 64,
    "do_sample": False,
}

# ── Label maps ────────────────────────────────────────────────────────────────
LABEL_MAPS = {
    2: {0: "real", 1: "fake"},
    3: {0: "true", 1: "fake_with_true_text", 2: "fake_with_false_text"},
    6: {
        0: "true",
        1: "satire/parody",
        2: "misleading content",
        3: "imposter content",
        4: "false connection",
        5: "manipulated content",
    },
}

# ── Column names expected in the TSV files ────────────────────────────────────
LABEL_COLUMN = {
    2: "2_way_label",
    3: "3_way_label",
    6: "6_way_label",
}

# ── Expected TSV filenames when loading from a local directory ────────────────
# These match the default filenames used by the official Fakeddit dataset.
# Users can place pre-downloaded TSV files in a directory and pass --data-dir.
LOCAL_TSV_FILENAMES = {
    "train":    "multimodal_train.tsv",
    "validate": "multimodal_validate.tsv",
    "test":     "multimodal_test_public.tsv",
}
