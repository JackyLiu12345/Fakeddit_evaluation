"""
train_lora_qwen2vl.py — LoRA SFT training script for Qwen2-VL on Fakeddit.

Performs parameter-efficient fine-tuning using LoRA (QLoRA when CUDA is
available and bitsandbytes is installed) via the Hugging Face PEFT library.

Usage examples
--------------
# Basic training run (10% of train split, 2-way classification):
    python train_lora_qwen2vl.py

# 3-way classification, 5% data, bf16:
    python train_lora_qwen2vl.py --task 3 --sample-fraction 0.05 --bf16

# Disable QLoRA (use full fp16 LoRA):
    python train_lora_qwen2vl.py --no-use-qlora --fp16

# Use a larger model and cap training samples:
    python train_lora_qwen2vl.py --model-name Qwen/Qwen2-VL-7B-Instruct --max-train-samples 500
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="LoRA SFT training for Qwen2-VL on the Fakeddit dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2-VL-2B-Instruct",
        dest="model_name",
        help="HuggingFace Hub base model ID.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validate", "test"],
        default="train",
        help="Dataset split to train on.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.1,
        dest="sample_fraction",
        help="Fraction of the split to use for training.",
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=[2, 3, 6],
        default=2,
        help="Classification task: 2-way, 3-way, or 6-way.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/qwen2vl-lora",
        dest="output_dir",
        help="Directory to save the LoRA adapter checkpoint.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        dest="per_device_train_batch_size",
        help="Training batch size per GPU/CPU.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        dest="gradient_accumulation_steps",
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=1,
        dest="num_train_epochs",
        help="Number of full training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        dest="learning_rate",
        help="Peak learning rate for AdamW.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        dest="max_train_samples",
        help="Optional cap on the number of training samples.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 mixed precision (requires Ampere GPU or later).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 mixed precision.",
    )
    parser.add_argument(
        "--use-qlora",
        action=argparse.BooleanOptionalAction,
        default=None,  # None = auto-detect
        dest="use_qlora",
        help=(
            "Enable QLoRA (4-bit quantisation via bitsandbytes). "
            "Defaults to True when CUDA is available and bitsandbytes is installed."
        ),
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        dest="lora_r",
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        dest="lora_alpha",
        help="LoRA alpha (scaling factor).",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        dest="lora_dropout",
        help="Dropout probability applied to LoRA layers.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        dest="data_dir",
        help=(
            "Path to a local directory containing pre-downloaded Fakeddit TSV "
            "files.  When provided, Google Drive download is skipped entirely. "
            "Use this if gdown fails due to rate-limiting or permission errors."
        ),
    )
    return parser.parse_args(argv)


# ── QLoRA auto-detection ──────────────────────────────────────────────────────

def _should_use_qlora(requested: Optional[bool]) -> bool:
    """
    Resolve whether to use QLoRA.

    If *requested* is None, auto-detect based on CUDA availability and
    whether bitsandbytes can be imported.
    """
    if requested is False:
        return False
    if requested is True:
        return True
    # Auto-detect.
    if not torch.cuda.is_available():
        logger.info("QLoRA disabled: CUDA not available.")
        return False
    try:
        import bitsandbytes  # noqa: F401
        logger.info("QLoRA enabled (CUDA + bitsandbytes detected).")
        return True
    except ImportError:
        logger.info(
            "QLoRA disabled: bitsandbytes not installed. "
            "Install with: pip install bitsandbytes"
        )
        return False


# ── Dataset class ─────────────────────────────────────────────────────────────

class FakedditSFTDataset(torch.utils.data.Dataset):
    """
    Supervised fine-tuning dataset for Fakeddit multimodal rows.

    Each item is a dict with:
    - ``"input_ids"``: tokenised instruction + title (with image placeholder).
    - ``"labels"``: tokenised gold label (for language-model loss).
    - ``"pixel_values"``: preprocessed image tensor.
    - ``"image_grid_thw"``: Qwen2-VL grid metadata (if applicable).

    Rows where the image download fails are skipped silently.
    """

    def __init__(
        self,
        df,
        task: int,
        label_col: str,
        processor,
        max_train_samples: Optional[int] = None,
    ):
        import config as cfg
        from utils import download_image

        self.processor = processor
        self.label_map = cfg.LABEL_MAPS[task]
        self._items = []

        rows = df.iterrows()
        total_rows = min(len(df), max_train_samples) if max_train_samples else len(df)
        count = 0
        skipped = 0
        for idx, (_, row) in enumerate(rows):
            if max_train_samples is not None and count >= max_train_samples:
                break

            clean_title = str(row.get("clean_title", "")).strip()
            image_url = str(row.get("image_url", "")).strip()
            label_int = row.get(label_col)

            if not clean_title or not image_url:
                continue
            try:
                gold_label = self.label_map[int(label_int)]
            except (KeyError, ValueError, TypeError):
                continue

            # Download image; skip on failure.
            try:
                image = download_image(image_url)
            except Exception as exc:
                skipped += 1
                logger.debug("Skipping row — image download failed: %s", exc)
                if skipped % 50 == 0:
                    logger.warning(
                        "Image download: %d images skipped so far "
                        "(latest: %s).",
                        skipped, exc,
                    )
                continue

            self._items.append(
                {
                    "clean_title": clean_title,
                    "gold_label": gold_label,
                    "image": image,
                }
            )
            count += 1
            if count % 100 == 0:
                logger.info(
                    "Image download progress: %d/%d loaded, %d skipped.",
                    count, total_rows, skipped,
                )

        logger.info(
            "FakedditSFTDataset: %d valid samples loaded (%d skipped).",
            len(self._items), skipped,
        )

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        item = self._items[idx]
        clean_title = item["clean_title"]
        gold_label = item["gold_label"]
        image = item["image"]

        # Build chat messages for Qwen2-VL.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            f'Examine the image and classify this news post title:\n'
                            f'"{clean_title}"\n'
                            f"Respond with ONLY the label."
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": gold_label,
            },
        ]

        # Apply chat template to get the full formatted string.
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Encode inputs + image together.
        encoding = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
            padding=False,
        )

        # Squeeze batch dimension added by the processor.
        input_ids = encoding["input_ids"].squeeze(0)
        pixel_values = encoding.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.squeeze(0)

        # Build labels: mask the user / system portion with -100 so the loss
        # is computed only on the assistant's token(s).
        labels = input_ids.clone()

        # Locate the assistant's reply in the token sequence.
        # We encode the gold label alone to determine how many tokens it occupies,
        # then mask all preceding tokens with -100.
        # Assumption: the chat template always places the assistant reply at the
        # very end of the sequence (standard for decoder-only models).
        assistant_tokens = self.processor.tokenizer(
            gold_label, return_tensors="pt", add_special_tokens=False
        )["input_ids"].squeeze(0)
        assistant_len = len(assistant_tokens)

        # Mask everything except the last assistant_len tokens with -100 so that
        # the language-model loss is computed only on the assistant's output.
        if assistant_len < len(labels):
            labels[: len(labels) - assistant_len] = -100

        result = {
            "input_ids": input_ids,
            "labels": labels,
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        # Include image_grid_thw if returned by the processor (Qwen2-VL).
        if "image_grid_thw" in encoding:
            result["image_grid_thw"] = encoding["image_grid_thw"].squeeze(0)

        return result


# ── Collate function ──────────────────────────────────────────────────────────

def _collate_fn(batch: List[Dict], pad_token_id: int):
    """
    Pad sequences to the same length within a batch.
    Handles variable-size pixel_values by stacking only when shapes match.
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence

    input_ids = pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    labels = pad_sequence(
        [item["labels"] for item in batch],
        batch_first=True,
        padding_value=-100,
    )
    attention_mask = (input_ids != pad_token_id).long()

    result = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }

    # Stack pixel_values only if all items have them and shapes match.
    if all("pixel_values" in item for item in batch):
        shapes = [item["pixel_values"].shape for item in batch]
        if len(set(shapes)) == 1:
            result["pixel_values"] = torch.stack(
                [item["pixel_values"] for item in batch]
            )
        else:
            # Variable image sizes: concatenate along the batch dimension.
            # Each tensor is unsqueezed to add a leading batch dimension before
            # concatenation, producing a (batch_size, ...) tensor.
            result["pixel_values"] = torch.cat(
                [item["pixel_values"].unsqueeze(0) for item in batch], dim=0
            )

    if all("image_grid_thw" in item for item in batch):
        result["image_grid_thw"] = torch.stack(
            [item["image_grid_thw"] for item in batch]
        )

    return result


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    """Main training function."""
    from functools import partial

    import config as cfg
    from data_loader import load_split

    # ── 1. Resolve QLoRA ──────────────────────────────────────────────────────
    use_qlora = _should_use_qlora(args.use_qlora)

    # ── 2. Load data ──────────────────────────────────────────────────────────
    logger.info(
        "Loading '%s' split (%.0f%% sample) …", args.split, args.sample_fraction * 100
    )
    df = load_split(args.split, sample_fraction=args.sample_fraction,
                    data_dir=args.data_dir)

    label_col = cfg.LABEL_COLUMN[args.task]
    if label_col not in df.columns:
        logger.error("Label column '%s' not found in the dataset.", label_col)
        sys.exit(1)

    # ── 3. Load processor ─────────────────────────────────────────────────────
    logger.info("Loading processor for '%s' …", args.model_name)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_name)
    # Ensure a pad token exists.
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ── 4. Build dataset ──────────────────────────────────────────────────────
    logger.info("Building SFT dataset …")
    dataset = FakedditSFTDataset(
        df,
        task=args.task,
        label_col=label_col,
        processor=processor,
        max_train_samples=args.max_train_samples,
    )

    if len(dataset) == 0:
        logger.error("Dataset is empty — no valid samples. Aborting.")
        sys.exit(1)

    # ── 5. Load model (with optional 4-bit quantisation) ──────────────────────
    logger.info("Loading base model '%s' …", args.model_name)
    from transformers import Qwen2VLForConditionalGeneration

    load_kwargs = {"device_map": "auto"}

    if use_qlora:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
        logger.info("4-bit QLoRA quantisation enabled.")
    else:
        dtype = torch.bfloat16 if args.bf16 else (
            torch.float16 if args.fp16 else torch.float32
        )
        load_kwargs["torch_dtype"] = dtype

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name, **load_kwargs
    )

    # Prepare model for k-bit training (required by PEFT for QLoRA).
    if use_qlora:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    # ── 6. Apply LoRA ─────────────────────────────────────────────────────────
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 7. Set up Trainer ─────────────────────────────────────────────────────
    from transformers import Trainer, TrainingArguments

    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",  # Disable W&B / TensorBoard by default.
        remove_unused_columns=False,
    )

    collate = partial(
        _collate_fn, pad_token_id=processor.tokenizer.pad_token_id
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate,
    )

    # ── 8. Train ──────────────────────────────────────────────────────────────
    logger.info("Starting training …")
    trainer.train()

    # ── 9. Save adapter ───────────────────────────────────────────────────────
    adapter_path = os.path.join(args.output_dir, "adapter")
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    logger.info("LoRA adapter saved to '%s'.", adapter_path)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Adapter checkpoint: {adapter_path}")
    print("\nTo run inference with this adapter:")
    print(
        f"  python lora_infer_qwen2vl.py \\\n"
        f"    --base-model {args.model_name} \\\n"
        f"    --adapter-path {adapter_path} \\\n"
        f"    --title 'Your news title here' \\\n"
        f"    --image-url 'https://example.com/image.jpg'"
    )
    print("=" * 60 + "\n")


# ── Entry-point ───────────────────────────────────────────────────────────────

def main(argv=None):
    args = _parse_args(argv)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Validate precision flags.
    if args.bf16 and args.fp16:
        logger.error("--bf16 and --fp16 are mutually exclusive. Pick one.")
        sys.exit(1)

    # Import check: peft is required.
    try:
        import peft  # noqa: F401
    except ImportError:
        logger.error(
            "peft is not installed. Install with: pip install peft"
        )
        sys.exit(1)

    train(args)


if __name__ == "__main__":
    main()
