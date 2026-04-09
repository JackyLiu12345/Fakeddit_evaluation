"""
lora_infer_qwen2vl.py — Single-sample inference using a trained LoRA adapter
for Qwen2-VL on the Fakeddit dataset.

Usage examples
--------------
# Infer with a local adapter:
    python lora_infer_qwen2vl.py \\
        --base-model Qwen/Qwen2-VL-2B-Instruct \\
        --adapter-path outputs/qwen2vl-lora/adapter \\
        --title "Breaking: Scientists discover cure for all diseases" \\
        --image-url "https://example.com/image.jpg"

# Specify task and device explicitly:
    python lora_infer_qwen2vl.py \\
        --base-model Qwen/Qwen2-VL-7B-Instruct \\
        --adapter-path outputs/qwen2vl-lora/adapter \\
        --task 3 \\
        --title "Your news title here" \\
        --image-url "https://example.com/image.jpg" \\
        --device cuda
"""

import argparse
import logging
import sys

import torch

logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run single-sample inference on a Qwen2-VL model with a LoRA adapter."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2-VL-2B-Instruct",
        dest="base_model",
        help="HuggingFace Hub ID of the base (pre-trained) model.",
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        dest="adapter_path",
        help="Path to the saved LoRA adapter directory.",
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=[2, 3, 6],
        default=2,
        help="Classification task (controls the label set shown to the model).",
    )
    parser.add_argument(
        "--title",
        required=True,
        help="News post title to classify.",
    )
    parser.add_argument(
        "--image-url",
        required=True,
        dest="image_url",
        help="Publicly accessible URL of the post image.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Compute device.  'auto' picks CUDA if available.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        dest="max_new_tokens",
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


# ── Inference ─────────────────────────────────────────────────────────────────

def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def run_inference(args):
    """Load model + adapter and run inference on a single sample."""
    import config as cfg
    from utils import download_image

    resolved_device = _resolve_device(args.device)
    label_map = cfg.LABEL_MAPS[args.task]
    label_names = ", ".join(label_map.values())

    # ── Load processor ────────────────────────────────────────────────────────
    logger.info("Loading processor from adapter path '%s' …", args.adapter_path)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.adapter_path)

    # ── Load base model ───────────────────────────────────────────────────────
    logger.info("Loading base model '%s' …", args.base_model)
    from transformers import Qwen2VLForConditionalGeneration

    dtype = torch.float16 if resolved_device == "cuda" else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto" if resolved_device == "cuda" else None,
    )

    if resolved_device != "cuda":
        model = model.to(resolved_device)

    # ── Attach LoRA adapter ───────────────────────────────────────────────────
    logger.info("Loading LoRA adapter from '%s' …", args.adapter_path)
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    logger.info("Model ready.")

    # ── Download image ────────────────────────────────────────────────────────
    logger.info("Downloading image from '%s' …", args.image_url)
    try:
        image = download_image(args.image_url)
    except Exception as exc:
        logger.error("Failed to download image: %s", exc)
        sys.exit(1)

    # ── Build prompt ──────────────────────────────────────────────────────────
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        f'Examine the image and classify this news post title:\n'
                        f'"{args.title}"\n'
                        f"Respond with ONLY the label from: {label_names}."
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=text, images=[image], return_tensors="pt")

    # Move inputs to the model's device.
    target_device = next(model.parameters()).device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    # ── Generate ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    raw_text = processor.decode(generated_ids[0], skip_special_tokens=True).strip()

    # ── Parse label ───────────────────────────────────────────────────────────
    labels_by_length = sorted(label_map.values(), key=len, reverse=True)
    predicted_label = "unknown"
    for label in labels_by_length:
        if label.lower() in raw_text.lower():
            predicted_label = label
            break

    print("\n" + "=" * 60)
    print(f"  Title       : {args.title}")
    print(f"  Image URL   : {args.image_url}")
    print(f"  Task        : {args.task}-way classification")
    print(f"  Raw output  : {raw_text!r}")
    print(f"  Predicted   : {predicted_label}")
    print("=" * 60 + "\n")

    return predicted_label


# ── Entry-point ───────────────────────────────────────────────────────────────

def main(argv=None):
    args = _parse_args(argv)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        import peft  # noqa: F401
    except ImportError:
        logger.error("peft is not installed. Install with: pip install peft")
        sys.exit(1)

    run_inference(args)


if __name__ == "__main__":
    main()
