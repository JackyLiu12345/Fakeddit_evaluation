"""
utils.py — Shared helpers: image downloading, logging setup, and HF model
loading / inference.
"""

import logging
import time
from io import BytesIO
from typing import Any, List, Tuple, Union

import requests
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ── Retry configuration ───────────────────────────────────────────────────────
_MAX_RETRIES = 3             # retries for non-429 errors
_MAX_RETRIES_429 = 6         # extra patience for rate-limit (429) responses
_BACKOFF_BASE = 2.0          # seconds; doubles on each retry
_BACKOFF_429_MIN = 5.0       # minimum wait (seconds) on 429 before first retry
_REQUEST_TIMEOUT = 15        # seconds per HTTP request
_INTER_REQUEST_DELAY = 0.2   # small pause between consecutive downloads (seconds)
_last_request_time: float = 0.0  # monotonic timestamp of last HTTP request

# HTTP headers — a User-Agent is required by many CDNs (including imgur)
# to avoid being blocked outright.
_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; FakedditEval/1.0; "
        "+https://github.com/JackyLiu12345/Fakeddit_evaluation)"
    ),
}


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root-level logging with a consistent format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ── Image downloading ─────────────────────────────────────────────────────────

def _get_429_wait(response: requests.Response, attempt: int) -> float:
    """
    Determine how long to wait after an HTTP 429 response.

    Checks the ``Retry-After`` header first; falls back to exponential
    back-off starting from ``_BACKOFF_429_MIN``.
    """
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(float(retry_after), 1.0)
        except (ValueError, TypeError):
            pass
    # Exponential back-off: 5 s, 10 s, 20 s, 40 s …
    return _BACKOFF_429_MIN * (_BACKOFF_BASE ** (attempt - 1))


def download_image(url: str) -> Image.Image:
    """
    Fetch an image from *url* and return a PIL Image in RGB mode.

    Rate-limit handling
    -------------------
    - A small inter-request delay (``_INTER_REQUEST_DELAY``) is enforced
      between consecutive calls to avoid triggering CDN rate limits.
    - HTTP 429 responses are retried with longer back-off (up to
      ``_MAX_RETRIES_429`` attempts), and the ``Retry-After`` header is
      respected when present.
    - Other errors are retried up to ``_MAX_RETRIES`` times with standard
      exponential back-off.

    Parameters
    ----------
    url:
        Publicly accessible image URL.

    Returns
    -------
    PIL.Image.Image (RGB)

    Raises
    ------
    RuntimeError
        If the image cannot be fetched after all retries.
    """
    global _last_request_time

    # Enforce a small pause between successive downloads to stay below
    # CDN rate limits when processing many images in a loop.
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _INTER_REQUEST_DELAY:
        time.sleep(_INTER_REQUEST_DELAY - elapsed)

    last_exc: Exception = RuntimeError(f"Failed to download image from {url!r}")
    max_attempts = _MAX_RETRIES_429   # use the larger retry budget

    for attempt in range(1, max_attempts + 1):
        try:
            _last_request_time = time.monotonic()
            response = requests.get(
                url, timeout=_REQUEST_TIMEOUT, headers=_HTTP_HEADERS,
            )
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return image
        except requests.exceptions.HTTPError as exc:
            last_exc = exc
            if response.status_code == 429:
                # Rate-limited — use longer, smarter back-off.
                wait = _get_429_wait(response, attempt)
                logger.warning(
                    "Rate-limited (429) downloading %s (attempt %d/%d). "
                    "Waiting %.1fs …",
                    url, attempt, max_attempts, wait,
                )
                time.sleep(wait)
            else:
                # Non-429 HTTP error — use standard back-off with the
                # smaller retry budget.
                if attempt > _MAX_RETRIES:
                    break
                wait = _BACKOFF_BASE ** (attempt - 1)
                logger.warning(
                    "Image download attempt %d/%d failed (%s). "
                    "Retrying in %.1fs …",
                    attempt, _MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
        except Exception as exc:
            last_exc = exc
            if attempt > _MAX_RETRIES:
                break
            wait = _BACKOFF_BASE ** (attempt - 1)
            logger.warning(
                "Image download attempt %d/%d failed (%s). "
                "Retrying in %.1fs …",
                attempt, _MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Could not download image from {url!r} after retries."
    ) from last_exc


# ── HuggingFace model loading ─────────────────────────────────────────────────

def _resolve_device(device: str) -> str:
    """Resolve the ``"auto"`` device alias to ``"cuda"`` or ``"cpu"``."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model_and_processor(
    model_name: str,
    device: str = "auto",
    dtype: torch.dtype = None,
) -> Tuple[Any, Any]:
    """
    Factory function — load the appropriate HuggingFace model and processor for
    *model_name*.

    Detection logic (case-insensitive model name matching):
    - Contains ``"llava"``  → ``LlavaNextForConditionalGeneration`` + ``LlavaNextProcessor``
    - Contains ``"qwen"``   → ``Qwen2VLForConditionalGeneration`` + ``AutoProcessor``
    - Otherwise             → ``AutoModelForVision2Seq`` + ``AutoProcessor``

    Parameters
    ----------
    model_name:
        HuggingFace Hub model ID (e.g. ``"llava-hf/llava-v1.6-mistral-7b-hf"``).
    device:
        ``"cuda"``, ``"cpu"``, or ``"auto"`` (default).
    dtype:
        Torch dtype.  If *None*, ``torch.float16`` is used on CUDA and
        ``torch.float32`` on CPU.

    Returns
    -------
    (model, processor)
    """
    resolved_device = _resolve_device(device)

    if dtype is None:
        dtype = torch.float16 if resolved_device == "cuda" else torch.float32

    model_name_lower = model_name.lower()

    logger.info("Loading model '%s' …", model_name)

    if "llava" in model_name_lower:
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        processor = LlavaNextProcessor.from_pretrained(model_name)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )

    elif "qwen" in model_name_lower:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )

    else:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )

    model.eval()
    logger.info("Model loaded on device_map='auto' (dtype=%s).", dtype)
    return model, processor


# ── Unified inference helper ──────────────────────────────────────────────────

def generate_response(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt_text: Union[str, List[dict]],
    device: str = "auto",
    max_new_tokens: int = 64,
) -> str:
    """
    Run inference with *model* + *processor* on a single image–text pair.

    Handles the input-formatting differences between LLaVA-style and Qwen2-VL-
    style models transparently.

    Parameters
    ----------
    model:
        A loaded HuggingFace model.
    processor:
        The matching processor.
    image:
        PIL Image (RGB) to pass as visual context.
    prompt_text:
        Either a formatted string (LLaVA) or a chat-messages list (Qwen2-VL).
    device:
        Device string (``"cuda"``, ``"cpu"``, or ``"auto"``).
    max_new_tokens:
        Maximum tokens the model may generate.

    Returns
    -------
    str
        The raw decoded text generated by the model (input prompt stripped).
    """
    # Determine whether we have a Qwen2-VL style messages list or a plain string.
    if isinstance(prompt_text, list):
        # Qwen2-VL: apply chat template to get the formatted string, then encode.
        formatted_prompt = processor.apply_chat_template(
            prompt_text, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=formatted_prompt,
            images=[image],
            return_tensors="pt",
        )
    else:
        # LLaVA (and generic Vision2Seq): prompt_text is already a string.
        inputs = processor(
            text=prompt_text,
            images=[image],
            return_tensors="pt",
        )

    # Move inputs to the model's first parameter device.
    # When device_map="auto" is used the model may span multiple devices, so
    # we target the device that holds the embedding layer (first parameter).
    target_device = next(model.parameters()).device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Strip the input tokens from the output to get only the generated text.
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    raw_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    return raw_text.strip()
