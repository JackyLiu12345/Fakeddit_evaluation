"""
prompts.py — Zero-shot prompt templates for Fakeddit classification tasks.

Supports 2-way, 3-way, and 6-way classification for both LLaVA-style and
Qwen2-VL-style models.
"""

from typing import Dict, List, Tuple, Union

from config import LABEL_MAPS

# ── Human-readable label descriptions used inside prompts ─────────────────────
_LABEL_DESCRIPTIONS: Dict[int, str] = {
    2: (
        "- real\n"
        "- fake"
    ),
    3: (
        "- true\n"
        "- fake_with_true_text\n"
        "- fake_with_false_text"
    ),
    6: (
        "- true\n"
        "- satire/parody\n"
        "- misleading content\n"
        "- imposter content\n"
        "- false connection\n"
        "- manipulated content"
    ),
}

_SYSTEM_PROMPT = (
    "You are a multimodal fake-news detection assistant. "
    "Given a news post (title + image), classify it accurately. "
    "Respond with ONLY the label name — no explanation, no punctuation."
)


def get_prompts(n_way: int, clean_title: str) -> Tuple[str, str]:
    """
    Return ``(system_prompt, user_prompt)`` for a zero-shot classification task.

    Parameters
    ----------
    n_way:
        Number of classes: 2, 3, or 6.
    clean_title:
        The post's cleaned title text.

    Returns
    -------
    (system_prompt, user_prompt)
    """
    if n_way not in LABEL_MAPS:
        raise ValueError(f"n_way must be one of {list(LABEL_MAPS.keys())}, got {n_way}.")

    label_list = _LABEL_DESCRIPTIONS[n_way]
    label_names = ", ".join(LABEL_MAPS[n_way].values())

    user_prompt = (
        f"Examine the image and the following news post title carefully.\n\n"
        f"Title: \"{clean_title}\"\n\n"
        f"Classify this news post into exactly one of these {n_way} categories:\n"
        f"{label_list}\n\n"
        f"Respond with ONLY the label name from this list: {label_names}."
    )

    return _SYSTEM_PROMPT, user_prompt


def format_llava_prompt(user_prompt: str) -> str:
    """
    Wrap *user_prompt* in the LLaVA instruction template.

    The ``<image>`` token must appear inside the [INST] … [/INST] block so
    the LLaVA processor knows where to insert image features.
    """
    return f"[INST] <image>\n{user_prompt} [/INST]"


def format_qwen2vl_messages(
    system_prompt: str, user_prompt: str
) -> List[Dict[str, object]]:
    """
    Return a chat-message list suitable for Qwen2-VL's processor.

    The image placeholder is represented as a dict with ``type: "image"``
    following the Qwen2-VL conversation format; the actual PIL image object
    is passed separately when calling the processor.
    """
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
