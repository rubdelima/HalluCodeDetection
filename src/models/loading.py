"""Shared Hugging Face loaders for the Gemma model families."""

from __future__ import annotations

import torch
from transformers import AutoModelForImageTextToText, AutoModelForMultimodalLM, AutoTokenizer


def is_gemma4(model_id: str) -> bool:
    """Return whether a checkpoint uses the Gemma 4 multimodal architecture."""
    return "gemma-4-" in model_id.lower()


def load_text_tokenizer(model_id: str):
    """Load the text tokenizer without importing Gemma's video processor."""
    return AutoTokenizer.from_pretrained(model_id)


def load_model(
    model_id: str,
    *,
    dtype: torch.dtype | None = None,
    **kwargs,
):
    """Load Gemma 3 or Gemma 4 with the appropriate Transformers auto class."""
    model_class = AutoModelForMultimodalLM if is_gemma4(model_id) else AutoModelForImageTextToText
    if dtype is not None:
        kwargs["dtype"] = dtype
    return model_class.from_pretrained(model_id, **kwargs)
