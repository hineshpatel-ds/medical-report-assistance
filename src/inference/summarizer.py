"""Thin inference wrapper around the summarization model, used by the
Streamlit app and by tests. Loads the fine-tuned checkpoint if one has been
trained (see src/modeling/train_summarizer.py); otherwise falls back to the
plain pretrained model so the app still runs before any training happens.

Generation is done directly with AutoModelForSeq2SeqLM + tokenizer rather
than transformers' `pipeline("summarization", ...)` helper: transformers 5.x
dropped the "summarization" task from the pipeline registry (only
classification/generation-style tasks remain), so the encoder-decoder
generate() call is done by hand here instead.
"""
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

FINE_TUNED_DIR = Path("models/summarizer/final")
BASE_MODEL = "t5-small"
TASK_PREFIX = "summarize: "
MAX_TARGET_TOKENS = 64
MAX_INPUT_TOKENS = 512


class Summarizer:
    """Callable wrapper: `summarizer(text) -> summary string`."""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.eval()

    def __call__(self, text: str, max_length: int = MAX_TARGET_TOKENS) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def resolve_model_path() -> str:
    return str(FINE_TUNED_DIR) if FINE_TUNED_DIR.exists() else BASE_MODEL


def load_summarizer(model_path: str | None = None) -> Summarizer:
    """Load the summarizer for the given model path (or the fine-tuned model
    if available, else the base pretrained model).
    """
    return Summarizer(model_path or resolve_model_path())


def summarize(text: str, summarizer: Callable[..., str] | None = None, max_length: int = MAX_TARGET_TOKENS) -> str:
    """Summarize a single piece of report text. Empty/whitespace-only input
    returns an empty string rather than calling the model.
    """
    text = (text or "").strip()
    if not text:
        return ""

    summarizer = summarizer or load_summarizer()
    return summarizer(TASK_PREFIX + text, max_length=max_length)
