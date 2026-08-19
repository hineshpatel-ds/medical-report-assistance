"""Thin inference wrapper around the summarization model, used by the
Streamlit app and by tests. Loads the fine-tuned checkpoint if one has been
trained (see src/modeling/train_summarizer.py); otherwise falls back to the
plain pretrained model so the app still runs before any training happens.
"""
from pathlib import Path

from transformers import Pipeline, pipeline

FINE_TUNED_DIR = Path("models/summarizer/final")
BASE_MODEL = "t5-small"
TASK_PREFIX = "summarize: "
MAX_TARGET_TOKENS = 64


def resolve_model_path() -> str:
    return str(FINE_TUNED_DIR) if FINE_TUNED_DIR.exists() else BASE_MODEL


def load_summarizer(model_path: str | None = None) -> Pipeline:
    """Load a Hugging Face summarization pipeline for the given model path
    (or the fine-tuned model if available, else the base pretrained model).
    """
    model_path = model_path or resolve_model_path()
    return pipeline("summarization", model=model_path, tokenizer=model_path)


def summarize(text: str, summarizer: Pipeline | None = None, max_length: int = MAX_TARGET_TOKENS) -> str:
    """Summarize a single piece of report text. Empty/whitespace-only input
    returns an empty string rather than calling the model.
    """
    text = (text or "").strip()
    if not text:
        return ""

    summarizer = summarizer or load_summarizer()
    output = summarizer(TASK_PREFIX + text, max_length=max_length, truncation=True)
    return output[0]["summary_text"]
