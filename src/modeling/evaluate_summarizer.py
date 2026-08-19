"""Evaluate the fine-tuned summarizer against the pretrained t5-small baseline
on two real, held-out datasets:

1. MTSamples test split (in-domain: clinical transcription -> description)
2. A sample of real PubMed article/abstract pairs (out-of-domain: checks
   whether fine-tuning on clinical notes hurt general summarization ability)

Prints a ROUGE comparison table. Used to write the honest numbers into the
README rather than asserting the model is "good" without evidence.

Usage:
    python -m src.modeling.evaluate_summarizer
"""
from pathlib import Path

import evaluate
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from src.data_ingestion.load_mtsamples import load_mtsamples
from src.data_ingestion.load_pubmed import load_pubmed_sample
from src.modeling.train_summarizer import MAX_INPUT_TOKENS, MAX_TARGET_TOKENS, MODEL_NAME, TASK_PREFIX

FINE_TUNED_DIR = Path("models/summarizer/final")
ROUGE = evaluate.load("rouge")


def summarize_batch(model_path: str, texts: list[str]) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device_map="auto")

    prefixed = [TASK_PREFIX + t for t in texts]
    outputs = summarizer(
        prefixed,
        max_length=MAX_TARGET_TOKENS,
        truncation=True,
        batch_size=4,
    )
    return [o["summary_text"] for o in outputs]


def score(predictions: list[str], references: list[str]) -> dict:
    result = ROUGE.compute(predictions=predictions, references=references, use_stemmer=True)
    return {k: round(v, 4) for k, v in result.items()}


def evaluate_on_mtsamples(model_path: str, n: int = 200) -> dict:
    df = load_mtsamples()
    sample = df.sample(n=min(n, len(df)), random_state=42)
    preds = summarize_batch(model_path, sample["transcription"].tolist())
    return score(preds, sample["description"].tolist())


def evaluate_on_pubmed(model_path: str, n: int = 100) -> dict:
    df = load_pubmed_sample(n=n)
    preds = summarize_batch(model_path, df["article"].tolist())
    return score(preds, df["abstract"].tolist())


def main() -> None:
    rows = []
    models = {"pretrained-t5-small (baseline)": MODEL_NAME}
    if FINE_TUNED_DIR.exists():
        models["fine-tuned-t5-small (ours)"] = str(FINE_TUNED_DIR)
    else:
        print(f"No fine-tuned model found at {FINE_TUNED_DIR}; run train_summarizer.py first.")

    for label, path in models.items():
        mt_scores = evaluate_on_mtsamples(path)
        pubmed_scores = evaluate_on_pubmed(path)
        rows.append({"model": label, "dataset": "MTSamples (in-domain)", **mt_scores})
        rows.append({"model": label, "dataset": "PubMed (out-of-domain)", **pubmed_scores})

    results = pd.DataFrame(rows)
    print(results.to_string(index=False))

    results_path = Path("models/summarizer/eval_results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
