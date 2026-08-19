"""Fine-tune a small T5 model to summarize real clinical transcriptions.

Trains on MTSamples (src/data_ingestion/load_mtsamples.py): the real
`transcription` field is the input, the real `description` field (a short,
human-written one-liner) is the summarization target.

t5-small (~60M params) is used deliberately so this fine-tune runs on a
single consumer GPU with 4GB VRAM (tested on a GTX 1650 Ti) in well under an
hour, rather than requiring a datacenter GPU. See README for the resulting
ROUGE scores and honest limitations.

Usage:
    python -m src.modeling.train_summarizer
"""
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.data_ingestion.load_mtsamples import load_mtsamples

MODEL_NAME = "t5-small"
OUTPUT_DIR = Path("models/summarizer")
MAX_INPUT_TOKENS = 512
MAX_TARGET_TOKENS = 64
TASK_PREFIX = "summarize: "


def build_dataset(tokenizer, test_size: float = 0.1, seed: int = 42):
    df = load_mtsamples()
    df = df[["transcription", "description"]].dropna()

    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)

    def tokenize(batch):
        inputs = [TASK_PREFIX + t for t in batch["transcription"]]
        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_TOKENS, truncation=True)
        labels = tokenizer(text_target=batch["description"], max_length=MAX_TARGET_TOKENS, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    return tokenized


def build_compute_metrics(tokenizer):
    import evaluate

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(v, 4) for k, v in result.items()}

    return compute_metrics


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    tokenized = build_dataset(tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=3e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_TOKENS,
        fp16=True,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)

    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved fine-tuned model to {final_dir}")


if __name__ == "__main__":
    main()
