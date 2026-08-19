"""Load a real sample from the PubMed summarization dataset: full-text PubMed
articles paired with their real published abstracts.

Source: https://huggingface.co/datasets/ccdv/pubmed-summarization

Used as an independent, out-of-domain evaluation set for the summarizer —
MTSamples (clinical notes -> one-line description) is what the model is
fine-tuned on; PubMed (research article -> abstract) checks whether the
summarization behaviour generalizes beyond that one source.
"""
from pathlib import Path

import pandas as pd
import requests

PUBMED_VALIDATION_URL = (
    "https://huggingface.co/datasets/ccdv/pubmed-summarization/"
    "resolve/main/document/validation-00000-of-00001.parquet"
)
RAW_DIR = Path("data/raw")
RAW_PARQUET = RAW_DIR / "pubmed_validation_full.parquet"
SAMPLE_CSV = RAW_DIR / "pubmed_summarization_sample.csv"


def download_pubmed_validation(dest: Path = RAW_PARQUET, url: str = PUBMED_VALIDATION_URL) -> Path:
    """Download the real PubMed validation shard (~55MB) if not already cached."""
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def load_pubmed_sample(n: int = 300, seed: int = 42, dest: Path = SAMPLE_CSV) -> pd.DataFrame:
    """Load a random sample of `n` real (article, abstract) pairs, caching the
    sampled subset as a small CSV so repeat runs skip the 55MB parquet download.
    """
    if dest.exists():
        return pd.read_csv(dest)

    download_pubmed_validation()
    full = pd.read_parquet(RAW_PARQUET, columns=["article", "abstract"])
    sample = full.sample(n=min(n, len(full)), random_state=seed).reset_index(drop=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(dest, index=False)
    return sample


if __name__ == "__main__":
    data = load_pubmed_sample()
    print(f"Loaded {len(data)} real PubMed article/abstract pairs")
    print(data["abstract"].iloc[0][:300])
