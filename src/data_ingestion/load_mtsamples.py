"""Load the real MTSamples dataset: de-identified medical transcription samples
paired with a short human-written description, spanning ~40 clinical specialties.

Source: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
(mirrored as CSV at the URL below; original data is public and already de-identified).
"""
from pathlib import Path

import pandas as pd
import requests

MTSAMPLES_URL = "https://raw.githubusercontent.com/socd06/medical-nlp/master/data/mtsamples.csv"
RAW_PATH = Path("data/raw/mtsamples.csv")


def download_mtsamples(dest: Path = RAW_PATH, url: str = MTSAMPLES_URL) -> Path:
    """Download the MTSamples CSV to `dest` if it isn't already cached locally."""
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def load_mtsamples(dest: Path = RAW_PATH) -> pd.DataFrame:
    """Load MTSamples as a DataFrame with columns:
    medical_specialty, sample_name, transcription, description, keywords.

    Rows missing a transcription or description (used as the summarization
    target) are dropped since they can't be used for that task.
    """
    download_mtsamples(dest)
    df = pd.read_csv(dest, index_col=0)
    df.columns = [c.strip() for c in df.columns]
    for col in ["medical_specialty", "sample_name", "transcription", "description", "keywords"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.strip()

    df = df.dropna(subset=["transcription", "description"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    data = load_mtsamples()
    print(f"Loaded {len(data)} real MTSamples clinical transcription records")
    print(data["medical_specialty"].value_counts().head(10))
