"""Reusable text-cleaning pipeline: lowercase -> strip punctuation/numbers ->
tokenize -> remove stopwords -> lemmatize -> rejoin.

Refactored out of notebooks/04_text_preprocessing.ipynb so the same cleaning
logic can be applied to any text column (synthetic clinical notes or the real
MTSamples transcriptions) from both notebooks and scripts.
"""
import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_NLTK_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def ensure_nltk_resources() -> None:
    """Download the NLTK corpora this module needs, if not already present."""
    for package, path in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


def clean_text(text: str, stop_words: set[str] | None = None, lemmatizer: WordNetLemmatizer | None = None) -> str:
    """Apply the lowercase -> depunctuate -> tokenize -> destopword -> lemmatize
    pipeline to a single string of text.
    """
    if stop_words is None or lemmatizer is None:
        ensure_nltk_resources()
        stop_words = stop_words or set(stopwords.words("english"))
        lemmatizer = lemmatizer or WordNetLemmatizer()

    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}0-9]", " ", text)
    tokens = [word for word in text.split() if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


def clean_dataframe_column(df: pd.DataFrame, column: str, out_column: str = "processed_text") -> pd.DataFrame:
    """Return a copy of `df` with `out_column` holding the cleaned text of `column`."""
    ensure_nltk_resources()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    df = df.copy()
    df[out_column] = df[column].astype(str).apply(lambda t: clean_text(t, stop_words, lemmatizer))
    return df
