from pathlib import Path

from src.data_ingestion.load_mtsamples import RAW_PATH as MTSAMPLES_PATH
from src.data_ingestion.load_mtsamples import load_mtsamples
from src.data_ingestion.load_pubmed import SAMPLE_CSV as PUBMED_SAMPLE_PATH
from src.data_ingestion.load_pubmed import load_pubmed_sample


def test_mtsamples_csv_is_cached_in_repo():
    assert Path(MTSAMPLES_PATH).exists(), "real MTSamples data should be committed to data/raw/"


def test_load_mtsamples_has_expected_columns_and_no_missing_targets():
    df = load_mtsamples()
    assert {"transcription", "description", "medical_specialty"}.issubset(df.columns)
    assert df["transcription"].isna().sum() == 0
    assert df["description"].isna().sum() == 0
    assert len(df) > 1000


def test_pubmed_sample_is_cached_in_repo():
    assert Path(PUBMED_SAMPLE_PATH).exists(), "real PubMed sample should be committed to data/raw/"


def test_load_pubmed_sample_has_expected_columns():
    df = load_pubmed_sample()
    assert {"article", "abstract"}.issubset(df.columns)
    assert len(df) > 0
    assert df["article"].isna().sum() == 0
