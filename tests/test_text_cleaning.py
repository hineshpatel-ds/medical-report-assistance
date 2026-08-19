import pandas as pd

from src.preprocessing.text_cleaning import clean_dataframe_column, clean_text


def test_clean_text_lowercases_and_strips_punctuation_and_numbers():
    result = clean_text("Patient, age 45, has Diabetes!!")
    assert result == result.lower()
    assert not any(ch.isdigit() for ch in result)
    assert "," not in result and "!" not in result


def test_clean_text_removes_stopwords():
    result = clean_text("the patient is presenting with a history of asthma")
    tokens = result.split()
    assert "the" not in tokens
    assert "is" not in tokens
    assert "a" not in tokens


def test_clean_text_lemmatizes():
    # WordNetLemmatizer defaults to noun POS, so "diagnoses" -> "diagnosis";
    # "running" (default noun reading) is left as-is.
    result = clean_text("running diagnoses")
    tokens = result.split()
    assert "diagnosis" in tokens
    assert "diagnoses" not in tokens


def test_clean_text_empty_string_returns_empty():
    assert clean_text("") == ""


def test_clean_dataframe_column_adds_processed_column():
    df = pd.DataFrame({"note_text": ["Patient has Diabetes.", "Routine check-up, no complaints."]})
    result = clean_dataframe_column(df, "note_text")

    assert "processed_text" in result.columns
    assert len(result) == len(df)
    assert result["processed_text"].iloc[0] != df["note_text"].iloc[0]
