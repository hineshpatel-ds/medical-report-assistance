import pytest

from src.inference.summarizer import TASK_PREFIX, load_summarizer, summarize


def _fake_summarizer(calls):
    def run(prompt, max_length):
        calls.append(prompt)
        return "fake summary"

    return run


def test_summarize_empty_input_returns_empty_without_calling_model():
    calls = []
    result = summarize("   ", summarizer=_fake_summarizer(calls))
    assert result == ""
    assert calls == []


def test_summarize_prefixes_text_for_the_model():
    calls = []
    summarize("Patient presents with chest pain.", summarizer=_fake_summarizer(calls))
    assert calls[0] == TASK_PREFIX + "Patient presents with chest pain."


def test_summarize_returns_model_output():
    result = summarize("Some report text.", summarizer=_fake_summarizer([]))
    assert result == "fake summary"


@pytest.mark.slow
def test_load_summarizer_and_summarize_real_model():
    summarizer = load_summarizer("t5-small")
    result = summarize(
        "The patient is a 45 year old male presenting with chest pain and shortness of breath.",
        summarizer=summarizer,
    )
    assert isinstance(result, str)
    assert len(result) > 0
