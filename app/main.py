"""Streamlit app: paste or upload a medical report and get a plain-language
summary from the fine-tuned t5-small summarizer (src/inference/summarizer.py).
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_ingestion.load_mtsamples import load_mtsamples  # noqa: E402
from src.inference.summarizer import load_summarizer, resolve_model_path, summarize  # noqa: E402

st.set_page_config(page_title="Medical Report Assistant", page_icon="🩺", layout="centered")


@st.cache_resource
def get_summarizer():
    return load_summarizer()


@st.cache_data
def get_example_reports(n: int = 5):
    df = load_mtsamples()
    sample = df.sample(n=n, random_state=7)
    return dict(zip(sample["sample_name"], sample["transcription"]))


st.title("🩺 Medical Report Assistant")
st.caption(
    "Summarizes clinical report text with a t5-small model fine-tuned on the real "
    "MTSamples clinical transcription dataset. Educational demo only — not medical advice, "
    "not for use on real patient data."
)

model_path = resolve_model_path()
if "final" in model_path:
    st.success("Using the fine-tuned MTSamples model.")
else:
    st.warning(
        "No fine-tuned model found — falling back to the pretrained t5-small baseline. "
        "Run `python -m src.modeling.train_summarizer` to fine-tune on real data."
    )

examples = get_example_reports()
choice = st.selectbox("Load a real example report (from MTSamples)", ["-- none --", *examples.keys()])

default_text = examples[choice] if choice != "-- none --" else ""
report_text = st.text_area("Medical report text", value=default_text, height=280)

if st.button("Summarize", type="primary", disabled=not report_text.strip()):
    with st.spinner("Summarizing..."):
        summarizer = get_summarizer()
        result = summarize(report_text, summarizer=summarizer)
    st.subheader("Summary")
    st.write(result if result else "_Nothing to summarize._")
