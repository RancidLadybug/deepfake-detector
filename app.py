"""
app.py
------
Simple local browser app for the image authenticity checker.

Reuses the exact same model-loading and decision logic from detect.py,
just swaps the command-line interface for a Streamlit upload box.

Run with:
    streamlit run app.py

This opens a browser tab at http://localhost:8501
"""

import streamlit as st
from PIL import Image

from detect import load_model, decide_verdict, MODEL_ID

st.set_page_config(page_title="Image Authenticity Checker", page_icon="🔍", layout="centered")


@st.cache_resource(show_spinner="Loading model (first run downloads it, then it's cached)...")
def get_pipeline():
    return load_model()


st.title("🔍 Image Authenticity Checker")
st.caption(
    f"Proof-of-concept using the pretrained model `{MODEL_ID}`. "
    "Not a certified detector — treat results as a screening signal, not "
    "a definitive verdict."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption=uploaded_file.name, use_container_width=True)

    with st.spinner("Analysing..."):
        pipe = get_pipeline()
        results = pipe(image)
        verdict = decide_verdict(results)

    st.subheader(f"Verdict: {verdict['verdict']}")
    st.progress(verdict["confidence"])
    st.write(f"Confidence: **{verdict['confidence']*100:.1f}%**")

    st.markdown("**All scores:**")
    for label, score in sorted(verdict["raw_scores"].items(), key=lambda x: -x[1]):
        st.write(f"{label}: {score*100:.2f}%")

    with st.expander("Raw JSON result"):
        st.json(verdict)
else:
    st.info("Upload a .jpg, .jpeg, .png, or .webp file to analyse.")
