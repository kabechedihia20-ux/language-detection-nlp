from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.predict import load_pipeline, predict_language_app

st.set_page_config(page_title="Language Detection", page_icon="🌍", layout="centered")

st.title("Détection de langue")
st.caption("Modèle : BiLSTM + SentencePiece")

@st.cache_resource
def init_pipeline():
    return load_pipeline(str(PROJECT_ROOT))

model, sp, id2label, device = init_pipeline()

text = st.text_area(
    "Texte à analyser",
    height=180,
    placeholder="Collez un texte ici (idéalement une phrase complète ou un paragraphe)."
)

col1, col2, col3 = st.columns(3)
with col1:
    top_k = st.selectbox("Top-K", [1, 3, 5], index=1)
with col2:
    min_chars = st.number_input("Min caractères", min_value=5, max_value=300, value=20, step=5)
with col3:
    conf_thresh = st.slider("Seuil confiance", min_value=0.10, max_value=0.95, value=0.50, step=0.05)

if st.button("Détecter"):
    result = predict_language_app(
        text=text,
        model=model,
        sp=sp,
        id2label=id2label,
        device=device,
        top_k=int(top_k),
        min_chars=int(min_chars),
        confidence_threshold=float(conf_thresh)
    )

    if result["status"] == "too_short":
        st.warning(result["message"])
    else:
        best = result["predictions"][0]
        st.subheader(f"Langue détectée : {best['language']}")
        st.metric("Confiance", f"{best['confidence']*100:.2f}%")

        if result["status"] == "low_confidence":
            st.warning("Confiance faible : le texte peut être ambigu ou trop court.")

        st.write("Détails (Top-K) :")
        st.table(result["predictions"])

st.divider()
st.caption(f"Exécution sur : {device}")
