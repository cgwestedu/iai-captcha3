import streamlit as st
import tempfile
import os
from PIL import Image
import pandas as pd
import numpy as np


from text.testing.solver import Solver

solver = Solver()

def dummy_solver(image_path):
    return solver.solve(image_path)


def render():
    st.title("🔍 Solve Text CAPTCHA")
    st.markdown("""
    Upload a CAPTCHA image. The model will predict the text and display per-character confidence.
    """)

    uploaded_file = st.file_uploader("Upload CAPTCHA Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded CAPTCHA", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(uploaded_file.read())
            image_path = tmp.name

        # ✅ Use your actual solver
        predicted_text, confidences = solver.solve(image_path)

        st.subheader("🧠 Predicted Text")
        st.success(f"Prediction: `{predicted_text}`")

        st.subheader("🔍 Character Confidence")
        characters = list(predicted_text)
        df = pd.DataFrame({
            "Character": characters,
            "Confidence": [round(c, 3) for c in confidences]
        })
        st.dataframe(df, use_container_width=True)

        st.subheader("📊 Confidence by Character")
        st.bar_chart(df.set_index("Character"))

        st.markdown("---")
        st.subheader("📌 Optional: Enter Ground Truth")
        ground_truth = st.text_input("Expected CAPTCHA Text")

        if ground_truth:
            if ground_truth.upper() == predicted_text.upper():
                st.success("✅ Correct prediction!")
            else:
                st.error("❌ Mismatch!")
                st.info(f"Expected: `{ground_truth.upper()}`")

        os.remove(image_path)