import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import tempfile
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import glob

from object.testing.solver import Solver


def render():
    st.title("🔍 Solve CAPTCHA")
    st.markdown(
        "Upload a `.html` CAPTCHA file to extract tile images, solve using the trained model, "
        "and visualize predictions vs. actual matches."
    )

    uploaded_file = st.file_uploader("Upload CAPTCHA HTML File", type=["html"], accept_multiple_files=False)

    if uploaded_file:
        # Save HTML file to temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            html_path = tmp_file.name

        st.markdown(f"---\n### Solving `{uploaded_file.name}`")

        solver = Solver()
        predictions, confidences, all_tile_ids = solver.solve(html_path, return_confidence=True)

        # Load actual ground truth
        actual_path = html_path.replace(".html", "_sol.txt")
        actual_ids = []
        if os.path.exists(actual_path):
            with open(actual_path, "r") as f:
                actual_ids = list(map(int, f.read().strip().split(",")))

        # Load images
        image_dir = html_path.replace(".html", "") + "_images"
        tile_images = sorted(glob.glob(f"{image_dir}/*.png"))

        if not tile_images:
            st.error("No tiles extracted from CAPTCHA.")
            return

        records = []
        for tile_path in tile_images:
            tile_id = int(os.path.basename(tile_path).replace(".png", ""))
            predicted = tile_id in predictions
            actual = tile_id in actual_ids
            label = "✅ MATCH" if predicted and actual else "❌ MISS"
            confidence = confidences.get(tile_id, "-")
            records.append({
                "ID": tile_id,
                "Filename": os.path.basename(tile_path),
                "Prediction": predicted,
                "Actual": actual,
                "Label": label,
                "Confidence": round(confidence, 3) if isinstance(confidence, float) else "-",
                "Image Path": tile_path
            })

        df = pd.DataFrame(records)
        if df.empty:
            st.warning("No tiles found for prediction display.")
            return

        df = df.sort_values("ID")

        # Tile Grid Preview
        st.subheader("CAPTCHA Tile Preview")
        cols = st.columns(4)
        for i, row in df.iterrows():
            with cols[i % 4]:
                st.image(Image.open(row["Image Path"]), caption=f"ID {row['ID']}", use_container_width=True)

        # Prediction Result
        st.markdown("---")
        st.subheader("Predicted Matching Tile IDs")
        st.success(f"Predicted: `{sorted(predictions)}`")
        if actual_ids:
            st.info(f"Actual: `{sorted(actual_ids)}`")
        else:
            st.warning("No ground truth found.")

        # CSV Export
        st.download_button(
            label="📥 Download Predictions CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{uploaded_file.name}_predictions.csv",
            mime="text/csv"
        )

        # Bar Chart
        st.markdown("---")
        st.subheader("Prediction Accuracy Metrics")
        match_count = sum((df["Prediction"]) & (df["Actual"]))
        miss_count = sum((df["Prediction"]) & (~df["Actual"]))
        missed_truth = sum((~df["Prediction"]) & (df["Actual"]))

        metric_df = pd.DataFrame({
            "Type": ["Correct Match", "Incorrect Prediction", "Missed Ground Truth"],
            "Count": [match_count, miss_count, missed_truth]
        })

        st.bar_chart(metric_df.set_index("Type"))

        # Missed Tiles
        st.markdown("---")
        st.subheader("Mismatched Tiles")
        mismatches = df[df["Label"] == "❌ MISS"]
        if mismatches.empty:
            st.success("✅ No mismatches!")
        else:
            for _, row in mismatches.iterrows():
                st.markdown(f"**ID:** {row['ID']} | **Conf:** `{row['Confidence']}`")
                try:
                    st.image(Image.open(row["Image Path"]), width=100, caption=row["Filename"])
                except Exception:
                    st.warning(f"Cannot display: {row['Image Path']}")

        # Cleanup
        shutil.rmtree(image_dir, ignore_errors=True)
        os.remove(html_path)
