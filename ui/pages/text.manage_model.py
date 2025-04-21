import streamlit as st
import os
import pandas as pd
from PIL import Image

import text.config as config
from text.preprocessing import create_training_data
from text.training import trainer
from text.testing import tester


def render():
    st.title("🛠️ Manage Text CAPTCHA Model")
    st.markdown("This unified interface lets you preprocess data, train a model, and evaluate it all in one place.")

    tab1, tab2, tab3 = st.tabs(["🧹 Preprocess Data", "🧠 Train Model", "🧪 Evaluate Model"])

    # --- Tab 1: Preprocessing ---
    with tab1:
        st.header("🧹 Preprocess CAPTCHA Dataset")

        if st.button("Run Preprocessing"):
            create_training_data.create()
            st.success("✅ Preprocessing complete!")

        # Sample check
        if os.path.exists(config.TRAIN_DIR):
            classes = os.listdir(config.TRAIN_DIR)
            if classes:
                first_class = classes[0]
                class_path = os.path.join(config.TRAIN_DIR, first_class)
                images = os.listdir(class_path)
                if images:
                    image_path = os.path.join(class_path, images[0])
                    st.image(Image.open(image_path), caption=f"Sample: {first_class}", width=150)

        # Class distribution
        st.subheader("📊 Class Distribution")
        class_counts = {
            cls: len(os.listdir(os.path.join(config.TRAIN_DIR, cls)))
            for cls in os.listdir(config.TRAIN_DIR)
            if os.path.isdir(os.path.join(config.TRAIN_DIR, cls))
        }

        if class_counts:
            df = pd.DataFrame(class_counts.items(), columns=["Class", "Count"]).sort_values("Class")
            st.bar_chart(df.set_index("Class"))
            st.dataframe(df)
        else:
            st.info("No training data found.")

    # --- Tab 2: Training ---
    with tab2:
        st.header("🧠 Train CAPTCHA Model")

        if st.button("Start Training"):
            with st.spinner("Training in progress..."):
                trainer.train_model()
            st.success("✅ Training finished!")

    # --- Tab 3: Evaluation ---
    with tab3:
        st.header("🧪 Test CAPTCHA Model")

        if st.button("Run Evaluation"):
            with st.spinner("Testing model..."):
                tester.test()
            st.success("✅ Evaluation complete! Check terminal for output.")
