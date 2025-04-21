import os
import random
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from object.preprocessing.create_training_data import create
import object.config as config

def load_sample_image(base_path):
    """Load first available image from the first class folder."""
    if not os.path.exists(base_path):
        return None, None
    classes = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    if not classes:
        return None, None
    for cls in classes:
        cls_dir = os.path.join(base_path, cls)
        images = [f for f in os.listdir(cls_dir) if f.endswith(".png")]
        if images:
            img_path = os.path.join(cls_dir, images[0])
            return Image.open(img_path), cls
    return None, None

def get_class_counts(folder):
    """Returns dictionary of class -> image count"""
    if not os.path.exists(folder):
        return {}
    return {
        cls: len(os.listdir(os.path.join(folder, cls)))
        for cls in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, cls))
    }

def render():
    st.header("Data Preprocessing")
    st.markdown("Run the data preprocessing pipeline and inspect transformations across stages.")

    if st.button("Run Preprocessing"):
        create()
        st.success("✅ Data preprocessed successfully!")

    st.subheader("Sample Image at Each Stage")
    raw_img, raw_cls = load_sample_image(config.RAW_DIR)
    train_img, train_cls = load_sample_image(config.TRAIN_DIR)
    val_img, val_cls = load_sample_image(config.VAL_DIR)

    col1, col2, col3 = st.columns(3)
    if raw_img:
        col1.image(raw_img, caption=f"Raw - {raw_cls}", use_container_width=True)
    else:
        col1.warning("Raw image not found.")

    if train_img:
        col2.image(train_img, caption=f"Augmented/Rotated (Train) - {train_cls}", use_container_width=True)
    else:
        col2.warning("Train image not found.")

    if val_img:
        col3.image(val_img, caption=f"Validation - {val_cls}", use_container_width=True)
    else:
        col3.warning("Validation image not found.")

    st.markdown("---")
    st.subheader("Class Distribution in Training Set")
    train_counts = get_class_counts(config.TRAIN_DIR)
    if train_counts:
        df = pd.DataFrame(train_counts.items(), columns=["Class", "Count"]).sort_values("Count", ascending=False)
        st.bar_chart(data=df.set_index("Class"))
        st.dataframe(df)
    else:
        st.info("Training directory is empty or missing.")
