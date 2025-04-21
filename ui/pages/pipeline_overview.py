import os
import random
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from imutils import paths

import object.config as config


def load_random_image_from_folder(folder_path):
    """Returns a random image path from a folder (recursively)."""
    image_files = list(paths.list_images(folder_path))
    return random.choice(image_files) if image_files else None


def render():
    st.title("📈 Pipeline Overview")
    st.markdown("""
    This page offers a step-by-step walkthrough of the CAPTCHA solving pipeline.  
    It helps users visualize how raw images are processed and used to train, validate, and test the AI model.
    """)

    # --- Step 1: Input Image ---
    st.header("🔹 Step 1: Raw CAPTCHA Tile (Input Image)")
    raw_img_path = load_random_image_from_folder(config.RAW_DIR)
    if raw_img_path:
        st.image(raw_img_path, caption="Sample Raw Image", use_container_width=True)
    else:
        st.warning("No raw training images found. Please check your raw data folder.")

    # --- Step 2: Preprocessing & Augmentations ---
    st.header("🔹 Step 2: Image Preprocessing & Augmentation")
    col1, col2, col3 = st.columns(3)

    if raw_img_path:
        image = Image.open(raw_img_path).convert("RGBA")

        with col1:
            st.markdown("**Original Image**")
            st.image(image, use_container_width=True)

        with col2:
            rotated = image.rotate(90)
            st.markdown("**Rotated (Augmented)**")
            st.image(rotated, use_container_width=True)

        with col3:
            noisy = image.point(lambda x: max(0, min(255, x + random.randint(-30, 30))))
            st.markdown("**Noisy/Blurred (Augmented)**")
            st.image(noisy, use_container_width=True)

    # --- Step 3: Dataset Splitting ---
    st.header("🔹 Step 3: Dataset Splitting")
    split_labels = ['Training', 'Validation']
    sizes = [config.SPLIT_RATE, 1 - config.SPLIT_RATE]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=split_labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # --- Step 4: Training vs Validation Data ---
    st.header("🔹 Step 4: How Model Trains and Validates")
    col1, col2 = st.columns(2)

    train_img = load_random_image_from_folder(config.TRAIN_DIR)
    val_img = load_random_image_from_folder(config.VAL_DIR)

    with col1:
        if train_img:
            st.markdown("**📚 Sample Training Image**")
            st.image(train_img, use_container_width=True)
        st.markdown("""
        - Used by the model to **learn patterns** (features).
        - Trained on many rotated and augmented samples.
        """)

    with col2:
        if val_img:
            st.markdown("**🧪 Sample Validation Image**")
            st.image(val_img, use_container_width=True)
        st.markdown("""
        - Used to **evaluate performance** during training.
        - Helps prevent overfitting.
        """)

    # --- Step 5: Prediction Pipeline ---
    st.header("🔹 Step 5: Solving a CAPTCHA")
    st.markdown("""
    - A `.html` CAPTCHA file is uploaded.
    - The system extracts its tile images.
    - The CNN model classifies each tile into a category (e.g., 'bus').
    - It selects the tiles matching the target class.
    """)

    st.info("💡 Try it yourself under the **🔍 Solve CAPTCHA** tab!")

    # --- Glossary ---
    with st.expander("📘 Glossary"):
        st.markdown("""
        - **CAPTCHA**: A test to distinguish human users from bots using images.
        - **Tile**: Each square image within a CAPTCHA grid.
        - **Augmentation**: Enhancing training data by applying transformations.
        - **Validation**: Data used to tune model performance but not used in training.
        - **CNN**: A Convolutional Neural Network, great for image recognition.
        """)
