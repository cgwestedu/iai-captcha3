import sys
import os

# Append root directory for proper submodule imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st

# Page config must be the FIRST Streamlit command
st.set_page_config(page_title="CAPTCHA Solver", layout="wide")

# Title & Sidebar
st.title("🤖 CAPTCHA Solver Dashboard")
st.sidebar.title("Navigation")

# --- Page Navigation ---
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Landing Page",
        "🧹 Preprocess Data",
        "🧠 Train Model",
        "🧪 Test CAPTCHA",
        "🔍 Solve CAPTCHA",
        "📊 Compare Models",
        "📈 Pipeline Overview",
        "📊 Text - Pipeline Overview",
        "🛠️ Text - Manage Model",
        "📊 Text Model Details"

    ]
)

# --- Page Routing ---
if page == "🏠 Landing Page":
    st.header("📌 Project Overview")
    st.markdown("""
    Welcome to the **CAPTCHA Solver**!  
    This system automates the solution of image-based object recognition CAPTCHAs.

    ### 📂 Tabs:
    - **🧹 Object Preprocess Data**: Prepare and augment the dataset for training.
    - **🧠 Train Model**: Train a CNN on the augmented dataset.
    - **🧪 Test CAPTCHA**: Evaluate performance on held-out HTML CAPTCHA files.
    - **🔍 Solve CAPTCHA**: Upload one or more HTML files and get instant predictions.
    - **📊 Compare Models**: Visualize and compare performance of saved models.
    - **📈 Pipeline Overview**: Visual walkthrough of data → training → inference.

    #### 🟣 **Text-Based CAPTCHA**
    - **🧹 Preprocess Data**: Extract and organize character images from CAPTCHA files.
    - **🧠 Train & Test Model**: Unified tab to preprocess, train, and evaluate in one place.
    - **🔍 Solve CAPTCHA**: Upload a single text CAPTCHA image and get the predicted string.
    - **📊 Model Details**: View architecture, parameters, and training history.
    
    👈 Use the sidebar to switch between sections.
    """)

elif page == "🧹 Object Preprocess Data":
    from ui.pages import preprocess_data
    preprocess_data.render()

elif page == "🧠 Train Model":
    from ui.pages import train_model
    train_model.render()

elif page == "🧪 Test CAPTCHA":
    from ui.pages import test_captcha
    test_captcha.render()

elif page == "🔍 Solve CAPTCHA":
    from ui.pages import solve_captcha
    solve_captcha.render()

elif page == "📊 Compare Models":
    from ui.pages import compare_models
    compare_models.render()

elif page == "📈 Pipeline Overview":
    from ui.pages import pipeline_overview
    pipeline_overview.render()

# ----- TEXT CAPTCHA ROUTING -----

elif page == "📊 Text - Pipeline Overview":
    from text.ui import pipeline_overview
    pipeline_overview.render()

elif page == "🛠️ Text - Manage Model":
    from text.ui import manage_model
    manage_model.render()

elif page == "📊 Text Model Details":
    from text.ui import model_details
    model_details.render()