def render():
    import streamlit as st
    import os
    import pandas as pd
    from glob import glob
    import object.config as config

    # --- UI Header ---
    st.header("🧠 Train Model")
    st.markdown("""
    Launch CNN model training on the preprocessed CAPTCHA dataset.  
    Visualize accuracy, loss, and learning rate schedules post-training.
    """)

    # --- Trigger training from UI ---
    if st.button("Start Training"):
        with st.spinner("Training in progress..."):
            # Run the trainer script (ensure this works relative to project root)
            exit_code = os.system("PYTHONPATH=. python object/training/trainer.py")
            if exit_code != 0:
                st.error("raining failed. Check console logs.")
            else:
                st.success("Training completed successfully!")

    # --- Load latest training history ---
    history_files = sorted(glob(os.path.join(config.MODELS_DIR, "history_*.csv")))
    if not history_files:
        st.info("No training history found yet.")
        return

    latest_history = history_files[-1]
    df = pd.read_csv(latest_history)

    # --- Plot accuracy and loss ---
    st.subheader("📈 Accuracy and Loss")
    st.line_chart(df[["accuracy", "val_accuracy"]], use_container_width=True)
    st.line_chart(df[["loss", "val_loss"]], use_container_width=True)

    # --- Learning rate chart (if available) ---
    lr_col = next((col for col in ["learning_rate", "lr"] if col in df.columns), None)
    if lr_col:
        st.subheader("📉 Learning Rate")
        st.line_chart(df[[lr_col]], use_container_width=True)

    # --- Full table of metrics ---
    with st.expander("Full Training Metrics Table"):
        st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)
