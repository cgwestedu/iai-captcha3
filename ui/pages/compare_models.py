import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.utils import plot_model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import object.config as config

# --------------------- FLOPs Calculator ---------------------
def calculate_flops(model):
    try:
        concrete_func = tf.function(lambda inputs: model(inputs))
        concrete_func = concrete_func.get_concrete_function(
            tf.TensorSpec([1, 64, 64, 4], tf.float32)
        )
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        graph_def = frozen_func.graph.as_graph_def()

        with tf.Graph().as_default() as graph:
            tf.compat.v1.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            flops = tf.compat.v1.profiler.profile(
                graph=graph,
                run_meta=run_meta,
                cmd='op',
                options=opts
            )
            return flops.total_float_ops
    except Exception as e:
        return f"Unavailable ({e})"


# --------------------- Display One Model's Details ---------------------
def display_model_details(selected_model, col):
    model_path = os.path.join(config.MODELS_DIR, selected_model)

    try:
        model = load_model(model_path)
    except Exception as e:
        col.error(f"❌ Failed to load `{selected_model}`: {e}")
        return

    col.markdown("### Model Summary")
    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    col.text("\n".join(summary))

    col.markdown("### Architecture Diagram")
    try:
        plot_path = os.path.join(config.MODELS_DIR, f"{selected_model}_arch.png")
        plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
        col.image(plot_path, caption="Model Architecture", use_container_width=True)
    except Exception as e:
        col.warning(f"Diagram not rendered: {e}")

    col.markdown("### Configuration & Hyperparameters")
    col.markdown(f"- **Epochs**: `{config.EPOCHS}`")
    col.markdown(f"- **Batch Size**: `{config.BATCH_SIZE}`")
    col.markdown("- **Optimizer**: `Adam`")

    col.markdown("### Model Metrics")
    try:
        trainable = model.count_params()
        non_trainable = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
        total = trainable + non_trainable
        layers = len(model.layers)
        flops = calculate_flops(model)
        flops_str = f"{flops:,}" if isinstance(flops, int) else flops

        col.markdown(f"- **Trainable Parameters**: `{trainable:,}`")
        col.markdown(f"- **Non-trainable Parameters**: `{non_trainable:,}`")
        col.markdown(f"- **Total Parameters**: `{total:,}`")
        col.markdown(f"- **Total Layers**: `{layers}`")
        col.markdown(f"- **FLOPs Estimate**: `{flops_str}`")

    except Exception as e:
        col.warning(f"Metric computation failed: {e}")

    timestamp = selected_model.replace("model_", "").replace(".keras", "")
    history_path = os.path.join(config.MODELS_DIR, f"history_{timestamp}.csv")
    col.markdown("### Training History")

    if os.path.exists(history_path):
        df = pd.read_csv(history_path)
        col.success(f"History loaded: `{os.path.basename(history_path)}`")

        fig1, ax1 = plt.subplots()
        ax1.plot(df["accuracy"], label="Train")
        ax1.plot(df["val_accuracy"], label="Validation")
        ax1.set_title("Accuracy Over Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        col.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.plot(df["loss"], label="Train")
        ax2.plot(df["val_loss"], label="Validation")
        ax2.set_title("Loss Over Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        col.pyplot(fig2)

        with col.expander("Full Training Metrics Table"):
            col.dataframe(df.style.highlight_max(axis=0), use_container_width=True)
    else:
        col.warning("No training history found for this model.")


# --------------------- Main Render Function ---------------------
def render():
    st.title("Compare Model Architectures & Metrics")

    model_files = sorted([f for f in os.listdir(config.MODELS_DIR) if f.endswith(".keras")])
    if len(model_files) < 2:
        st.error("At least two trained models are required for comparison.")
        return

    col1, col2 = st.columns(2)
    with col1:
        model_1 = st.selectbox("Select Model 1", model_files, key="model1")
    with col2:
        model_2 = st.selectbox("Select Model 2", model_files, key="model2")

    if model_1 == model_2:
        st.warning("Select two different models to compare.")
        return

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.markdown(f"## 🔍 Model: `{model_1}`")
        display_model_details(model_1, left)
    with right:
        st.markdown(f"## 🔍 Model: `{model_2}`")
        display_model_details(model_2, right)


if __name__ == "__main__":
    render()
