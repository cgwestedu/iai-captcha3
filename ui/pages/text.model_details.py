import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import plot_model
import text.config as config
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def calculate_flops(model):
    try:
        concrete_func = tf.function(lambda x: model(x))
        concrete_func = concrete_func.get_concrete_function(
            tf.TensorSpec([1, 35, 35, 1], tf.float32)  #May need changed
        )
        frozen_func = convert_variables_to_constants_v2(concrete_func)
        graph_def = frozen_func.graph.as_graph_def()

        with tf.Graph().as_default() as graph:
            tf.compat.v1.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            return flops.total_float_ops
    except Exception as e:
        return f"Unavailable ({e})"

# --- Render UI ---
def render():
    st.title("📊 Text CAPTCHA Model Details")

    # Find latest model
    model_files = sorted([f for f in os.listdir(config.MODELS_DIR) if f.endswith(".keras")])
    if not model_files:
        st.error("No trained models found.")
        return

    latest_model = model_files[-1]
    model_path = os.path.join(config.MODELS_DIR, latest_model)

    st.markdown(f"### Loaded Model: `{latest_model}`")

    # Load model
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # Model summary
    with st.expander("📋 Model Summary"):
        summary = []
        model.summary(print_fn=lambda x: summary.append(x))
        st.text("\n".join(summary))

    # Model diagram
    try:
        arch_path = os.path.join(config.MODELS_DIR, f"{latest_model}_arch.png")
        plot_model(model, to_file=arch_path, show_shapes=True, show_layer_names=True)
        st.image(arch_path, caption="Model Architecture", use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render model diagram: {e}")

    # Parameter counts
    st.subheader("⚙️ Model Parameters")
    try:
        trainable = model.count_params()
        non_trainable = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
        total = trainable + non_trainable
        layers = len(model.layers)
        flops = calculate_flops(model)
        flops_str = f"{flops:,}" if isinstance(flops, int) else flops

        st.markdown(f"- **Trainable Parameters**: `{trainable:,}`")
        st.markdown(f"- **Non-trainable Parameters**: `{non_trainable:,}`")
        st.markdown(f"- **Total Layers**: `{layers}`")
        st.markdown(f"- **Estimated FLOPs**: `{flops_str}`")
    except Exception as e:
        st.warning(f"Error computing model metrics: {e}")

    # Training history
    history_name = latest_model.replace("model_", "history_").replace(".keras", ".csv")
    history_path = os.path.join(config.MODELS_DIR, history_name)
    if os.path.exists(history_path):
        df = pd.read_csv(history_path)
        st.subheader("📈 Training Metrics")
        st.line_chart(df[["accuracy", "val_accuracy"]])
        st.line_chart(df[["loss", "val_loss"]])

        with st.expander("Full Metrics Table"):
            st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)
    else:
        st.info("No training history found for this model.")
