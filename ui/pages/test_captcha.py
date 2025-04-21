def render():
    import streamlit as st
    import os
    import glob
    import pandas as pd
    from object.testing.solver import Solver
    import object.config as config

    # --- Header ---
    st.header("CAPTCHA Testing")
    st.markdown("Run model evaluation on test CAPTCHA `.html` files. Accuracy is computed based on exact match with ground truth.")

    # --- Button to Start Evaluation ---
    if st.button("Run Test Evaluation"):
        # Locate test files
        test_files = sorted(glob.glob(os.path.join(config.TEST_DIR, "*.html")))
        if not test_files:
            st.warning("No test files found in the test directory.")
            return

        # Get latest trained model
        model_paths = sorted(glob.glob(os.path.join(config.MODELS_DIR, "*.keras")))
        if not model_paths:
            st.error("No trained model found. Please train a model first.")
            return

        model_path = model_paths[-1]
        solver = Solver(network_path=model_path)

        # --- Run Evaluation ---
        results = []
        for html_file in test_files:
            prediction = solver.solve(html_file)

            # Load ground truth solution
            solution_file = html_file.replace(".html", "_sol.txt")
            if not os.path.exists(solution_file):
                continue

            with open(solution_file, "r") as f:
                actual = list(map(int, f.read().strip().split(",")))

            is_correct = set(prediction) == set(actual)

            results.append({
                "Filename": os.path.basename(html_file),
                "Predicted": prediction,
                "Actual": actual,
                "Result": "✅" if is_correct else "❌"
            })

        # --- Display Metrics ---
        df = pd.DataFrame(results)
        accuracy = df["Result"].value_counts(normalize=True).get("✅", 0) * 100

        st.metric("Overall Accuracy", f"{accuracy:.2f}%")
        st.dataframe(df, use_container_width=True)

        # Optional download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Test Results", data=csv, file_name="test_results.csv", mime="text/csv")
