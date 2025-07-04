import streamlit as st
import os
import pandas as pd
import json

# Load API key for OpenRouter LLM
os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

# App Config
st.set_page_config(page_title="AI Consultant", layout="wide")
st.title("🤖 AI Consultant App")

# Mode Selector
mode = st.sidebar.selectbox("Select App Mode", ["🥑 Avocado Market", "🎓 School Dropout Predictor"])

# ---------------- Avocado Market Analysis ----------------
if mode == "🥑 Avocado Market":
    from avocado.src.data.make_dataset import load_raw_data
    from avocado.src.features.build_features import preprocess_data
    from avocado.src.models.train_model import train_lgb_model, load_trained_model, make_predictions
    from avocado.src.llm.llm_consultant import consult_llm_with_metrics

    st.header("🥑 Avocado Market: Price & Sales Analysis")

    model_path = "avocado/models/model_lgbm.pkl"

    # st.subheader("1️⃣ Train Model")
    #train_file = st.file_uploader("Upload training CSV file", type="csv", key="avocado_train")
   # if train_file:
    #    df = pd.read_csv(train_file)
    #else:
    df = load_raw_data("avocado/data/raw/avocados.csv")

    X, y = preprocess_data(df)
    feature_columns = X.columns.tolist()
    rmse, r2 = train_lgb_model(X, y, model_path)
    #st.success(f"✅ Model trained\nRMSE: {rmse:.4f} | R²: {r2:.4f}")

    # if st.button("🧠 Get LLM Insights (Training Data)"):
    #     metrics = {"RMSE": rmse, "R2 Score": r2}
    #     llm_response = consult_llm_with_metrics(metrics, feature_columns)
    #     st.markdown("### 📌 LLM Recommendations:")
    #     st.write(llm_response)

    st.subheader("Upload your data here")
    infer_file = st.file_uploader("Upload CSV for prediction", type="csv", key="avocado_infer")
    if infer_file:
        new_df = pd.read_csv(infer_file)
        new_X, _ = preprocess_data(new_df, training=False)

        model = load_trained_model(model_path)
        feature_path = model_path.replace(".pkl", "_features.json")
        with open(feature_path, "r") as f:
            trained_features = json.load(f)

        for col in trained_features:
            if col not in new_X.columns:
                new_X[col] = 0
        new_X = new_X[trained_features]

        preds = make_predictions(model, new_X)
        new_df["Predicted AveragePrice"] = preds

        # st.markdown("### 📊 Predictions")
        # st.dataframe(new_df)

        pred_metrics = {
            "Predicted Mean Price": float(preds.mean()),
            "Predicted Min Price": float(preds.min()),
            "Predicted Max Price": float(preds.max()),
            "Predicted Std Dev": float(preds.std())
        }

        if st.button("🧠 Get data driven Insights "):
            llm_pred_response = consult_llm_with_metrics(pred_metrics, trained_features)
            st.markdown("### 📌 Recommendations:")
            st.write(llm_pred_response)

# ---------------- School Dropout Prediction ----------------
elif mode == "🎓 School Dropout Predictor":
    from Education.src.data.make_dataset import load_data, preprocess_data
    from Education.src.features.build_features import split_features_labels, train_test_split_data
    from Education.src.models.train_model import train_xgboost_model, evaluate_model
    from Education.src.visualization.visualize import plot_feature_importance
    from Education.src.llm.llm_consultant import consult_llm_with_metrics
    from Education.src.predict.inference import load_model, evaluate_on_unseen

    st.header("🎓 Keeping Students on Track: Evidence-Based Retention Strategies")

    # Section 1: Load and show data
    df = load_data("Education/data/raw/school_data.csv")
    df = preprocess_data(df)
    # st.subheader("1️⃣ Dataset Preview")
    # st.dataframe(df.head())

    # Section 2: Train + Evaluate
    X, y = split_features_labels(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    model = train_xgboost_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    # # st.subheader("2️⃣ Model Evaluation")
    # for key, value in metrics.items():
    #     st.metric(label=key.capitalize(), value=f"{value:.2f}")

    # # Section 3: LLM Recommendations
    # st.subheader("3️⃣ LLM Recommendations")
    # if st.button("💡 Get LLM Insights (Base Model)"):
    #     llm_response = consult_llm_with_metrics(metrics, X.columns.tolist())
    #     st.text_area("LLM Recommendations", llm_response, height=400)

    # Section 4: Inference on New Data
    st.subheader("Upload your data here")
    uploaded_file = st.file_uploader("📂 Upload labeled CSV", type=["csv"], key="edu_infer")
    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)
        #st.write("📄 Uploaded Data Preview")
        #st.dataframe(new_data.head())

        try:
            loaded_model = load_model()
            result_df, uploaded_metrics = evaluate_on_unseen(loaded_model, new_data)

            #st.write("✅ Predictions on Uploaded Data")
            #st.dataframe(result_df.head())

            # st.subheader("📊 Evaluation Metrics")
            # for key, value in uploaded_metrics.items():
            #     st.metric(label=key.capitalize(), value=f"{value:.2f}")

            if st.button("🧠 Get data driven Insights"):
                llm_response = consult_llm_with_metrics(uploaded_metrics, result_df.columns.tolist())
                st.text_area("Recommendations", llm_response, height=400)

        except Exception as e:
            st.error(f"❌ Error during inference: {e}")
