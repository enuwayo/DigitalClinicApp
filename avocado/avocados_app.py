import streamlit as st
import os
import pandas as pd
import json
from src.data.make_dataset import load_raw_data
from src.features.build_features import preprocess_data
from src.models.train_model import train_lgb_model, load_trained_model, make_predictions
from src.llm.llm_consultant import consult_llm_with_metrics

# Load API key securely
os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

st.set_page_config(page_title="Avocado Market Insights", layout="wide")
st.title("🥑 Avocado Market: Price & Sales Analysis + LLM Insights")

model_path = "models/model_lgbm.pkl"

# ---------- 1️⃣ Train Model Section ----------
st.header("1️⃣ Train LightGBM Model")

train_file = st.file_uploader("Upload training CSV file", type="csv", key="train_csv")
if train_file:
    df = pd.read_csv(train_file)
else:
    df = load_raw_data("data/raw/avocados.csv")

X, y = preprocess_data(df)
feature_columns = X.columns.tolist()

rmse, r2 = train_lgb_model(X, y, model_path)
st.success(f"✅ Model trained and saved as model_lgbm.pkl\nRMSE: {rmse:.4f} | R²: {r2:.4f}")

if st.button("🧠 Get LLM Insights (Training Data)"):
    metrics = {"RMSE": rmse, "R2 Score": r2}
    llm_response = consult_llm_with_metrics(metrics, feature_columns)
    st.markdown("### 📌 LLM Recommendations:")
    st.write(llm_response)

# ---------- 2️⃣ Inference Section ----------
st.header("2️⃣ Run Inference on New Data with Saved Model")

infer_file = st.file_uploader("Upload new dataset (CSV) for prediction", type="csv", key="infer_csv")
if infer_file:
    new_df = pd.read_csv(infer_file)
    new_X, _ = preprocess_data(new_df, training=False)

    # Load trained model
    model = load_trained_model(model_path)

    # Load feature list used during training
    feature_path = model_path.replace(".pkl", "_features.json")
    with open(feature_path, "r") as f:
        trained_features = json.load(f)

    # Align columns: add missing, drop extra, order correctly
    for col in trained_features:
        if col not in new_X.columns:
            new_X[col] = 0
    new_X = new_X[trained_features]

    # Predict
    preds = make_predictions(model, new_X)
    new_df["Predicted AveragePrice"] = preds

    st.markdown("### 📊 Predictions:")
    st.dataframe(new_df)

    # LLM insights from prediction stats
    pred_metrics = {
        "Predicted Mean Price": float(preds.mean()),
        "Predicted Min Price": float(preds.min()),
        "Predicted Max Price": float(preds.max()),
        "Predicted Std Dev": float(preds.std())
    }

    if st.button("🧠 Get LLM Insights (New Predictions)"):
        llm_pred_response = consult_llm_with_metrics(pred_metrics, trained_features)
        st.markdown("### 📌 LLM Recommendations (Inferred Data):")
        st.write(llm_pred_response)
