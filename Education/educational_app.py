import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from src.data.make_dataset import load_data, preprocess_data
from src.features.build_features import split_features_labels, train_test_split_data
from src.models.train_model import train_xgboost_model, evaluate_model
from src.visualization.visualize import plot_feature_importance
from src.llm.llm_consultant import consult_llm_with_metrics
from src.predict.inference import load_model, evaluate_on_unseen

# Load OpenRouter API key for cloud-compatible LLM
os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

st.set_page_config(layout="wide")
st.title("🎓 School Dropout Prediction Dashboard")

# 🔹 Section 1: Load and Preview Base Data
df = load_data("data/raw/school_data.csv")
df = preprocess_data(df)

st.subheader("1️⃣ Dataset Preview")
st.dataframe(df.head())

# 🔹 Section 2: Model Training and Base Evaluation
X, y = split_features_labels(df)
X_train, X_test, y_train, y_test = train_test_split_data(X, y)

model = train_xgboost_model(X_train, y_train)
metrics = evaluate_model(model, X_test, y_test)

st.subheader("2️⃣ Model Evaluation Metrics")
for key, value in metrics.items():
    st.metric(label=key.capitalize(), value=f"{value:.2f}")

# 🔹 Section 3: LLM Recommendations (Base Model)
st.subheader("3️⃣ LLM Recommendations Based on Model Metrics")
if st.button("💡 Get LLM Insights"):
    llm_response = consult_llm_with_metrics(metrics, X.columns.tolist())
    st.text_area("LLM Recommendations", llm_response, height=400)

# 🔹 Section 4: Inference & Evaluation on Uploaded Data
st.subheader("4️⃣ Inference and Evaluation on New Data")

uploaded_file = st.file_uploader("📂 Upload new labeled CSV (must include `target` column)", type=["csv"])

if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded Data Preview")
    st.dataframe(new_data.head())

    try:
        loaded_model = load_model()
        result_df, uploaded_metrics = evaluate_on_unseen(loaded_model, new_data)

        st.write("✅ Predictions on Uploaded Data")
        st.dataframe(result_df.head())

        st.subheader("📊 Evaluation Metrics on Uploaded Data")
        for key, value in uploaded_metrics.items():
            st.metric(label=key.capitalize(), value=f"{value:.2f}")

        if st.button("🧠 Consult LLM Based on Uploaded Data"):
            llm_response = consult_llm_with_metrics(uploaded_metrics, result_df.columns.tolist())
            st.text_area("LLM Recommendations", llm_response, height=400)

    except Exception as e:
        st.error(f"❌ Error during inference or metric computation: {e}")
