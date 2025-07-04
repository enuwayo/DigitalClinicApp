
from src.data.make_dataset import load_raw_data
from src.features.build_features import preprocess_data
from src.models.train_model import train_lgb_model
from src.llm.llm_consultant import consult_llm_with_metrics

if __name__ == "__main__":
    filepath = "data/raw/avocados.csv"
    model_path = "models/model_lgbm.pkl"

    # Load and preprocess data
    df = load_raw_data(filepath)
    X, y = preprocess_data(df)
    feature_columns = X.columns.tolist()

    # Train model and get performance metrics
    rmse, r2 = train_lgb_model(X, y, model_path)
    print(f"✅ Model trained successfully\nRMSE: {rmse:.4f} | R²: {r2:.4f}")

    # Prepare metrics dictionary for LLM
    metrics = {
        "RMSE": rmse,
        "R2 Score": r2
    }

    # Get LLM insights
    print("\n🧠 Consulting LLM for business recommendations...\n")
    llm_response = consult_llm_with_metrics(metrics, feature_columns)
    print("📌 LLM Recommendations:\n")
    print(llm_response)

